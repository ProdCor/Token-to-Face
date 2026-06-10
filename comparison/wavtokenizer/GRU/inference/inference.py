#!/usr/bin/env python3
"""
Test trained CosyVoice2-based model on a custom audio file
"""
import argparse
import os
import torch
import numpy as np
import torchaudio
import onnxruntime as ort
from model_adapt import FaceDiffBeatCosyVoice2
from utils_adapt import create_gaussian_diffusion


def load_onnx_tokenizer(onnx_path):
    """Load CosyVoice2 ONNX speech tokenizer"""
    print(f"Loading ONNX tokenizer: {onnx_path}")
    
    # Set session options to avoid threading warnings
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 1
    
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']  # Use CPU for tokenizer
    )
    print(f"✓ Tokenizer loaded")
    return session


def extract_cosyvoice2_tokens(audio_path, tokenizer_session, device="cuda"):
    """Extract CosyVoice2 speech tokens from audio file"""
    print(f"\nLoading audio: {audio_path}")
    
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        print(f"Resampling from {sr}Hz to 16000Hz")
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    audio = audio.squeeze().numpy()
    audio_duration = len(audio) / 16000
    print(f"Audio duration: {audio_duration:.2f}s")
    
    # Compute mel spectrogram (matching CosyVoice2 preprocessing)
    mel = compute_mel_spectrogram(audio)
    
    # Extract tokens using ONNX model
    print(f"Extracting speech tokens...")
    print(f"Mel spectrogram shape: {mel.shape}")
    
    # Get input/output names
    input_names = [inp.name for inp in tokenizer_session.get_inputs()]
    output_names = [out.name for out in tokenizer_session.get_outputs()]
    
    print(f"ONNX inputs: {input_names}")
    print(f"ONNX outputs: {output_names}")
    
    # Prepare inputs: feats and feats_length
    feats_length = np.array([mel.shape[2]], dtype=np.int32)  # Length of mel spectrogram (int32!)
    
    tokens = tokenizer_session.run(
        output_names,
        {
            'feats': mel,
            'feats_length': feats_length
        }
    )[0]
    
    # tokens shape: (1, seq_len) - discrete indices
    tokens = torch.from_numpy(tokens).long()
    tokens = tokens.to(device)
    
    token_count = tokens.shape[1]
    token_rate = token_count / audio_duration
    print(f"✓ Extracted {token_count} tokens (~{token_rate:.1f} Hz)")
    print(f"Token shape: {tokens.shape}")
    
    return tokens


def compute_mel_spectrogram(audio, n_fft=512, hop_length=160, n_mels=128):
    """
    Compute mel spectrogram matching CosyVoice2's preprocessing
    
    Args:
        audio: (seq_len,) numpy array at 16kHz
        n_fft: FFT size
        hop_length: Hop length (160 samples = 10ms at 16kHz)
        n_mels: Number of mel bins
    
    Returns:
        mel: (1, n_mels, time) numpy array
    """
    import librosa
    
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=8000
    )
    
    # Convert to log scale
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    
    # Add batch dimension
    mel = mel[np.newaxis, :, :]  # (1, n_mels, time)
    
    return mel.astype(np.float32)


def detect_model_config(model_path):
    """Detect model configuration from checkpoint"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Detect dimensions from weight shapes
    # time_mlp.0.weight shape is (latent_dim, 1000)
    latent_dim = checkpoint['time_mlp.0.weight'].shape[0]
    
    # final_layer.weight shape is (vertice_dim, gru_dim)
    vertice_dim = checkpoint['final_layer.weight'].shape[0]
    gru_dim = checkpoint['final_layer.weight'].shape[1]
    
    # Count GRU layers
    gru_layers = sum(1 for k in checkpoint.keys() if k.startswith('gru.weight_ih_l'))
    
    # Detect token embedding dimension
    if 'token_embedding.weight' in checkpoint:
        token_embedding_dim = checkpoint['token_embedding.weight'].shape[1]
        num_embeddings = checkpoint['token_embedding.weight'].shape[0]
    else:
        token_embedding_dim = 512  # default
        num_embeddings = 6561  # default CosyVoice2 codebook size
    
    return {
        'feature_dim': latent_dim,
        'gru_dim': gru_dim,
        'vertice_dim': vertice_dim,
        'gru_layers': gru_layers,
        'token_embedding_dim': token_embedding_dim,
        'num_embeddings': num_embeddings
    }


@torch.no_grad()
def test_audio(args, model, diffusion, wav_file, tokenizer_session, device="cuda"):
    """Generate prediction for audio file"""
    
    # Load trained model
    model_path = os.path.join(args.save_path, f'{args.model}_{args.dataset}_{args.epoch}.pth')
    
    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found!")
        print(f"\nAvailable models in {args.save_path}:")
        for f in sorted(os.listdir(args.save_path)):
            if f.endswith('.pth'):
                print(f"  - {f}")
        return
    
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded")
    print(f"{'='*60}\n")
    
    # Extract CosyVoice2 tokens
    tokens = extract_cosyvoice2_tokens(wav_file, tokenizer_session, device=device)
    print(f"Tokens shape: {tokens.shape}")
    
    # Create dummy template and one-hot (BEAT2 doesn't use subject conditioning)
    template = torch.zeros(1, args.vertice_dim).to(device)
    one_hot = torch.zeros(1, 8).to(device)
    one_hot[0, 0] = 1.0
    
    # Calculate output shape based on token count and fps conversion
    # Tokens are at 25 Hz, output is at 30 fps
    num_tokens = tokens.shape[1]
    audio_duration = num_tokens / 25.0  # 25 Hz token rate
    num_frames = int(audio_duration * args.output_fps)
    shape = (1, num_frames, args.vertice_dim)
    
    print(f"\nInference parameters:")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Input tokens: {num_tokens} @ 25 Hz")
    print(f"Output frames: {num_frames} @ {args.output_fps} fps")
    print(f"Output shape: {shape}")
    print(f"\nGenerating prediction (this may take a minute)...\n")

    # Generate prediction
    sample = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs={
            "cond_embed": tokens,  # Pass tokens directly (already LongTensor)
            "one_hot": one_hot,
            "template": template,
        },
        skip_timesteps=args.skip_steps,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
        device=device
    )
    
    sample = sample.squeeze().detach().cpu().numpy()

    # Save prediction
    basename = os.path.basename(wav_file).replace('.wav', '').replace('.mp3', '')
    output_file = f"{basename}_epoch{args.epoch}_cosyvoice2_prediction.npy"
    output_path = os.path.join(args.result_path, output_file)
    np.save(output_path, sample)
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS!")
    print(f"{'='*60}")
    print(f"Saved: {output_path}")
    print(f"Shape: {sample.shape}")
    print(f"Frames: {sample.shape[0]}")
    print(f"Blendshapes: {sample.shape[1]}")
    print(f"Value range: [{sample.min():.4f}, {sample.max():.4f}]")
    print(f"Mean: {sample.mean():.4f}, Std: {sample.std():.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Test CosyVoice2 model on custom audio')
    
    # Required
    parser.add_argument("--audio", type=str, required=True,
                        help='Path to audio file (.wav)')
    
    # Model configuration (will be auto-detected if not provided)
    parser.add_argument("--dataset", type=str, default="beat2")
    parser.add_argument("--vertice_dim", type=int, default=None)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--gru_dim", type=int, default=None)
    parser.add_argument("--gru_layers", type=int, default=None)
    parser.add_argument("--token_embedding_dim", type=int, default=None)
    parser.add_argument("--output_fps", type=int, default=30)
    parser.add_argument("--diff_steps", type=int, default=1000)
    
    # Paths
    parser.add_argument("--save_path", type=str, default="save_no_overlap")
    parser.add_argument("--result_path", type=str, default="result_cosyvoice2_no_overlap")
    parser.add_argument("--tokenizer_path", type=str,
                        default="pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx",
                        help='Path to CosyVoice2 ONNX tokenizer')
    
    # Testing options
    parser.add_argument("--epoch", type=int, default=100,
                        help='Which epoch checkpoint to use')
    parser.add_argument("--skip_steps", type=int, default=0,
                        help='Skip diffusion steps for faster generation (0 = no skip)')
    parser.add_argument("--model", type=str, default="face_diffuser")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input_fps", type=int, default=25,
                        help='Token rate (25 Hz for CosyVoice2)')
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio):
        print(f"❌ ERROR: Audio file not found: {args.audio}")
        return
    
    # Validate tokenizer
    if not os.path.exists(args.tokenizer_path):
        print(f"❌ ERROR: Tokenizer not found: {args.tokenizer_path}")
        return
    
    # Auto-detect model configuration
    model_path = os.path.join(args.save_path, f'{args.model}_{args.dataset}_{args.epoch}.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found: {model_path}")
        print(f"\nAvailable models in {args.save_path}:")
        for f in sorted(os.listdir(args.save_path)):
            if f.endswith('.pth'):
                print(f"  - {f}")
        return
    
    print(f"\n{'='*60}")
    print(f"Auto-detecting model configuration...")
    print(f"{'='*60}")
    
    config = detect_model_config(model_path)
    
    # Override with detected values if not provided
    if args.vertice_dim is None:
        args.vertice_dim = config['vertice_dim']
    if args.feature_dim is None:
        args.feature_dim = config['feature_dim']
    if args.gru_dim is None:
        args.gru_dim = config['gru_dim']
    if args.gru_layers is None:
        args.gru_layers = config['gru_layers']
    if args.token_embedding_dim is None:
        args.token_embedding_dim = config['token_embedding_dim']
    
    print(f"Detected configuration:")
    print(f"  vertice_dim: {args.vertice_dim}")
    print(f"  feature_dim: {args.feature_dim}")
    print(f"  gru_dim: {args.gru_dim}")
    print(f"  gru_layers: {args.gru_layers}")
    print(f"  token_embedding_dim: {args.token_embedding_dim}")
    print(f"{'='*60}")
    
    # Create result directory
    os.makedirs(args.result_path, exist_ok=True)

    # Print configuration
    print("\n" + "="*60)
    print("CosyVoice2 Custom Audio Test")
    print("="*60)
    print(f"Audio file: {args.audio}")
    print(f"Model: {args.model}_{args.dataset}_{args.epoch}.pth")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Output: {args.result_path}/")
    print(f"Output FPS: {args.output_fps}")
    print(f"Skip steps: {args.skip_steps}")
    print("="*60)

    # Load tokenizer
    tokenizer_session = load_onnx_tokenizer(args.tokenizer_path)

    # Create diffusion
    diffusion = create_gaussian_diffusion(args)

    # Create model
    print("\nCreating CosyVoice2 model...")
    model = FaceDiffBeatCosyVoice2(
        args,
        vertice_dim=args.vertice_dim,
        latent_dim=args.feature_dim,
        diffusion_steps=args.diff_steps,
        gru_latent_dim=args.gru_dim,
        num_layers=args.gru_layers,
    )
    
    device = torch.device(args.device)
    model = model.to(device)
    print("✓ Model created\n")

    # Test audio
    test_audio(args, model, diffusion, args.audio, tokenizer_session, device=args.device)


if __name__ == "__main__":
    main()


'''
python inference.py --audio chunk_0_audio.wav --epoch 5

'''