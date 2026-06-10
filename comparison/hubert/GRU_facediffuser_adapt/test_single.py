#!/usr/bin/env python3
"""
Test trained model on custom audio with auto-detected config
"""
import argparse
import os
import torch
import numpy as np
import torchaudio
from models import FaceDiffBeat
from utils import create_gaussian_diffusion
from transformers import Wav2Vec2Processor

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
    
    return {
        'feature_dim': latent_dim,
        'gru_dim': gru_dim,
        'vertice_dim': vertice_dim,
        'gru_layers': gru_layers
    }


def load_audio(wav_path, processor):
    """Load and process audio file"""
    print(f"Loading audio: {wav_path}")
    audio, sr = torchaudio.load(wav_path)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        print(f"Resampling from {sr}Hz to 16000Hz")
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Process with Wav2Vec2
    audio = audio.squeeze().numpy()
    audio = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    
    return audio


@torch.no_grad()
def test_audio(args, model, diffusion, wav_file, device="cuda"):
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

    # Load processor
    print(f"Loading Wav2Vec2Processor...")
    processor = Wav2Vec2Processor.from_pretrained(
        args.processor_path,
        local_files_only=True
    )
    print(f"✓ Processor loaded\n")
    
    # Load audio
    audio = load_audio(wav_file, processor)
    audio = audio.to(device)
    print(f"Audio shape: {audio.shape}")
    
    # Create dummy template and one-hot (BEAT2 doesn't use subject conditioning)
    template = torch.zeros(1, args.vertice_dim).to(device)
    one_hot = torch.zeros(1, 8).to(device)
    one_hot[0, 0] = 1.0
    
    # Calculate output shape
    sr = 16000
    audio_duration = audio.shape[-1] / sr
    num_frames = int(audio.shape[-1] / sr * args.output_fps)
    shape = (1, num_frames, args.vertice_dim)
    
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Output frames: {num_frames} @ {args.output_fps} fps")
    print(f"Output shape: {shape}")
    print(f"\nGenerating prediction (this may take a minute)...\n")

    # Generate prediction
    sample = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs={
            "cond_embed": audio,
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
    output_file = f"{basename}_epoch{args.epoch}_prediction.npy"
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
    parser = argparse.ArgumentParser(description='Test model on custom audio')
    
    # Required
    parser.add_argument("--audio", type=str, required=True,
                        help='Path to audio file (.wav)')
    
    # Model configuration (will be auto-detected if not provided)
    parser.add_argument("--dataset", type=str, default="beat2")
    parser.add_argument("--vertice_dim", type=int, default=None)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--gru_dim", type=int, default=None)
    parser.add_argument("--gru_layers", type=int, default=None)
    parser.add_argument("--output_fps", type=int, default=30)
    parser.add_argument("--diff_steps", type=int, default=1000)
    
    # Paths
    parser.add_argument("--save_path", type=str, default="save")
    parser.add_argument("--result_path", type=str, default="result")
    parser.add_argument("--hubert_path", type=str, 
                        default="pretrained_models/hubert/hubert-base-ls960")
    parser.add_argument("--processor_path", type=str,
                        default="pretrained_models/hubert/hubert-xlarge-ls960-ft")
    
    # Testing options
    parser.add_argument("--epoch", type=int, default=100,
                        help='Which epoch checkpoint to use')
    parser.add_argument("--skip_steps", type=int, default=0,
                        help='Skip diffusion steps for faster generation')
    parser.add_argument("--model", type=str, default="face_diffuser")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input_fps", type=int, default=50)
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio):
        print(f"❌ ERROR: Audio file not found: {args.audio}")
        return
    
    # Auto-detect model configuration
    model_path = os.path.join(args.save_path, f'{args.model}_{args.dataset}_{args.epoch}.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found: {model_path}")
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
    
    print(f"Detected configuration:")
    print(f"  vertice_dim: {args.vertice_dim}")
    print(f"  feature_dim: {args.feature_dim}")
    print(f"  gru_dim: {args.gru_dim}")
    print(f"  gru_layers: {args.gru_layers}")
    print(f"{'='*60}")
    
    # Create result directory
    os.makedirs(args.result_path, exist_ok=True)

    # Print configuration
    print("\n" + "="*60)
    print("Custom Audio Test")
    print("="*60)
    print(f"Audio file: {args.audio}")
    print(f"Model: {args.model}_{args.dataset}_{args.epoch}.pth")
    print(f"Output: {args.result_path}/")
    print(f"Output FPS: {args.output_fps}")
    print("="*60)

    # Create diffusion
    diffusion = create_gaussian_diffusion(args)

    # Create model
    print("\nCreating model...")
    model = FaceDiffBeat(
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
    test_audio(args, model, diffusion, args.audio, device=args.device)


if __name__ == "__main__":
    main()