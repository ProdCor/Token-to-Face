#!/usr/bin/env python3
"""
Test trained SpeechTokenizer-based GRU+Diffusion model on a custom audio file
"""
import argparse
import os
import torch
import numpy as np
import torchaudio
from speechtokenizer import SpeechTokenizer
from model import FaceDiffBeatSpeechTokenizer
from utils import create_gaussian_diffusion


def load_speechtokenizer(config_path, ckpt_path, device):
    print(f"Loading SpeechTokenizer from: {ckpt_path}")
    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model = model.to(device)
    model.eval()
    print(f"✓ SpeechTokenizer loaded (sample_rate={model.sample_rate}Hz, "
          f"token_rate={model.sample_rate / model.downsample_rate}Hz)")
    return model


def load_preextracted_tokens(token_path, utt_id, device):
    print(f"\nLoading pre-extracted tokens from: {token_path}")
    utt2token = torch.load(token_path, map_location='cpu')
    
    matches = sorted([k for k in utt2token.keys() if k.startswith(utt_id)])
    if not matches:
        print(f"Available keys (first 10): {list(utt2token.keys())[:10]}")
        raise KeyError(f"Utterance '{utt_id}' not found in {token_path}")
    
    print(f"Found {len(matches)} chunks: {matches}")
    
    def parse_chunk(val):
        if isinstance(val, dict):
            # Old format: {'semantic': (T,), 'acoustic': (n_q-1, T)}
            semantic = val['semantic']  # (T,)
            acoustic = val['acoustic']  # (n_q-1, T)
            if semantic.ndim == 1:
                semantic = semantic[np.newaxis, :]  # (1, T)
            return np.concatenate([semantic, acoustic], axis=0)  # (n_q, T)
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                return val[np.newaxis, :]
            return val  # already (n_q, T)
        else:
            raise ValueError(f"Unexpected token type: {type(val)}")
    
    chunks = [parse_chunk(utt2token[k]) for k in matches]
    tokens = np.concatenate(chunks, axis=1)  # (n_q, T_total)
    tokens = torch.from_numpy(tokens).long().unsqueeze(0).to(device)  # (1, n_q, T)
    print(f"✓ Loaded and concatenated {len(chunks)} chunks → {tokens.shape}")
    return tokens


def detect_model_config(model_path):
    """Detect model configuration from checkpoint weights"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # time_mlp.0.weight: (latent_dim, diff_steps)
    latent_dim = checkpoint['time_mlp.0.weight'].shape[0]
    
    # final_layer.weight: (vertice_dim, gru_dim)
    vertice_dim = checkpoint['final_layer.weight'].shape[0]
    gru_dim = checkpoint['final_layer.weight'].shape[1]
    
    # Count GRU layers
    gru_layers = sum(1 for k in checkpoint.keys() if k.startswith('gru.weight_ih_l'))
    
    # SpeechTokenizer embedder: token_embedding.embeddings.0.weight: (vocab_size, embedding_dim)
    if 'token_embedding.embeddings.0.weight' in checkpoint:
        vocab_size = checkpoint['token_embedding.embeddings.0.weight'].shape[0]
        token_embedding_dim = checkpoint['token_embedding.embeddings.0.weight'].shape[1]
        n_q = sum(1 for k in checkpoint.keys() 
                  if k.startswith('token_embedding.embeddings.') and k.endswith('.weight'))
    else:
        vocab_size = 1024
        token_embedding_dim = 512
        n_q = 8

    return {
        'feature_dim': latent_dim,
        'gru_dim': gru_dim,
        'vertice_dim': vertice_dim,
        'gru_layers': gru_layers,
        'token_embedding_dim': token_embedding_dim,
        'vocab_size': vocab_size,
        'n_q': n_q
    }


@torch.no_grad()
def test_audio(args, model, diffusion, device="cuda"):
    """Generate prediction for audio file"""
    
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
    
    tokens = load_preextracted_tokens(args.token_path, args.utt_id, device)
    token_fps = 50.0  # SpeechTokenizer rate
    num_tokens = tokens.shape[2]
    audio_duration = num_tokens / token_fps
    
    # Dummy conditioning (BEAT2 doesn't use subject conditioning)
    template = torch.zeros(1, args.vertice_dim).to(device)
    one_hot = torch.zeros(1, 8).to(device)
    one_hot[0, 0] = 1.0
    
    # Calculate output shape: tokens at 50Hz, output at 30fps
    num_tokens = tokens.shape[2]
    num_frames = int(audio_duration * args.output_fps)
    shape = (1, num_frames, args.vertice_dim)
    
    print(f"\nInference parameters:")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Input tokens: {num_tokens} @ {token_fps:.0f}Hz, n_q={tokens.shape[1]}")
    print(f"Output frames: {num_frames} @ {args.output_fps}fps")
    print(f"Output shape: {shape}")
    print(f"\nGenerating prediction...\n")

    sample = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs={
            "cond_embed": tokens,   # (1, n_q, T) LongTensor
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
    basename = args.utt_id
    output_file = f"{basename}_epoch{args.epoch}_speechtokenizer_prediction.npy"
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
    parser = argparse.ArgumentParser(description='Test SpeechTokenizer GRU+Diffusion model on custom audio')
    
    # Required
    parser.add_argument("--audio", type=str,
                        help='Path to audio file (.wav)')
    
    # Model configuration (auto-detected if not provided)
    parser.add_argument("--dataset", type=str, default="beat2")
    parser.add_argument("--vertice_dim", type=int, default=None)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--gru_dim", type=int, default=None)
    parser.add_argument("--gru_layers", type=int, default=None)
    parser.add_argument("--token_embedding_dim", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--n_q", type=int, default=None)
    parser.add_argument("--output_fps", type=int, default=30)
    parser.add_argument("--diff_steps", type=int, default=1000)
    
    # Paths
    parser.add_argument("--save_path", type=str, default="save/mj_100ep_GRU")
    parser.add_argument("--result_path", type=str, default="result/mj_100ep_GRU_test")
    parser.add_argument("--token_path", type=str, required=True,
                    help="Path to .pt file with pre-extracted tokens (e.g. test_utt2speech_token.pt)")
    parser.add_argument("--utt_id", type=str, required=True,
                        help="Utterance ID to look up in the token file")
    
    # Testing options
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--skip_steps", type=int, default=0)
    parser.add_argument("--model", type=str, default="face_diffuser_speechtokenizer")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.audio is None and (args.token_path is None or args.utt_id is None):
        parser.error("Either --audio or both --token_path and --utt_id must be provided")
    
    # Auto-detect model config
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
    config = detect_model_config(model_path)
    
    if args.vertice_dim is None:   args.vertice_dim = config['vertice_dim']
    if args.feature_dim is None:   args.feature_dim = config['feature_dim']
    if args.gru_dim is None:       args.gru_dim = config['gru_dim']
    if args.gru_layers is None:    args.gru_layers = config['gru_layers']
    if args.token_embedding_dim is None: args.token_embedding_dim = config['token_embedding_dim']
    if args.vocab_size is None:    args.vocab_size = config['vocab_size']
    if args.n_q is None:           args.n_q = config['n_q']
    
    # Required by create_gaussian_diffusion
    args.codebook_size = args.vocab_size

    print(f"Detected configuration:")
    print(f"  vertice_dim:        {args.vertice_dim}")
    print(f"  feature_dim:        {args.feature_dim}")
    print(f"  gru_dim:            {args.gru_dim}")
    print(f"  gru_layers:         {args.gru_layers}")
    print(f"  token_embedding_dim:{args.token_embedding_dim}")
    print(f"  vocab_size:         {args.vocab_size}")
    print(f"  n_q:                {args.n_q}")
    print(f"{'='*60}")
    
    os.makedirs(args.result_path, exist_ok=True)

    print("\n" + "="*60)
    print("SpeechTokenizer GRU+Diffusion Inference")
    print("="*60)
    print(f"Audio file:  {args.audio}")
    print(f"Model:       {args.model}_{args.dataset}_{args.epoch}.pth")
    print(f"Output:      {args.result_path}/")
    print(f"Output FPS:  {args.output_fps}")
    print(f"Skip steps:  {args.skip_steps}")
    print("="*60)

    device = torch.device(args.device)

    # Create diffusion
    diffusion = create_gaussian_diffusion(args)

    # Create model
    print("\nCreating SpeechTokenizer GRU+Diffusion model...")
    model = FaceDiffBeatSpeechTokenizer(
        args,
        vertice_dim=args.vertice_dim,
        latent_dim=args.feature_dim,
        diffusion_steps=args.diff_steps,
        gru_latent_dim=args.gru_dim,
        num_layers=args.gru_layers,
    )
    model = model.to(device)
    print("✓ Model created\n")

    test_audio(args, model, diffusion, device=args.device)


if __name__ == "__main__":
    main()


'''
python inference.py \
    --token_path speechtokenizer_tokens/test_utt2speech_token.pt \
    --utt_id 1_wayne_0_1_1 \
    --epoch 100
'''