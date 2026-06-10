#!/usr/bin/env python3
"""
Inference script for the BlendshapeDecoder model with SpeechTokenizer.
Generates ARKit blendshape sequences from SpeechTokenizer speech tokens using a trained model.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
import sys
import os
import torchaudio
from speechtokenizer import SpeechTokenizer

from model import BlendshapeDecoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config(config_path):
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_config_file(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    
    config_candidates = [
        checkpoint_dir / 'training_config.yaml',
        checkpoint_dir / 'config.yaml',
        checkpoint_dir / 'config2.yaml',
    ]
    
    for config_path in config_candidates:
        if config_path.exists():
            logging.info(f"Found config file: {config_path}")
            return config_path
    
    raise FileNotFoundError(
        f"Could not find config file in {checkpoint_dir}. "
        f"Tried: {[c.name for c in config_candidates]}"
    )


def load_model_from_checkpoint(checkpoint_path, config, device):
    logging.info("Initializing model from configuration...")
    
    if 'training' in config:
        model_config = config['training']
    else:
        model_config = config
    
    model = BlendshapeDecoder(
        vocab_size=model_config.get('vocab_size', 1024),   # SpeechTokenizer per-level vocab
        n_q=model_config.get('n_q', 8),                    # RVQ levels
        d_model=model_config.get('d_model', 512),
        nhead=model_config.get('nhead', 8),
        num_layers=model_config.get('num_layers', 6),
        dim_feedforward=model_config.get('dim_feedforward', 2048),
        dropout=model_config.get('dropout', 0.1)
    )
    
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
        'global_step': checkpoint.get('global_step', 'unknown')
    }
    
    logging.info("Model loaded successfully.")
    logging.info(f"  Checkpoint epoch: {checkpoint_info['epoch']}")
    logging.info(f"  Best validation loss: {checkpoint_info['best_val_loss']}")
    
    return model, checkpoint_info


def extract_speechtokenizer_tokens(audio_path, speechtokenizer, device, start_time=None, duration=None):
    """
    Extract SpeechTokenizer tokens from audio file.
    Returns all RVQ layers as (n_q, T) tensor.
    """
    logging.info(f"Loading audio from: {audio_path}")
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    wav, sr = torchaudio.load(str(audio_path))
    
    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav[:1, :]
    
    # Resample to SpeechTokenizer sample rate (16kHz)
    if sr != speechtokenizer.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, speechtokenizer.sample_rate)
    
    # Extract chunk if specified
    if start_time is not None and duration is not None:
        start_sample = int(start_time * speechtokenizer.sample_rate)
        end_sample = int((start_time + duration) * speechtokenizer.sample_rate)
        end_sample = min(end_sample, wav.shape[1])
        wav = wav[:, start_sample:end_sample]
        logging.info(f"Extracted chunk: {start_time:.2f}s to {start_time + duration:.2f}s")
    
    # SpeechTokenizer expects (B, C, T)
    wav = wav.unsqueeze(0).to(device)
    
    with torch.no_grad():
        codes = speechtokenizer.encode(wav)  # (n_q, 1, T)
    
    tokens = codes[:, 0, :].cpu()  # (n_q, T)
    
    duration_sec = wav.shape[-1] / speechtokenizer.sample_rate
    token_fps = speechtokenizer.sample_rate / speechtokenizer.downsample_rate  # 50Hz

    metadata = {
        'audio_path': str(audio_path),
        'num_tokens': tokens.shape[1],          # T
        'n_q': tokens.shape[0],                 # number of RVQ levels
        'duration_seconds': duration_sec,
        'sample_rate': speechtokenizer.sample_rate,
        'token_fps': token_fps
    }
    
    logging.info(f"Extracted {metadata['num_tokens']} tokens per RVQ level from {duration_sec:.2f}s audio")
    logging.info(f"RVQ levels: {metadata['n_q']}, Token rate: {metadata['token_fps']} Hz")
    
    return tokens, metadata


def calculate_target_length(duration_seconds, target_fps):
    target_length = int(duration_seconds * target_fps)
    logging.info(f"Audio duration: {duration_seconds:.2f} seconds")
    logging.info(f"Output: {target_length} frames at {target_fps} FPS")
    return target_length


def run_inference(args):
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        config_path = find_config_file(checkpoint_path)
    
    config = load_config(config_path)
    
    # Load SpeechTokenizer
    logging.info("Loading SpeechTokenizer...")
    speechtokenizer = SpeechTokenizer.load_from_checkpoint(
        args.speechtokenizer_config,
        args.speechtokenizer_model
    )
    speechtokenizer = speechtokenizer.to(device)
    speechtokenizer.eval()
    logging.info(f"SpeechTokenizer loaded (sample_rate={speechtokenizer.sample_rate}Hz, "
                 f"token_rate={speechtokenizer.sample_rate / speechtokenizer.downsample_rate}Hz)")
    
    # Load BlendshapeDecoder
    model, checkpoint_info = load_model_from_checkpoint(checkpoint_path, config, device)

    # Extract tokens
    tokens, metadata = extract_speechtokenizer_tokens(
        args.audio,
        speechtokenizer,
        device,
        start_time=args.start_time,
        duration=args.duration
    )
    
    # Add batch dimension: (n_q, T) -> (1, n_q, T)
    tokens = tokens.unsqueeze(0).to(device)

    # Determine target length
    # target_length = calculate_target_length(
    #     metadata['duration_seconds'],
    #     args.target_fps
    # )

    num_tokens = metadata['num_tokens']  # T at 50Hz
    target_length = int(num_tokens * args.target_fps / metadata['token_fps'])

    # Run inference
    logging.info("Running model inference...")
    with torch.no_grad():
        predicted_blendshapes = model.predict(tokens, target_length=target_length)
    
    # Remove batch dimension: (1, T, 51) -> (T, 51)
    if predicted_blendshapes.dim() == 3:
        predicted_blendshapes = predicted_blendshapes.squeeze(0)
    
    predicted_blendshapes_np = predicted_blendshapes.cpu().numpy()
    
    logging.info(f"Generated blendshapes shape: {predicted_blendshapes_np.shape}")
    logging.info(f"  Frames: {predicted_blendshapes_np.shape[0]}")
    logging.info(f"  Blendshapes: {predicted_blendshapes_np.shape[1]}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_metadata = {
        'arkit_blendshapes': predicted_blendshapes_np,
        'fps': args.target_fps,
        'audio_path': metadata['audio_path'],
        'duration_seconds': metadata['duration_seconds'],
        'num_frames': predicted_blendshapes_np.shape[0],
        'num_tokens': metadata['num_tokens'],
        'n_q': metadata['n_q'],
        'token_fps': metadata['token_fps'],
        'checkpoint_epoch': checkpoint_info['epoch'],
        'mask_categories': config.get('training', config).get('mask_categories', ['jaw', 'mouth'])
    }
    
    np.savez(output_path, **output_metadata)
    logging.info(f"✅ Successfully saved generated blendshapes to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ARKit blendshapes from audio using SpeechTokenizer and trained BlendshapeDecoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--audio', type=str, required=True,
                        help="Path to input audio file (.wav).")
    parser.add_argument('--start_time', type=float, default=None,
                        help="Start time in seconds.")
    parser.add_argument('--duration', type=float, default=None,
                        help="Duration in seconds.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to trained model checkpoint (.pth).")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to config YAML. Auto-detected if not provided.")
    parser.add_argument('--speechtokenizer_config', type=str,
                        default='SpeechTokenizer/config/spt_base_cfg.json',
                        help="Path to SpeechTokenizer config JSON.")
    parser.add_argument('--speechtokenizer_model', type=str,
                        default='pretrained_models/SpeechTokenizer.pt',
                        help="Path to SpeechTokenizer checkpoint.")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to save output blendshapes (.npz).")
    parser.add_argument('--target_fps', type=int, default=30,
                        help="Target frame rate for output animation (default: 30).")
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'])
    args = parser.parse_args()

    run_inference(args)


'''
python inference.py \
    --audio ../chunk_0_audio.wav \
    --checkpoint checkpoints/mj_32bs_frozen/jaw_mouth_model_epoch_999.pth \
    --output ../results/frozen_test_1000ep.npz \
    --speechtokenizer_config ../SpeechTokenizer/config/spt_base_cfg.json \
    --speechtokenizer_model ../pretrained_models/SpeechTokenizer.pt
'''