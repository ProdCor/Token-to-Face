#!/usr/bin/env python3
"""
Inference script for the BlendshapeDecoder model with WavTokenizer.

Generates ARKit blendshape sequences from WavTokenizer speech tokens using a trained model.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
import sys
import os

# Add WavTokenizer to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, 'WavTokenizer'))
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer
import torchaudio

# Import the model definition
from model import BlendshapeDecoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config(config_path):
    """Load configuration from YAML file"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_config_file(checkpoint_path):
    """
    Find the config file associated with a checkpoint.
    Tries multiple naming conventions.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Path to config file
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    
    # Try different config file names
    config_candidates = [
        checkpoint_dir / 'training_config.yaml',  # Standard name
        checkpoint_dir / 'config.yaml',
        checkpoint_dir / 'config2.yaml',
    ]
    
    for config_path in config_candidates:
        if config_path.exists():
            logging.info(f"Found config file: {config_path}")
            return config_path
    
    # If no config found, raise error with helpful message
    raise FileNotFoundError(
        f"Could not find config file in {checkpoint_dir}. "
        f"Tried: {[c.name for c in config_candidates]}"
    )


def load_model_from_checkpoint(checkpoint_path, config, device):
    """
    Load the model and its trained weights from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        config (dict): The model configuration dictionary.
        device (torch.device): The device to load the model onto.
        
    Returns:
        tuple: (model, checkpoint_info) where checkpoint_info contains training metadata
    """
    logging.info("Initializing model from configuration...")
    
    # Handle both old and new config structures
    if 'training' in config:
        model_config = config['training']
    else:
        model_config = config
    
    model = BlendshapeDecoder(
        vocab_size=model_config.get('vocab_size', 4096),
        d_model=model_config.get('d_model', 512),
        nhead=model_config.get('nhead', 8),
        num_layers=model_config.get('num_layers', 6),
        dim_feedforward=model_config.get('dim_feedforward', 2048),
        dropout=model_config.get('dropout', 0.1)
    )
    
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Extract checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
        'global_step': checkpoint.get('global_step', 'unknown')
    }
    
    logging.info("Model loaded successfully.")
    logging.info(f"  Checkpoint epoch: {checkpoint_info['epoch']}")
    logging.info(f"  Best validation loss: {checkpoint_info['best_val_loss']}")
    
    return model, checkpoint_info


def extract_wavtokenizer_tokens(audio_path, wavtokenizer, device, start_time=None, duration=None):
    """
    Extract WavTokenizer tokens from audio file
    
    Args:
        audio_path: Path to .wav file
        wavtokenizer: WavTokenizer model
        device: torch device
        start_time: Start time in seconds (for chunked audio)
        duration: Duration in seconds (for chunked audio)
        
    Returns:
        tuple: (tokens, metadata)
    """
    logging.info(f"Loading audio from: {audio_path}")
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio
    wav, sr = torchaudio.load(str(audio_path))
    wav = convert_audio(wav, sr, 24000, 1)  # Convert to 24kHz mono for WavTokenizer
    
    # Extract chunk if specified
    if start_time is not None and duration is not None:
        start_sample = int(start_time * 24000)
        end_sample = int((start_time + duration) * 24000)
        end_sample = min(end_sample, wav.shape[1])
        wav = wav[:, start_sample:end_sample]
        logging.info(f"Extracted chunk: {start_time:.2f}s to {start_time + duration:.2f}s")
    
    wav = wav.to(device)
    
    # Extract tokens
    bandwidth_id = torch.tensor([0]).to(device)
    
    with torch.no_grad():
        _, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    
    # Extract first quantizer tokens
    tokens = discrete_code[0, 0, :].cpu()  # (num_tokens,)
    
    # Calculate metadata
    duration_sec = wav.shape[1] / 24000
    
    metadata = {
        'audio_path': str(audio_path),
        'num_tokens': tokens.shape[0],
        'duration_seconds': duration_sec,
        'sample_rate': 24000,
        'token_fps': 75  # WavTokenizer frame rate
    }
    
    logging.info(f"Extracted {metadata['num_tokens']} tokens from {duration_sec:.2f}s audio")
    logging.info(f"Token rate: {metadata['token_fps']} Hz")
    
    return tokens, metadata


def calculate_target_length(duration_seconds, target_fps):
    """
    Calculate target frame length for animation
    
    Args:
        duration_seconds: Audio duration in seconds
        target_fps: Desired output FPS (default: 30)
        
    Returns:
        int: Target frame length
    """
    target_length = int(duration_seconds * target_fps)
    
    logging.info(f"Audio duration: {duration_seconds:.2f} seconds")
    logging.info(f"Output: {target_length} frames at {target_fps} FPS")
    
    return target_length


def run_inference(args):
    """
    Main function to run the inference process.
    """
    # --- 1. Setup ---
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logging.info(f"Using provided config file: {config_path}")
    else:
        config_path = find_config_file(checkpoint_path)
        logging.info(f"Auto-detected config file: {config_path}")
    
    config = load_config(config_path)
    
    # --- 2. Load WavTokenizer ---
    logging.info("Loading WavTokenizer...")
    wavtokenizer = WavTokenizer.from_pretrained0802(
        args.wavtokenizer_config,
        args.wavtokenizer_model
    )
    wavtokenizer = wavtokenizer.to(device)
    wavtokenizer.eval()
    logging.info("WavTokenizer loaded successfully")
    
    # --- 3. Load Model ---
    model, checkpoint_info = load_model_from_checkpoint(checkpoint_path, config, device)

    # --- 4. Extract Tokens from Audio ---
    tokens, metadata = extract_wavtokenizer_tokens(
        args.audio,
        wavtokenizer,
        device,
        start_time=args.start_time,
        duration=args.duration
    )
    
    # Add batch dimension and move to device
    tokens = tokens.unsqueeze(0).to(device)  # (1, num_tokens)

    # --- 5. Determine Target Length ---
    target_length = calculate_target_length(
        metadata['duration_seconds'],
        args.target_fps
    )

    # --- 6. Run Prediction ---
    logging.info("Running model inference...")
    with torch.no_grad():
        predicted_blendshapes = model.predict(tokens, target_length=target_length)
    
    # Remove batch dimension if present
    if predicted_blendshapes.dim() == 3:
        predicted_blendshapes = predicted_blendshapes.squeeze(0)  # (1, T, 51) -> (T, 51)
    
    predicted_blendshapes_np = predicted_blendshapes.cpu().numpy()
    
    logging.info(f"Generated blendshapes with shape: {predicted_blendshapes_np.shape}")
    logging.info(f"  Frames: {predicted_blendshapes_np.shape[0]}")
    logging.info(f"  Blendshapes: {predicted_blendshapes_np.shape[1]}")

    # --- 7. Save Output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_metadata = {
        'arkit_blendshapes': predicted_blendshapes_np,
        'fps': args.target_fps,
        'audio_path': metadata['audio_path'],
        'duration_seconds': metadata['duration_seconds'],
        'num_frames': predicted_blendshapes_np.shape[0],
        'num_tokens': metadata['num_tokens'],
        'token_fps': metadata['token_fps'],
        'checkpoint_epoch': checkpoint_info['epoch'],
        'mask_categories': config.get('training', config).get('mask_categories', ['jaw', 'mouth'])
    }
    
    np.savez(output_path, **output_metadata)
    
    logging.info(f"✅ Successfully saved generated blendshapes to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ARKit blendshapes from audio using WavTokenizer and trained BlendshapeDecoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help="Path to input audio file (.wav)."
    )
    parser.add_argument(
        '--start_time',
        type=float,
        default=None,
        help="Start time in seconds (for extracting chunk from long audio)."
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help="Duration in seconds (for extracting chunk from long audio)."
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="Path to config file (YAML). If not provided, will auto-detect from checkpoint directory."
    )
    parser.add_argument(
        '--wavtokenizer_config',
        type=str,
        default='WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
        help="Path to WavTokenizer config YAML."
    )
    parser.add_argument(
        '--wavtokenizer_model',
        type=str,
        default='pretrained_models/WavTokenizer_small_320_24k_4096.ckpt',
        help="Path to WavTokenizer checkpoint."
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Path to save the output blendshapes (.npz file)."
    )
    parser.add_argument(
        '--target_fps',
        type=int,
        default=30,
        help="Target frame rate for output animation (default: 30)."
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help="Device to run inference on."
    )
    args = parser.parse_args()

    run_inference(args)


'''

python inference.py \
    --audio ../chunk_0_audio.wav \
    --checkpoint checkpoints/mouth_jaw_wavtokenizer_1000ep/jaw_mouth_model_epoch_999.pth \
    --output results/chunk_0_audio_1000ep_wavtokenizer_transformer.npz \
    --wavtokenizer_config ../WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
    --wavtokenizer_model ../pretrained_models/WavTokenizer_small_320_24k_4096.ckpt
'''