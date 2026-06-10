#!/usr/bin/env python3
"""
Inference script for the BlendshapeDecoder model.

Generates ARKit blendshape sequences from speech token inputs using a trained model.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from transformers import Wav2Vec2Processor
import librosa
import torchaudio

# Import the model definition from your model.py file
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
        checkpoint_dir / 'training_config2.yaml',  # New standard name
        checkpoint_dir / 'config.yaml',
        checkpoint_dir / 'config2.yaml',
        checkpoint_dir / 'config7.yaml',
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
        # New structure with nested training config
        model_config = config['training']
    else:
        # Old flat structure
        model_config = config
    
    model = BlendshapeDecoder(
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


def load_audio(audio_path, processor, start_time=None, duration=None):
    """
    Load and preprocess audio file for HuBERT
    
    Args:
        audio_path: Path to .wav file
        processor: Wav2Vec2Processor instance
        start_time: Start time in seconds (for chunked audio)
        duration: Duration in seconds (for chunked audio)
        
    Returns:
        tuple: (processed_audio, metadata)
    """
    logging.info(f"Loading audio from: {audio_path}")
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio with librosa (16kHz for HuBERT)
    speech_array, sr = librosa.load(str(audio_path), sr=16000)
    
    # Extract chunk if specified
    if start_time is not None and duration is not None:
        start_sample = int(start_time * 16000)
        end_sample = int((start_time + duration) * 16000)
        end_sample = min(end_sample, len(speech_array))
        speech_array = speech_array[start_sample:end_sample]
        logging.info(f"Extracted chunk: {start_time:.2f}s to {start_time + duration:.2f}s")
    
    # Process with Wav2Vec2Processor
    input_values = processor(
        speech_array,
        return_tensors="pt",
        padding="longest",
        sampling_rate=16000
    ).input_values
    
    audio_waveform = input_values.squeeze(0)  # (samples,)
    
    # Calculate metadata
    duration_sec = len(speech_array) / 16000
    
    metadata = {
        'audio_path': str(audio_path),
        'num_samples': audio_waveform.shape[0],
        'duration_seconds': duration_sec,
        'sample_rate': 16000
    }
    
    logging.info(f"Loaded audio: {duration_sec:.2f}s ({metadata['num_samples']} samples)")
    
    return audio_waveform, metadata


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
    
    # --- 2. Load Processor ---
    logging.info("Loading Wav2Vec2Processor...")
    processor = Wav2Vec2Processor.from_pretrained(
        "pretrained_models/hubert/hubert-xlarge-ls960-ft",
        local_files_only = True
        )
    
    # --- 3. Load Model ---
    model, checkpoint_info = load_model_from_checkpoint(checkpoint_path, config, device)

    # --- 4. Load Input Audio ---
    audio_waveform, metadata = load_audio(
        args.audio,
        processor,
        start_time=args.start_time,
        duration=args.duration
    )
    
    audio_waveform = audio_waveform.to(device).unsqueeze(0)  # Add batch dimension

    # --- 5. Determine Target Length ---
    target_length = calculate_target_length(
        metadata['duration_seconds'],
        args.target_fps
    )

    # --- 6. Run Prediction ---
    # NEW:
    predicted_blendshapes = model.predict(audio_waveform, target_length=target_length)
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
        'checkpoint_epoch': checkpoint_info['epoch'],
        'mask_categories': config.get('training', config).get('mask_categories', ['jaw', 'mouth'])
    }
    
    np.savez(output_path, **output_metadata)
    
    logging.info(f"✅ Successfully saved generated blendshapes to: {output_path}")


def list_available_utterances(tokens_path, limit=20):
    """
    List available utterances in a token file
    
    Args:
        tokens_path: Path to tokens file
        limit: Maximum number of utterances to display
    """
    logging.info(f"Loading token file: {tokens_path}")
    tokens_data = torch.load(tokens_path, map_location='cpu')
    
    if not isinstance(tokens_data, dict):
        logging.error(f"Token file is not a dictionary")
        return
    
    all_keys = list(tokens_data.keys())
    
    print("\n" + "="*70)
    print(f"Available Utterances in {Path(tokens_path).name}")
    print("="*70)
    print(f"Total utterances: {len(all_keys)}")
    
    # Group by base utterance ID if chunked
    utterance_groups = {}
    for key in all_keys:
        if '_chunk_' in key:
            base_id = key.split('_chunk_')[0]
            if base_id not in utterance_groups:
                utterance_groups[base_id] = []
            utterance_groups[base_id].append(key)
        else:
            utterance_groups[key] = [key]
    
    print(f"\nShowing first {limit} utterances:")
    for i, (base_id, keys) in enumerate(list(utterance_groups.items())[:limit]):
        num_tokens = sum(len(tokens_data[k]) for k in keys)
        if len(keys) > 1:
            print(f"  {base_id}: {len(keys)} chunks, {num_tokens} total tokens")
        else:
            print(f"  {base_id}: {num_tokens} tokens")
    
    if len(utterance_groups) > limit:
        print(f"  ... and {len(utterance_groups) - limit} more")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ARKit blendshapes from speech tokens using a trained BlendshapeEncoderDecoder model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="Path to config file (YAML). If not provided, will auto-detect from checkpoint directory."
    )
    parser.add_argument(
        '--output',
        type=str,
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
    
    # Handle list mode
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if not args.output:
        parser.error("--output is required")

    run_inference(args)


''''

python inference.py \
    --checkpoint checkpoints/mouth_jaw_1000ep_4bs_no_overlap_hubert_encode/jaw_mouth_model_epoch_499.pth \
    --audio chunk_0_audio.wav \
    --output results/chunk_0_audio_1000ep_4bs_no_overlap_hubert_encode.npz

'''