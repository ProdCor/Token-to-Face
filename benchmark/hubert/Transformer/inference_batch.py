#!/usr/bin/env python3
"""
Batch Inference script for the BlendshapeDecoder model with HuBERT.

Generates ARKit blendshape sequences from ALL audio files using a trained model.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
from transformers import Wav2Vec2Processor
import librosa

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
        checkpoint_dir / 'training_config2.yaml',
        checkpoint_dir / 'training_config.yaml',
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
        model_config = config['training']
    else:
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


def load_all_audio_files(audio_dir, split_csv=None, split='test'):
    """
    Load all audio files from directory
    
    Args:
        audio_dir: Directory containing .wav files
        split_csv: Optional CSV file with filename,split
        split: Which split to use ('train', 'val', 'test')
        
    Returns:
        dict: Dictionary mapping utterance IDs to audio file paths
    """
    logging.info(f"Scanning audio files in: {audio_dir}")
    audio_dir = Path(audio_dir)
    
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    # Load split filter if provided
    allowed_files = None
    if split_csv:
        import pandas as pd
        split_df = pd.read_csv(split_csv, header=None, names=['filename', 'split'])
        allowed_files = set(split_df[split_df['split'] == split]['filename'])
        logging.info(f"Loaded split CSV: {len(allowed_files)} files for '{split}' split")
    
    # Scan for .wav files
    audio_files = {}
    for wav_path in audio_dir.glob("*.wav"):
        utterance_id = wav_path.stem
        
        # Filter by split if CSV provided
        if allowed_files is not None and utterance_id not in allowed_files:
            continue
        
        audio_files[utterance_id] = wav_path
    
    logging.info(f"Found {len(audio_files)} audio files for '{split}' split")
    
    return audio_files


def calculate_target_length(duration_seconds, target_fps):
    """
    Calculate target frame length for animation
    
    Args:
        duration_seconds: Audio duration in seconds
        target_fps: Desired output FPS
        
    Returns:
        int: Target frame length
    """
    target_length = int(duration_seconds * target_fps)
    return target_length


def process_single_utterance(model, audio_waveform, target_fps, device):
    """
    Process a single utterance through the model
    
    Args:
        model: The trained model
        audio_waveform: Audio tensor (samples,)
        target_fps: Target output FPS
        device: Device to run on
        
    Returns:
        tuple: (blendshapes_numpy, duration_seconds)
    """
    audio_waveform = audio_waveform.to(device).unsqueeze(0)  # Add batch dimension
    
    # Calculate target length from audio duration
    duration_seconds = audio_waveform.shape[1] / 16000
    target_length = calculate_target_length(duration_seconds, target_fps)
    
    # Run prediction
    with torch.no_grad():
        predicted_blendshapes = model.predict(audio_waveform, target_length=target_length)
        # Remove batch dimension if present
        if predicted_blendshapes.dim() == 3:
            predicted_blendshapes = predicted_blendshapes.squeeze(0)
    
    return predicted_blendshapes.cpu().numpy(), duration_seconds


def run_batch_inference(args):
    """
    Main function to run batch inference on all utterances
    """
    start_time = time.time()
    
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
    
    # --- 2. Load Model ---
    model, checkpoint_info = load_model_from_checkpoint(checkpoint_path, config, device)

    # --- 2.5. Load Processor ---
    logging.info("Loading Wav2Vec2Processor...")
    processor = Wav2Vec2Processor.from_pretrained(
        "pretrained_models/hubert/hubert-xlarge-ls960-ft",
        local_files_only = True
        )

    # --- 3. Load ALL Audio Files ---
    all_audio_files = load_all_audio_files(
        args.audio_dir,
        split_csv=args.split_csv,
        split=args.split
    )
    
    if len(all_audio_files) == 0:
        logging.error(f"No audio files found for split '{args.split}'")
        return
    
    # --- 4. Setup Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization if requested
    if args.organize_by_prefix:
        logging.info("Will organize outputs by utterance prefix")
    
    # --- 5. Process Each Utterance ---
    logging.info(f"\nProcessing {len(all_audio_files)} utterances...")
    
    results_summary = {
        'total_utterances': len(all_audio_files),
        'successful': 0,
        'failed': 0,
        'failed_keys': [],
        'checkpoint_epoch': checkpoint_info['epoch'],
        'split': args.split
    }
    
    # Use tqdm for progress bar
    for utt_key, audio_path in tqdm(all_audio_files.items(), desc="Processing utterances"):
        try:
            # Load and preprocess audio
            speech_array, sr = librosa.load(str(audio_path), sr=16000)
            
            # Process with Wav2Vec2Processor
            input_values = processor(
                speech_array,
                return_tensors="pt",
                padding="longest",
                sampling_rate=16000
            ).input_values
            
            audio_waveform = input_values.squeeze(0)  # (samples,)
            
            # Generate blendshapes
            predicted_blendshapes_np, duration_seconds = process_single_utterance(
                model, audio_waveform, args.target_fps, device
            )
            
            # Determine output path
            if args.organize_by_prefix:
                prefix = utt_key.split('_')[0]
                subdir = output_dir / prefix
                subdir.mkdir(exist_ok=True)
                output_path = subdir / f"{utt_key}.npz"
            else:
                output_path = output_dir / f"{utt_key}.npz"
            
            # Prepare metadata
            output_metadata = {
                'arkit_blendshapes': predicted_blendshapes_np,
                'fps': args.target_fps,
                'utterance_id': utt_key,
                'audio_path': str(audio_path),
                'duration_seconds': duration_seconds,
                'num_frames': predicted_blendshapes_np.shape[0],
                'checkpoint_epoch': checkpoint_info['epoch'],
                'mask_categories': config.get('training', config).get('mask_categories', ['jaw', 'mouth'])
            }
            
            # Save
            np.savez(output_path, **output_metadata)
            
            results_summary['successful'] += 1
            
        except Exception as e:
            logging.error(f"Failed to process '{utt_key}': {str(e)}")
            results_summary['failed'] += 1
            results_summary['failed_keys'].append(utt_key)
            
            if args.stop_on_error:
                raise
    
    # --- 6. Summary ---
    elapsed_time = time.time() - start_time
    
    logging.info("\n" + "="*70)
    logging.info("BATCH INFERENCE COMPLETE")
    logging.info("="*70)
    logging.info(f"Split: {args.split}")
    logging.info(f"Total utterances: {results_summary['total_utterances']}")
    logging.info(f"Successful: {results_summary['successful']}")
    logging.info(f"Failed: {results_summary['failed']}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    if results_summary['successful'] > 0:
        logging.info(f"Average time per utterance: {elapsed_time/results_summary['successful']:.2f} seconds")
    
    if results_summary['failed'] > 0:
        logging.warning(f"\nFailed utterances ({len(results_summary['failed_keys'])}):")
        for key in results_summary['failed_keys'][:10]:  # Show first 10
            logging.warning(f"  - {key}")
        if len(results_summary['failed_keys']) > 10:
            logging.warning(f"  ... and {len(results_summary['failed_keys']) - 10} more")
    
    # Save summary
    summary_path = output_dir / 'inference_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False)
    logging.info(f"\nSummary saved to: {summary_path}")
    logging.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch inference: Generate ARKit blendshapes for ALL audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
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
        '--audio_dir',
        type=str,
        required=True,
        help="Directory containing audio files (.wav)."
    )
    parser.add_argument(
        '--split_csv',
        type=str,
        default=None,
        help="Optional CSV file with filename,split for filtering."
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help="Which split to process (default: test)."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory to save all output blendshapes (.npz files)."
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
    parser.add_argument(
        '--organize_by_prefix',
        action='store_true',
        help="Organize output files into subdirectories based on utterance ID prefix."
    )
    parser.add_argument(
        '--stop_on_error',
        action='store_true',
        help="Stop processing if any utterance fails (default: continue and log errors)."
    )
    
    args = parser.parse_args()
    
    run_batch_inference(args)


'''
Usage examples:

# With split CSV filtering (recommended for BEAT2)
python inference_batch.py \
    --checkpoint checkpoints/cben_1000ep_8bs_no_overlap_hubert_encode/eye_blink_eye_look_eye_shape_brow_cheek_nose_model_epoch_499.pth \
    --audio_dir ../../BEAT2/beat_english_v2.0.0/wave16k \
    --split_csv ../../BEAT2/beat_english_v2.0.0/train_test_split.csv \
    --split test \
    --output_dir results/test_predictions_cben_500ep
'''
