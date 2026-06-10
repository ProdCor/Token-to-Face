#!/usr/bin/env python3
"""
Batch Inference script for the BlendshapeDecoder model.

Generates ARKit blendshape sequences from ALL speech token inputs in a file using a trained model.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time

# Import the model definition from your model.py file
from model import BlendshapeDecoder, BlendshapeEncoderDecoder, BlendshapeDecoderConv1D

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
        vocab_size=model_config.get('vocab_size', 6561),
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


def load_all_speech_tokens(tokens_path):
    """
    Load ALL speech tokens from file
    
    Args:
        tokens_path: Path to tokens file
        
    Returns:
        dict: Dictionary mapping utterance keys to token tensors
    """
    logging.info(f"Loading all speech tokens from: {tokens_path}")
    tokens_path = Path(tokens_path)
    
    if not tokens_path.exists():
        raise FileNotFoundError(f"Token file not found: {tokens_path}")

    tokens_data = torch.load(tokens_path, map_location='cpu')
    
    if not isinstance(tokens_data, dict):
        raise TypeError(f"Expected tokens file to contain a dictionary, but got {type(tokens_data)}")
    
    # Convert all to tensors
    processed_tokens = {}
    for key, tokens in tokens_data.items():
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        elif not isinstance(tokens, torch.Tensor):
            logging.warning(f"Skipping key '{key}': unexpected type {type(tokens)}")
            continue
        
        processed_tokens[key] = tokens.long()
    
    logging.info(f"Loaded {len(processed_tokens)} utterances")
    
    return processed_tokens


def calculate_target_length(num_tokens, target_fps, token_fps=25):
    """
    Calculate target frame length for animation
    
    Args:
        num_tokens: Number of input tokens
        target_fps: Desired output FPS
        token_fps: Token rate (default: 25 Hz for CosyVoice)
        
    Returns:
        int: Target frame length
    """
    target_length = int(num_tokens * (target_fps / token_fps))
    return target_length


def process_single_utterance(model, speech_tokens, target_fps, token_fps, device):
    """
    Process a single utterance through the model
    
    Args:
        model: The trained model
        speech_tokens: Token tensor for one utterance
        target_fps: Target output FPS
        token_fps: Input token FPS
        device: Device to run on
        
    Returns:
        numpy array: Generated blendshapes
    """
    speech_tokens = speech_tokens.to(device)
    num_tokens = speech_tokens.shape[0]
    target_length = calculate_target_length(num_tokens, target_fps, token_fps)
    
    # Run prediction
    with torch.no_grad():
        predicted_blendshapes = model.predict(speech_tokens, target_length=target_length)
    
    return predicted_blendshapes.cpu().numpy()


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

    # Load config - use provided path or try to find it
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

    # --- 3. Load ALL Input Tokens ---
    all_tokens = load_all_speech_tokens(args.tokens)
    
    # --- 4. Setup Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization if requested
    if args.organize_by_prefix:
        logging.info("Will organize outputs by utterance prefix")
    
    # --- 5. Process Each Utterance ---
    logging.info(f"\nProcessing {len(all_tokens)} utterances...")
    
    results_summary = {
        'total_utterances': len(all_tokens),
        'successful': 0,
        'failed': 0,
        'failed_keys': [],
        'checkpoint_epoch': checkpoint_info['epoch']
    }
    
    # Use tqdm for progress bar
    for utt_key, speech_tokens in tqdm(all_tokens.items(), desc="Processing utterances"):
        try:
            # Generate blendshapes
            predicted_blendshapes_np = process_single_utterance(
                model, speech_tokens, args.target_fps, args.token_fps, device
            )
            
            # Determine output path
            if args.organize_by_prefix:
                # Extract prefix (e.g., "1_wayne_0_1_1" -> "1")
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
                'num_tokens': speech_tokens.shape[0],
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
    logging.info(f"Total utterances: {results_summary['total_utterances']}")
    logging.info(f"Successful: {results_summary['successful']}")
    logging.info(f"Failed: {results_summary['failed']}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Time elapsed: {elapsed_time:.2f} seconds")
    logging.info(f"Average time per utterance: {elapsed_time/results_summary['total_utterances']:.2f} seconds")
    
    if results_summary['failed'] > 0:
        logging.warning(f"\nFailed utterances: {results_summary['failed_keys']}")
    
    # Save summary
    summary_path = output_dir / 'inference_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False)
    logging.info(f"\nSummary saved to: {summary_path}")
    logging.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch inference: Generate ARKit blendshapes for ALL utterances in a token file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all utterances in test set
  python batch_inference.py \
    --checkpoint experiments/exp1/best_model.pth \
    --tokens data/test_tokens.pt \
    --output_dir outputs/test_predictions
  
  # Organize outputs by prefix
  python batch_inference.py \
    --checkpoint experiments/exp1/best_model.pth \
    --tokens data/test_tokens.pt \
    --output_dir outputs/test_predictions \
    --organize_by_prefix
        """
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
        '--tokens',
        type=str,
        required=True,
        help="Path to the input speech tokens file (.pt file containing all test utterances)."
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
        '--token_fps',
        type=int,
        default=25,
        help="Token rate in Hz (default: 25 for CosyVoice)."
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

python inference_batch.py \
    --checkpoint checkpoints/mouth_jaw_1000ep_4bs_no_overlap_l1_pure/jaw_mouth_model_epoch_899.pth \
    --tokens ../all_speakers/splits/utt2speech_token_test.pt \
    --output_dir results/mouth_jaw_test_set_1000ep_4bs_no_overlap_l1_pure
'''