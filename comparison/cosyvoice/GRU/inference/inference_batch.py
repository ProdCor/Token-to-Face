#!/usr/bin/env python3
"""
Batch inference for trained CosyVoice2-based model using pre-extracted tokens
"""
import argparse
import os
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
import yaml

from model_adapt import FaceDiffBeatCosyVoice2
from utils_adapt import create_gaussian_diffusion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def detect_model_config(model_path):
    """Detect model configuration from checkpoint"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Detect dimensions from weight shapes
    latent_dim = checkpoint['time_mlp.0.weight'].shape[0]
    vertice_dim = checkpoint['final_layer.weight'].shape[0]
    gru_dim = checkpoint['final_layer.weight'].shape[1]
    gru_layers = sum(1 for k in checkpoint.keys() if k.startswith('gru.weight_ih_l'))
    
    if 'token_embedding.weight' in checkpoint:
        token_embedding_dim = checkpoint['token_embedding.weight'].shape[1]
        num_embeddings = checkpoint['token_embedding.weight'].shape[0]
    else:
        token_embedding_dim = 512
        num_embeddings = 6561
    
    return {
        'feature_dim': latent_dim,
        'gru_dim': gru_dim,
        'vertice_dim': vertice_dim,
        'gru_layers': gru_layers,
        'token_embedding_dim': token_embedding_dim,
        'num_embeddings': num_embeddings
    }


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
        elif isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens).long()
        elif not isinstance(tokens, torch.Tensor):
            logging.warning(f"Skipping key '{key}': unexpected type {type(tokens)}")
            continue
        
        processed_tokens[key] = tokens.long()
    
    logging.info(f"Loaded {len(processed_tokens)} utterances")
    
    return processed_tokens


def calculate_num_frames(num_tokens, output_fps, token_fps=25):
    """
    Calculate output frame count from token count
    
    Args:
        num_tokens: Number of input tokens
        output_fps: Desired output FPS (default: 30)
        token_fps: Token rate (default: 25 Hz for CosyVoice2)
        
    Returns:
        int: Number of output frames
    """
    audio_duration = num_tokens / token_fps
    num_frames = int(audio_duration * output_fps)
    return num_frames


@torch.no_grad()
def process_single_utterance(model, diffusion, tokens, args, device):
    """
    Generate prediction for a single utterance
    
    Args:
        model: The trained model
        diffusion: Diffusion process
        tokens: Token tensor (seq_len,) or (1, seq_len)
        args: Arguments
        device: Device
        
    Returns:
        numpy array: Generated blendshapes (num_frames, vertice_dim)
    """
    # Ensure tokens have batch dimension
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    
    tokens = tokens.to(device)
    
    # Create dummy template and one-hot (BEAT2 doesn't use subject conditioning)
    template = torch.zeros(1, args.vertice_dim).to(device)
    one_hot = torch.zeros(1, 8).to(device)
    one_hot[0, 0] = 1.0
    
    # Calculate output shape
    num_tokens = tokens.shape[1]
    num_frames = calculate_num_frames(num_tokens, args.output_fps, args.token_fps)
    shape = (1, num_frames, args.vertice_dim)
    
    # Generate prediction
    sample = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs={
            "cond_embed": tokens,
            "one_hot": one_hot,
            "template": template,
        },
        skip_timesteps=args.skip_steps,
        init_image=None,
        progress=False,  # Disable progress bar for batch processing
        dump_steps=None,
        noise=None,
        const_noise=False,
        device=device
    )
    
    sample = sample.squeeze().detach().cpu().numpy()
    return sample


def run_batch_inference(args):
    """
    Main function to run batch inference on all utterances
    """
    start_time = time.time()
    
    # --- 1. Setup ---
    device = torch.device(args.device)
    model_path = os.path.join(args.save_path, f'{args.model}_{args.dataset}_{args.epoch}.pth')
    
    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        logging.info(f"\nAvailable models in {args.save_path}:")
        for f in sorted(os.listdir(args.save_path)):
            if f.endswith('.pth'):
                logging.info(f"  - {f}")
        return
    
    # --- 2. Auto-detect model configuration ---
    logging.info("="*70)
    logging.info("Auto-detecting model configuration...")
    logging.info("="*70)
    
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
    
    logging.info(f"Detected configuration:")
    logging.info(f"  vertice_dim: {args.vertice_dim}")
    logging.info(f"  feature_dim: {args.feature_dim}")
    logging.info(f"  gru_dim: {args.gru_dim}")
    logging.info(f"  gru_layers: {args.gru_layers}")
    logging.info(f"  token_embedding_dim: {args.token_embedding_dim}")
    
    # --- 3. Create model ---
    logging.info("\nCreating CosyVoice2 model...")
    diffusion = create_gaussian_diffusion(args)
    
    model = FaceDiffBeatCosyVoice2(
        args,
        vertice_dim=args.vertice_dim,
        latent_dim=args.feature_dim,
        diffusion_steps=args.diff_steps,
        gru_latent_dim=args.gru_dim,
        num_layers=args.gru_layers,
    )
    
    # Load checkpoint
    logging.info(f"Loading model: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    logging.info("✓ Model loaded")
    
    # --- 4. Load ALL Input Tokens ---
    all_tokens = load_all_speech_tokens(args.tokens)
    
    # --- 5. Setup Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.organize_by_prefix:
        logging.info("Will organize outputs by utterance prefix")
    
    # --- 6. Process Each Utterance ---
    logging.info(f"\nProcessing {len(all_tokens)} utterances...")
    logging.info("="*70)
    
    results_summary = {
        'total_utterances': len(all_tokens),
        'successful': 0,
        'failed': 0,
        'failed_keys': [],
        'model_epoch': args.epoch,
        'token_fps': args.token_fps,
        'output_fps': args.output_fps
    }
    
    # Use tqdm for progress bar
    for utt_key, speech_tokens in tqdm(all_tokens.items(), desc="Processing utterances"):
        try:
            # Generate blendshapes
            predicted_blendshapes = process_single_utterance(
                model, diffusion, speech_tokens, args, device
            )
            
            # Determine output path
            if args.organize_by_prefix:
                prefix = utt_key.split('_')[0]
                subdir = output_dir / prefix
                subdir.mkdir(exist_ok=True)
                output_path = subdir / f"{utt_key}.npy"
            else:
                output_path = output_dir / f"{utt_key}.npy"
            
            # Save as .npy
            np.save(output_path, predicted_blendshapes)
            
            results_summary['successful'] += 1
            
        except Exception as e:
            logging.error(f"Failed to process '{utt_key}': {str(e)}")
            results_summary['failed'] += 1
            results_summary['failed_keys'].append(utt_key)
            
            if args.stop_on_error:
                raise
    
    # --- 7. Summary ---
    elapsed_time = time.time() - start_time
    
    logging.info("\n" + "="*70)
    logging.info("BATCH INFERENCE COMPLETE")
    logging.info("="*70)
    logging.info(f"Total utterances: {results_summary['total_utterances']}")
    logging.info(f"Successful: {results_summary['successful']}")
    logging.info(f"Failed: {results_summary['failed']}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    if results_summary['successful'] > 0:
        logging.info(f"Average time per utterance: {elapsed_time/results_summary['successful']:.2f} seconds")
    
    if results_summary['failed'] > 0:
        logging.warning(f"\nFailed utterances ({len(results_summary['failed_keys'])}):")
        for key in results_summary['failed_keys'][:10]:
            logging.warning(f"  - {key}")
        if len(results_summary['failed_keys']) > 10:
            logging.warning(f"  ... and {len(results_summary['failed_keys']) - 10} more")
    
    # Save summary
    summary_path = output_dir / 'inference_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False)
    logging.info(f"\nSummary saved to: {summary_path}")
    logging.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Batch inference for CosyVoice2 model using pre-extracted tokens',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument("--tokens", type=str, required=True,
                        help='Path to speech tokens file (.pt)')
    parser.add_argument("--output_dir", type=str, required=True,
                        help='Directory to save output predictions')
    
    # Model configuration (auto-detected if not provided)
    parser.add_argument("--dataset", type=str, default="beat2")
    parser.add_argument("--vertice_dim", type=int, default=None)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--gru_dim", type=int, default=None)
    parser.add_argument("--gru_layers", type=int, default=None)
    parser.add_argument("--token_embedding_dim", type=int, default=None)
    parser.add_argument("--output_fps", type=int, default=30)
    parser.add_argument("--token_fps", type=int, default=25,
                        help='Token rate (25 Hz for CosyVoice2)')
    parser.add_argument("--diff_steps", type=int, default=1000)
    
    # Paths
    parser.add_argument("--save_path", type=str, default="save_no_overlap")
    parser.add_argument("--model", type=str, default="face_diffuser")
    
    # Testing options
    parser.add_argument("--epoch", type=int, default=100,
                        help='Which epoch checkpoint to use')
    parser.add_argument("--skip_steps", type=int, default=0,
                        help='Skip diffusion steps for faster generation')
    parser.add_argument("--device", type=str, default="cuda")
    
    # Organization
    parser.add_argument("--organize_by_prefix", action="store_true",
                        help='Organize outputs into subdirectories by prefix')
    parser.add_argument("--stop_on_error", action="store_true",
                        help='Stop processing if any utterance fails')
    
    args = parser.parse_args()
    
    # Validate tokens file
    if not os.path.exists(args.tokens):
        print(f"❌ ERROR: Tokens file not found: {args.tokens}")
        return
    
    run_batch_inference(args)


if __name__ == "__main__":
    main()

'''

python inference_batch.py \
    --tokens ../data/beat2/train_split_tokens/splits/utt2speech_token_test.pt \
    --output_dir results/test_set_results_100ep_GRU_CV2_cben \
    --save_path save/cben_100ep_GRU_COSY \
    --epoch 100

'''