#!/usr/bin/env python3
"""
Inference script for the BlendshapeDecoder model with SpeechTokenizer.
Uses pre-extracted tokens (same as GRU script) instead of live extraction.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging

from model import BlendshapeDecoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config(config_path):
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        import yaml
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
        vocab_size=model_config.get('vocab_size', 1024),
        n_q=model_config.get('n_q', 8),
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
    logging.info(f"  Checkpoint epoch: {checkpoint_info['epoch']}")
    logging.info(f"  Best validation loss: {checkpoint_info['best_val_loss']}")
    return model, checkpoint_info


def load_preextracted_tokens(token_path, utt_id, device):
    """Load and concatenate pre-extracted tokens for an utterance (same as GRU script)"""
    logging.info(f"Loading pre-extracted tokens from: {token_path}")
    utt2token = torch.load(token_path, map_location='cpu')

    matches = sorted([k for k in utt2token.keys() if k.startswith(utt_id)])
    if not matches:
        logging.error(f"Available keys (first 10): {list(utt2token.keys())[:10]}")
        raise KeyError(f"Utterance '{utt_id}' not found in {token_path}")

    logging.info(f"Found {len(matches)} chunks: {matches}")

    def parse_chunk(val):
        if isinstance(val, dict):
            # Old format: {'semantic': (T,), 'acoustic': (n_q-1, T)}
            semantic = val['semantic']
            acoustic = val['acoustic']
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
    logging.info(f"Loaded and concatenated {len(matches)} chunks → {tokens.shape}")
    return tokens


def run_inference(args):
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = find_config_file(checkpoint_path)

    config = load_config(config_path)

    # Load model
    model, checkpoint_info = load_model_from_checkpoint(checkpoint_path, config, device)

    # Load pre-extracted tokens (same as GRU)
    tokens = load_preextracted_tokens(args.token_path, args.utt_id, device)

    token_fps = 50.0  # SpeechTokenizer rate
    num_tokens = tokens.shape[2]
    target_length = int(num_tokens * args.target_fps / token_fps)

    logging.info(f"Input tokens: {num_tokens} @ {token_fps:.0f}Hz, n_q={tokens.shape[1]}")
    logging.info(f"Output frames: {target_length} @ {args.target_fps}fps")

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

    duration_seconds = num_tokens / token_fps
    output_metadata = {
        'arkit_blendshapes': predicted_blendshapes_np,
        'fps': args.target_fps,
        'utt_id': args.utt_id,
        'duration_seconds': duration_seconds,
        'num_frames': predicted_blendshapes_np.shape[0],
        'num_tokens': num_tokens,
        'n_q': tokens.shape[1],
        'token_fps': token_fps,
        'checkpoint_epoch': checkpoint_info['epoch'],
        'mask_categories': config.get('training', config).get('mask_categories', ['jaw', 'mouth'])
    }

    np.savez(output_path, **output_metadata)
    logging.info(f"✅ Successfully saved generated blendshapes to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ARKit blendshapes from pre-extracted SpeechTokenizer tokens.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--token_path', type=str, required=True,
                        help="Path to pre-extracted tokens .pt file.")
    parser.add_argument('--utt_id', type=str, required=True,
                        help="Utterance ID to look up in the token file (e.g. 1_wayne_0_1_1).")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to trained model checkpoint (.pth).")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to config YAML. Auto-detected if not provided.")
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
python inference_pre_extracted.py \
    --token_path ../speechtokenizer_tokens/frozen/test_utt2speech_token.pt \
    --utt_id 1_wayne_0_1_1_chunk_0 \
    --checkpoint checkpoints/mj_32bs_frozen/jaw_mouth_model_epoch_999.pth \
    --output ../results/sanity_sync_test.npz
'''