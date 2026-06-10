#!/usr/bin/env python3
"""
Batch inference for BlendshapeDecoder using pre-extracted SpeechTokenizer tokens.
Generates blendshape sequences per chunk (30s) for all chunks in the test set.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time

from model import BlendshapeDecoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_config_file(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    for name in ['training_config.yaml', 'config.yaml', 'config2.yaml']:
        p = checkpoint_path.parent / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No config found in {checkpoint_path.parent}")


def load_model_from_checkpoint(checkpoint_path, config, device):
    model_config = config.get('training', config)
    model = BlendshapeDecoder(
        vocab_size=model_config.get('vocab_size', 1024),
        n_q=model_config.get('n_q', 8),
        d_model=model_config.get('d_model', 512),
        nhead=model_config.get('nhead', 8),
        num_layers=model_config.get('num_layers', 6),
        dim_feedforward=model_config.get('dim_feedforward', 2048),
        dropout=model_config.get('dropout', 0.1)
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint.get('epoch', 'unknown')


def parse_chunk(val):
    if isinstance(val, dict):
        semantic = val['semantic']
        acoustic = val['acoustic']
        if semantic.ndim == 1:
            semantic = semantic[np.newaxis, :]
        return np.concatenate([semantic, acoustic], axis=0)  # (n_q, T)
    elif isinstance(val, np.ndarray):
        if val.ndim == 1:
            return val[np.newaxis, :]
        return val
    else:
        raise ValueError(f"Unexpected token type: {type(val)}")


def run_batch_inference(args):
    start_time = time.time()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else find_config_file(checkpoint_path)
    config = load_config(config_path)

    model, epoch = load_model_from_checkpoint(checkpoint_path, config, device)
    mask_categories = config.get('training', config).get('mask_categories', ['jaw', 'mouth'])

    logging.info(f"Loading tokens from {args.token_path}")
    utt2token = torch.load(args.token_path, map_location='cpu')

    # Iterate over chunk keys directly — no grouping or concatenation
    all_keys = sorted(utt2token.keys())
    logging.info(f"Found {len(all_keys)} chunks")

    token_fps = 50.0
    stats = {'success': 0, 'failed': 0, 'failed_ids': []}

    for chunk_key in tqdm(all_keys, desc="Generating"):
        try:
            tokens = parse_chunk(utt2token[chunk_key])                        # (n_q, T)
            tokens = torch.from_numpy(tokens).long().unsqueeze(0).to(device)  # (1, n_q, T)

            num_tokens = tokens.shape[2]
            target_length = int(num_tokens * args.target_fps / token_fps)

            with torch.no_grad():
                predicted = model.predict(tokens, target_length=target_length)

            if predicted.dim() == 3:
                predicted = predicted.squeeze(0)
            predicted_np = predicted.cpu().numpy()

            out_path = output_dir / f"{chunk_key}.npz"
            np.savez(
                out_path,
                arkit_blendshapes=predicted_np,
                fps=args.target_fps,
                chunk_key=chunk_key,
                num_frames=predicted_np.shape[0],
                num_tokens=num_tokens,
                token_fps=token_fps,
                checkpoint_epoch=epoch,
                mask_categories=mask_categories
            )
            stats['success'] += 1

        except Exception as e:
            logging.warning(f"Failed {chunk_key}: {e}")
            stats['failed'] += 1
            stats['failed_ids'].append(chunk_key)

    elapsed = time.time() - start_time
    logging.info(f"\n✅ Done: {stats['success']} succeeded, {stats['failed']} failed")
    logging.info(f"Time elapsed: {elapsed:.2f}s")
    if stats['failed_ids']:
        logging.warning(f"Failed: {stats['failed_ids'][:10]}")

    summary_path = output_dir / 'inference_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    logging.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--target_fps', type=int, default=30)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    run_batch_inference(args)


'''
python inference_batch.py \
    --token_path ../speechtokenizer_tokens/avhubert_768D_proj_RVQ8/test_utt2speech_token.pt \
    --checkpoint checkpoints/mj_32bs_avhubert/jaw_mouth_model_epoch_999.pth \
    --output_dir ../results/test_set_results_mj_avhubert
'''