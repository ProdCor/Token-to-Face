#!/usr/bin/env python3
"""
Batch inference for trained SpeechTokenizer-based GRU+Diffusion model
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

from model import FaceDiffBeatSpeechTokenizer
from utils import create_gaussian_diffusion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def detect_model_config(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    
    latent_dim = checkpoint['time_mlp.0.weight'].shape[0]
    vertice_dim = checkpoint['final_layer.weight'].shape[0]
    gru_dim = checkpoint['final_layer.weight'].shape[1]
    gru_layers = sum(1 for k in checkpoint.keys() if k.startswith('gru.weight_ih_l'))
    
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


def parse_chunk(val):
    """Handle old dict format or new (n_q, T) numpy array format."""
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


def load_all_speech_tokens(tokens_path):
    """
    Load all tokens, grouping chunks per utterance and concatenating them.
    Returns dict mapping utt_id → (n_q, T_total) numpy array.
    """
    logging.info(f"Loading tokens from: {tokens_path}")
    raw = torch.load(tokens_path, map_location='cpu')

    # Group chunk keys by utterance prefix (strip _chunk_N suffix)
    from collections import defaultdict
    utt_chunks = defaultdict(list)
    for key in sorted(raw.keys()):
        if '_chunk_' in key:
            utt_id = key.rsplit('_chunk_', 1)[0]
            chunk_idx = int(key.rsplit('_chunk_', 1)[1])
            utt_chunks[utt_id].append((chunk_idx, key))
        else:
            utt_chunks[key].append((0, key))

    processed = {}
    for utt_id, chunk_list in utt_chunks.items():
        chunk_list = sorted(chunk_list, key=lambda x: x[0])  # sort by chunk index
        arrays = [parse_chunk(raw[k]) for _, k in chunk_list]
        tokens = np.concatenate(arrays, axis=1)  # (n_q, T_total)
        processed[utt_id] = tokens

    logging.info(f"Loaded {len(processed)} utterances")
    return processed


@torch.no_grad()
def process_single_utterance(model, diffusion, tokens_np, args, device):
    """
    tokens_np: (n_q, T) numpy array
    Returns: (num_frames, vertice_dim) numpy array
    """
    tokens = torch.from_numpy(tokens_np).long().unsqueeze(0).to(device)  # (1, n_q, T)

    token_fps = 50.0
    num_tokens = tokens.shape[2]
    audio_duration = num_tokens / token_fps
    num_frames = int(audio_duration * args.output_fps)
    shape = (1, num_frames, args.vertice_dim)

    template = torch.zeros(1, args.vertice_dim).to(device)
    one_hot = torch.zeros(1, 8).to(device)
    one_hot[0, 0] = 1.0

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
        progress=False,
        dump_steps=None,
        noise=None,
        const_noise=False,
        device=device
    )

    return sample.squeeze().detach().cpu().numpy()


def run_batch_inference(args):
    start_time = time.time()
    device = torch.device(args.device)

    model_path = os.path.join(args.save_path, f'{args.model}_{args.dataset}_{args.epoch}.pth')
    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        for f in sorted(os.listdir(args.save_path)):
            if f.endswith('.pth'):
                logging.info(f"  - {f}")
        return

    logging.info("Auto-detecting model configuration...")
    config = detect_model_config(model_path)

    if args.vertice_dim is None:         args.vertice_dim = config['vertice_dim']
    if args.feature_dim is None:         args.feature_dim = config['feature_dim']
    if args.gru_dim is None:             args.gru_dim = config['gru_dim']
    if args.gru_layers is None:          args.gru_layers = config['gru_layers']
    if args.token_embedding_dim is None: args.token_embedding_dim = config['token_embedding_dim']
    if args.vocab_size is None:          args.vocab_size = config['vocab_size']
    if args.n_q is None:                 args.n_q = config['n_q']
    args.codebook_size = args.vocab_size

    logging.info(f"  vertice_dim={args.vertice_dim}, feature_dim={args.feature_dim}, "
                 f"gru_dim={args.gru_dim}, gru_layers={args.gru_layers}, "
                 f"vocab_size={args.vocab_size}, n_q={args.n_q}")

    diffusion = create_gaussian_diffusion(args)
    model = FaceDiffBeatSpeechTokenizer(
        args,
        vertice_dim=args.vertice_dim,
        latent_dim=args.feature_dim,
        diffusion_steps=args.diff_steps,
        gru_latent_dim=args.gru_dim,
        num_layers=args.gru_layers,
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    logging.info("✓ Model loaded")

    all_tokens = load_all_speech_tokens(args.tokens)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = {
        'total': len(all_tokens), 'successful': 0,
        'failed': 0, 'failed_keys': [],
        'epoch': args.epoch, 'token_fps': 50, 'output_fps': args.output_fps
    }

    for utt_id, tokens_np in tqdm(all_tokens.items(), desc="Processing"):
        try:
            blendshapes = process_single_utterance(model, diffusion, tokens_np, args, device)

            if args.organize_by_prefix:
                subdir = output_dir / utt_id.split('_')[0]
                subdir.mkdir(exist_ok=True)
                output_path = subdir / f"{utt_id}.npy"
            else:
                output_path = output_dir / f"{utt_id}.npy"

            np.save(output_path, blendshapes)
            results_summary['successful'] += 1

        except Exception as e:
            logging.error(f"Failed '{utt_id}': {e}")
            results_summary['failed'] += 1
            results_summary['failed_keys'].append(utt_id)
            if args.stop_on_error:
                raise

    elapsed = time.time() - start_time
    logging.info(f"\n✅ Done: {results_summary['successful']}/{results_summary['total']} successful in {elapsed:.1f}s")
    if results_summary['successful'] > 0:
        logging.info(f"   Average: {elapsed/results_summary['successful']:.2f}s per utterance")
    if results_summary['failed']:
        logging.warning(f"   Failed: {results_summary['failed_keys'][:10]}")

    summary_path = output_dir / 'inference_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False)
    logging.info(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch inference for SpeechTokenizer GRU+Diffusion model')

    parser.add_argument("--tokens", type=str, required=True,
                        help='Path to pre-extracted tokens .pt file')
    parser.add_argument("--output_dir", type=str, required=True,
                        help='Directory to save output .npy predictions')
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
    parser.add_argument("--save_path", type=str, default="save/mj_100ep_GRU")
    parser.add_argument("--model", type=str, default="face_diffuser_speechtokenizer")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--skip_steps", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--organize_by_prefix", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.tokens):
        print(f"❌ Tokens file not found: {args.tokens}")
        return

    run_batch_inference(args)


if __name__ == "__main__":
    main()


'''
python inference_batch.py \
    --save_path save/frozen/mj_100ep_GRU \
    --tokens speechtokenizer_tokens/frozen/test_utt2speech_token.pt \
    --output_dir results/test_set_results_100ep_GRU_frozen_mj \
    --epoch 100

python inference_batch.py \
    --save_path save/frozen/cben_100ep_GRU \
    --tokens speechtokenizer_tokens/frozen/test_utt2speech_token.pt \
    --output_dir results/test_set_results_100ep_GRU_frozen_cben \
    --epoch 100
'''