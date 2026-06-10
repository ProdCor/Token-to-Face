#!/usr/bin/env python3
"""
DataLoader for SpeechTokenizer Speech Tokens + ARKit Blendshapes
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import logging

from utils import (
    load_flame_to_arkit_transform,
    flame_to_arkit_blendshapes,
    apply_arkit_mask,
    get_arkit_mask
)

logging.basicConfig(level=logging.INFO)


class SpeechBlendshapeDataset(Dataset):
    """
    Dataset pairing SpeechTokenizer speech tokens with ARKit blendshapes.

    Tokens saved as (n_q, T) numpy arrays — all RVQ layers.
    """
    
    def __init__(
        self,
        speech_tokens_path,
        beat2_dir,
        arkit_transform_path='../arkit_to_flame.npy',
        mask_categories=['jaw', 'mouth'],
        fps=30,
        chunk_duration=30.0,
        overlap_duration=0.0
    ):
        self.beat2_dir = Path(beat2_dir)
        self.fps = fps
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
        logging.info(f"Loading speech tokens from {speech_tokens_path}")
        self.speech_tokens = torch.load(speech_tokens_path)
        logging.info(f"  Loaded {len(self.speech_tokens)} chunks")

        # Log n_q and T from first sample
        first = next(iter(self.speech_tokens.values()))
        if isinstance(first, dict):
            self.n_q = first['acoustic'].shape[0] + 1
        elif isinstance(first, list):
            # Single RVQ layer extracted as list -> treat as n_q=1
            self.n_q = 1
        else:
            self.n_q, _ = first.shape
            
        logging.info(f"  RVQ levels (n_q): {self.n_q}")
        
        logging.info(f"Loading transformation matrix from {arkit_transform_path}")
        self.flame_to_arkit = load_flame_to_arkit_transform(arkit_transform_path)
        
        self.arkit_mask = get_arkit_mask(mask_categories)
        active = self.arkit_mask.sum()
        logging.info(f"  ARKit mask: {active}/51 blendshapes active ({mask_categories})")
        
        self.samples = self._build_index()
        logging.info(f"Dataset ready: {len(self.samples)} samples")
    
    def _build_index(self):
        samples = []
        for chunk_key in self.speech_tokens.keys():
            parts = chunk_key.rsplit('_chunk_', 1)
            if len(parts) != 2:
                logging.warning(f"Invalid chunk key: {chunk_key}")
                continue
            
            utterance_id = parts[0]
            chunk_idx = int(parts[1])
            npz_path = self.beat2_dir / f"{utterance_id}.npz"
            
            if not npz_path.exists():
                logging.warning(f"NPZ not found: {npz_path}")
                continue
            
            samples.append({
                'chunk_key': chunk_key,
                'utterance_id': utterance_id,
                'chunk_idx': chunk_idx,
                'npz_path': npz_path
            })
        
        return samples
    
    def _extract_flame_params(self, npz_path, chunk_idx):
        data = np.load(npz_path, allow_pickle=True)
        expressions = data['expressions']  # (num_frames, 100)
        poses = data['poses']              # (num_frames, 165)
        
        step = self.chunk_duration - self.overlap_duration
        start_time = chunk_idx * step
        end_time = start_time + self.chunk_duration
        
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        end_frame = min(end_frame, expressions.shape[0], poses.shape[0])
        
        expressions_chunk = expressions[start_frame:end_frame]   # (T, 100)
        jaw_chunk = poses[start_frame:end_frame, 66:69]          # (T, 3)
        flame_params = np.concatenate([expressions_chunk, jaw_chunk], axis=1)
        
        return flame_params.astype(np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - speech_tokens: (n_q, T) LongTensor  ← changed from (T,)
                - arkit_blendshapes: (num_frames, 51) FloatTensor
                - chunk_key, utterance_id, chunk_idx, num_tokens, num_frames
        """
        sample = self.samples[idx]
        
        # tokens shape: (n_q, T)
        tokens = self.speech_tokens[sample['chunk_key']]
        if isinstance(tokens, dict):
            semantic = tokens['semantic']                          # (T,)
            acoustic = tokens['acoustic']                         # (n_q-1, T)
            tokens = np.concatenate([semantic[None, :], acoustic], axis=0)  # (n_q, T)
        tokens = torch.LongTensor(tokens)  
        
        flame_params = self._extract_flame_params(
            sample['npz_path'],
            sample['chunk_idx']
        )
        arkit_blendshapes = flame_to_arkit_blendshapes(flame_params, self.flame_to_arkit)
        arkit_blendshapes = apply_arkit_mask(arkit_blendshapes, self.arkit_mask)
        arkit_blendshapes = torch.FloatTensor(arkit_blendshapes)
        
        return {
            'speech_tokens': tokens,                      # (n_q, T)
            'arkit_blendshapes': arkit_blendshapes,       # (num_frames, 51)
            'chunk_key': sample['chunk_key'],
            'utterance_id': sample['utterance_id'],
            'chunk_idx': sample['chunk_idx'],
            'num_tokens': tokens.shape[1],                # T, not shape[0]
            'num_frames': arkit_blendshapes.shape[0]
        }


def collate_fn(batch):
    """Collate function with padding for (n_q, T) tokens"""
    max_tokens = max(s['num_tokens'] for s in batch)
    max_frames = max(s['num_frames'] for s in batch)
    batch_size = len(batch)
    n_q = batch[0]['speech_tokens'].shape[0]
    
    # (B, n_q, T_max) instead of (B, T_max)
    padded_tokens = torch.zeros(batch_size, n_q, max_tokens, dtype=torch.long)
    padded_blendshapes = torch.zeros(batch_size, max_frames, 51, dtype=torch.float32)
    token_lengths = torch.zeros(batch_size, dtype=torch.long)
    frame_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, sample in enumerate(batch):
        t = sample['num_tokens']
        padded_tokens[i, :, :t] = sample['speech_tokens']   # fill along T dim
        padded_blendshapes[i, :sample['num_frames']] = sample['arkit_blendshapes']
        token_lengths[i] = t
        frame_lengths[i] = sample['num_frames']
    
    return {
        'speech_tokens': padded_tokens,       # (B, n_q, T_max)
        'arkit_blendshapes': padded_blendshapes,
        'token_lengths': token_lengths,
        'frame_lengths': frame_lengths
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech_tokens_path', default='../speechtokenizer_tokens/train_utt2speech_token.pt')
    parser.add_argument('--beat2_dir', default='../../BEAT2/beat_english_v2.0.0/smplxflame_30')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    dataset = SpeechBlendshapeDataset(
        speech_tokens_path=args.speech_tokens_path,
        beat2_dir=args.beat2_dir
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print("\nTesting DataLoader...")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Tokens: {batch['speech_tokens'].shape}")    # (B, n_q, T)
        print(f"  Blendshapes: {batch['arkit_blendshapes'].shape}")
        print(f"  Token lengths: {batch['token_lengths'].tolist()}")
        print(f"  Frame lengths: {batch['frame_lengths'].tolist()}")
        if i >= 2:
            break
    
    print("\n✅ DataLoader OK!")