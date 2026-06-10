#!/usr/bin/env python3
"""
DataLoader for Speech Tokens + ARKit Blendshapes
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import logging
import torchaudio
import pandas as pd
from transformers import Wav2Vec2Processor
import librosa

from utils import (
    load_flame_to_arkit_transform,
    flame_to_arkit_blendshapes,
    apply_arkit_mask,
    get_arkit_mask
)

logging.basicConfig(level=logging.INFO)


class SpeechBlendshapeDataset(Dataset):
    """
    Dataset pairing speech tokens with ARKit blendshapes
    
    Pipeline:
    1. Load FLAME params from BEAT2 (100 expressions + 3 jaw)
    2. Transform to ARKit blendshapes (51)
    3. Apply mask to keep only desired blendshapes (e.g., mouth+jaw)
    """
    
    def __init__(
        self,
        audio_dir,
        beat2_dir,
        split_csv=None, 
        split='train',   
        arkit_transform_path='../arkit_to_flame.npy',
        mask_categories=['jaw', 'mouth'],
        fps=30,
        chunk_duration=30.0,
        overlap_duration=0.0
    ):
        """
        Args:
            speech_tokens_path: Path to utt2speech_token.pt
            beat2_dir: Directory with BEAT2 .npz files
            arkit_transform_path: Path to transformation matrix
            mask_categories: ARKit categories to keep (e.g., ['jaw', 'mouth'])
            fps: Frame rate (default: 30)
            chunk_duration: Chunk size in seconds (default: 30)
            overlap_duration: Overlap in seconds (default: 0)
        """
        self.beat2_dir = Path(beat2_dir)
        self.fps = fps
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration

        # Load audio    
        self.audio_dir = Path(audio_dir)
        logging.info(f"Audio directory: {self.audio_dir}")

        # Load processor for HuBERT
        self.processor = Wav2Vec2Processor.from_pretrained(
            "pretrained_models/hubert/hubert-xlarge-ls960-ft",
            local_files_only = True
        )
        logging.info("Loaded Wav2Vec2Processor for HuBERT")

        # Load split CSV
        if split_csv is not None:
            self.split_df = pd.read_csv(split_csv, header=None, names=['filename', 'split'])
            self.allowed_files = set(self.split_df[self.split_df['split'] == split]['filename'])
            logging.info(f"Loaded split CSV: {len(self.allowed_files)} files for '{split}' split")
        else:
            self.allowed_files = None
            logging.info(f"No split CSV provided, using all files")
        
        # Load transformation matrix
        logging.info(f"Loading transformation matrix from {arkit_transform_path}")
        self.flame_to_arkit = load_flame_to_arkit_transform(arkit_transform_path)
        logging.info(f"  Matrix shape: (103, 51)")
        
        # Create mask
        self.arkit_mask = get_arkit_mask(mask_categories)
        active = self.arkit_mask.sum()
        logging.info(f"  ARKit mask: {active}/51 blendshapes active ({mask_categories})")
        
        # Build sample index
        self.samples = self._build_index()
        logging.info(f"Dataset ready: {len(self.samples)} samples")
    
    def _build_index(self):
        """Build index of valid samples"""
        samples = []

        # Scan audio files in audio directory
        audio_files = list(self.audio_dir.glob("*.wav"))
        for audio_path in audio_files:
            utterance_id = audio_path.stem  # filename without extension

            # Filter by split CSV
            if self.allowed_files is not None and utterance_id not in self.allowed_files:
                continue  # Skip this file
            
            # Find corresponding .npz file
            npz_path = self.beat2_dir / f"{utterance_id}.npz"
            if not npz_path.exists():
                logging.warning(f"NPZ not found for {utterance_id}")
                continue
            
            # Calculate number of chunks for this audio
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            step = self.chunk_duration - self.overlap_duration
            num_chunks = int((duration - self.chunk_duration) / step) + 1
            
            # Create samples for each chunk
            for chunk_idx in range(num_chunks):
                samples.append({
                    'utterance_id': utterance_id,
                    'chunk_idx': chunk_idx,
                    'npz_path': npz_path,
                    'audio_path': audio_path
                })
        
        return samples
    
    def _extract_flame_params(self, npz_path, chunk_idx):
        """Extract FLAME parameters for chunk"""
        data = np.load(npz_path, allow_pickle=True)
        expressions = data['expressions']  # (num_frames, 100)
        poses = data['poses']  # (num_frames, 165)
        
        # Calculate frame range
        step = self.chunk_duration - self.overlap_duration
        start_time = chunk_idx * step
        end_time = start_time + self.chunk_duration
        
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        end_frame = min(end_frame, expressions.shape[0], poses.shape[0])
        
        # Extract FLAME params
        expressions_chunk = expressions[start_frame:end_frame]  # (T, 100)
        jaw_chunk = poses[start_frame:end_frame, 66:69]  # (T, 3)
        
        # Concatenate to FLAME (103)
        flame_params = np.concatenate([expressions_chunk, jaw_chunk], axis=1)
        
        return flame_params.astype(np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - speech_tokens: (num_tokens,) LongTensor
                - arkit_blendshapes: (num_frames, 51) FloatTensor (masked)
                - chunk_key: str
                - utterance_id: str
                - chunk_idx: int
        """
        sample = self.samples[idx]

        # Load audio waveform
        # Load with librosa (like FaceDiffuser does)
        speech_array, sr = librosa.load(str(sample['audio_path']), sr=16000)

        # Calculate chunk timestamps
        step = self.chunk_duration - self.overlap_duration
        start_time = sample['chunk_idx'] * step
        end_time = start_time + self.chunk_duration

        # Extract chunk
        step = self.chunk_duration - self.overlap_duration
        start_time = sample['chunk_idx'] * step
        end_time = start_time + self.chunk_duration
        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)
        end_sample = min(end_sample, len(speech_array))
        audio_chunk = speech_array[start_sample:end_sample]

        # Process with Wav2Vec2Processor (IMPORTANT!)
        input_values = self.processor(
            audio_chunk, 
            return_tensors="pt", 
            padding="longest",
            sampling_rate=16000
        ).input_values
        audio_chunk = input_values.squeeze(0)  # (samples,) - now properly normalized
        
        # Extract FLAME and transform to ARKit
        flame_params = self._extract_flame_params(
            sample['npz_path'],
            sample['chunk_idx']
        )
        
        # Transform to ARKit
        arkit_blendshapes = flame_to_arkit_blendshapes(
            flame_params,
            self.flame_to_arkit
        )
        
        # Apply mask
        arkit_blendshapes = apply_arkit_mask(arkit_blendshapes, self.arkit_mask)
        arkit_blendshapes = torch.FloatTensor(arkit_blendshapes)
        
        return {
            'audio_waveform': audio_chunk,  # (samples,)
            'arkit_blendshapes': arkit_blendshapes,
            'utterance_id': sample['utterance_id'],
            'chunk_idx': sample['chunk_idx'],
            'num_audio_samples': audio_chunk.shape[0],
            'num_frames': arkit_blendshapes.shape[0]
        }


def collate_fn(batch):
    """Collate function with padding"""
    max_audio_samples = max(s['num_audio_samples'] for s in batch)
    max_frames = max(s['num_frames'] for s in batch)
    batch_size = len(batch)
    
    # Initialize padded tensors
    padded_audio = torch.zeros(batch_size, max_audio_samples, dtype=torch.float32)
    padded_blendshapes = torch.zeros(batch_size, max_frames, 51, dtype=torch.float32)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    frame_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, sample in enumerate(batch):
        padded_audio[i, :sample['num_audio_samples']] = sample['audio_waveform']
        padded_blendshapes[i, :sample['num_frames']] = sample['arkit_blendshapes']
        audio_lengths[i] = sample['num_audio_samples']
        frame_lengths[i] = sample['num_frames']
    
    return {
        'audio_waveforms': padded_audio,
        'arkit_blendshapes': padded_blendshapes,
        'audio_lengths': audio_lengths,
        'frame_lengths': frame_lengths
    }


# Test
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--beat2_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    dataset = SpeechBlendshapeDataset(
        audio_dir=args.audio_dir,
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
        print(f"  Tokens: {batch['speech_tokens'].shape}")
        print(f"  Blendshapes: {batch['arkit_blendshapes'].shape}")
        print(f"  Token lengths: {batch['token_lengths'].tolist()}")
        print(f"  Frame lengths: {batch['frame_lengths'].tolist()}")
        
        if i >= 2:
            break
    
    print("\n✅ DataLoader OK!")