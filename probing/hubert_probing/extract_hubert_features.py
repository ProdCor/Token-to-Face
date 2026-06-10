#!/usr/bin/env python3
"""
Extract HuBERT base last-hidden-state features from BEAT2 audio files.

Saves features incrementally to avoid OOM, then merge with merge script.

Usage:
    python extract_hubert_features.py \
        --audio_dir ../BEAT2/beat_english_v2.0.0/wave16k \
        --output_dir hubert_features \
        --split_csv ../BEAT2/beat_english_v2.0.0/train_test_split.csv

    # Then merge:
    python merge_hubert_parts.py --input_dir hubert_features
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
from transformers import HubertModel


def extract_features_for_file(audio_path, model, device, chunk_duration=30.0):
    """
    Extract HuBERT last-hidden-state features from audio file with chunking.

    Returns:
        features_dict: {chunk_key: np.array (T, 768) float16}
    """
    sample_rate = 16000

    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav[:1, :]
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    wav = wav.squeeze(0)  # (samples,)

    chunk_samples = int(chunk_duration * sample_rate)
    total_samples = wav.shape[0]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    features_dict = {}

    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        wav_chunk = wav[start_sample:end_sample]

        # Normalize (same as Wav2Vec2Processor would do)
        wav_chunk = (wav_chunk - wav_chunk.mean()) / (wav_chunk.std() + 1e-6)

        input_values = wav_chunk.unsqueeze(0).to(device)  # (1, samples)

        with torch.no_grad():
            outputs = model(input_values)
            features = outputs.last_hidden_state  # (1, T, 768)

        features_np = features[0].cpu().half().numpy()  # (T, 768) float16

        utterance_id = audio_path.stem
        chunk_key = f"{utterance_id}_chunk_{chunk_idx}"
        features_dict[chunk_key] = features_np

    return features_dict


def main(args):
    print("=" * 70)
    print("HuBERT Feature Extraction for BEAT2")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading HuBERT base from {args.model_path}...")
    model = HubertModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    print(f"✓ HuBERT loaded (hidden_size={model.config.hidden_size}, output_fps=50Hz)")

    # Load split CSV
    split_df = None
    if args.split_csv:
        split_df = pd.read_csv(args.split_csv, header=None, names=['filename', 'split'])
        print(f"\n✓ Loaded split CSV: {args.split_csv}")
        for s in ['train', 'val', 'test']:
            print(f"  {s.capitalize()}: {len(split_df[split_df['split'] == s])} files")

    audio_dir = Path(args.audio_dir)
    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"\nFound {len(audio_files)} audio files in {audio_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        print(f"\n{'=' * 70}")
        print(f"Processing {split.upper()} split")
        print(f"{'=' * 70}")

        if split_df is not None:
            split_files = set(split_df[split_df['split'] == split]['filename'])
            files_to_process = [f for f in audio_files if f.stem in split_files]
        else:
            files_to_process = audio_files if split == 'train' else []

        if not files_to_process:
            print(f"No files for {split} split, skipping...")
            continue

        print(f"Processing {len(files_to_process)} files...")

        all_features = {}
        part_idx = 0

        for i, audio_path in enumerate(tqdm(files_to_process, desc=f"Extracting {split}")):
            try:
                feat_dict = extract_features_for_file(
                    audio_path, model, device,
                    chunk_duration=args.chunk_duration
                )
                all_features.update(feat_dict)
            except Exception as e:
                print(f"\n✗ Error processing {audio_path.name}: {e}")
                continue

            # Periodic save to free memory
            if (i + 1) % args.save_every == 0:
                part_path = output_dir / f"{split}_part_{part_idx:03d}.pt"
                torch.save(all_features, part_path)
                print(f"\n  Checkpoint: {len(all_features)} chunks → {part_path}")
                all_features = {}
                part_idx += 1

        # Save remaining
        if all_features:
            part_path = output_dir / f"{split}_part_{part_idx:03d}.pt"
            torch.save(all_features, part_path)
            print(f"\n  Final save: {len(all_features)} chunks → {part_path}")

        # Count total chunks for this split
        total_chunks = 0
        for pt_file in output_dir.glob(f"{split}_part_*.pt"):
            data = torch.load(pt_file, map_location="cpu")
            total_chunks += len(data)
            del data
        print(f"\n✓ {split}: {total_chunks} total chunks saved")

    print(f"\n{'=' * 70}")
    print("✓ Extraction complete!")
    print(f"  Run merge script to combine parts:")
    print(f"  python merge_hubert_parts.py --input_dir {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract HuBERT features from BEAT2 audio")
    parser.add_argument('--model_path', type=str,
                        default='../../FaceDiffuser/pretrained_models/hubert/hubert-base-ls960')
    parser.add_argument('--audio_dir', type=str,
                        default='../../BEAT2/beat_english_v2.0.0/wave16k')
    parser.add_argument('--split_csv', type=str,
                        default='../../BEAT2/beat_english_v2.0.0/train_test_split.csv')
    parser.add_argument('--output_dir', type=str,
                        default='hubert_features')
    parser.add_argument('--chunk_duration', type=float, default=30.0)
    parser.add_argument('--save_every', type=int, default=100,
                        help="Save checkpoint every N files to avoid OOM")
    args = parser.parse_args()
    main(args)