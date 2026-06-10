#!/usr/bin/env python3
"""
Extract SpeechTokenizer tokens from BEAT2 audio files with train/val/test splits
"""
import torch
import torchaudio
import sys
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
from speechtokenizer import SpeechTokenizer


def extract_tokens_for_file(audio_path, model, device, chunk_duration=30.0):
    """
    Extract SpeechTokenizer tokens from audio file with chunking

    Args:
        audio_path: Path to .wav file
        model: SpeechTokenizer model
        device: torch device
        chunk_duration: Chunk size in seconds (default: 30)

    Returns:
        tokens_dict: Dict mapping chunk_key to dict with 'semantic' and 'acoustic' tokens
    """
    sample_rate = model.sample_rate

    # Load and preprocess audio
    wav, sr = torchaudio.load(str(audio_path))

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav[:1, :]

    # Resample if needed
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    wav = wav.to(device)

    # Calculate chunks
    chunk_samples = int(chunk_duration * sample_rate)
    total_samples = wav.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    tokens_dict = {}

    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, total_samples)

        wav_chunk = wav[:, start_sample:end_sample]

        # Pad last chunk if shorter
        if wav_chunk.shape[1] < chunk_samples:
            padding = chunk_samples - wav_chunk.shape[1]
            wav_chunk = torch.nn.functional.pad(wav_chunk, (0, padding))

        # SpeechTokenizer expects (B, C, T)
        wav_chunk = wav_chunk.unsqueeze(0)

        with torch.no_grad():
            codes = model.encode(wav_chunk)  # (n_q, B, T)

        # RVQ layer 1: semantic tokens (content info)
        semantic_tokens = codes[0, 0, :].cpu().numpy()   # (T,)
        # RVQ layers 2+: acoustic tokens (timbre info)
        acoustic_tokens = codes[1:, 0, :].cpu().numpy()  # (n_q-1, T)

        utterance_id = audio_path.stem
        chunk_key = f"{utterance_id}_chunk_{chunk_idx}"
        tokens_dict[chunk_key] = {
            'semantic': semantic_tokens,   # (T,)
            'acoustic': acoustic_tokens,   # (n_q-1, T)
        }

    return tokens_dict


def main(args):
    print("="*70)
    print("SpeechTokenizer Token Extraction for BEAT2")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nLoading SpeechTokenizer...")
    print(f"  Config: {args.config_path}")
    print(f"  Model:  {args.model_path}")

    model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.model_path)
    model = model.to(device)
    model.eval()
    print(f"✓ SpeechTokenizer loaded (sample_rate={model.sample_rate}Hz)")

    # Load split CSV
    if args.split_csv:
        split_df = pd.read_csv(args.split_csv, header=None, names=['filename', 'split'])
        print(f"\n✓ Loaded split CSV: {args.split_csv}")
        for s in ['train', 'val', 'test']:
            print(f"  {s.capitalize()}: {len(split_df[split_df['split'] == s])} files")
    else:
        split_df = None
        print("\n⚠ No split CSV provided, processing all files as train")

    audio_dir = Path(args.audio_dir)
    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"\nFound {len(audio_files)} audio files in {audio_dir}")

    for split in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}")

        if split_df is not None:
            split_files = set(split_df[split_df['split'] == split]['filename'])
            files_to_process = [f for f in audio_files if f.stem in split_files]
        else:
            files_to_process = audio_files if split == 'train' else []

        if not files_to_process:
            print(f"No files for {split} split, skipping...")
            continue

        print(f"Processing {len(files_to_process)} files...")

        all_tokens = {}
        for audio_path in tqdm(files_to_process, desc=f"Extracting {split}"):
            try:
                tokens_dict = extract_tokens_for_file(
                    audio_path, model, device,
                    chunk_duration=args.chunk_duration
                )
                all_tokens.update(tokens_dict)
            except Exception as e:
                print(f"\n✗ Error processing {audio_path.name}: {e}")
                continue

        output_path = Path(args.output_dir) / f"{split}_utt2speech_token.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_tokens, output_path)
        print(f"\n✓ Saved {len(all_tokens)} chunks to {output_path}")

        if all_tokens:
            lengths = [v['semantic'].shape[0] for v in all_tokens.values()]
            print(f"  Semantic token sequence lengths:")
            print(f"    Min:  {min(lengths)}")
            print(f"    Max:  {max(lengths)}")
            print(f"    Mean: {sum(lengths)/len(lengths):.1f}")

    print(f"\n{'='*70}")
    print("✓ Extraction complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SpeechTokenizer tokens from BEAT2 audio with splits"
    )

    parser.add_argument('--config_path', type=str,
                        default='SpeechTokenizer/config/spt_base_cfg.json')
    parser.add_argument('--model_path', type=str,
                        default='pretrained_models/SpeechTokenizer.pt')
    parser.add_argument('--audio_dir', type=str,
                        default='../BEAT2/beat_english_v2.0.0/wave16k')
    parser.add_argument('--split_csv', type=str,
                        default='../BEAT2/beat_english_v2.0.0/train_test_split.csv')
    parser.add_argument('--output_dir', type=str,
                        default='speechtokenizer_tokens')
    parser.add_argument('--chunk_duration', type=float, default=30.0)

    args = parser.parse_args()
    main(args)