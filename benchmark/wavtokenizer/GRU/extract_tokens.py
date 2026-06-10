'''import torchaudio
import torch
import sys
import os

# Add the 'WavTokenizer' directory to Python's search path
sys.path.append(os.path.join(os.getcwd(), 'WavTokenizer'))

from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer

device=torch.device('cuda')

config_path = "./WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "./pretrained_models/WavTokenizer_small_320_24k_4096.ckpt"
audio_path = 'chunk_0_audio.wav'

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
_,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
print(discrete_code.shape)'''


#!/usr/bin/env python3
"""
Extract WavTokenizer tokens from BEAT2 audio files with train/val/test splits
"""
import torch
import torchaudio
import sys
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd

# Add WavTokenizer to path
sys.path.append(os.path.join(os.getcwd(), 'WavTokenizer'))
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer


def extract_tokens_for_file(audio_path, wavtokenizer, device, chunk_duration=30.0, sample_rate=24000):
    """
    Extract WavTokenizer tokens from audio file with chunking
    
    Args:
        audio_path: Path to .wav file
        wavtokenizer: WavTokenizer model
        device: torch device
        chunk_duration: Chunk size in seconds (default: 30)
        sample_rate: Target sample rate (24kHz for WavTokenizer)
    
    Returns:
        tokens_dict: Dict mapping chunk_key to tokens
    """
    # Load audio
    wav, sr = torchaudio.load(str(audio_path))
    wav = convert_audio(wav, sr, sample_rate, 1)  # Convert to 24kHz mono
    wav = wav.to(device)
    
    # Calculate chunks
    chunk_samples = int(chunk_duration * sample_rate)
    total_samples = wav.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples  # Ceiling division
    
    tokens_dict = {}
    bandwidth_id = torch.tensor([0]).to(device)
    
    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        
        # Extract chunk
        wav_chunk = wav[:, start_sample:end_sample]
        
        # Pad if last chunk is shorter
        if wav_chunk.shape[1] < chunk_samples:
            padding = chunk_samples - wav_chunk.shape[1]
            wav_chunk = torch.nn.functional.pad(wav_chunk, (0, padding))
        
        # Extract tokens
        with torch.no_grad():
            _, discrete_code = wavtokenizer.encode_infer(wav_chunk, bandwidth_id=bandwidth_id)
        
        # discrete_code shape: (1, num_quantizers, num_tokens)
        # We only use the first quantizer (bandwidth_id=0)
        tokens = discrete_code[0, 0, :].cpu().numpy()  # (num_tokens,)
        
        # Create chunk key
        utterance_id = audio_path.stem
        chunk_key = f"{utterance_id}_chunk_{chunk_idx}"
        tokens_dict[chunk_key] = tokens
    
    return tokens_dict


def main(args):
    """Main extraction function"""
    print("="*70)
    print("WavTokenizer Token Extraction for BEAT2")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load WavTokenizer
    print(f"\nLoading WavTokenizer...")
    print(f"  Config: {args.config_path}")
    print(f"  Model: {args.model_path}")
    
    wavtokenizer = WavTokenizer.from_pretrained0802(args.config_path, args.model_path)
    wavtokenizer = wavtokenizer.to(device)
    wavtokenizer.eval()
    print("✓ WavTokenizer loaded")
    
    # Load split CSV
    if args.split_csv:
        split_df = pd.read_csv(args.split_csv, header=None, names=['filename', 'split'])
        print(f"\n✓ Loaded split CSV: {args.split_csv}")
        print(f"  Train: {len(split_df[split_df['split'] == 'train'])} files")
        print(f"  Val: {len(split_df[split_df['split'] == 'val'])} files")
        print(f"  Test: {len(split_df[split_df['split'] == 'test'])} files")
    else:
        split_df = None
        print("\n⚠ No split CSV provided, processing all files")
    
    # Find audio files
    audio_dir = Path(args.audio_dir)
    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"\nFound {len(audio_files)} audio files in {audio_dir}")
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}")
        
        # Filter files for this split
        if split_df is not None:
            split_files = set(split_df[split_df['split'] == split]['filename'])
            files_to_process = [f for f in audio_files if f.stem in split_files]
        else:
            # If no CSV, put everything in train
            if split == 'train':
                files_to_process = audio_files
            else:
                files_to_process = []
        
        if not files_to_process:
            print(f"No files for {split} split, skipping...")
            continue
        
        print(f"Processing {len(files_to_process)} files...")
        
        # Extract tokens
        all_tokens = {}
        
        for audio_path in tqdm(files_to_process, desc=f"Extracting {split}"):
            try:
                tokens_dict = extract_tokens_for_file(
                    audio_path,
                    wavtokenizer,
                    device,
                    chunk_duration=args.chunk_duration
                )
                all_tokens.update(tokens_dict)
            except Exception as e:
                print(f"\n✗ Error processing {audio_path.name}: {e}")
                continue
        
        # Save tokens
        output_path = Path(args.output_dir) / f"{split}_utt2speech_token.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(all_tokens, output_path)
        
        print(f"\n✓ Saved {len(all_tokens)} chunks to {output_path}")
        
        # Print statistics
        if all_tokens:
            token_lengths = [len(tokens) for tokens in all_tokens.values()]
            print(f"  Token sequence lengths:")
            print(f"    Min: {min(token_lengths)}")
            print(f"    Max: {max(token_lengths)}")
            print(f"    Mean: {sum(token_lengths) / len(token_lengths):.1f}")
    
    print(f"\n{'='*70}")
    print("✓ Extraction complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract WavTokenizer tokens from BEAT2 audio with splits"
    )
    
    # WavTokenizer config
    parser.add_argument('--config_path', type=str, default='WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                        help='Path to WavTokenizer config YAML')
    parser.add_argument('--model_path', type=str, default='pretrained_models/WavTokenizer_small_320_24k_4096.ckpt',
                        help='Path to WavTokenizer checkpoint')
    
    # Data paths
    parser.add_argument('--audio_dir', type=str, default='../BEAT2/beat_english_v2.0.0/wave16k',
                        help='Directory containing all .wav files')
    parser.add_argument('--split_csv', type=str, default='../BEAT2/beat_english_v2.0.0/train_test_split.csv',
                        help='CSV file with columns: filename,split (train/val/test)')
    parser.add_argument('--output_dir', type=str, default='data/wavtokenizer_tokens',
                        help='Output directory for token files')
    
    # Chunking
    parser.add_argument('--chunk_duration', type=float, default=30.0,
                        help='Chunk duration in seconds (default: 30)')
    
    args = parser.parse_args()
    
    main(args)