#!/usr/bin/env python3
"""
Helper script to create wav.scp file from a directory of .wav files
"""
import argparse
from pathlib import Path

def create_wav_scp(input_dir, output_file):
    """
    Create wav.scp file from directory containing .wav files
    
    Format: <utterance_id> <path_to_wav>
    """
    input_path = Path(input_dir)
    wav_files = sorted(input_path.rglob("*.wav"))
    
    if not wav_files:
        print(f"Warning: No .wav files found in {input_dir}")
        return
    
    with open(output_file, 'w') as f:
        for wav_file in wav_files:
            # Use filename without extension as utterance ID
            utt_id = wav_file.stem
            # Use absolute path
            wav_path = wav_file.absolute()
            f.write(f"{utt_id} {wav_path}\n")
    
    print(f"Created {output_file} with {len(wav_files)} entries")
    print(f"First few entries:")
    with open(output_file) as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"  {line.strip()}")
            else:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create wav.scp file from directory of .wav files")
    parser.add_argument("--input_dir", type=str,
                        default="../BEAT2/beat_english_v2.0.0/wave16k", 
                        help="Directory containing .wav files")
    parser.add_argument("--output", type=str, default="all_speakers/all_speakers.scp",
                        help="Output wav.scp file path")
    args = parser.parse_args()
    
    create_wav_scp(args.input_dir, args.output)