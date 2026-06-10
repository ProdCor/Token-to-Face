#!/usr/bin/env python3
"""
Merge HuBERT feature parts into a single file.

Usage:
    python merge_hubert_parts.py --input_dir hubert_features
"""

import torch
from pathlib import Path
import argparse


def main(args):
    input_dir = Path(args.input_dir)
    part_files = sorted(input_dir.glob("*_part_*.pt"))

    if not part_files:
        print(f"No part files found in {input_dir}")
        return

    print(f"Found {len(part_files)} part files:")
    for f in part_files:
        print(f"  {f.name}")

    merged = {}
    for pt_file in part_files:
        print(f"Loading {pt_file.name}...")
        data = torch.load(pt_file, map_location="cpu")
        merged.update(data)
        del data

    print(f"\nTotal: {len(merged)} chunks")

    # Show stats
    lengths = [v.shape[0] for v in merged.values()]
    feat_dim = next(iter(merged.values())).shape[1]
    print(f"  Feature dim: {feat_dim}")
    print(f"  Lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")
    print(f"  Dtype: {next(iter(merged.values())).dtype}")

    output_path = input_dir / "utt2hubert_features_all.pt"
    torch.save(merged, output_path)
    print(f"\n✓ Saved merged features to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge HuBERT feature parts")
    parser.add_argument('--input_dir', type=str, default='hubert_features')
    args = parser.parse_args()
    main(args)