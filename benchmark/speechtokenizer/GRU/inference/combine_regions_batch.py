#!/usr/bin/env python3
"""
Combine blendshape predictions from two folders by adding them together.
One folder typically has mouth predictions, the other has upper face predictions.

The script matches files by name and adds the blendshape values together.
"""

import argparse
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_blendshapes(npz_path):
    """
    Load blendshape data from .npz or .npy file
    
    Args:
        npz_path: Path to file
        
    Returns:
        numpy array: blendshapes [frames, 51]
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # For .npz files, try to find blendshape data
    if isinstance(data, np.lib.npyio.NpzFile):
        blendshapes = None
        for key in ['arkit_blendshapes', 'blendshapes', 'predictions']:
            if key in data.files:
                blendshapes = data[key]
                break
        
        if blendshapes is None:
            raise ValueError(f"No blendshape data found in {npz_path}. Available keys: {data.files}")
        return blendshapes
    else:
        # .npy file - direct array
        return data


def add_blendshapes(seq1, seq2, clip=True):
    """
    Add two blendshape sequences together
    
    Args:
        seq1: First blendshape sequence [frames, 51]
        seq2: Second blendshape sequence [frames, 51]
        clip: Whether to clip final values to [0, 1]
        
    Returns:
        numpy array: Combined blendshapes [frames, 51]
    """
    # Ensure same length
    min_frames = min(seq1.shape[0], seq2.shape[0])
    
    if seq1.shape[0] != seq2.shape[0]:
        logging.warning(f"Length mismatch: seq1={seq1.shape[0]}, seq2={seq2.shape[0]}. Using {min_frames} frames.")
        seq1 = seq1[:min_frames]
        seq2 = seq2[:min_frames]
    
    # Add
    result = seq1 + seq2
    
    # Clip to valid range
    if clip:
        result = np.clip(result, 0.0, 1.0)
    
    return result


def find_matching_files(dir1, dir2, extensions=['.npz', '.npy']):
    """
    Find matching files in both directories
    
    Args:
        dir1: First directory
        dir2: Second directory
        extensions: File extensions to look for
        
    Returns:
        list: List of tuples (filename, path1, path2)
    """
    dir1 = Path(dir1)
    dir2 = Path(dir2)
    
    # Get all files from first directory
    files1 = {}
    for ext in extensions:
        for f in dir1.glob(f"*{ext}"):
            files1[f.name] = f
    
    # Find matching files in second directory
    matching = []
    missing_in_dir2 = []
    
    for filename, path1 in files1.items():
        path2 = dir2 / filename
        
        if path2.exists():
            matching.append((filename, path1, path2))
        else:
            missing_in_dir2.append(filename)
    
    # Check for files only in dir2
    files2 = set()
    for ext in extensions:
        for f in dir2.glob(f"*{ext}"):
            files2.add(f.name)
    
    only_in_dir2 = files2 - set(files1.keys())
    
    # Report
    logging.info(f"\nFile matching:")
    logging.info(f"  Directory 1: {len(files1)} files")
    logging.info(f"  Directory 2: {len(files2)} files")
    logging.info(f"  Matched pairs: {len(matching)}")
    
    if missing_in_dir2:
        logging.warning(f"  Missing in directory 2: {len(missing_in_dir2)} files")
        if len(missing_in_dir2) <= 5:
            for f in missing_in_dir2:
                logging.warning(f"    - {f}")
    
    if only_in_dir2:
        logging.warning(f"  Only in directory 2: {len(only_in_dir2)} files")
        if len(only_in_dir2) <= 5:
            for f in only_in_dir2:
                logging.warning(f"    - {f}")
    
    return matching


def main():
    parser = argparse.ArgumentParser(
        description="Combine predictions from two folders by adding them together.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dir1',
        type=str,
        required=True,
        help="First predictions directory (e.g., mouth model)"
    )
    parser.add_argument(
        '--dir2',
        type=str,
        required=True,
        help="Second predictions directory (e.g., upper face model)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Output directory for combined predictions"
    )
    parser.add_argument(
        '--no_clip',
        action='store_true',
        help="Don't clip final values to [0, 1] range"
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help="FPS for output file metadata (default: 30)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COMBINE PREDICTIONS FROM TWO FOLDERS")
    print("="*80)
    
    # Find matching files
    matching_files = find_matching_files(args.dir1, args.dir2)
    
    if len(matching_files) == 0:
        logging.error("No matching files found!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each pair
    print(f"\nProcessing {len(matching_files)} file pairs...")
    
    stats = {
        'successful': 0,
        'failed': 0,
        'failed_files': []
    }
    
    for filename, path1, path2 in tqdm(matching_files, desc="Combining predictions"):
        try:
            # Load both predictions
            seq1 = load_blendshapes(path1)
            seq2 = load_blendshapes(path2)
            
            # Add them together
            combined = add_blendshapes(seq1, seq2, clip=not args.no_clip)
            
            # Save
            output_path = output_dir / filename
            
            # Determine output format from input
            if filename.endswith('.npz'):
                np.savez(
                    output_path,
                    arkit_blendshapes=combined,
                    fps=args.fps,
                    num_frames=combined.shape[0],
                    source_dir1=str(path1.parent.name),
                    source_dir2=str(path2.parent.name)
                )
            else:  # .npy
                np.save(output_path, combined)
            
            stats['successful'] += 1
            
        except Exception as e:
            logging.error(f"Failed to process {filename}: {str(e)}")
            stats['failed'] += 1
            stats['failed_files'].append(filename)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files: {len(matching_files)}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        print(f"\nFailed files:")
        for f in stats['failed_files'][:10]:
            print(f"  - {f}")
        if len(stats['failed_files']) > 10:
            print(f"  ... and {len(stats['failed_files']) - 10} more")
    
    print(f"\n✅ Combined predictions saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


'''

python combine_regions_batch.py \
    --dir1 results/test_set_results_100ep_GRU_frozen_cben \
    --dir2 results/test_set_results_100ep_GRU_frozen_mj \
    --output_dir results/test_set_results_100ep_GRU_WAV_combined
    --pred_ext .npy

'''