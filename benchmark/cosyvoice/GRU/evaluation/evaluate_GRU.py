import numpy as np
import argparse
import os
from pathlib import Path
import csv
import sys
from scipy.fft import rfft, rfftfreq
import torch

# Import utilities for FLAME to ARKit conversion
sys.path.append('../')
#from utils_adapt import flame_to_arkit_blendshapes


# =============================================================================
# SMOOTHNESS METRICS
# =============================================================================

def compute_jitter(seq, fps=30):
    """
    Jitter = mean magnitude of acceleration (second derivative)
    Lower = smoother animation
    """
    velocity = np.diff(seq, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jitter = np.mean(np.linalg.norm(acceleration, axis=1))
    return jitter


def compute_mav(seq, fps=30):
    """
    Mean Absolute Velocity - captures overall motion intensity
    """
    velocity = np.diff(seq, axis=0) * fps
    return np.mean(np.linalg.norm(velocity, axis=1))


def compute_hf_energy(seq, fps=30, cutoff_hz=5):
    """
    Ratio of high-frequency energy to total energy.
    High-frequency motion often corresponds to jitter/noise.
    Lower ratio = smoother motion
    """
    n = seq.shape[0]
    if n < 4:  # Need minimum frames for FFT
        return 0.0
        
    freqs = rfftfreq(n, 1/fps)
    
    total_energy = 0
    hf_energy = 0
    
    for dim in range(seq.shape[1]):
        fft_vals = np.abs(rfft(seq[:, dim]))**2
        total_energy += np.sum(fft_vals)
        hf_energy += np.sum(fft_vals[freqs > cutoff_hz])
    
    return hf_energy / (total_energy + 1e-8)


def compute_temporal_consistency(seq):
    """
    Measures frame-to-frame consistency via autocorrelation.
    Higher = more temporally consistent/smooth
    """
    if seq.shape[0] < 5:  # Need minimum frames
        return 1.0
        
    autocorr = []
    for lag in [1, 2, 3]:
        correlations = []
        for d in range(seq.shape[1]):
            if np.std(seq[:-lag, d]) > 1e-8 and np.std(seq[lag:, d]) > 1e-8:
                corr = np.corrcoef(seq[:-lag, d], seq[lag:, d])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        if correlations:
            autocorr.append(np.mean(correlations))
    
    return np.mean(autocorr) if autocorr else 1.0


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

'''def load_blendshapes_from_npz(npz_path, is_prediction=True):
    npz_path = Path(npz_path)
    
    # Handle .npy files directly
    if npz_path.suffix == '.npy':
        return np.load(npz_path)
        
    data = np.load(npz_path, allow_pickle=True)
    
    if isinstance(data, np.ndarray):
        return data
    
    if is_prediction:
        possible_keys = ['arkit_blendshapes', 'blendshapes', 'pred_blendshapes', 'data']
        for key in possible_keys:
            if key in data:
                return data[key]
        # Fallback to first array
        arrays = [data[k] for k in data.files if isinstance(data[k], np.ndarray) and data[k].ndim == 2]
        if arrays: return arrays[0]
        raise KeyError(f"Could not find blendshapes in {npz_path}")
    else:
        return data'''

def load_blendshapes_from_npz(npz_path, is_prediction=True):
    npz_path = Path(npz_path)
    
    # Handle .npy files directly
    if npz_path.suffix == '.npy':
        return np.load(npz_path)
        
    data = np.load(npz_path, allow_pickle=True)
    
    # PREDICTION FILES
    if is_prediction:
        if isinstance(data, np.ndarray):
            return data
        
        possible_keys = ['arkit_blendshapes', 'blendshapes', 'pred_blendshapes', 'data']
        for key in possible_keys:
            if key in data.files:
                return data[key]
        
        arrays = [data[k] for k in data.files if isinstance(data[k], np.ndarray) and data[k].ndim == 2]
        if arrays: return arrays[0]
        raise KeyError(f"Could not find blendshapes in {npz_path}")
    
    # GROUND TRUTH FILES - Handle NumPy version differences
    else:
        # NumPy 1.26+: Returns NpzFile directly
        if isinstance(data, np.lib.npyio.NpzFile):
            return data
        
        # NumPy 1.23: Might return 0-d object array containing dict
        elif isinstance(data, np.ndarray) and data.shape == ():
            # Extract the actual dict from the 0-d array
            inner_data = data.item()
            if isinstance(inner_data, dict):
                return inner_data
            else:
                return data
        
        # Fallback
        else:
            return data


'''def extract_flame_facial_params(flame_data):
    # Handle NPZ file object
    if isinstance(flame_data, np.lib.npyio.NpzFile):
        if 'expressions' in flame_data.files and 'poses' in flame_data.files:
            expressions = flame_data['expressions']
            poses = flame_data['poses']
            # BEAT usually has jaw at 66:69 in poses
            jaw = poses[:, 66:69] if poses.shape[-1] >= 69 else poses[:, 6:9]
            return np.concatenate([expressions, jaw], axis=-1)
        elif 'expression' in flame_data.files and 'pose' in flame_data.files:
            expression = flame_data['expression']
            pose = flame_data['pose']
            jaw = pose[:, 66:69] if pose.shape[-1] >= 69 else pose[:, 6:9]
            return np.concatenate([expression, jaw], axis=-1)
            
    # Handle Dictionary
    elif isinstance(flame_data, dict):
        if 'expressions' in flame_data and 'poses' in flame_data:
            jaw = flame_data['poses'][:, 66:69]
            return np.concatenate([flame_data['expressions'], jaw], axis=-1)
            
    # Handle Combined Array (fallback)
    if isinstance(flame_data, np.ndarray) or (hasattr(flame_data, 'files') and 'params' in flame_data):
        arr = flame_data if isinstance(flame_data, np.ndarray) else flame_data['params']
        if arr.shape[-1] >= 300:
            return np.concatenate([arr[:, :100], arr[:, 156:159]], axis=-1) 
            
    raise ValueError("Could not extract FLAME parameters")'''


def extract_flame_facial_params(flame_data):
    """
    Extract facial parameters from BEAT2 FLAME/SMPLX NPZ files
    Handles both NumPy 1.23 and 1.26+ formats
    """
    # Case 1: NpzFile object (NumPy 1.26+)
    if isinstance(flame_data, np.lib.npyio.NpzFile):
        if 'expressions' in flame_data.files and 'poses' in flame_data.files:
            expressions = flame_data['expressions']
            poses = flame_data['poses']
            jaw = poses[:, 66:69] if poses.shape[-1] >= 69 else poses[:, 6:9]
            return np.concatenate([expressions, jaw], axis=-1)
        elif 'expression' in flame_data.files and 'pose' in flame_data.files:
            expression = flame_data['expression']
            pose = flame_data['pose']
            jaw = pose[:, 66:69] if pose.shape[-1] >= 69 else pose[:, 6:9]
            return np.concatenate([expression, jaw], axis=-1)
    
    # Case 2: Dictionary (NumPy 1.23 with allow_pickle)
    elif isinstance(flame_data, dict):
        if 'expressions' in flame_data and 'poses' in flame_data:
            expressions = flame_data['expressions']
            poses = flame_data['poses']
            jaw = poses[:, 66:69] if poses.shape[-1] >= 69 else poses[:, 6:9]
            return np.concatenate([expressions, jaw], axis=-1)
        elif 'expression' in flame_data and 'pose' in flame_data:
            expression = flame_data['expression']
            pose = flame_data['pose']
            jaw = pose[:, 66:69] if pose.shape[-1] >= 69 else pose[:, 6:9]
            return np.concatenate([expression, jaw], axis=-1)
    
    # Case 3: Already extracted array
    elif isinstance(flame_data, np.ndarray):
        if flame_data.shape == ():
            # 0-d object array - extract inner data
            return extract_flame_facial_params(flame_data.item())
        elif flame_data.shape[-1] == 103:
            return flame_data
        elif flame_data.shape[-1] >= 300:
            return np.concatenate([flame_data[:, :100], flame_data[:, 156:159]], axis=-1)
    
    raise ValueError(f"Could not extract FLAME parameters from type: {type(flame_data)}")


def convert_flame_to_arkit(flame_params, transform_matrix):
    """
    Transform FLAME parameters to ARKit blendshapes

    Args:
        flame_params: (*, 103) tensor/array of FLAME parameters
        transform_matrix: (103, 51) transformation matrix

    Returns:
        arkit_blendshapes: (*, 51) tensor/array
    """
    if isinstance(flame_params, torch.Tensor):
        if not isinstance(transform_matrix, torch.Tensor):
            transform_matrix = torch.from_numpy(transform_matrix).float().to(flame_params.device)
        return torch.matmul(flame_params, transform_matrix)
    else:
        if isinstance(transform_matrix, torch.Tensor):
            transform_matrix = transform_matrix.cpu().numpy()
        return np.dot(flame_params, transform_matrix)


def parse_chunk_info(pred_filename):
    """
    Parse prediction filename to extract base_id.
    Handles various formats including _output suffix.
    """
    p = Path(pred_filename)
    name = p.stem
    
    # Remove common suffixes
    for suffix in ['_output', '_pred', '_prediction']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    info = {
        'base_id': name,
        'chunk_id': 0,
        'duration': 30  # Default
    }
    
    # CASE 1: Chunked format (e.g., 1_wayne_0_1_1_chunk_3)
    if '_chunk_' in name:
        parts = name.split('_chunk_')
        info['base_id'] = parts[0]
        if len(parts) > 1 and parts[1].isdigit():
            info['chunk_id'] = int(parts[1])
        info['duration'] = 30
        
    # CASE 2: Test prefix with chunk (e.g., test_1_wayne_0_5_5_004)
    elif name.startswith('test_'):
        clean_name = name[5:]  # Remove 'test_'
        parts = clean_name.split('_')
        
        if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 3:
            # Last part looks like a chunk ID (e.g., "004")
            info['chunk_id'] = int(parts[-1])
            info['base_id'] = "_".join(parts[:-1])
            info['duration'] = 10
        else:
            info['base_id'] = clean_name
            
    return info


def match_gt_file(base_id, gt_path, gt_ext):
    """Find the ground truth file ignoring prefixes/chunks"""
    gt_path = Path(gt_path)
    
    # Strategy 1: Exact match
    f = gt_path / f"{base_id}{gt_ext}"
    if f.exists(): return f
    
    # Strategy 2: Prefix match
    parts = base_id.split('_')
    while len(parts) > 2:
        curr_id = "_".join(parts)
        f = gt_path / f"{curr_id}{gt_ext}"
        if f.exists(): return f
        parts.pop()
        
    return None


def load_split_csv(csv_path, split='test'):
    if not csv_path: return None
    utterance_ids = set()
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[1].strip() == split:
                    utterance_ids.add(row[0].strip())
        print(f"Loaded {len(utterance_ids)} IDs from split '{split}'")
    except Exception as e:
        print(f"Warning: Could not load split CSV: {e}")
        return None
    return utterance_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True, help="Folder with predictions")
    parser.add_argument("--gt_path", type=str, default='../../../BEAT2/beat_english_v2.0.0/smplxflame_30', help="Folder with GT")
    parser.add_argument("--pred_ext", type=str, default=".npz", help="Extension of prediction files")
    parser.add_argument("--gt_ext", type=str, default=".npz", help="Extension of GT files")
    parser.add_argument("--gt_format", type=str, default="flame", choices=["flame", "arkit"])
    parser.add_argument("--flame_transform", type=str, default="../../arkit_to_flame.npy")
    parser.add_argument("--dataset", type=str, default="beat", choices=["beat", "vocaset"])
    parser.add_argument("--split_csv", type=str, default='../../../BEAT2/beat_english_v2.0.0/train_test_split.csv')
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hf_cutoff", type=float, default=5.0, help="Cutoff frequency (Hz) for high-frequency energy")
    parser.add_argument("--verbose", action="store_true", help="Print detailed matching info")
    args = parser.parse_args()

    # Load resources
    if args.gt_format == "flame":
        if not os.path.exists(args.flame_transform):
            raise FileNotFoundError(f"Transform file missing: {args.flame_transform}")
        arkit_to_flame = np.load(args.flame_transform)
        flame_transform = np.linalg.pinv(arkit_to_flame)
    
    split_ids = load_split_csv(args.split_csv, args.split)

    # BEAT Masks
    if args.dataset == "beat":
        upper_mask = list(range(0,22)) + [49,50]
        mouth_mask = list(range(22,49))
        print("Using BEAT dataset masks")
    else:
        mouth_mask = list(range(94, 114)) + list(range(146, 178)) + list(range(183, 192))
        upper_mask = [x for x in range(192) if x not in mouth_mask]

    # Metrics accumulators
    mve, lve = 0, 0
    motion_std_diff, abs_motion_std_diff = [], []
    jitter_pred_list, jitter_gt_list = [], []
    mav_pred_list, mav_gt_list = [], []
    hf_pred_list, hf_gt_list = [], []
    tc_pred_list, tc_gt_list = [], []
    
    num_seq = 0
    skipped = []

    # Find prediction files
    pred_path = Path(args.pred_path)
    search_pattern = f"*{args.pred_ext}"
    pred_files = list(pred_path.glob(search_pattern))
    
    print(f"Searching {pred_path} for {search_pattern}")
    print(f"Found {len(pred_files)} files. Processing...\n")

    for pred_f in pred_files:
        try:
            # 1. Parse filename
            info = parse_chunk_info(pred_f.name)
            base_id = info['base_id']
            
            if args.verbose:
                print(f"File: {pred_f.name} -> base_id: {base_id}")
            
            # 2. Check if in split
            if split_ids is not None and base_id not in split_ids:
                if args.verbose:
                    print(f"  -> Not in split, skipping")
                continue

            # 3. Find GT file
            gt_f = match_gt_file(base_id, args.gt_path, args.gt_ext)
            if not gt_f:
                skipped.append((pred_f.name, f"No GT found for base_id: {base_id}"))
                continue
                
            if args.verbose:
                print(f"  -> Matched GT: {gt_f.name}")
            
            # Load data
            pred_seq = load_blendshapes_from_npz(pred_f, is_prediction=True)
            
            if args.gt_format == "flame":
                gt_raw = load_blendshapes_from_npz(gt_f, is_prediction=False)
                gt_facial = extract_flame_facial_params(gt_raw)
                gt_seq_full = convert_flame_to_arkit(gt_facial, flame_transform)
            else:
                gt_seq_full = load_blendshapes_from_npz(gt_f, is_prediction=True)

            # 4. Temporal alignment
            chunk_start_sec = info['chunk_id'] * info['duration']
            start_frame = int(chunk_start_sec * args.fps)
            
            if start_frame >= gt_seq_full.shape[0]:
                skipped.append((pred_f.name, f"Start frame {start_frame} > GT len {gt_seq_full.shape[0]}"))
                continue

            gt_seq = gt_seq_full[start_frame:]
            
            # 5. Length alignment
            min_len = min(pred_seq.shape[0], gt_seq.shape[0])
            
            if min_len < 10:
                 skipped.append((pred_f.name, f"Overlap too short: {min_len} frames"))
                 continue

            pred_seq = pred_seq[:min_len]
            gt_seq = gt_seq[:min_len]

            # 6. Compute metrics
            # MVE
            mve += np.linalg.norm(pred_seq - gt_seq, axis=1).mean()
            # LVE
            lve += np.linalg.norm(pred_seq[:, mouth_mask] - gt_seq[:, mouth_mask], axis=1).mean()
            
            # FDD
            gt_std = np.std(np.sum(np.square(gt_seq[:, upper_mask]), axis=1))
            pred_std = np.std(np.sum(np.square(pred_seq[:, upper_mask]), axis=1))
            motion_std_diff.append(gt_std - pred_std)
            abs_motion_std_diff.append(abs(gt_std - pred_std))
            
            # Smoothness metrics
            jitter_pred_list.append(compute_jitter(pred_seq, args.fps))
            jitter_gt_list.append(compute_jitter(gt_seq, args.fps))
            mav_pred_list.append(compute_mav(pred_seq, args.fps))
            mav_gt_list.append(compute_mav(gt_seq, args.fps))
            hf_pred_list.append(compute_hf_energy(pred_seq, args.fps, args.hf_cutoff))
            hf_gt_list.append(compute_hf_energy(gt_seq, args.fps, args.hf_cutoff))
            tc_pred_list.append(compute_temporal_consistency(pred_seq))
            tc_gt_list.append(compute_temporal_consistency(gt_seq))
            
            num_seq += 1

        except Exception as e:
            skipped.append((pred_f.name, str(e)))

    # Report
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS")
    print(f"Processed: {num_seq} sequences | Skipped: {len(skipped)}")
    print("="*70)
    
    if num_seq > 0:
        print("\n--- RECONSTRUCTION METRICS ---")
        print(f"  MVE (Mean Vertex Error):      {mve/num_seq:.5f}")
        print(f"  LVE (Lip Vertex Error):       {lve/num_seq:.5f}")
        print(f"  FDD (Face Dynamics Dev):      {np.mean(motion_std_diff):.5f}")
        print(f"  FDD Absolute:                 {np.mean(abs_motion_std_diff):.5f}")
        
        print("\n--- SMOOTHNESS METRICS ---")
        print(f"  {'Metric':<25} {'Pred':>12} {'GT':>12} {'Ratio (P/G)':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
        
        # Jitter
        jitter_pred_mean = np.mean(jitter_pred_list)
        jitter_gt_mean = np.mean(jitter_gt_list)
        jitter_ratio = jitter_pred_mean / (jitter_gt_mean + 1e-8)
        print(f"  {'Jitter (↓ better)':<25} {jitter_pred_mean:>12.5f} {jitter_gt_mean:>12.5f} {jitter_ratio:>12.3f}")
        
        # MAV
        mav_pred_mean = np.mean(mav_pred_list)
        mav_gt_mean = np.mean(mav_gt_list)
        mav_ratio = mav_pred_mean / (mav_gt_mean + 1e-8)
        print(f"  {'MAV (motion intensity)':<25} {mav_pred_mean:>12.5f} {mav_gt_mean:>12.5f} {mav_ratio:>12.3f}")
        
        # HF Energy
        hf_pred_mean = np.mean(hf_pred_list)
        hf_gt_mean = np.mean(hf_gt_list)
        hf_ratio = hf_pred_mean / (hf_gt_mean + 1e-8)
        print(f"  {'HF Energy (↓ better)':<25} {hf_pred_mean:>12.5f} {hf_gt_mean:>12.5f} {hf_ratio:>12.3f}")
        
        # Temporal Consistency
        tc_pred_mean = np.mean(tc_pred_list)
        tc_gt_mean = np.mean(tc_gt_list)
        tc_ratio = tc_pred_mean / (tc_gt_mean + 1e-8)
        print(f"  {'Temp. Consistency (↑)':<25} {tc_pred_mean:>12.5f} {tc_gt_mean:>12.5f} {tc_ratio:>12.3f}")
        
        print("\n--- INTERPRETATION GUIDE ---")
        print("  Jitter Ratio < 1.0  →  Your model is SMOOTHER than GT")
        print("  HF Energy Ratio < 1.0  →  Less high-frequency noise")
        print("  Temp. Consistency Ratio > 1.0  →  More temporally coherent")
        print("  MAV Ratio ≈ 1.0  →  Similar motion intensity to GT")
    
    if len(skipped) > 0:
        print(f"\n--- SKIPPED FILES ({len(skipped)} total) ---")
        for s in skipped[:10]: 
            print(f"  - {s[0]}: {s[1]}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped)-10} more.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
    

'''

 python evaluate4.py \
    --pred_path ../results/test_set_results_100ep_GRU_CV2_combined/ \
    --pred_ext .npy

'''