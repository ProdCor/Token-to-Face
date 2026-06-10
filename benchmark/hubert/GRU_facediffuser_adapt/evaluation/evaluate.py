"""
evaluate_compare.py
--------------------
Evaluate and compare two (or more) facial animation models against ground truth.

Supports:
  - .npz prediction files  (model that uses ARKit blendshape NPZ outputs)
  - .npy prediction files  (model that uses raw NumPy array outputs)
  - Any mix of the above across models

Usage example:
  python evaluate_compare.py \
      --models \
          "NPZ Model:../results/model_npz/::.npz" \
          "NpyModel:../results/model_npy/::.npy" \
      --gt_path ../../../BEAT2/beat_english_v2.0.0/smplxflame_30 \
      --flame_transform ../../arkit_to_flame.npy \
      --output_dir ./comparison_results
"""

import numpy as np
import argparse
import os
from pathlib import Path
import csv
import sys
from scipy.fft import rfft, rfftfreq


# =============================================================================
# FLAME → ARKit CONVERSION  (no external import needed)
# =============================================================================

def convert_flame_to_arkit(flame_params, transform_matrix):
    """
    flame_params   : (T, 103) ndarray
    transform_matrix: (103, N_arkit) ndarray  — the *flame→arkit* matrix
                      (i.e. pinv of the arkit_to_flame matrix)
    """
    if isinstance(transform_matrix, np.ndarray):
        return np.dot(flame_params, transform_matrix)
    # torch fallback (unlikely in eval, but kept for safety)
    import torch
    tm = torch.from_numpy(transform_matrix).float()
    fp = torch.from_numpy(flame_params).float()
    return torch.matmul(fp, tm).numpy()


# =============================================================================
# SMOOTHNESS METRICS
# =============================================================================

def compute_jitter(seq, fps=30):
    """Mean magnitude of acceleration (second derivative). Lower = smoother."""
    velocity     = np.diff(seq, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    return np.mean(np.linalg.norm(acceleration, axis=1))


def compute_mav(seq, fps=30):
    """Mean Absolute Velocity — captures overall motion intensity."""
    velocity = np.diff(seq, axis=0) * fps
    return np.mean(np.linalg.norm(velocity, axis=1))


def compute_hf_energy(seq, fps=30, cutoff_hz=5.0):
    """Ratio of high-frequency energy to total energy. Lower = smoother."""
    n = seq.shape[0]
    if n < 4:
        return 0.0
    freqs        = rfftfreq(n, 1 / fps)
    total_energy = 0.0
    hf_energy    = 0.0
    for dim in range(seq.shape[1]):
        fft_vals      = np.abs(rfft(seq[:, dim])) ** 2
        total_energy += np.sum(fft_vals)
        hf_energy    += np.sum(fft_vals[freqs > cutoff_hz])
    return hf_energy / (total_energy + 1e-8)


def compute_temporal_consistency(seq):
    """Frame-to-frame consistency via autocorrelation. Higher = smoother."""
    if seq.shape[0] < 5:
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
# DATA LOADING
# =============================================================================

def load_prediction(npz_path):
    """Load a prediction file (.npy or .npz)."""
    npz_path = Path(npz_path)

    if npz_path.suffix == '.npy':
        return np.load(npz_path)

    data = np.load(npz_path, allow_pickle=True)

    if isinstance(data, np.ndarray):
        return data  # already a plain array

    # NpzFile
    for key in ['arkit_blendshapes', 'blendshapes', 'pred_blendshapes', 'data']:
        if key in data.files:
            return data[key]

    # fallback: first 2-D array
    arrays = [data[k] for k in data.files
              if isinstance(data[k], np.ndarray) and data[k].ndim == 2]
    if arrays:
        return arrays[0]

    raise KeyError(f"Could not find blendshapes array in {npz_path}")


def load_gt_file(gt_path):
    """
    Load a ground-truth NPZ, handling both NumPy 1.23 (returns 0-d object array)
    and NumPy 1.26+ (returns NpzFile directly).
    """
    gt_path = Path(gt_path)
    data = np.load(gt_path, allow_pickle=True)

    # NumPy 1.26+: NpzFile
    if isinstance(data, np.lib.npyio.NpzFile):
        return data

    # NumPy 1.23: 0-d object array wrapping a dict
    if isinstance(data, np.ndarray) and data.shape == ():
        inner = data.item()
        if isinstance(inner, dict):
            return inner
        return data

    return data


def extract_flame_facial_params(flame_data):
    """
    Extract (T, 103) facial params from BEAT2 SMPLX-FLAME NPZ data.
    Handles NpzFile, dict, plain array, and 0-d object array.
    """
    # 0-d object array → recurse on inner value
    if isinstance(flame_data, np.ndarray) and flame_data.shape == ():
        return extract_flame_facial_params(flame_data.item())

    def _expressions_jaw(expressions, poses):
        jaw = poses[:, 66:69] if poses.shape[-1] >= 69 else poses[:, 6:9]
        return np.concatenate([expressions, jaw], axis=-1)

    # NpzFile
    if isinstance(flame_data, np.lib.npyio.NpzFile):
        files = flame_data.files
        if 'expressions' in files and 'poses' in files:
            return _expressions_jaw(flame_data['expressions'], flame_data['poses'])
        if 'expression' in files and 'pose' in files:
            return _expressions_jaw(flame_data['expression'], flame_data['pose'])

    # Dict
    if isinstance(flame_data, dict):
        if 'expressions' in flame_data and 'poses' in flame_data:
            return _expressions_jaw(flame_data['expressions'], flame_data['poses'])
        if 'expression' in flame_data and 'pose' in flame_data:
            return _expressions_jaw(flame_data['expression'], flame_data['pose'])

    # Already extracted / combined array
    if isinstance(flame_data, np.ndarray):
        if flame_data.ndim == 2:
            if flame_data.shape[-1] == 103:
                return flame_data
            if flame_data.shape[-1] >= 300:
                return np.concatenate([flame_data[:, :100], flame_data[:, 156:159]], axis=-1)

    raise ValueError(f"Could not extract FLAME parameters from type: {type(flame_data)}")


# =============================================================================
# FILE MATCHING HELPERS
# =============================================================================

def parse_chunk_info(pred_filename):
    """Return base_id, chunk_id, and chunk duration from a prediction filename."""
    name = Path(pred_filename).stem

    for suffix in ['_output', '_pred', '_prediction']:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    info = {'base_id': name, 'chunk_id': 0, 'duration': 30}

    if '_chunk_' in name:
        parts = name.split('_chunk_')
        info['base_id'] = parts[0]
        if len(parts) > 1 and parts[1].isdigit():
            info['chunk_id'] = int(parts[1])
        info['duration'] = 30

    elif name.startswith('test_'):
        clean = name[5:]
        parts = clean.split('_')
        if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 3:
            info['chunk_id'] = int(parts[-1])
            info['base_id']  = '_'.join(parts[:-1])
            info['duration'] = 10
        else:
            info['base_id'] = clean

    return info


def match_gt_file(base_id, gt_path, gt_ext):
    """Find a GT file by exact match first, then by progressively shorter prefix."""
    gt_path = Path(gt_path)

    f = gt_path / f"{base_id}{gt_ext}"
    if f.exists():
        return f

    parts = base_id.split('_')
    while len(parts) > 2:
        f = gt_path / f"{'_'.join(parts)}{gt_ext}"
        if f.exists():
            return f
        parts.pop()

    return None


def load_split_csv(csv_path, split='test'):
    if not csv_path:
        return None
    ids = set()
    try:
        with open(csv_path) as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[1].strip() == split:
                    ids.add(row[0].strip())
        print(f"  Split CSV: loaded {len(ids)} IDs for split='{split}'")
    except Exception as e:
        print(f"  Warning: could not load split CSV: {e}")
        return None
    return ids


# =============================================================================
# PER-SAMPLE CSV EXPORT
# =============================================================================

FIELDNAMES = [
    'model_name', 'sample_id', 'num_frames',
    'mve', 'lve', 'fdd', 'fdd_abs',
    'jitter_pred', 'jitter_gt', 'jitter_ratio',
    'mav_pred',    'mav_gt',    'mav_ratio',
    'hf_pred',     'hf_gt',     'hf_ratio',
    'tc_pred',     'tc_gt',     'tc_ratio',
]


def write_combined_csv(output_path, all_rows_by_model):
    """
    Write a single CSV with all models' per-sample results.
    all_rows_by_model: list of (model_name, rows) tuples.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for model_name, rows in all_rows_by_model:
            for r in rows:
                writer.writerow({
                    'model_name':   model_name,
                    'sample_id':    r['sample_id'],
                    'num_frames':   r['num_frames'],
                    'mve':          r['mve'],
                    'lve':          r['lve'],
                    'fdd':          r['fdd'],
                    'fdd_abs':      r['fdd_abs'],
                    'jitter_pred':  r['jitter_pred'],
                    'jitter_gt':    r['jitter_gt'],
                    'jitter_ratio': r['jitter_pred'] / (r['jitter_gt'] + 1e-8),
                    'mav_pred':     r['mav_pred'],
                    'mav_gt':       r['mav_gt'],
                    'mav_ratio':    r['mav_pred'] / (r['mav_gt'] + 1e-8),
                    'hf_pred':      r['hf_pred'],
                    'hf_gt':        r['hf_gt'],
                    'hf_ratio':     r['hf_pred'] / (r['hf_gt'] + 1e-8),
                    'tc_pred':      r['tc_pred'],
                    'tc_gt':        r['tc_gt'],
                    'tc_ratio':     r['tc_pred'] / (r['tc_gt'] + 1e-8),
                })


# =============================================================================
# SINGLE-MODEL EVALUATION
# =============================================================================

def evaluate_model(
    pred_path, pred_ext, gt_path, gt_ext, gt_format, flame_transform,
    mouth_mask, upper_mask, split_ids, fps, hf_cutoff, verbose
):
    """
    Run evaluation for one model directory.

    Returns
    -------
    rows     : list of per-sample metric dicts
    skipped  : list of (filename, reason) tuples
    """
    pred_path = Path(pred_path)
    pred_files = list(pred_path.glob(f"*{pred_ext}"))
    print(f"  Found {len(pred_files)} prediction files in {pred_path}")

    rows    = []
    skipped = []

    for pred_f in pred_files:
        try:
            info    = parse_chunk_info(pred_f.name)
            base_id = info['base_id']

            if split_ids is not None and base_id not in split_ids:
                continue

            gt_f = match_gt_file(base_id, gt_path, gt_ext)
            if not gt_f:
                skipped.append((pred_f.name, f"No GT for base_id={base_id}"))
                continue

            if verbose:
                print(f"    {pred_f.name}  →  GT: {gt_f.name}")

            # Load predictions
            pred_seq = load_prediction(pred_f)

            # Load and convert GT
            if gt_format == 'flame':
                gt_raw     = load_gt_file(gt_f)
                gt_facial  = extract_flame_facial_params(gt_raw)
                gt_seq_full = convert_flame_to_arkit(gt_facial, flame_transform)
            else:
                gt_seq_full = load_prediction(gt_f)

            # Temporal alignment
            start_frame = int(info['chunk_id'] * info['duration'] * fps)
            if start_frame >= gt_seq_full.shape[0]:
                skipped.append((pred_f.name,
                    f"start_frame={start_frame} >= GT len={gt_seq_full.shape[0]}"))
                continue

            gt_seq  = gt_seq_full[start_frame:]
            min_len = min(pred_seq.shape[0], gt_seq.shape[0])

            if min_len < 10:
                skipped.append((pred_f.name, f"Overlap too short: {min_len} frames"))
                continue

            pred_seq = pred_seq[:min_len]
            gt_seq   = gt_seq[:min_len]

            # ── Reconstruction metrics ───────────────────────────────────────
            mve = np.linalg.norm(pred_seq - gt_seq, axis=1).mean()
            lve = np.linalg.norm(
                pred_seq[:, mouth_mask] - gt_seq[:, mouth_mask], axis=1
            ).mean()

            gt_std   = np.std(np.sum(np.square(gt_seq[:, upper_mask]),   axis=1))
            pred_std = np.std(np.sum(np.square(pred_seq[:, upper_mask]), axis=1))
            fdd      = gt_std - pred_std
            fdd_abs  = abs(fdd)

            # ── Smoothness metrics ───────────────────────────────────────────
            rows.append({
                'sample_id':    pred_f.stem,
                'num_frames':   min_len,
                'mve':          mve,
                'lve':          lve,
                'fdd':          fdd,
                'fdd_abs':      fdd_abs,
                'jitter_pred':  compute_jitter(pred_seq, fps),
                'jitter_gt':    compute_jitter(gt_seq,   fps),
                'mav_pred':     compute_mav(pred_seq, fps),
                'mav_gt':       compute_mav(gt_seq,   fps),
                'hf_pred':      compute_hf_energy(pred_seq, fps, hf_cutoff),
                'hf_gt':        compute_hf_energy(gt_seq,   fps, hf_cutoff),
                'tc_pred':      compute_temporal_consistency(pred_seq),
                'tc_gt':        compute_temporal_consistency(gt_seq),
            })

        except Exception as e:
            skipped.append((pred_f.name, str(e)))

    return rows, skipped


# =============================================================================
# CSV EXPORT
# =============================================================================

FIELDNAMES = [
    'model_name', 'sample_id', 'num_frames',
    'mve', 'lve', 'fdd', 'fdd_abs',
    'jitter_pred', 'jitter_gt', 'jitter_ratio',
    'mav_pred',    'mav_gt',    'mav_ratio',
    'hf_pred',     'hf_gt',     'hf_ratio',
    'tc_pred',     'tc_gt',     'tc_ratio',
]


def write_combined_csv(output_path, all_rows_by_model):
    """Write a single CSV with all models' per-sample results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for model_name, rows in all_rows_by_model:
            for r in rows:
                writer.writerow({
                    'model_name':   model_name,
                    'sample_id':    r['sample_id'],
                    'num_frames':   r['num_frames'],
                    'mve':          r['mve'],
                    'lve':          r['lve'],
                    'fdd':          r['fdd'],
                    'fdd_abs':      r['fdd_abs'],
                    'jitter_pred':  r['jitter_pred'],
                    'jitter_gt':    r['jitter_gt'],
                    'jitter_ratio': r['jitter_pred'] / (r['jitter_gt'] + 1e-8),
                    'mav_pred':     r['mav_pred'],
                    'mav_gt':       r['mav_gt'],
                    'mav_ratio':    r['mav_pred'] / (r['mav_gt'] + 1e-8),
                    'hf_pred':      r['hf_pred'],
                    'hf_gt':        r['hf_gt'],
                    'hf_ratio':     r['hf_pred'] / (r['hf_gt'] + 1e-8),
                    'tc_pred':      r['tc_pred'],
                    'tc_gt':        r['tc_gt'],
                    'tc_ratio':     r['tc_pred'] / (r['tc_gt'] + 1e-8),
                })


# =============================================================================
# TXT SUMMARY EXPORT
# =============================================================================

def write_summary_txt(output_path, all_rows_by_model):
    """Write a human-readable .txt with mean scores per model per metric."""
    output_path = Path(output_path)

    RECON_METRICS = [
        ('mve',     'MVE  (Mean Vertex Error)  ↓'),
        ('lve',     'LVE  (Lip Vertex Error)   ↓'),
        ('fdd',     'FDD  (Face Dynamics Dev)   '),
        ('fdd_abs', 'FDD  Absolute             ↓'),
    ]
    SMOOTH_METRICS = [
        ('jitter_pred',  'Jitter         pred  ↓'),
        ('jitter_gt',    'Jitter         GT      '),
        ('jitter_ratio', 'Jitter         ratio   '),
        ('mav_pred',     'MAV            pred    '),
        ('mav_gt',       'MAV            GT      '),
        ('mav_ratio',    'MAV            ratio   '),
        ('hf_pred',      'HF Energy      pred  ↓'),
        ('hf_gt',        'HF Energy      GT      '),
        ('hf_ratio',     'HF Energy      ratio   '),
        ('tc_pred',      'Temp.Consist.  pred  ↑'),
        ('tc_gt',        'Temp.Consist.  GT      '),
        ('tc_ratio',     'Temp.Consist.  ratio   '),
    ]

    valid = [(name, rows) for name, rows in all_rows_by_model if rows]

    lines = []
    lines.append('=' * 70)
    lines.append('EVALUATION SUMMARY — MEAN SCORES PER MODEL')
    lines.append('=' * 70)

    for model_name, rows in valid:
        n = len(rows)
        lines.append(f'\nModel : {model_name}')
        lines.append(f'Seqs  : {n}')
        lines.append('-' * 50)

        lines.append('  RECONSTRUCTION')
        for key, label in RECON_METRICS:
            mean = np.mean([r[key] for r in rows])
            lines.append(f'    {label:<35}  {mean:.6f}')

        lines.append('  SMOOTHNESS')
        for key, label in SMOOTH_METRICS:
            if key.endswith('_ratio'):
                base     = key[:-6]
                pred_key = f'{base}_pred'
                gt_key   = f'{base}_gt'
                vals = [r[pred_key] / (r[gt_key] + 1e-8) for r in rows]
            else:
                vals = [r[key] for r in rows]
            mean = np.mean(vals)
            lines.append(f'    {label:<35}  {mean:.6f}')

    lines.append('\n' + '=' * 70)
    lines.append('INTERPRETATION GUIDE')
    lines.append('  Ratio < 1.0  (Jitter / HF Energy)  ->  smoother than GT')
    lines.append('  Ratio > 1.0  (Temp. Consistency)   ->  more temporally coherent')
    lines.append('  Ratio ~= 1.0 (MAV)                 ->  similar motion intensity')
    lines.append('  Lower MVE / LVE                    ->  better reconstruction')
    lines.append('=' * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n')


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_model_spec(spec):
    """
    Parse a model specification string: "name::path::ext"
      name : model name that appears in the CSV model_name column
      path : folder with prediction files
      ext  : file extension (.npz or .npy); defaults to .npz if omitted

    Examples:
      "GRU_CV2::../results/gru_cv2/::.npy"
      "Transformer::../results/transformer/::.npz"
      "Transformer::../results/transformer"   (ext defaults to .npz)
    """
    parts = spec.split('::')
    if len(parts) == 3:
        name, path, ext = parts
    elif len(parts) == 2:
        name, path = parts
        ext = '.npz'
    else:
        path = parts[0]
        name = Path(path).name or path
        ext  = '.npz'

    if not ext.startswith('.'):
        ext = '.' + ext

    return name.strip(), path.strip(), ext.strip()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare multiple facial animation models.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--models', nargs='+', required=True,
        metavar='NAME::PATH::EXT',
        help=(
            'One or more model specs: "name::path::ext"\n'
            'Examples:\n'
            '  "GRU_CV2::../results/gru_cv2/::.npy"\n'
            '  "Transformer::../results/transformer/::.npz"\n'
            'Extension defaults to .npz if omitted.'
        ),
    )
    parser.add_argument('--gt_path',         type=str,
                        default='../BEAT2/beat_english_v2.0.0/smplxflame_30')
    parser.add_argument('--gt_ext',          type=str,   default='.npz')
    parser.add_argument('--gt_format',       type=str,   default='flame',
                        choices=['flame', 'arkit'])
    parser.add_argument('--flame_transform', type=str,   default='arkit_to_flame.npy')
    parser.add_argument('--dataset',         type=str,   default='beat',
                        choices=['beat', 'vocaset'])
    parser.add_argument('--split_csv',       type=str,
                        default='../BEAT2/beat_english_v2.0.0/train_test_split.csv')
    parser.add_argument('--split',           type=str,   default='test')
    parser.add_argument('--fps',             type=int,   default=30)
    parser.add_argument('--hf_cutoff',       type=float, default=5.0)
    parser.add_argument('--output_csv',      type=str,   default='./results.csv',
                        help='Path for the combined output CSV (summary .txt saved alongside).')
    parser.add_argument('--verbose',         action='store_true')
    args = parser.parse_args()

    # Resources
    flame_transform = None
    if args.gt_format == 'flame':
        if not os.path.exists(args.flame_transform):
            raise FileNotFoundError(f"Transform file missing: {args.flame_transform}")
        arkit_to_flame  = np.load(args.flame_transform)
        flame_transform = np.linalg.pinv(arkit_to_flame)

    split_ids = load_split_csv(args.split_csv, args.split)

    # Masks
    if args.dataset == 'beat':
        upper_mask = list(range(0, 22)) + [49, 50]
        mouth_mask = [24, 26, 27, 28, 31, 33, 34, 37, 41, 42, 43, 44, 45, 46, 47, 48]
        print("Using BEAT dataset masks")
    else:
        mouth_mask = (list(range(94, 114)) + list(range(146, 178)) +
                      list(range(183, 192)))
        upper_mask = [x for x in range(192) if x not in mouth_mask]

    # Evaluate each model
    all_rows_by_model = []

    for spec in args.models:
        model_name, pred_path, pred_ext = parse_model_spec(spec)
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  |  path: {pred_path}  |  ext: {pred_ext}")
        print(f"{'='*60}")

        rows, skipped = evaluate_model(
            pred_path       = pred_path,
            pred_ext        = pred_ext,
            gt_path         = args.gt_path,
            gt_ext          = args.gt_ext,
            gt_format       = args.gt_format,
            flame_transform = flame_transform,
            mouth_mask      = mouth_mask,
            upper_mask      = upper_mask,
            split_ids       = split_ids,
            fps             = args.fps,
            hf_cutoff       = args.hf_cutoff,
            verbose         = args.verbose,
        )

        print(f"  Processed: {len(rows)} sequences | Skipped: {len(skipped)}")
        if skipped:
            for s in skipped[:5]:
                print(f"    - {s[0]}: {s[1]}")
            if len(skipped) > 5:
                print(f"    ... and {len(skipped) - 5} more.")

        all_rows_by_model.append((model_name, rows))

    # Write outputs
    total_rows = sum(len(r) for _, r in all_rows_by_model)
    if total_rows == 0:
        print("\nNo valid results to save.")
        return

    csv_path = Path(args.output_csv)
    txt_path = csv_path.with_suffix('.txt')

    write_combined_csv(csv_path, all_rows_by_model)
    write_summary_txt(txt_path, all_rows_by_model)

    print(f"\nSaved {total_rows} rows  ->  {csv_path}")
    print(f"Saved summary          ->  {txt_path}")


if __name__ == '__main__':
    main()

'''
python evaluate4_fine_v3.py \
    --models \
        "hubert_gru::../FaceDiffuser/result_BEAT2::.npy" \
        "hubert_trans::../cosyvoice2_decode_v3/modular_hubert/results/test_predictions_combined_500ep::.npz" \
        "speechtokenizer_trans::../speechtokenizer_decode/results/test_set_results_combined_frozen::.npz" \
        "speechtokenizer_gru::../FaceDiffuser/speechtokenizer_adaptation/results/test_set_results_100ep_GRU_frozen_combined::.npy" \
        "wavtokenizer_gru::../FaceDiffuser/wavtokenizer_adaptation/results/test_set_results_100ep_GRU_WAV_combined::.npy" \
        "wavtokenizer_trans::../wavtokenizer_decode/modular/results/test_set_results_combined::.npz" \
        "cosyvoice2_trans::../cosyvoice2_decode_v3/modular/results/combined_test_set_1000ep::.npz" \
        "cosyvoice2_gru::../FaceDiffuser/cosyvoice_adaptation/results/test_set_results_100ep_GRU_CV2_combined::.npy" \
    --output_csv ./results_final.csv
'''