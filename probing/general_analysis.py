import argparse
import os
import json
import logging
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    mutual_info_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        logger.info(f"▶ START: {self.name}")
        return self
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        logger.info(f"✓ DONE:  {self.name} ({elapsed:.1f}s)")


# ──────────────────────────────────────────────────────────────────────
# ARKit blendshape names — alphabetical order matching arkit_to_flame.npy
# ──────────────────────────────────────────────────────────────────────
ARKIT_NAMES_51 = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
    'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
    'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
    'mouthRight', 'mouthRollLower', 'mouthRollUpper',
    'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
    'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'noseSneerLeft', 'noseSneerRight',
]

BLENDSHAPE_GROUPS = {
    "brow":       [0, 1, 2, 3, 4],
    "cheek+nose": [5, 6, 7, 49, 50],
    "eye":        [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "jaw+mouth":  [22, 23, 24, 25] + list(range(26, 49)),
}

BS_FPS = 30  # BEAT2 motion capture rate


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
def flame_to_arkit(expressions, jaw, transform_path):
    """Convert FLAME params (100 expr + 3 jaw = 103) to ARKit 51 blendshapes."""
    T = np.load(transform_path)
    T_pinv = np.linalg.pinv(T)
    combined = np.concatenate([expressions, jaw], axis=1)
    return combined @ T_pinv


def load_beat2_blendshapes(beat2_dir, transform_path, split=None):
    """Load BEAT2 motion data, convert to ARKit blendshapes."""
    motion_dir = Path(beat2_dir) / "smplxflame_30"
    utt2bs = {}

    test_utts = None
    if split is not None:
        csv_file = Path(beat2_dir) / "train_test_split.csv"
        split_file_txt = Path(beat2_dir) / f"{split}.txt"

        if csv_file.exists():
            import csv
            test_utts = set()
            with open(csv_file) as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2 and row[1].strip().lower() == split.lower():
                        test_utts.add(row[0].strip())
            logger.info(f"Loaded {len(test_utts)} utterances for split='{split}'")
        elif split_file_txt.exists():
            with open(split_file_txt) as f:
                test_utts = set(l.strip() for l in f if l.strip())
            logger.info(f"Loaded {len(test_utts)} utterances from {split_file_txt}")
    else:
        logger.info("No split specified, loading ALL motion files")

    npz_files = sorted(motion_dir.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} motion files in {motion_dir}")

    skipped = 0
    for npz_file in tqdm(npz_files, desc="Loading BEAT2 blendshapes"):
        utt_id = npz_file.stem
        if test_utts is not None and utt_id not in test_utts:
            continue
        try:
            data = np.load(npz_file)
            expressions = data["expressions"]
            poses = data["poses"]
            jaw = poses[:, 66:69]
            arkit = flame_to_arkit(expressions, jaw, transform_path)
            utt2bs[utt_id] = arkit
        except Exception as e:
            skipped += 1
            logger.warning(f"Skipping {utt_id}: {e}")

    logger.info(f"Loaded blendshapes for {len(utt2bs)} utterances (skipped {skipped})")
    return utt2bs


def load_tokens(tokens_file, acoustic_layer=None):
    """
    Load speech tokens. Supports:
      - CosyVoice2 / WavTokenizer: {utt_id: array of int}
      - SpeechTokenizer: {utt_id: {'semantic': (T,), 'acoustic': (n_q-1, T)}}
    
    For SpeechTokenizer, creates composite token IDs from all layers.
    """
    logger.info(f"Loading tokens from {tokens_file}")
    raw = torch.load(tokens_file, map_location="cpu", weights_only=False)

    # Detect format
    first_val = next(iter(raw.values()))
    is_multilayer = isinstance(first_val, dict) and 'acoustic' in first_val

    utt2tok = {}
    if is_multilayer:
        if acoustic_layer is not None:
            logger.info(f"Using acoustic layer {acoustic_layer}...")
            for k, v in raw.items():
                utt2tok[k] = np.array(v['acoustic'][acoustic_layer], dtype=np.int64)
        else:
            logger.info("Using semantic layer only...")
            for k, v in raw.items():
                utt2tok[k] = np.array(v['semantic'], dtype=np.int64)
    else:
        utt2tok = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                tokens = v.get("tokens", v.get("token", []))
            else:
                tokens = v
            utt2tok[k] = np.array(tokens, dtype=np.int64)

    all_lens = [len(v) for v in utt2tok.values()]
    if all_lens:
        logger.info(f"Loaded tokens for {len(utt2tok)} utterances "
                    f"(lengths: min={min(all_lens)}, max={max(all_lens)}, "
                    f"mean={np.mean(all_lens):.0f})")
    return utt2tok


def load_phonemes_from_textgrid(beat2_dir, utt_id):
    """Load phoneme alignments from TextGrid files."""
    tg_dir = Path(beat2_dir) / "textgrid"
    tg_file = tg_dir / f"{utt_id}.TextGrid"
    if not tg_file.exists():
        tg_file = tg_dir / f"{utt_id}.textgrid"
    if not tg_file.exists():
        return None

    try:
        import textgrid
    except ImportError:
        logger.warning("textgrid package not installed.")
        return None

    try:
        tg = textgrid.TextGrid.fromFile(str(tg_file))
        phone_tier = None
        for tier in tg:
            if tier.name.lower() in ("phones", "phone", "phonemes", "phoneme", "phon"):
                phone_tier = tier
                break
        if phone_tier is None:
            phone_tier = tg[1] if len(tg) > 1 else tg[0]

        intervals = [(iv.minTime, iv.maxTime, iv.mark)
                      for iv in phone_tier if iv.mark and iv.mark.strip()]
        return intervals if intervals else None
    except Exception as e:
        logger.debug(f"Failed to parse TextGrid {tg_file}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
# Alignment
# ──────────────────────────────────────────────────────────────────────
def align_tokens_to_blendshapes(tokens, blendshapes, token_fps):
    """Align token sequence to blendshape sequence by nearest-neighbor."""
    n_tok = len(tokens)
    n_bs = blendshapes.shape[0]

    dur = min(n_tok / token_fps, n_bs / BS_FPS)
    n_tok_trim = int(dur * token_fps)

    tokens_trimmed = tokens[:n_tok_trim]
    tok_times = np.arange(n_tok_trim) / token_fps
    bs_indices = np.round(tok_times * BS_FPS).astype(int)
    bs_indices = np.clip(bs_indices, 0, n_bs - 1)
    bs_aligned = blendshapes[bs_indices]

    return tokens_trimmed, bs_aligned


def tokens_to_phonemes(tokens, phoneme_intervals, token_fps):
    """Assign phoneme label to each token frame based on TextGrid."""
    n = len(tokens)
    phonemes = np.array([""] * n, dtype=object)
    for start, end, phone in phoneme_intervals:
        i_start = max(0, int(start * token_fps))
        i_end = min(n, int(end * token_fps))
        phonemes[i_start:i_end] = phone
    return phonemes


# ──────────────────────────────────────────────────────────────────────
# Build aligned dataset
# ──────────────────────────────────────────────────────────────────────
def build_aligned_dataset(utt2tok, utt2bs, token_fps, beat2_dir=None):
    """Align tokens and blendshapes across all shared utterances."""
    all_tokens = []
    all_bs = []
    all_phonemes = []

    tok_keys = set(utt2tok.keys())
    bs_keys = set(utt2bs.keys())

    matched = []
    direct_matches = 0
    chunk_matches = 0
    for tk in tok_keys:
        if tk in bs_keys:
            matched.append((tk, tk))
            direct_matches += 1
            continue
        base = tk.rsplit("_chunk_", 1)[0] if "_chunk_" in tk else tk
        if base in bs_keys:
            matched.append((tk, base))
            chunk_matches += 1

    logger.info(f"Matched {len(matched)} utterance pairs "
                f"(direct={direct_matches}, chunk-stripped={chunk_matches})")
    if len(matched) == 0:
        logger.error(f"No matches! Token keys sample: {list(tok_keys)[:5]}")
        logger.error(f"Blendshape keys sample: {list(bs_keys)[:5]}")

    _phoneme_logged_once = False
    _phoneme_load_failures = 0
    _phoneme_load_successes = 0

    for tok_key, bs_key in tqdm(matched, desc="Aligning"):
        tokens = utt2tok[tok_key]
        bs = utt2bs[bs_key]

        if len(tokens) == 0 or bs.shape[0] == 0:
            continue

        tok_aligned, bs_aligned = align_tokens_to_blendshapes(tokens, bs, token_fps)
        if len(tok_aligned) == 0:
            continue

        all_tokens.append(tok_aligned)
        all_bs.append(bs_aligned)

        if beat2_dir is not None:
            phone_intervals = load_phonemes_from_textgrid(beat2_dir, bs_key)
            if phone_intervals is not None:
                _phoneme_load_successes += 1
                if "_chunk_" in tok_key:
                    chunk_idx = int(tok_key.rsplit("_chunk_", 1)[1])
                    chunk_offset = chunk_idx * 30.0
                    phone_intervals_offset = [
                        (max(0, s - chunk_offset), e - chunk_offset, p)
                        for s, e, p in phone_intervals
                        if e > chunk_offset and s < chunk_offset + len(tok_aligned) / token_fps + 1.0
                    ]
                    ph = tokens_to_phonemes(tok_aligned, phone_intervals_offset, token_fps)
                else:
                    ph = tokens_to_phonemes(tok_aligned, phone_intervals, token_fps)

                if not _phoneme_logged_once:
                    n_labeled = (ph != "").sum()
                    logger.info(f"  Phoneme sample ({bs_key}): {n_labeled}/{len(ph)} frames labeled")
                    _phoneme_logged_once = True
            else:
                _phoneme_load_failures += 1
                ph = np.array([""] * len(tok_aligned), dtype=object)
        else:
            ph = np.array([""] * len(tok_aligned), dtype=object)
        all_phonemes.append(ph)

    logger.info(f"Phoneme loading: {_phoneme_load_successes} succeeded, "
                f"{_phoneme_load_failures} failed")

    if not all_tokens:
        return np.array([]), np.array([]).reshape(0, 51), np.array([])

    all_tokens = np.concatenate(all_tokens)
    all_bs = np.concatenate(all_bs, axis=0)
    all_phonemes = np.concatenate(all_phonemes)

    logger.info(f"Total aligned frames: {len(all_tokens)}")
    logger.info(f"Phoneme coverage: {(all_phonemes != '').sum()}/{len(all_phonemes)} "
                f"({(all_phonemes != '').mean()*100:.1f}%)")
    return all_tokens, all_bs, all_phonemes


# ──────────────────────────────────────────────────────────────────────
# Analysis 1: Clustering + Mutual Information
# ──────────────────────────────────────────────────────────────────────
def run_clustering_mi(all_tokens, all_bs, all_phonemes, k_values=(32,),
                      output_dir="."):
    results = {}

    H_token = compute_entropy(all_tokens)
    logger.info(f"H(token) = {H_token:.4f} bits ({len(np.unique(all_tokens))} unique tokens)")

    ph_mask = all_phonemes != ""
    if ph_mask.sum() > 1000:
        logger.info(f"Computing phoneme MI baseline ({ph_mask.sum()} frames)...")
        ph_filtered = all_phonemes[ph_mask]
        tok_filtered = all_tokens[ph_mask]
        unique_ph, ph_encoded = np.unique(ph_filtered, return_inverse=True)

        mi_tok_ph = mutual_info_score(tok_filtered, ph_encoded)
        nmi_tok_ph = normalized_mutual_info_score(tok_filtered, ph_encoded)
        H_ph = compute_entropy(ph_encoded)
        logger.info(f"H(phoneme) = {H_ph:.4f} bits ({len(unique_ph)} unique phonemes)")
        logger.info(f"MI(token, phoneme) = {mi_tok_ph:.4f}")
        logger.info(f"NMI(token, phoneme) = {nmi_tok_ph:.4f}")
        results["phoneme"] = {
            "H_phoneme": H_ph,
            "MI_token_phoneme": mi_tok_ph,
            "NMI_token_phoneme": nmi_tok_ph,
            "n_frames": int(ph_mask.sum()),
            "n_unique_phonemes": len(unique_ph),
        }
    else:
        logger.warning(f"Insufficient phoneme data ({ph_mask.sum()} frames).")
        results["phoneme"] = None

    logger.info("Fitting StandardScaler on blendshapes...")
    scaler = StandardScaler()
    bs_scaled = scaler.fit_transform(all_bs)

    for k in k_values:
        with Timer(f"K-Means clustering k={k}"):
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            cluster_labels = km.fit_predict(bs_scaled)

        H_cluster = compute_entropy(cluster_labels)
        mi = mutual_info_score(all_tokens, cluster_labels)
        nmi = normalized_mutual_info_score(all_tokens, cluster_labels)
        H_cluster_given_token = H_cluster - mi
        info_reduction = mi / H_cluster if H_cluster > 0 else 0.0

        logger.info(f"  H(cluster) = {H_cluster:.4f}")
        logger.info(f"  MI(token, cluster) = {mi:.4f}")
        logger.info(f"  NMI(token, cluster) = {nmi:.4f}")
        logger.info(f"  H(cluster|token) = {H_cluster_given_token:.4f}")
        logger.info(f"  Info reduction = {info_reduction:.4f} "
                     f"({info_reduction*100:.1f}% visual uncertainty resolved)")

        results[f"k{k}"] = {
            "k": k,
            "H_cluster": H_cluster,
            "MI_token_cluster": mi,
            "NMI_token_cluster": nmi,
            "H_cluster_given_token": H_cluster_given_token,
            "information_reduction": info_reduction,
        }

        for region_name, indices in BLENDSHAPE_GROUPS.items():
            bs_region = all_bs[:, indices]
            bs_region_scaled = StandardScaler().fit_transform(bs_region)
            km_region = KMeans(n_clusters=min(k, len(indices) * 2),
                              random_state=42, n_init=10)
            cl_region = km_region.fit_predict(bs_region_scaled)
            nmi_region = normalized_mutual_info_score(all_tokens, cl_region)
            results[f"k{k}"][f"NMI_{region_name}"] = nmi_region
            logger.info(f"  NMI({region_name}) = {nmi_region:.4f}")

    return results


def compute_entropy(labels):
    counts = np.bincount(labels.astype(int))
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


# ──────────────────────────────────────────────────────────────────────
# Analysis 2: Linear Probes
# ──────────────────────────────────────────────────────────────────────
def run_linear_probes(all_tokens, all_bs, cluster_labels, output_dir=".",
                      context_window=0):
    print(f"\n=== Linear Probes (context_window={context_window}) ===")
    results = {}

    # Dynamic vocab size
    vocab_size = int(all_tokens.max()) + 1
    N = len(all_tokens)
    logger.info(f"Building features: N={N}, vocab_size={vocab_size}, "
                f"context_window={context_window}")

    if context_window == 0:
        from scipy.sparse import csr_matrix
        rows = np.arange(N)
        cols = all_tokens
        data = np.ones(N)
        X = csr_matrix((data, (rows, cols)), shape=(N, vocab_size))
    else:
        padded = np.pad(all_tokens, context_window, mode="edge")
        indices = []
        for offset in range(-context_window, context_window + 1):
            shifted = padded[context_window + offset: context_window + offset + N]
            indices.append(shifted)
        indices = np.stack(indices, axis=1)

        width = (2 * context_window + 1) * vocab_size
        from scipy.sparse import lil_matrix
        X = lil_matrix((N, width), dtype=np.float32)
        for i_off in range(2 * context_window + 1):
            col_offset = i_off * vocab_size
            for row in range(N):
                X[row, col_offset + indices[row, i_off]] = 1.0
        X = X.tocsr()

    max_samples = min(N, 50_000)
    rng = np.random.RandomState(42)
    idx = rng.choice(N, max_samples, replace=False)
    X_sub = X[idx]
    bs_sub = all_bs[idx]
    cl_sub = cluster_labels[idx]
    logger.info(f"Subsampled {max_samples} frames for probing")

    # Classification
    logger.info("Running classification probe (logistic regression, 5-fold CV)...")
    clf = LogisticRegression(max_iter=500, solver="saga", random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X_sub, cl_sub, cv=5, scoring="accuracy")
    acc_mean = scores.mean()
    acc_std = scores.std()
    chance = 1.0 / len(np.unique(cl_sub))
    logger.info(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f} (chance={chance:.4f})")
    results["classification"] = {
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "chance_level": chance,
        "n_samples": max_samples,
        "context_window": context_window,
    }

    # Regression
    logger.info("Running regression probe (Ridge, 5-fold CV)...")
    scaler_y = StandardScaler()
    bs_sub_scaled = scaler_y.fit_transform(bs_sub)

    ridge = Ridge(alpha=1.0)
    r2_scores = cross_val_score(ridge, X_sub, bs_sub_scaled, cv=5, scoring="r2")
    r2_mean = r2_scores.mean()
    r2_std = r2_scores.std()
    logger.info(f"  Overall R²: {r2_mean:.4f} ± {r2_std:.4f}")

    # Per-blendshape R²
    logger.info("Computing per-blendshape R² (train set)...")
    per_bs_r2 = {}
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(X_sub, bs_sub_scaled)
    preds = ridge_full.predict(X_sub)
    for j in range(bs_sub_scaled.shape[1]):
        ss_res = np.sum((bs_sub_scaled[:, j] - preds[:, j]) ** 2)
        ss_tot = np.sum((bs_sub_scaled[:, j] - bs_sub_scaled[:, j].mean()) ** 2)
        r2_j = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        name = ARKIT_NAMES_51[j] if j < len(ARKIT_NAMES_51) else f"bs_{j}"
        per_bs_r2[name] = r2_j

    sorted_bs = sorted(per_bs_r2.items(), key=lambda x: x[1], reverse=True)
    logger.info("  Per-blendshape R² (top 10):")
    for name, r2 in sorted_bs[:10]:
        logger.info(f"    {name}: R²={r2:.4f}")

    per_region_r2 = {}
    for region_name, indices in BLENDSHAPE_GROUPS.items():
        r2_vals = [per_bs_r2[ARKIT_NAMES_51[i]] for i in indices
                   if i < len(ARKIT_NAMES_51)]
        per_region_r2[region_name] = np.mean(r2_vals) if r2_vals else 0.0
        logger.info(f"  Region {region_name}: mean R²={per_region_r2[region_name]:.4f}")

    results["regression"] = {
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "per_blendshape_r2": per_bs_r2,
        "per_region_r2": per_region_r2,
        "n_samples": max_samples,
        "context_window": context_window,
    }

    return results


# ──────────────────────────────────────────────────────────────────────
# Analysis 3: Token Specificity
# ──────────────────────────────────────────────────────────────────────
def compute_conditional_entropies(all_tokens, target_labels, n_targets, min_count=20):
    token_counts = Counter(all_tokens)
    token_target_counts = {}
    for tok, tgt in zip(all_tokens, target_labels):
        if tok not in token_target_counts:
            token_target_counts[tok] = np.zeros(n_targets)
        token_target_counts[tok][tgt] += 1

    entropies = {}
    for tok, counts in token_target_counts.items():
        if token_counts[tok] < min_count:
            continue
        p = counts / counts.sum()
        p = p[p > 0]
        entropies[tok] = -np.sum(p * np.log2(p))

    return entropies, np.log2(n_targets)


def plot_specificity_single(all_tokens, target_labels, n_targets, k,
                            output_dir=".", min_count=20,
                            target_name="Blendshape Cluster",
                            filename_prefix="token_visual"):
    logger.info(f"Computing H({target_name} | token) specificity...")

    entropies, max_entropy = compute_conditional_entropies(
        all_tokens, target_labels, n_targets, min_count=min_count)
    ent_values = np.array(list(entropies.values()))

    pct_values = (ent_values / max_entropy) * 100
    mean_pct = pct_values.mean()
    specificity = 100 - mean_pct

    logger.info(f"  Tokens ≥{min_count} occurrences: {len(ent_values)}")
    logger.info(f"  Max entropy: {max_entropy:.4f} bits")
    logger.info(f"  Mean H({target_name}|token): {ent_values.mean():.4f} bits "
                f"({mean_pct:.1f}% of max)")
    logger.info(f"  Specificity: {specificity:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _draw_normalized_specificity_ax(ax, pct_values, target_name, len(ent_values))
    plt.tight_layout()
    path = os.path.join(output_dir, f"{filename_prefix}_specificity.pdf")
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"Saved: {path}")

    return {
        "n_tokens": len(ent_values),
        "max_entropy_bits": float(max_entropy),
        "mean_entropy_bits": float(ent_values.mean()),
        "mean_pct_of_max": float(mean_pct),
        "specificity_pct": float(specificity),
        "median_pct": float(np.median(pct_values)),
        "min_pct": float(pct_values.min()),
        "pct_values": pct_values,
    }


def _draw_normalized_specificity_ax(ax, pct_values, target_name, n_tokens):
    bins = np.linspace(0, 105, 40)
    ax.hist(pct_values, bins=bins, color="#3498db", edgecolor="white",
            linewidth=0.5, alpha=0.85, zorder=2)
    ax.axvline(100, color="#e74c3c", linewidth=2, linestyle="--",
               label="Uniform baseline (100%)", zorder=3)
    mean_pct = pct_values.mean()
    specificity = 100 - mean_pct
    ax.axvline(mean_pct, color="#2ecc71", linewidth=2, linestyle="-",
               label=f"Mean ({mean_pct:.1f}%)", zorder=3)
    ax.axvspan(mean_pct, 100, alpha=0.15, color="#e74c3c", zorder=1)
    mid_x = (mean_pct + 100) / 2
    ylim = ax.get_ylim()
    ax.text(mid_x, ylim[1] * 0.5, f"{specificity:.1f}%\nspecificity",
            ha="center", va="center", fontsize=9, color="#c0392b", fontweight="bold")
    ax.set_xlabel(f"H({target_name} | Token) as % of max entropy", fontsize=11)
    ax.set_ylabel("Number of Tokens", fontsize=11)
    ax.set_title(f"Token → {target_name} (n={n_tokens} tokens)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 108)


def plot_combined_specificity(visual_results, phoneme_results, output_dir="."):
    vis_pct = visual_results["pct_values"]
    ph_pct = phoneme_results["pct_values"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 105, 35)
    ax.hist(vis_pct, bins=bins, color="#e74c3c", alpha=0.55, edgecolor="white",
            linewidth=0.4, label=f"H(Blendshape Cluster | Token)\nμ = {vis_pct.mean():.1f}%", zorder=2)
    ax.hist(ph_pct, bins=bins, color="#3498db", alpha=0.55, edgecolor="white",
            linewidth=0.4, label=f"H(Phoneme | Token)\nμ = {ph_pct.mean():.1f}%", zorder=2)
    ax.axvline(vis_pct.mean(), color="#c0392b", linewidth=2, linestyle="-", zorder=3)
    ax.axvline(ph_pct.mean(), color="#2980b9", linewidth=2, linestyle="-", zorder=3)
    ax.axvline(100, color="black", linewidth=1.5, linestyle=":",
               label="Uniform (no specificity)", zorder=3)
    ax.set_xlabel("Conditional Entropy as % of Maximum", fontsize=11)
    ax.set_ylabel("Number of Tokens", fontsize=11)
    ax.set_title("Token Specificity: Visual vs. Linguistic", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 108)
    plt.tight_layout()
    path = os.path.join(output_dir, "combined_specificity.pdf")
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────
# Analysis 4: Temporal MI
# ──────────────────────────────────────────────────────────────────────
def run_temporal_mi(all_tokens, cluster_labels):
    print("\n=== Temporal (Bigram) MI ===")
    vocab_size = int(all_tokens.max()) + 1
    logger.info(f"Computing bigram MI on {len(all_tokens)-1} transitions (vocab={vocab_size})...")

    tok_bigrams = all_tokens[:-1].astype(np.int64) * vocab_size + all_tokens[1:].astype(np.int64)
    n_clusters = int(cluster_labels.max()) + 1
    cl_bigrams = cluster_labels[:-1] * n_clusters + cluster_labels[1:]

    mi_bigram = mutual_info_score(tok_bigrams, cl_bigrams)
    nmi_bigram = normalized_mutual_info_score(tok_bigrams, cl_bigrams)
    mi_unigram = mutual_info_score(all_tokens, cluster_labels)
    nmi_unigram = normalized_mutual_info_score(all_tokens, cluster_labels)

    logger.info(f"  Unigram: MI={mi_unigram:.4f}, NMI={nmi_unigram:.4f}")
    logger.info(f"  Bigram:  MI={mi_bigram:.4f}, NMI={nmi_bigram:.4f}")
    logger.info(f"  MI gain = {mi_bigram - mi_unigram:.4f}")

    return {
        "mi_unigram": mi_unigram,
        "nmi_unigram": nmi_unigram,
        "mi_bigram": mi_bigram,
        "nmi_bigram": nmi_bigram,
        "mi_gain": mi_bigram - mi_unigram,
    }


# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────
def plot_per_blendshape_r2(per_bs_r2, output_dir="."):
    sorted_items = sorted(per_bs_r2.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    name_to_region = {}
    for region, indices in BLENDSHAPE_GROUPS.items():
        for idx in indices:
            if idx < len(ARKIT_NAMES_51):
                name_to_region[ARKIT_NAMES_51[idx]] = region
    region_colors = {
        "jaw+mouth": "#e74c3c", "eye": "#3498db",
        "brow": "#2ecc71", "cheek+nose": "#f39c12",
    }
    colors = [region_colors.get(name_to_region.get(n, ""), "#95a5a6") for n in names]

    fig, ax = plt.subplots(figsize=(8, 12))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("R² (Ridge Regression)")
    ax.set_title("Per-Blendshape R² from Token Linear Probe")
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=r) for r, c in region_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "per_blendshape_r2.pdf")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")


def plot_mi_comparison(mi_results, output_dir="."):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, nmi_values = [], []

    if mi_results.get("phoneme") is not None:
        labels.append("Phoneme\n(baseline)")
        nmi_values.append(mi_results["phoneme"]["NMI_token_phoneme"])

    for key in sorted(mi_results.keys()):
        if key.startswith("k"):
            k = mi_results[key]["k"]
            labels.append(f"BS Cluster\nk={k}")
            nmi_values.append(mi_results[key]["NMI_token_cluster"])

    colors = ["#2ecc71"] + ["#3498db"] * (len(labels) - 1)
    ax.bar(labels, nmi_values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Normalized Mutual Information")
    ax.set_title("NMI: Speech Tokens vs. Phonemes / Blendshape Clusters")
    ax.set_ylim(0, max(nmi_values) * 1.2 if nmi_values else 1.0)
    for i, v in enumerate(nmi_values):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "nmi_comparison.pdf")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Probe speech tokens for visual (blendshape) information")
    parser.add_argument("--tokens_file",
                        default='../cosyvoice2_decode_v3/all_speakers/splits/utt2speech_token_test.pt')
    parser.add_argument("--beat2_dir", default='../BEAT2/beat_english_v2.0.0')
    parser.add_argument("--arkit_transform", default='arkit_to_flame.npy')
    parser.add_argument("--output_dir", default="results/token_analysis")
    parser.add_argument("--token_fps", type=int, default=25,
                        help="Token frame rate (CosyVoice2=25, SpeechTokenizer=50, WavTokenizer=75)")
    parser.add_argument("--k_values", nargs="+", type=int, default=[32])
    parser.add_argument("--context_windows", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--split", default=None)
    parser.add_argument("--top_n_tokens_heatmap", type=int, default=50)
    parser.add_argument("--acoustic_layer", type=int, default=None,
                    help="Which acoustic layer to use (0-6). None = semantic only")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(args.output_dir, "analysis.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                                 datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("Token-Blendshape Probing Analysis")
    logger.info(f"  tokens_file:     {args.tokens_file}")
    logger.info(f"  beat2_dir:       {args.beat2_dir}")
    logger.info(f"  arkit_transform: {args.arkit_transform}")
    logger.info(f"  output_dir:      {args.output_dir}")
    logger.info(f"  token_fps:       {args.token_fps}")
    logger.info(f"  k_values:        {args.k_values}")
    logger.info(f"  context_windows: {args.context_windows}")
    logger.info(f"  split:           {args.split}")
    logger.info("=" * 60)

    total_start = time.time()

    # 1. Load data
    with Timer("STEP 1: Loading data"):
        utt2tok = load_tokens(args.tokens_file)
        utt2bs = load_beat2_blendshapes(args.beat2_dir, args.arkit_transform, args.split)

    # 2. Align
    with Timer("STEP 2: Aligning tokens and blendshapes"):
        all_tokens, all_bs, all_phonemes = build_aligned_dataset(
            utt2tok, utt2bs, args.token_fps, args.beat2_dir)

    if len(all_tokens) == 0:
        logger.error("No aligned data found. Check utterance ID matching.")
        return

    # Auto-detect vocab size
    vocab_size = int(all_tokens.max()) + 1

    logger.info(f"Dataset stats:")
    logger.info(f"  Total frames:   {len(all_tokens)}")
    logger.info(f"  Unique tokens:  {len(np.unique(all_tokens))}")
    logger.info(f"  Vocab size:     {vocab_size}")
    logger.info(f"  Token FPS:      {args.token_fps}")
    logger.info(f"  Token range:    [{all_tokens.min()}, {all_tokens.max()}]")
    logger.info(f"  BS shape:       {all_bs.shape}")
    logger.info(f"  Phoneme frames: {(all_phonemes != '').sum()}")

    logger.info("Per-blendshape stats:")
    for i, name in enumerate(ARKIT_NAMES_51):
        m = all_bs[:, i].mean()
        s = all_bs[:, i].std()
        flag = "⚠" if abs(m) > 0.1 or s < 0.01 else " "
        logger.info(f"  {flag} {name:25s}  mean={m:.4f}  std={s:.4f}")

    # 3. Clustering + MI
    with Timer("STEP 3: Clustering & Mutual Information"):
        mi_results = run_clustering_mi(
            all_tokens, all_bs, all_phonemes,
            k_values=args.k_values, output_dir=args.output_dir)

    # 4. Linear Probes
    with Timer("STEP 4: Linear Probes"):
        scaler = StandardScaler()
        bs_scaled = scaler.fit_transform(all_bs)
        km = KMeans(n_clusters=32, random_state=42, n_init=10)
        cluster_labels_32 = km.fit_predict(bs_scaled)

        probe_results = {}
        for ctx in args.context_windows:
            with Timer(f"Linear probe context_window={ctx}"):
                probe_results[f"ctx_{ctx}"] = run_linear_probes(
                    all_tokens, all_bs, cluster_labels_32,
                    output_dir=args.output_dir, context_window=ctx)

    # 5. Specificity
    with Timer("STEP 5: Visual & Linguistic Specificity"):
        visual_specificity = plot_specificity_single(
            all_tokens, cluster_labels_32, n_targets=32, k=32,
            output_dir=args.output_dir, min_count=20,
            target_name="Blendshape Cluster", filename_prefix="token_visual")

        phoneme_specificity = None
        ph_mask = all_phonemes != ""
        if ph_mask.sum() > 1000:
            ph_filtered = all_phonemes[ph_mask]
            tok_filtered = all_tokens[ph_mask]
            unique_ph, ph_encoded = np.unique(ph_filtered, return_inverse=True)
            n_phonemes = len(unique_ph)
            logger.info(f"  {n_phonemes} unique phonemes, {ph_mask.sum()} frames")

            phoneme_specificity = plot_specificity_single(
                tok_filtered, ph_encoded, n_targets=n_phonemes, k=32,
                output_dir=args.output_dir, min_count=20,
                target_name="Phoneme", filename_prefix="token_linguistic")

            plot_combined_specificity(
                visual_specificity, phoneme_specificity, output_dir=args.output_dir)

    # 6. Temporal MI
    with Timer("STEP 6: Temporal MI"):
        temporal_results = run_temporal_mi(all_tokens, cluster_labels_32)

    # 7. Plots
    with Timer("STEP 7: Generating plots"):
        plot_mi_comparison(mi_results, args.output_dir)
        if "ctx_0" in probe_results and "regression" in probe_results["ctx_0"]:
            plot_per_blendshape_r2(
                probe_results["ctx_0"]["regression"]["per_blendshape_r2"],
                args.output_dir)

    # 8. Save results
    all_results = {
        "dataset_stats": {
            "total_frames": int(len(all_tokens)),
            "unique_tokens": int(len(np.unique(all_tokens))),
            "vocab_size": vocab_size,
            "token_fps": args.token_fps,
            "token_range": [int(all_tokens.min()), int(all_tokens.max())],
            "n_utterances_matched": int(len(set(utt2tok.keys()) & set(utt2bs.keys()))),
        },
        "mutual_information": mi_results,
        "linear_probes": probe_results,
        "temporal_mi": temporal_results,
        "visual_specificity": {k: v for k, v in visual_specificity.items()
                               if k != "pct_values"},
        "phoneme_specificity": ({k: v for k, v in phoneme_specificity.items()
                                 if k != "pct_values"} if phoneme_specificity else None),
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    total_elapsed = time.time() - total_start
    logger.info(f"All results saved to {results_path}")
    logger.info(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()


'''

python general_analysis_v4.py \
    --tokens_file ../tts_pipeline/results_10s_chunks/beat2_test_synth/utt2speech_token_synth.pt \
    --beat2_dir ../BEAT2/beat_english_v2.0.0\
    --arkit_transform arkit_to_flame.npy \
    --token_fps 25 \
    --k_values 32 \
    --context_windows 0 \
    --split test \
    --output_dir results/full_ttsf_pipeline_test_set

'''