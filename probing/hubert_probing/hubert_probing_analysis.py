#!/usr/bin/env python3
"""
Probing Analysis for HuBERT continuous features.

Pipeline:
1. Load HuBERT features (T, 768) and BEAT2 blendshapes
2. K-Means on HuBERT embeddings → pseudo-tokens (for MI / classification)
3. MI(pseudo-token, phoneme) + MI(pseudo-token, blendshape cluster)
4. Classification: logistic regression on pseudo-tokens → accuracy
5. Ridge R² on raw continuous embeddings → per-blendshape R²
6. Specificity histograms using pseudo-tokens
7. Temporal bigram MI

Usage:
    python hubert_probing_analysis.py \
        --features_file hubert_features/utt2hubert_features_all.pt \
        --beat2_dir ../BEAT2/beat_english_v2.0.0 \
        --arkit_transform arkit_to_flame.npy \
        --output_dir results/hubert_probing \
        --n_pseudo_tokens 1024
"""

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
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    mutual_info_score,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Timer:
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
# ARKit blendshape definitions
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

HUBERT_FPS = 50  # HuBERT output rate (16kHz / 320 stride)
BS_FPS = 30      # BEAT2 motion capture rate


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
def flame_to_arkit(expressions, jaw, transform_path):
    T = np.load(transform_path)
    T_pinv = np.linalg.pinv(T)
    combined = np.concatenate([expressions, jaw], axis=1)
    return combined @ T_pinv


def load_beat2_blendshapes(beat2_dir, transform_path, split=None):
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
    else:
        logger.info("No split specified, loading ALL motion files")

    npz_files = sorted(motion_dir.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} motion files")

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
            utt2bs[utt_id] = flame_to_arkit(expressions, jaw, transform_path)
        except Exception as e:
            skipped += 1
            logger.warning(f"Skipping {utt_id}: {e}")

    logger.info(f"Loaded blendshapes for {len(utt2bs)} utterances (skipped {skipped})")
    return utt2bs

def load_hubert_features(features_path):
    """Load HuBERT features from single file or directory of parts."""
    features_path = Path(features_path)
    
    if features_path.is_dir():
        # Load from parts
        logger.info(f"Loading HuBERT features from parts in {features_path}")
        utt2feat = {}
        for pt_file in sorted(features_path.glob("*_part_*.pt")):
            logger.info(f"  Loading {pt_file.name}...")
            data = torch.load(pt_file, map_location="cpu")
            for k, v in data.items():
                utt2feat[k] = np.array(v, dtype=np.float16)
            del data
    else:
        # Single file
        logger.info(f"Loading HuBERT features from {features_path}")
        raw = torch.load(features_path, map_location="cpu")
        utt2feat = {}
        for k, v in raw.items():
            utt2feat[k] = np.array(v, dtype=np.float16)
        del raw

    all_lens = [v.shape[0] for v in utt2feat.values()]
    feat_dim = next(iter(utt2feat.values())).shape[1]
    logger.info(f"Loaded {len(utt2feat)} utterances (feat_dim={feat_dim})")
    return utt2feat


def load_phonemes_from_textgrid(beat2_dir, utt_id):
    tg_dir = Path(beat2_dir) / "textgrid"
    tg_file = tg_dir / f"{utt_id}.TextGrid"
    if not tg_file.exists():
        tg_file = tg_dir / f"{utt_id}.textgrid"
    if not tg_file.exists():
        return None
    try:
        import textgrid
    except ImportError:
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
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Alignment
# ──────────────────────────────────────────────────────────────────────
def align_features_to_blendshapes(features, blendshapes):
    """Align HuBERT features (50Hz) to blendshapes (30Hz) via nearest-neighbor."""
    n_feat = features.shape[0]
    n_bs = blendshapes.shape[0]

    dur = min(n_feat / HUBERT_FPS, n_bs / BS_FPS)
    n_feat_trim = int(dur * HUBERT_FPS)

    feat_trimmed = features[:n_feat_trim]
    feat_times = np.arange(n_feat_trim) / HUBERT_FPS
    bs_indices = np.round(feat_times * BS_FPS).astype(int)
    bs_indices = np.clip(bs_indices, 0, n_bs - 1)
    bs_aligned = blendshapes[bs_indices]

    return feat_trimmed, bs_aligned


def features_to_phonemes(n_frames, phoneme_intervals):
    """Assign phoneme label to each HuBERT frame."""
    phonemes = np.array([""] * n_frames, dtype=object)
    for start, end, phone in phoneme_intervals:
        i_start = max(0, int(start * HUBERT_FPS))
        i_end = min(n_frames, int(end * HUBERT_FPS))
        phonemes[i_start:i_end] = phone
    return phonemes


# ──────────────────────────────────────────────────────────────────────
# Build aligned dataset
# ──────────────────────────────────────────────────────────────────────
def build_aligned_dataset(utt2feat, utt2bs, beat2_dir=None):
    """Returns: all_features (N, 768), all_bs (N, 51), all_phonemes (N,)"""
    all_features = []
    all_bs = []
    all_phonemes = []

    tok_keys = set(utt2feat.keys())
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

    logger.info(f"Matched {len(matched)} pairs "
                f"(direct={direct_matches}, chunk={chunk_matches})")

    _phoneme_logged = False
    _ph_ok = 0
    _ph_fail = 0

    for feat_key, bs_key in tqdm(matched, desc="Aligning"):
        features = utt2feat[feat_key]
        bs = utt2bs[bs_key]

        if features.shape[0] == 0 or bs.shape[0] == 0:
            continue

        feat_aligned, bs_aligned = align_features_to_blendshapes(features, bs)
        if feat_aligned.shape[0] == 0:
            continue

        all_features.append(feat_aligned)
        all_bs.append(bs_aligned)

        if beat2_dir is not None:
            phone_intervals = load_phonemes_from_textgrid(beat2_dir, bs_key)
            if phone_intervals is not None:
                _ph_ok += 1
                if "_chunk_" in feat_key:
                    chunk_idx = int(feat_key.rsplit("_chunk_", 1)[1])
                    chunk_offset = chunk_idx * 30.0
                    phone_intervals = [
                        (max(0, s - chunk_offset), e - chunk_offset, p)
                        for s, e, p in phone_intervals
                        if e > chunk_offset and s < chunk_offset + feat_aligned.shape[0] / HUBERT_FPS + 1.0
                    ]
                ph = features_to_phonemes(feat_aligned.shape[0], phone_intervals)
                if not _phoneme_logged:
                    logger.info(f"  Phoneme sample ({bs_key}): {(ph != '').sum()}/{len(ph)} labeled")
                    _phoneme_logged = True
            else:
                _ph_fail += 1
                ph = np.array([""] * feat_aligned.shape[0], dtype=object)
        else:
            ph = np.array([""] * feat_aligned.shape[0], dtype=object)
        all_phonemes.append(ph)

    logger.info(f"Phoneme loading: {_ph_ok} succeeded, {_ph_fail} failed")

    if not all_features:
        return np.array([]).reshape(0, 768), np.array([]).reshape(0, 51), np.array([])

    all_features = np.concatenate(all_features, axis=0)  # stays float16
    all_bs = np.concatenate(all_bs, axis=0)
    all_phonemes = np.concatenate(all_phonemes)

    logger.info(f"Total aligned frames: {len(all_features)}")
    logger.info(f"Feature shape: {all_features.shape}")
    logger.info(f"Phoneme coverage: {(all_phonemes != '').sum()}/{len(all_phonemes)} "
                f"({(all_phonemes != '').mean()*100:.1f}%)")
    return all_features, all_bs, all_phonemes


# ──────────────────────────────────────────────────────────────────────
# Quantize HuBERT → pseudo-tokens
# ──────────────────────────────────────────────────────────────────────
def quantize_features(all_features, n_pseudo_tokens=1024, subsample_for_kmeans=500_000):
    N = all_features.shape[0]
    logger.info(f"Quantizing {N} frames into {n_pseudo_tokens} pseudo-tokens...")

    if N > subsample_for_kmeans:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, subsample_for_kmeans, replace=False)
        fit_data = all_features[idx].astype(np.float32)
    else:
        fit_data = all_features.astype(np.float32)

    logger.info(f"  Fitting MiniBatchKMeans on {len(fit_data)} samples...")
    km = MiniBatchKMeans(
        n_clusters=n_pseudo_tokens, random_state=42,
        batch_size=10000, max_iter=300, n_init=3
    )
    km.fit(fit_data)

    # Predict in batches to avoid OOM
    logger.info("  Assigning pseudo-tokens in batches...")
    batch_size = 100_000
    pseudo_tokens = np.empty(N, dtype=np.int32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        pseudo_tokens[start:end] = km.predict(
            all_features[start:end].astype(np.float32)
        )
        if start % 500_000 == 0 and start > 0:
            logger.info(f"    {start}/{N} frames assigned...")

    n_unique = len(np.unique(pseudo_tokens))
    logger.info(f"  {n_unique}/{n_pseudo_tokens} pseudo-tokens actually used")
    return pseudo_tokens, km


def compute_entropy(labels):
    counts = np.bincount(labels.astype(int))
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


# ──────────────────────────────────────────────────────────────────────
# Analysis 1: MI (using pseudo-tokens)
# ──────────────────────────────────────────────────────────────────────
def run_mi_analysis(pseudo_tokens, all_bs, all_phonemes, k_bs=32):
    """MI between pseudo-tokens and phonemes / blendshape clusters."""
    results = {}

    H_token = compute_entropy(pseudo_tokens)
    logger.info(f"H(pseudo-token) = {H_token:.4f} bits "
                f"({len(np.unique(pseudo_tokens))} unique)")

    # Phoneme MI
    ph_mask = all_phonemes != ""
    if ph_mask.sum() > 1000:
        ph_filtered = all_phonemes[ph_mask]
        tok_filtered = pseudo_tokens[ph_mask]
        unique_ph, ph_encoded = np.unique(ph_filtered, return_inverse=True)

        mi = mutual_info_score(tok_filtered, ph_encoded)
        nmi = normalized_mutual_info_score(tok_filtered, ph_encoded)
        H_ph = compute_entropy(ph_encoded)

        logger.info(f"H(phoneme) = {H_ph:.4f} bits ({len(unique_ph)} phonemes)")
        logger.info(f"MI(pseudo-token, phoneme) = {mi:.4f}")
        logger.info(f"NMI(pseudo-token, phoneme) = {nmi:.4f}")
        results["phoneme"] = {
            "H_phoneme": H_ph,
            "MI_token_phoneme": mi,
            "NMI_token_phoneme": nmi,
            "n_frames": int(ph_mask.sum()),
            "n_unique_phonemes": len(unique_ph),
        }
    else:
        results["phoneme"] = None

    # Blendshape cluster MI
    # logger.info(f"K-Means on blendshapes (k={k_bs})...")
    # scaler = StandardScaler()
    # bs_scaled = scaler.fit_transform(all_bs)
    # km_bs = KMeans(n_clusters=k_bs, random_state=42, n_init=10, max_iter=300)
    # bs_clusters = km_bs.fit_predict(bs_scaled)

    # Blendshape cluster MI — subsample for fitting
    logger.info(f"K-Means on blendshapes (k={k_bs})...")
    scaler = StandardScaler()
    
    # Fit scaler on subsample, transform in batches
    N = all_bs.shape[0]
    rng = np.random.RandomState(42)
    fit_idx = rng.choice(N, min(N, 500_000), replace=False)
    scaler.fit(all_bs[fit_idx])
    
    # Fit KMeans on subsample
    bs_fit = scaler.transform(all_bs[fit_idx])
    km_bs = KMeans(n_clusters=k_bs, random_state=42, n_init=10, max_iter=300)
    km_bs.fit(bs_fit)
    del bs_fit
    
    # Predict in batches
    logger.info("  Assigning blendshape clusters in batches...")
    batch_size = 200_000
    bs_clusters = np.empty(N, dtype=np.int32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bs_clusters[start:end] = km_bs.predict(scaler.transform(all_bs[start:end]))

    H_cluster = compute_entropy(bs_clusters)
    mi = mutual_info_score(pseudo_tokens, bs_clusters)
    nmi = normalized_mutual_info_score(pseudo_tokens, bs_clusters)
    H_cond = H_cluster - mi
    info_red = mi / H_cluster if H_cluster > 0 else 0.0

    logger.info(f"H(cluster) = {H_cluster:.4f}")
    logger.info(f"MI(pseudo-token, cluster) = {mi:.4f}")
    logger.info(f"NMI(pseudo-token, cluster) = {nmi:.4f}")
    logger.info(f"H(cluster|pseudo-token) = {H_cond:.4f}")
    logger.info(f"Info reduction = {info_red:.4f} ({info_red*100:.1f}%)")

    results[f"k{k_bs}"] = {
        "k": k_bs,
        "H_cluster": H_cluster,
        "MI_token_cluster": mi,
        "NMI_token_cluster": nmi,
        "H_cluster_given_token": H_cond,
        "information_reduction": info_red,
    }
    
  logger.info(f"  NMI({region_name}) = {nmi_r:.4f}")

    # Per-region NMI
    for region_name, indices in BLENDSHAPE_GROUPS.items():
        bs_region = all_bs[:, indices]
        scaler_r = StandardScaler()
        scaler_r.fit(bs_region[fit_idx])
        
        bs_region_fit = scaler_r.transform(bs_region[fit_idx])
        km_r = KMeans(n_clusters=min(k_bs, len(indices) * 2),
                      random_state=42, n_init=10)
        km_r.fit(bs_region_fit)
        del bs_region_fit
        
        cl_r = np.empty(N, dtype=np.int32)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            cl_r[start:end] = km_r.predict(scaler_r.transform(bs_region[start:end]))
        
        nmi_r = normalized_mutual_info_score(pseudo_tokens, cl_r)
        results[f"k{k_bs}"][f"NMI_{region_name}"] = nmi_r
        logger.info(f"  NMI({region_name}) = {nmi_r:.4f}")

    return results, bs_clusters


# ──────────────────────────────────────────────────────────────────────
# Analysis 2: Classification (using pseudo-tokens)
# ──────────────────────────────────────────────────────────────────────
def run_classification_probe(pseudo_tokens, bs_clusters, max_samples=50_000):
    """Logistic regression on pseudo-token one-hot → blendshape cluster."""
    logger.info("Running classification probe...")
    from scipy.sparse import csr_matrix

    vocab_size = int(pseudo_tokens.max()) + 1
    N = len(pseudo_tokens)

    rows = np.arange(N)
    cols = pseudo_tokens
    data = np.ones(N)
    X = csr_matrix((data, (rows, cols)), shape=(N, vocab_size))

    n_sub = min(N, max_samples)
    rng = np.random.RandomState(42)
    idx = rng.choice(N, n_sub, replace=False)
    X_sub = X[idx]
    cl_sub = bs_clusters[idx]

    clf = LogisticRegression(max_iter=500, solver="saga", random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X_sub, cl_sub, cv=5, scoring="accuracy")
    acc_mean = scores.mean()
    acc_std = scores.std()
    chance = 1.0 / len(np.unique(cl_sub))
    logger.info(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f} (chance={chance:.4f})")

    return {
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "chance_level": chance,
        "n_samples": n_sub,
    }


# ──────────────────────────────────────────────────────────────────────
# Analysis 3: Ridge R² on continuous features
# ──────────────────────────────────────────────────────────────────────
def run_regression_probe(all_features, all_bs, max_samples=50_000):
    """Ridge regression on raw HuBERT embeddings → blendshape values."""
    logger.info("Running Ridge regression probe on continuous features...")

    N = all_features.shape[0]
    n_sub = min(N, max_samples)
    rng = np.random.RandomState(42)
    idx = rng.choice(N, n_sub, replace=False)

    # X_sub = all_features[idx]
    X_sub = all_features[idx].astype(np.float32)
    bs_sub = all_bs[idx]

    # Scale features and targets
    scaler_X = StandardScaler()
    X_sub_scaled = scaler_X.fit_transform(X_sub)
    scaler_y = StandardScaler()
    bs_sub_scaled = scaler_y.fit_transform(bs_sub)

    # Overall R² (5-fold CV)
    ridge = Ridge(alpha=1.0)
    r2_scores = cross_val_score(ridge, X_sub_scaled, bs_sub_scaled, cv=5, scoring="r2")
    r2_mean = r2_scores.mean()
    r2_std = r2_scores.std()
    logger.info(f"  Overall CV R²: {r2_mean:.4f} ± {r2_std:.4f}")

    # Per-blendshape R² (train set)
    logger.info("  Computing per-blendshape R² (train set)...")
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(X_sub_scaled, bs_sub_scaled)
    preds = ridge_full.predict(X_sub_scaled)

    per_bs_r2 = {}
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

    return {
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "per_blendshape_r2": per_bs_r2,
        "per_region_r2": per_region_r2,
        "n_samples": n_sub,
        "feature_type": "continuous_768d",
    }


# ──────────────────────────────────────────────────────────────────────
# Analysis 4: Specificity (using pseudo-tokens)
# ──────────────────────────────────────────────────────────────────────
def compute_conditional_entropies(tokens, target_labels, n_targets, min_count=20):
    token_counts = Counter(tokens)
    token_target_counts = {}
    for tok, tgt in zip(tokens, target_labels):
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


def plot_specificity(tokens, target_labels, n_targets, output_dir,
                     min_count=20, target_name="Target", prefix="specificity"):
    entropies, max_entropy = compute_conditional_entropies(
        tokens, target_labels, n_targets, min_count)
    ent_values = np.array(list(entropies.values()))

    if len(ent_values) == 0:
        logger.warning(f"No tokens ≥{min_count} occurrences for {target_name}. Skipping.")
        return None

    pct_values = (ent_values / max_entropy) * 100
    mean_pct = pct_values.mean()
    specificity = 100 - mean_pct

    logger.info(f"  {target_name}: {len(ent_values)} tokens, "
                f"mean H={ent_values.mean():.4f} bits ({mean_pct:.1f}%), "
                f"specificity={specificity:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 105, 40)
    ax.hist(pct_values, bins=bins, color="#3498db", edgecolor="white",
            linewidth=0.5, alpha=0.85, zorder=2)
    ax.axvline(100, color="#e74c3c", linewidth=2, linestyle="--",
               label="Uniform (100%)", zorder=3)
    ax.axvline(mean_pct, color="#2ecc71", linewidth=2, linestyle="-",
               label=f"Mean ({mean_pct:.1f}%)", zorder=3)
    ax.axvspan(mean_pct, 100, alpha=0.15, color="#e74c3c", zorder=1)
    mid_x = (mean_pct + 100) / 2
    ylim = ax.get_ylim()
    ax.text(mid_x, ylim[1] * 0.5, f"{specificity:.1f}%\nspecificity",
            ha="center", va="center", fontsize=9, color="#c0392b", fontweight="bold")
    ax.set_xlabel(f"H({target_name} | Pseudo-Token) as % of max", fontsize=11)
    ax.set_ylabel("Number of Pseudo-Tokens", fontsize=11)
    ax.set_title(f"Pseudo-Token → {target_name} (n={len(ent_values)})", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 108)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_specificity.pdf")
    plt.savefig(path, dpi=200)
    plt.close()

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


def plot_combined_specificity(visual_results, phoneme_results, output_dir):
    if visual_results is None or phoneme_results is None:
        return
    vis_pct = visual_results["pct_values"]
    ph_pct = phoneme_results["pct_values"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 105, 35)
    ax.hist(vis_pct, bins=bins, color="#e74c3c", alpha=0.55, edgecolor="white",
            linewidth=0.4, label=f"H(BS Cluster | Token)\nμ={vis_pct.mean():.1f}%")
    ax.hist(ph_pct, bins=bins, color="#3498db", alpha=0.55, edgecolor="white",
            linewidth=0.4, label=f"H(Phoneme | Token)\nμ={ph_pct.mean():.1f}%")
    ax.axvline(vis_pct.mean(), color="#c0392b", linewidth=2, linestyle="-")
    ax.axvline(ph_pct.mean(), color="#2980b9", linewidth=2, linestyle="-")
    ax.axvline(100, color="black", linewidth=1.5, linestyle=":",
               label="Uniform (no specificity)")
    ax.set_xlabel("Conditional Entropy as % of Maximum", fontsize=11)
    ax.set_ylabel("Number of Pseudo-Tokens", fontsize=11)
    ax.set_title("HuBERT Pseudo-Token Specificity: Visual vs. Linguistic", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 108)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_specificity.pdf"), dpi=200)
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Analysis 5: Temporal MI (using pseudo-tokens)
# ──────────────────────────────────────────────────────────────────────
def run_temporal_mi(pseudo_tokens, bs_clusters):
    vocab_size = int(pseudo_tokens.max()) + 1
    logger.info(f"Computing bigram MI (vocab={vocab_size})...")

    tok_bi = pseudo_tokens[:-1].astype(np.int64) * vocab_size + pseudo_tokens[1:].astype(np.int64)
    n_cl = int(bs_clusters.max()) + 1
    cl_bi = bs_clusters[:-1] * n_cl + bs_clusters[1:]

    mi_bi = mutual_info_score(tok_bi, cl_bi)
    nmi_bi = normalized_mutual_info_score(tok_bi, cl_bi)
    mi_uni = mutual_info_score(pseudo_tokens, bs_clusters)
    nmi_uni = normalized_mutual_info_score(pseudo_tokens, bs_clusters)

    logger.info(f"  Unigram: MI={mi_uni:.4f}, NMI={nmi_uni:.4f}")
    logger.info(f"  Bigram:  MI={mi_bi:.4f}, NMI={nmi_bi:.4f}")
    logger.info(f"  MI gain = {mi_bi - mi_uni:.4f}")

    return {
        "mi_unigram": mi_uni, "nmi_unigram": nmi_uni,
        "mi_bigram": mi_bi, "nmi_bigram": nmi_bi,
        "mi_gain": mi_bi - mi_uni,
    }


# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────
def plot_per_blendshape_r2(per_bs_r2, output_dir):
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
    ax.set_title("Per-Blendshape R² — HuBERT Continuous Features")
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=r) for r, c in region_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_blendshape_r2.pdf"), dpi=150)
    plt.close()


def plot_mi_comparison(mi_results, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, nmi_values = [], []
    if mi_results.get("phoneme") is not None:
        labels.append("Phoneme\n(baseline)")
        nmi_values.append(mi_results["phoneme"]["NMI_token_phoneme"])
    for key in sorted(mi_results.keys()):
        if key.startswith("k"):
            labels.append(f"BS Cluster\nk={mi_results[key]['k']}")
            nmi_values.append(mi_results[key]["NMI_token_cluster"])
    colors = ["#2ecc71"] + ["#3498db"] * (len(labels) - 1)
    ax.bar(labels, nmi_values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Normalized Mutual Information")
    ax.set_title("NMI: HuBERT Pseudo-Tokens vs. Phonemes / BS Clusters")
    ax.set_ylim(0, max(nmi_values) * 1.2 if nmi_values else 1.0)
    for i, v in enumerate(nmi_values):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nmi_comparison.pdf"), dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HuBERT probing analysis")
    parser.add_argument("--features_file",
                        default="hubert_features/")
    parser.add_argument("--beat2_dir", default="../../BEAT2/beat_english_v2.0.0")
    parser.add_argument("--arkit_transform", default="../arkit_to_flame.npy")
    parser.add_argument("--output_dir", default="results/hubert_probing")
    parser.add_argument("--n_pseudo_tokens", type=int, default=500,
                        help="K for K-Means quantization of HuBERT embeddings")
    parser.add_argument("--k_bs", type=int, default=32,
                        help="K for blendshape clustering")
    parser.add_argument("--split", default=None)
    parser.add_argument("--max_probe_samples", type=int, default=50_000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.output_dir, "analysis.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("HuBERT Probing Analysis")
    logger.info(f"  features_file:    {args.features_file}")
    logger.info(f"  beat2_dir:        {args.beat2_dir}")
    logger.info(f"  arkit_transform:  {args.arkit_transform}")
    logger.info(f"  output_dir:       {args.output_dir}")
    logger.info(f"  n_pseudo_tokens:  {args.n_pseudo_tokens}")
    logger.info(f"  k_bs:             {args.k_bs}")
    logger.info(f"  split:            {args.split}")
    logger.info("=" * 60)

    total_start = time.time()

    # 1. Load
    with Timer("STEP 1: Loading data"):
        utt2feat = load_hubert_features(args.features_file)
        utt2bs = load_beat2_blendshapes(args.beat2_dir, args.arkit_transform, args.split)

    # 2. Align
    with Timer("STEP 2: Aligning"):
        all_features, all_bs, all_phonemes = build_aligned_dataset(
            utt2feat, utt2bs, args.beat2_dir)

    if all_features.shape[0] == 0:
        logger.error("No aligned data. Check utterance matching.")
        return

    logger.info(f"Dataset: {all_features.shape[0]} frames, "
                f"features={all_features.shape[1]}d, "
                f"phonemes={((all_phonemes != '').sum())}")

    # 3. Quantize HuBERT → pseudo-tokens
    with Timer("STEP 3: Quantizing HuBERT → pseudo-tokens"):
        pseudo_tokens, km_hubert = quantize_features(
            all_features, n_pseudo_tokens=args.n_pseudo_tokens)

    # 4. MI analysis (pseudo-tokens)
    with Timer("STEP 4: MI analysis"):
        mi_results, bs_clusters = run_mi_analysis(
            pseudo_tokens, all_bs, all_phonemes, k_bs=args.k_bs)

    # 5. Classification probe (pseudo-tokens)
    with Timer("STEP 5: Classification probe"):
        classification_results = run_classification_probe(
            pseudo_tokens, bs_clusters, max_samples=args.max_probe_samples)

    # 6. Regression probe (continuous features)
    with Timer("STEP 6: Ridge regression probe (continuous)"):
        regression_results = run_regression_probe(
            all_features, all_bs, max_samples=args.max_probe_samples)

    # 7. Specificity (pseudo-tokens)
    with Timer("STEP 7: Specificity"):
        visual_spec = plot_specificity(
            pseudo_tokens, bs_clusters, n_targets=args.k_bs,
            output_dir=args.output_dir, target_name="Blendshape Cluster",
            prefix="token_visual")

        phoneme_spec = None
        ph_mask = all_phonemes != ""
        if ph_mask.sum() > 1000:
            ph_filtered = all_phonemes[ph_mask]
            tok_filtered = pseudo_tokens[ph_mask]
            unique_ph, ph_encoded = np.unique(ph_filtered, return_inverse=True)
            phoneme_spec = plot_specificity(
                tok_filtered, ph_encoded, n_targets=len(unique_ph),
                output_dir=args.output_dir, target_name="Phoneme",
                prefix="token_linguistic")
            if visual_spec and phoneme_spec:
                plot_combined_specificity(visual_spec, phoneme_spec, args.output_dir)

    # 8. Temporal MI
    with Timer("STEP 8: Temporal MI"):
        temporal_results = run_temporal_mi(pseudo_tokens, bs_clusters)

    # 9. Plots
    with Timer("STEP 9: Plots"):
        plot_mi_comparison(mi_results, args.output_dir)
        plot_per_blendshape_r2(regression_results["per_blendshape_r2"], args.output_dir)

    # 10. Save
    all_results = {
        "dataset_stats": {
            "total_frames": int(all_features.shape[0]),
            "feature_dim": int(all_features.shape[1]),
            "n_pseudo_tokens": args.n_pseudo_tokens,
            "n_pseudo_tokens_used": int(len(np.unique(pseudo_tokens))),
            "hubert_fps": HUBERT_FPS,
        },
        "mutual_information": mi_results,
        "linear_probes": {
            "classification": classification_results,
            "regression_continuous": regression_results,
        },
        "temporal_mi": temporal_results,
        "visual_specificity": ({k: v for k, v in visual_spec.items()
                                if k != "pct_values"} if visual_spec else None),
        "phoneme_specificity": ({k: v for k, v in phoneme_spec.items()
                                 if k != "pct_values"} if phoneme_spec else None),
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
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()