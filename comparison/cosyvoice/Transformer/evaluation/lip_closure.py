"""
Bilabial Lip Closure Metric for ARKit Blendshapes

Learns a lip distance function from GT data using phoneme labels as supervision:
bilabial phonemes (M, B, P) = closed, open vowels = open.

GT data pipeline:
  FLAME .npz (expressions + poses) → extract face params (103) → ARKit blendshapes (51)

Prediction data:
  Already ARKit blendshapes (51) in .npz files

Pipeline:
  1. fit:  Learn lip distance model + threshold from GT
  2. eval: Score predicted blendshapes against GT
"""

import numpy as np
import re
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =============================================================================
# FLAME → ARKit conversion (from your utils.py)
# =============================================================================

def load_flame_to_arkit_transform(path: str) -> np.ndarray:
    """Load and compute FLAME→ARKit transform matrix. Returns (103, 51)."""
    arkit_to_flame = np.load(path)  # (51, 103)
    return np.linalg.pinv(arkit_to_flame)  # (103, 51)


def extract_flame_face_params(npz_path: str) -> np.ndarray:
    """
    Extract FLAME face parameters (100 expressions + 3 jaw) from a BEAT2 .npz file.
    Returns full sequence (T, 103).
    """
    data = np.load(npz_path, allow_pickle=True)
    expressions = data['expressions']  # (T, 100)
    poses = data['poses']              # (T, 165)
    jaw = poses[:, 66:69]              # (T, 3)
    T = min(len(expressions), len(poses))
    flame_params = np.concatenate([expressions[:T], jaw[:T]], axis=1)  # (T, 103)
    return flame_params.astype(np.float32)


def flame_to_arkit(flame_params: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Convert FLAME params (T, 103) → ARKit blendshapes (T, 51)."""
    return flame_params @ transform


# =============================================================================
# Phoneme definitions
# =============================================================================

BILABIAL_PHONEMES = {"M", "B", "P"}
OPEN_PHONEMES = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "OW", "OY"}


@dataclass
class PhonemeInterval:
    xmin: float
    xmax: float
    text: str


def strip_stress(phone: str) -> str:
    return phone.rstrip("012")


def parse_textgrid(path: str) -> list[PhonemeInterval]:
    with open(path, "r") as f:
        content = f.read()
    intervals = []
    pattern = r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"'
    for match in re.finditer(pattern, content):
        xmin, xmax, text = float(match.group(1)), float(match.group(2)), match.group(3)
        intervals.append(PhonemeInterval(xmin=xmin, xmax=xmax, text=text))
    return intervals


def get_bilabial_intervals(intervals: list[PhonemeInterval]) -> list[PhonemeInterval]:
    return [iv for iv in intervals if strip_stress(iv.text) in BILABIAL_PHONEMES]


# =============================================================================
# Lip distance model
# =============================================================================

class LipDistanceModel:
    """
    Logistic regression: ARKit blendshapes → lip closed/open.
    
    Trained on GT frames labeled by phoneme type:
      - bilabial (M/B/P) → 0 (closed)
      - open vowels (AA/AE/...) → 1 (open)
    
    The continuous decision_function score is used as lip distance proxy.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, C=1.0)
        self.fitted = False

    def _extract_labeled_frames(
        self, blendshapes: np.ndarray, textgrid_path: str, fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        intervals = parse_textgrid(textgrid_path)
        T = len(blendshapes)
        X_list, y_list = [], []

        for iv in intervals:
            phone = strip_stress(iv.text)
            if phone in BILABIAL_PHONEMES:
                label = 0
            elif phone in OPEN_PHONEMES:
                label = 1
            else:
                continue

            fs = max(0, int(iv.xmin * fps))
            fe = min(T, int(iv.xmax * fps))
            if fs >= fe:
                continue

            X_list.append(blendshapes[fs:fe])
            y_list.extend([label] * (fe - fs))

        if not X_list:
            return np.zeros((0, blendshapes.shape[1])), np.array([])

        return np.concatenate(X_list, axis=0), np.array(y_list)

    def fit(
        self,
        blendshapes_list: list[np.ndarray],
        textgrid_paths: list[str],
        fps: float = 30.0,
        blendshape_names: list[str] = None,
    ) -> "LipDistanceModel":
        all_X, all_y = [], []
        for bs, tg in zip(blendshapes_list, textgrid_paths):
            X, y = self._extract_labeled_frames(bs, tg, fps)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        n_closed = (y == 0).sum()
        n_open = (y == 1).sum()
        print(f"Training data: {n_closed} closed frames (bilabial), "
              f"{n_open} open frames (vowel)")

        if n_closed == 0 or n_open == 0:
            raise ValueError("Need both bilabial and open vowel frames to fit")

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

        print(f"Classification accuracy: {self.model.score(X_scaled, y):.3f}")

        if blendshape_names and len(blendshape_names) == X.shape[1]:
            weights = self.model.coef_[0]
            effective_w = weights / self.scaler.scale_
            print("Top blendshapes (+open, -closed):")
            for i in np.argsort(np.abs(effective_w))[::-1][:10]:
                print(f"  {blendshape_names[i]:30s} w = {effective_w[i]:+.4f}")

        return self

    def predict(self, blendshapes: np.ndarray) -> np.ndarray:
        assert self.fitted, "Call .fit() first"
        return self.model.decision_function(self.scaler.transform(blendshapes))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model}, f)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LipDistanceModel":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.scaler, obj.model = data["scaler"], data["model"]
        obj.fitted = True
        return obj


# =============================================================================
# Threshold + scoring
# =============================================================================

def learn_threshold_from_gt(
    lip_model: LipDistanceModel,
    gt_blendshapes_list: list[np.ndarray],
    textgrid_paths: list[str],
    fps: float = 30.0,
    percentile: float = 95.0,
) -> float:
    all_mins = []
    for gt_bs, tg in zip(gt_blendshapes_list, textgrid_paths):
        scores = lip_model.predict(gt_bs)
        for biv in get_bilabial_intervals(parse_textgrid(tg)):
            fs, fe = max(0, int(biv.xmin * fps)), min(len(scores), int(biv.xmax * fps))
            if fs < fe:
                all_mins.append(scores[fs:fe].min())

    if not all_mins:
        print("WARNING: no bilabial phonemes found")
        return 0.0

    arr = np.array(all_mins)
    threshold = np.percentile(arr, percentile)
    print(f"Threshold from {len(arr)} GT bilabials:")
    print(f"  {percentile}th pct = {threshold:.4f}  "
          f"mean={arr.mean():.4f} std={arr.std():.4f} "
          f"range=[{arr.min():.4f}, {arr.max():.4f}]")
    return threshold


def bilabial_closure_score(
    blendshapes: np.ndarray,
    lip_model: LipDistanceModel,
    textgrid_path: str,
    fps: float = 30.0,
    threshold: float = 0.0,
    return_details: bool = False,
) -> dict:
    """
    Compute bilabial closure metrics.
    
    Returns dict with:
        - score: % of bilabials passing threshold (higher = better)
        - mean_lip_score: mean of min lip openness per bilabial (lower = better closure)
        - num_success / num_total
    """
    bilabials = get_bilabial_intervals(parse_textgrid(textgrid_path))
    if not bilabials:
        return {"score": float("nan"), "mean_lip_score": float("nan"),
                "num_success": 0, "num_total": 0}

    scores = lip_model.predict(blendshapes)
    T = len(scores)
    successes, min_scores_list, details = 0, [], []

    for biv in bilabials:
        fs, fe = max(0, int(biv.xmin * fps)), min(T, int(biv.xmax * fps))
        if fs >= fe:
            continue
        min_score = scores[fs:fe].min()
        min_scores_list.append(min_score)
        ok = min_score <= threshold
        if ok:
            successes += 1
        if return_details:
            details.append({
                "phoneme": biv.text, "xmin": biv.xmin, "xmax": biv.xmax,
                "min_lip_score": float(min_score), "success": ok,
            })

    total = len(bilabials)
    result = {
        "score": (successes / total * 100) if total else float("nan"),
        "mean_lip_score": float(np.mean(min_scores_list)) if min_scores_list else float("nan"),
        "num_success": successes,
        "num_total": total,
    }
    if return_details:
        result["details"] = details
    return result


# =============================================================================
# Dataset discovery + loading
# =============================================================================

def discover_sequences(gt_flame_dir: str, textgrid_dir: str) -> list[dict]:
    """Find matching .npz + .TextGrid pairs by filename stem."""
    gt_dir, tg_dir = Path(gt_flame_dir), Path(textgrid_dir)
    gt_files = {p.stem: p for p in gt_dir.glob("*.npz")}
    tg_files = {p.stem: p for p in tg_dir.glob("*.TextGrid")}
    common = sorted(set(gt_files) & set(tg_files))

    if not common:
        print(f"WARNING: No matching sequences")
        print(f"  FLAME .npz:  {len(gt_files)} in {gt_dir}")
        print(f"  TextGrids:   {len(tg_files)} in {tg_dir}")
        return []

    seqs = [{"stem": s, "flame_path": str(gt_files[s]),
             "textgrid_path": str(tg_files[s])} for s in common]
    print(f"Found {len(seqs)} sequences")
    return seqs


def load_gt_as_arkit(flame_path: str, transform: np.ndarray) -> np.ndarray:
    """Load FLAME .npz → extract face params → convert to ARKit (T, 51)."""
    flame_params = extract_flame_face_params(flame_path)
    return flame_to_arkit(flame_params, transform)


def load_pred_blendshapes(path: str, key: str = "blendshapes") -> np.ndarray:
    """Load predicted ARKit blendshapes from .npz or .npy."""
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path)
    else:
        data = np.load(path, allow_pickle=True)
        for k in [key, "blendshapes", "bs", "arr_0", "pred"]:
            if k in data:
                return data[k]
        return data[data.files[0]]


def parse_chunk_stem(filename_stem: str) -> tuple[str, int | None]:
    """
    Parse a filename stem that may contain a chunk ID.
    
    Supported patterns:
        "1_wayne_0_1_1_chunk_0"      -> ("1_wayne_0_1_1", 0)
        "1_wayne_0_1_1_chunk_12"     -> ("1_wayne_0_1_1", 12)
        "test_1_wayne_0_1_1_000"     -> ("1_wayne_0_1_1", 0)
        "test_1_wayne_0_1_1_003"     -> ("1_wayne_0_1_1", 3)
        "1_wayne_0_1_1"              -> ("1_wayne_0_1_1", None)
    """
    # Pattern 1: _chunk_N (explicit chunk marker)
    match = re.match(r'^(.+)_chunk_(\d+)$', filename_stem)
    if match:
        return match.group(1), int(match.group(2))
    
    # Pattern 2: test_<utterance>_NNN (test_ prefix + zero-padded chunk index)
    match = re.match(r'^test_(.+)_(\d{3,})$', filename_stem)
    if match:
        return match.group(1), int(match.group(2))
    
    # No chunk ID
    return filename_stem, None


def discover_pred_files(pred_dir: str) -> dict[str, list[dict]]:
    """
    Discover prediction files (.npz and .npy), grouping chunks by utterance.
    
    Returns:
        dict mapping utterance_id -> list of {path, chunk_idx, stem}
        sorted by chunk_idx within each utterance.
    """
    pred_path = Path(pred_dir)
    files = list(pred_path.glob("*.npz")) + list(pred_path.glob("*.npy"))
    
    utterances = {}
    for f in files:
        utt_id, chunk_idx = parse_chunk_stem(f.stem)
        utterances.setdefault(utt_id, []).append({
            "path": str(f),
            "chunk_idx": chunk_idx,
            "stem": f.stem,
        })
    
    # Sort chunks within each utterance
    for utt_id in utterances:
        utterances[utt_id].sort(
            key=lambda x: x["chunk_idx"] if x["chunk_idx"] is not None else 0
        )
    
    return utterances


# =============================================================================
# Main pipeline
# =============================================================================

def fit_lip_model(
    gt_flame_dir: str,
    textgrid_dir: str,
    arkit_transform_path: str,
    save_path: str = "lip_distance_model.pkl",
    test_stems_from: str = None,
    max_sequences: int = None,
    fps: float = 30.0,
    threshold_percentile: float = 95.0,
) -> tuple[LipDistanceModel, float]:
    """
    Stage 1: Fit lip distance model + learn threshold.
    
    Loads GT FLAME params → converts to ARKit → trains on phoneme labels.
    
    Args:
        test_stems_from: if provided, path to a prediction directory;
                         only utterances present there will be used for fitting.
    """
    print("=" * 60)
    print("STAGE 1: Fitting lip distance model")
    print("=" * 60)

    transform = load_flame_to_arkit_transform(arkit_transform_path)
    print(f"Loaded FLAME→ARKit transform from {arkit_transform_path}")

    sequences = discover_sequences(gt_flame_dir, textgrid_dir)
    if not sequences:
        raise RuntimeError("No sequences found")

    # Filter to test set if requested
    if test_stems_from is not None:
        pred_utts = discover_pred_files(test_stems_from)
        test_stems = set(pred_utts.keys())
        sequences = [s for s in sequences if s["stem"] in test_stems]
        print(f"Filtered to {len(sequences)} test set utterances "
              f"(from {test_stems_from})")

    if max_sequences:
        sequences = sequences[:max_sequences]
        print(f"Using {len(sequences)} sequences")

    # Load and convert GT
    all_bs, all_tg = [], []
    for seq in sequences:
        try:
            arkit_bs = load_gt_as_arkit(seq["flame_path"], transform)
            all_bs.append(arkit_bs)
            all_tg.append(seq["textgrid_path"])
        except Exception as e:
            print(f"  Skipping {seq['stem']}: {e}")

    print(f"Loaded {len(all_bs)} sequences\n")

    # ARKit blendshape names for diagnostics
    from utils import ARKIT_BLENDSHAPE_NAMES

    model = LipDistanceModel()
    model.fit(all_bs, all_tg, fps=fps, blendshape_names=ARKIT_BLENDSHAPE_NAMES)

    print()
    threshold = learn_threshold_from_gt(model, all_bs, all_tg, fps=fps,
                                        percentile=threshold_percentile)

    model.save(save_path)
    meta_path = save_path.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "threshold": float(threshold),
            "threshold_percentile": threshold_percentile,
            "num_sequences": len(all_bs),
            "fps": fps,
            "arkit_transform_path": arkit_transform_path,
        }, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    return model, threshold


def evaluate_models(
    gt_flame_dir: str,
    textgrid_dir: str,
    pred_dirs: dict[str, str],
    arkit_transform_path: str,
    lip_model_path: str = "lip_distance_model.pkl",
    pred_bs_key: str = "blendshapes",
    threshold: float = None,
    fps: float = 30.0,
    chunk_duration: float = 30.0,
    pred_chunk_durations: dict[str, float] = None,
    output_path: str = "bilabial_results.json",
) -> dict:
    """
    Stage 2: Evaluate prediction models against GT.
    
    Args:
        gt_flame_dir: directory with GT FLAME .npz files
        textgrid_dir: directory with .TextGrid files
        pred_dirs: {"model_name": "path/to/predictions/", ...}
        arkit_transform_path: path to arkit_to_flame.npy
        chunk_duration: default chunk duration in seconds
        pred_chunk_durations: per-model override, e.g. {"hubert_gru": 10.0}
    """
    if pred_chunk_durations is None:
        pred_chunk_durations = {}
    print("=" * 60)
    print("STAGE 2: Evaluating bilabial closure scores")
    print("=" * 60)

    lip_model = LipDistanceModel.load(lip_model_path)
    transform = load_flame_to_arkit_transform(arkit_transform_path)

    if threshold is None:
        meta_path = lip_model_path.replace(".pkl", "_meta.json")
        try:
            with open(meta_path) as f:
                threshold = json.load(f)["threshold"]
            print(f"Loaded threshold = {threshold:.4f}")
        except FileNotFoundError:
            print("WARNING: no meta file, using threshold = 0.0")
            threshold = 0.0

    # Discover sequences
    gt_dir, tg_dir = Path(gt_flame_dir), Path(textgrid_dir)
    gt_files = {p.stem: p for p in gt_dir.glob("*.npz")}
    tg_files = {p.stem: p for p in tg_dir.glob("*.TextGrid")}
    all_stems = sorted(set(gt_files) & set(tg_files))
    print(f"Found {len(all_stems)} GT sequences total")

    # Discover all prediction utterances to find test set stems
    all_pred_utterances = {}
    for model_name, pred_dir in pred_dirs.items():
        all_pred_utterances[model_name] = discover_pred_files(pred_dir)

    # Test stems = utterances that exist in ANY prediction directory
    test_stems = set()
    for model_utts in all_pred_utterances.values():
        test_stems.update(model_utts.keys())
    test_stems = sorted(test_stems & set(all_stems))
    print(f"Test set: {len(test_stems)} utterances (present in predictions)\n")

    results = {"threshold": threshold, "fps": fps,
               "num_test_utterances": len(test_stems),
               "per_model": {}, "per_sequence": {}}

    # --- GT (evaluated only on test set utterances) ---
    print("Evaluating GT (test set only)...")
    gt_scores, gt_lip_scores, gt_succ, gt_total = [], [], 0, 0
    for stem in test_stems:
        gt_bs = load_gt_as_arkit(str(gt_files[stem]), transform)
        res = bilabial_closure_score(gt_bs, lip_model, str(tg_files[stem]),
                                     fps=fps, threshold=threshold)
        if not np.isnan(res["score"]):
            gt_scores.append(res["score"])
            gt_lip_scores.append(res["mean_lip_score"])
            gt_succ += res["num_success"]
            gt_total += res["num_total"]
        results["per_sequence"].setdefault(stem, {})["gt"] = res

    gt_micro = (gt_succ / gt_total * 100) if gt_total else float("nan")
    gt_macro = float(np.mean(gt_scores)) if gt_scores else float("nan")
    gt_mean_lip = float(np.mean(gt_lip_scores)) if gt_lip_scores else float("nan")
    results["per_model"]["gt"] = {
        "micro_score": gt_micro, "macro_score": gt_macro,
        "mean_lip_score": gt_mean_lip,
        "total_success": gt_succ, "total_bilabials": gt_total,
    }
    print(f"  GT: micro={gt_micro:.1f}%, mean_lip={gt_mean_lip:.3f} ({gt_succ}/{gt_total})\n")

    # --- Predictions ---
    for model_name, pred_dir in pred_dirs.items():
        print(f"Evaluating {model_name}...")
        model_chunk_dur = pred_chunk_durations.get(model_name, chunk_duration)
        pred_utterances = all_pred_utterances[model_name]
        scores_list, lip_scores_list, succ, total, missing = [], [], 0, 0, 0

        for stem in test_stems:
            if stem not in pred_utterances:
                missing += 1
                continue

            tg_path = str(tg_files[stem])

            for chunk_info in pred_utterances[stem]:
                pred_bs = load_pred_blendshapes(chunk_info["path"], key=pred_bs_key)
                chunk_idx = chunk_info["chunk_idx"]

                # Compute time offset for this chunk
                if chunk_idx is not None:
                    chunk_offset = chunk_idx * model_chunk_dur
                else:
                    chunk_offset = 0.0

                # Filter bilabials that fall within this chunk's time range
                chunk_end_time = chunk_offset + len(pred_bs) / fps
                intervals = parse_textgrid(tg_path)
                bilabials = get_bilabial_intervals(intervals)

                chunk_succ, chunk_total = 0, 0
                chunk_min_scores = []
                lip_scores = lip_model.predict(pred_bs)

                for biv in bilabials:
                    # Skip bilabials outside this chunk's time range
                    if biv.xmax <= chunk_offset or biv.xmin >= chunk_end_time:
                        continue

                    # Convert to chunk-local frame indices
                    fs = max(0, int((biv.xmin - chunk_offset) * fps))
                    fe = min(len(lip_scores), int((biv.xmax - chunk_offset) * fps))
                    if fs >= fe:
                        continue

                    min_score = lip_scores[fs:fe].min()
                    chunk_min_scores.append(min_score)
                    chunk_total += 1
                    if min_score <= threshold:
                        chunk_succ += 1

                succ += chunk_succ
                total += chunk_total
                if chunk_min_scores:
                    lip_scores_list.append(float(np.mean(chunk_min_scores)))
                if chunk_total > 0:
                    scores_list.append(chunk_succ / chunk_total * 100)

                results["per_sequence"].setdefault(stem, {}).setdefault(model_name, []).append({
                    "chunk_idx": chunk_idx,
                    "score": (chunk_succ / chunk_total * 100) if chunk_total else float("nan"),
                    "mean_lip_score": float(np.mean(chunk_min_scores)) if chunk_min_scores else float("nan"),
                    "num_success": chunk_succ,
                    "num_total": chunk_total,
                })

        micro = (succ / total * 100) if total else float("nan")
        macro = float(np.mean(scores_list)) if scores_list else float("nan")
        mean_lip = float(np.mean(lip_scores_list)) if lip_scores_list else float("nan")
        results["per_model"][model_name] = {
            "micro_score": micro, "macro_score": macro,
            "mean_lip_score": mean_lip,
            "total_success": succ, "total_bilabials": total,
            "num_sequences": len(pred_utterances), "missing": missing,
        }
        print(f"  {model_name}: micro={micro:.1f}%, mean_lip={mean_lip:.3f} ({succ}/{total})")
        if missing:
            print(f"  WARNING: {missing} utterances missing from {pred_dir}")
        print()

    # --- Summary ---
    print("=" * 70)
    print(f"{'Model':<25} {'BCS ↑':>8} {'MLS ↓':>10} {'Success':>10} {'Total':>8}")
    print(f"{'':25s} {'(%)':>8} {'(lower=better)':>10}")
    print("-" * 70)
    for name, m in results["per_model"].items():
        print(f"{name:<25} {m['micro_score']:>7.1f}% {m['mean_lip_score']:>10.3f} "
              f"{m['total_success']:>10} {m['total_bilabials']:>8}")
    print("=" * 70)
    print("BCS = Bilabial Closure Score (% passing threshold)")
    print("MLS = Mean Lip Score during bilabials (lower = better closure)")

    # Save
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_path}")
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bilabial Lip Closure Metric")
    subparsers = parser.add_subparsers(dest="command")

    # --- fit ---
    fp = subparsers.add_parser("fit", help="Fit lip distance model from GT")
    fp.add_argument("--gt-flame-dir", required=True,
                    help="Directory with GT FLAME .npz files (expressions + poses)")
    fp.add_argument("--textgrid-dir", required=True,
                    help="Directory with .TextGrid files")
    fp.add_argument("--arkit-transform", required=True,
                    help="Path to arkit_to_flame.npy")
    fp.add_argument("--save-path", default="lip_distance_model.pkl")
    fp.add_argument("--test-stems-from", default=None,
                    help="Path to a prediction directory; only fit on utterances present there")
    fp.add_argument("--max-sequences", type=int, default=None)
    fp.add_argument("--fps", type=float, default=30.0)
    fp.add_argument("--threshold-percentile", type=float, default=95.0)

    # --- eval ---
    ep = subparsers.add_parser("eval", help="Evaluate models")
    ep.add_argument("--gt-flame-dir", required=False,
                    help="Directory with GT FLAME .npz files")
    ep.add_argument("--textgrid-dir", required=False,
                    help="Directory with .TextGrid files")
    ep.add_argument("--arkit-transform", required=True,
                    help="Path to arkit_to_flame.npy")
    ep.add_argument("--pred-dirs", nargs="+", required=True,
                    help="model_name:directory[:chunk_duration] pairs, e.g. "
                         "facediffuser:outputs/fd/ hubert_gru:outputs/gru/:10")
    ep.add_argument("--lip-model-path", default="lip_distance_model.pkl")
    ep.add_argument("--pred-bs-key", default="blendshapes",
                    help="Key in prediction .npz files (default: blendshapes)")
    ep.add_argument("--threshold", type=float, default=None)
    ep.add_argument("--fps", type=float, default=30.0)
    ep.add_argument("--chunk-duration", type=float, default=30.0,
                    help="Chunk duration in seconds for chunked predictions (default: 30)")
    ep.add_argument("--output", default="bilabial_results.json")

    args = parser.parse_args()

    if args.command == "fit":
        fit_lip_model(
            gt_flame_dir=args.gt_flame_dir,
            textgrid_dir=args.textgrid_dir,
            arkit_transform_path=args.arkit_transform,
            save_path=args.save_path,
            test_stems_from=args.test_stems_from,
            max_sequences=args.max_sequences,
            fps=args.fps,
            threshold_percentile=args.threshold_percentile,
        )

    elif args.command == "eval":
        # Parse pred_dirs from "name:path" or "name:path:chunk_duration" format
        pred_dirs = {}
        pred_chunk_durations = {}
        for item in args.pred_dirs:
            parts = item.split(":")
            if len(parts) == 2:
                name, path = parts
            elif len(parts) == 3:
                name, path = parts[0], parts[1]
                pred_chunk_durations[name] = float(parts[2])
            else:
                parser.error(f"--pred-dirs must be name:path or name:path:chunk_dur, got '{item}'")
            pred_dirs[name] = path

        evaluate_models(
            gt_flame_dir=args.gt_flame_dir,
            textgrid_dir=args.textgrid_dir,
            pred_dirs=pred_dirs,
            arkit_transform_path=args.arkit_transform,
            lip_model_path=args.lip_model_path,
            pred_bs_key=args.pred_bs_key,
            threshold=args.threshold,
            fps=args.fps,
            chunk_duration=args.chunk_duration,
            pred_chunk_durations=pred_chunk_durations,
            output_path=args.output,
        )

    else:
        parser.print_help()


'''

# Step 1: Fit
python lip_closure_v2.py fit \
  --gt-flame-dir ../BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --textgrid-dir ../BEAT2/beat_english_v2.0.0/textgrid \
  --arkit-transform arkit_to_flame.npy \
  --threshold-percentile 70.0 \
  --save-path lip_distance_model_70%.pkl \
  --test-stems-from ../cosyvoice2_decode_v3/modular/results/combined_test_set_1000ep

  

# Step 2: Eval
python lip_closure.py eval \
  --gt-flame-dir ../BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --textgrid-dir ../BEAT2/beat_english_v2.0.0/textgrid \
  --arkit-transform arkit_to_flame.npy \
  --lip-model-path lip_distance_model_70%.pkl \
  --pred-dirs cosyvoice2_trans:../cosyvoice2_decode_v3/modular/results/combined_test_set_1000ep \
              cosyvoice2_gru:../FaceDiffuser/cosyvoice_adaptation/results/test_set_results_100ep_GRU_CV2_combined \
              hubert_trans:../cosyvoice2_decode_v3/modular_hubert/results/test_predictions_combined_500ep \
              hubert_gru:../FaceDiffuser/result_BEAT2:10 \
              wavtokenizer_trans:../wavtokenizer_decode/modular/results/test_set_results_combined \
              wavtokenizer_gru:../FaceDiffuser/wavtokenizer_adaptation/results/test_set_results_100ep_GRU_WAV_combined \
              speechtokenizer_gru:../FaceDiffuser/speechtokenizer_adaptation/results/test_set_results_100ep_GRU_frozen_combined \
              speechtokenizer_trans:../speechtokenizer_decode/results/test_set_results_combined_frozen \
  --chunk-duration 30.0 \
  --output bilabial_results_70%.json 

# Step 2: Eval
python lip_closure.py eval \
  --gt-flame-dir ../BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --textgrid-dir ../BEAT2/beat_english_v2.0.0/textgrid \
  --arkit-transform arkit_to_flame.npy \
  --lip-model-path lip_distance_model_70%.pkl \
  --pred-dirs cosyvoice2_trans_synt:../tts_pipeline/results_10s_chunks/beat2_test_synth/blendshapes:10 \
  --chunk-duration 10.0 \
  --output bilabial_results_70%_cv2_trans_synt.json 


'''