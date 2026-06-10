#!/usr/bin/env python3
"""
Synthesize speech from a TextGrid file and time-warp it to match
the ground truth duration using DTW.

Pipeline:
  TextGrid → extract full text + GT duration
           → CosyVoice2 → synthetic audio
           → DTW warp → time-aligned synthetic audio (matches GT duration)

Outputs:
  {output_name}_raw.wav      — raw CosyVoice2 output
  {output_name}_warped.wav   — DTW-warped to GT duration
  {output_name}_tokens.pt    — CosyVoice2 speech tokens (for later blendshape prediction)
  {output_name}_metadata.json

Usage:
    python synth_from_textgrid.py \
        --textgrid   path/to/file.TextGrid \
        --reference  ref_audios/chunk_0_audio.wav \
        --prompt_text "The first thing I like to do on weekends is relaxing." \
        --output_dir results/synth_aligned/

Requirements:
    pip install librosa soundfile
"""

import sys
from unittest.mock import MagicMock

class MockNormalizer:
    def __init__(self, *args, **kwargs): pass
    def normalize(self, text): return text

wetext_mock = MagicMock()
wetext_mock.Normalizer = MockNormalizer
sys.modules['wetext'] = wetext_mock

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_into_sentences(text: str, max_words: int = 10) -> List[str]:
    # First split on punctuation boundaries
    sentences = re.split(r'(?<=[.!?,])\s+', text.strip())
    
    chunks = []
    for sent in sentences:
        words = sent.split()
        if len(words) <= max_words:
            chunks.append(sent)
        else:
            # Only force-split if truly too long
            while len(words) > max_words:
                # Try to find a natural break point (conjunction/preposition)
                break_at = max_words
                for i in range(max_words - 1, max_words // 2, -1):
                    if words[i].lower() in {'and', 'but', 'or', 'so', 'because',
                                            'when', 'if', 'that', 'which', 'the'}:
                        break_at = i
                        break
                chunks.append(" ".join(words[:break_at]))
                words = words[break_at:]
            if words:
                chunks.append(" ".join(words))
    
    return chunks


EMOTION_INSTRUCTIONS = {
    'happy':     'Speak with clear joy and excitement, use a bright and enthusiastic tone',
    'sad':       'Speak with clear sadness and sorrow, use a slow and melancholic tone',
    'angry':     'Speak with clear anger and frustration, use a sharp and forceful tone',
    'neutral':   'Speak in a calm and natural tone without emotional emphasis',
    'surprised': 'Speak with clear surprise and amazement, use varied pitch and excitement',
    'fear':      'Speak with clear fear and anxiety, use a tense and worried tone',
    'disgust':   'Speak with clear disgust and aversion, use a dismissive tone',
}


# =============================================================================
# TextGrid parser
# =============================================================================

def parse_textgrid(path: str) -> Tuple[str, float, List[dict]]:
    """
    Parse a Praat TextGrid file.

    Returns:
        full_text:   all words joined by spaces
        gt_duration: total recording duration (xmax from file header)
        words:       list of {'text', 't_start', 't_end'}
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Total duration from header
    xmax_match = re.search(r'^xmax = ([\d.]+)', content, re.MULTILINE)
    if not xmax_match:
        raise ValueError("Could not find xmax in TextGrid header.")
    gt_duration = float(xmax_match.group(1))

    # Find words tier
    tier_match = re.search(
        r'name = "words".*?intervals: size = \d+(.*?)(?=item \[\d+\]:|$)',
        content, re.DOTALL
    )
    if not tier_match:
        raise ValueError("Could not find 'words' tier in TextGrid.")

    words = []
    for m in re.finditer(
        r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"',
        tier_match.group(1)
    ):
        text = m.group(3).strip()
        if text:
            words.append({
                'text':    text,
                't_start': float(m.group(1)),
                't_end':   float(m.group(2)),
            })

    full_text = " ".join(w['text'] for w in words)
    logger.info(f"  Parsed {len(words)} words  |  GT duration: {gt_duration:.2f}s")
    logger.info(f"  Text: \"{full_text[:100]}{'...' if len(full_text)>100 else ''}\"")
    return full_text, gt_duration, words


# =============================================================================
# DTW-based time warping
# =============================================================================

def dtw_warp_audio(
    syn_audio: np.ndarray,
    gt_duration: float,
    sample_rate: int,
    hop: int = 512
) -> np.ndarray:
    """
    Time-warp synthetic audio to match gt_duration using DTW on RMS energy.

    The DTW finds the optimal alignment path between evenly-spaced GT frames
    and synthetic frames, then uses that path to resample the synthetic audio
    sample-by-sample to the GT length.

    Args:
        syn_audio:   (N,) float32 mono audio
        gt_duration: target duration in seconds
        sample_rate: audio sample rate
        hop:         frame hop size in samples for DTW feature extraction

    Returns:
        warped (M,) float32 audio at the same sample_rate,
        where M = int(gt_duration * sample_rate)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa not installed. Run: pip install librosa")

    syn_duration = len(syn_audio) / sample_rate
    gt_samples   = int(gt_duration * sample_rate)
    logger.info(f"  DTW: {syn_duration:.2f}s → {gt_duration:.2f}s  "
                f"(ratio {gt_duration/syn_duration:.2f}x)")

    # --- 1D energy features per hop ---
    syn_energy = librosa.feature.rms(y=syn_audio, hop_length=hop)[0]   # (T_syn,)
    T_syn = len(syn_energy)

    # Create GT energy by resampling syn_energy to target frame count
    # (we have no GT audio, so we use a uniform warp as the reference signal)
    T_gt = max(1, int(gt_duration * sample_rate / hop))
    gt_energy = np.interp(
        np.linspace(0, T_syn - 1, T_gt),
        np.arange(T_syn),
        syn_energy
    )

    # --- DTW accumulated cost ---
    cost = np.full((T_gt, T_syn), np.inf)
    cost[0, 0] = abs(gt_energy[0] - syn_energy[0])
    for i in range(1, T_gt):
        cost[i, 0] = cost[i-1, 0] + abs(gt_energy[i] - syn_energy[0])
    for j in range(1, T_syn):
        cost[0, j] = cost[0, j-1] + abs(gt_energy[0] - syn_energy[j])
    for i in range(1, T_gt):
        for j in range(1, T_syn):
            cost[i, j] = abs(gt_energy[i] - syn_energy[j]) + min(
                cost[i-1, j-1], cost[i-1, j], cost[i, j-1]
            )

    # --- Traceback ---
    path_gt, path_syn = [], []
    i, j = T_gt - 1, T_syn - 1
    while i > 0 or j > 0:
        path_gt.append(i)
        path_syn.append(j)
        if   i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            step = np.argmin([cost[i-1, j-1], cost[i-1, j], cost[i, j-1]])
            if   step == 0: i -= 1; j -= 1
            elif step == 1: i -= 1
            else:           j -= 1
    path_gt.append(0); path_syn.append(0)
    path_gt  = np.array(path_gt[::-1])   # frame indices in GT timeline
    path_syn = np.array(path_syn[::-1])  # corresponding synthetic frame indices

    # --- Map DTW frame path → sample-level resample ---
    # Centre sample of each frame
    gt_frame_samples  = path_gt  * hop + hop // 2
    syn_frame_samples = path_syn * hop + hop // 2

    gt_frame_samples  = np.clip(gt_frame_samples,  0, gt_samples       - 1)
    syn_frame_samples = np.clip(syn_frame_samples, 0, len(syn_audio)   - 1)

    # For every output sample position (0..gt_samples-1), interpolate
    # which synthetic sample to read from
    out_positions  = np.arange(gt_samples, dtype=np.float64)
    syn_read_pos   = np.interp(out_positions, gt_frame_samples, syn_frame_samples)
    syn_read_pos   = np.clip(syn_read_pos, 0, len(syn_audio) - 1)

    # Linear interpolation in the synthetic signal
    idx_lo  = syn_read_pos.astype(np.int64)
    idx_hi  = np.minimum(idx_lo + 1, len(syn_audio) - 1)
    frac    = syn_read_pos - idx_lo
    warped  = (1.0 - frac) * syn_audio[idx_lo] + frac * syn_audio[idx_hi]

    return warped.astype(np.float32)


# =============================================================================
# Main pipeline
# =============================================================================

def synthesize_and_warp(
    textgrid_path: str,
    reference_audio: str,
    output_dir: str = "results/synth_aligned/",
    cosyvoice_model_path: str = "../pretrained_models/CosyVoice2-0.5B",
    audios_dir: str = "ref_audios",
    prompt_text: Optional[str] = None,
    emotion: Optional[str] = None,
    ref_duration: float = 10.0,
    output_name: Optional[str] = None,
    device: str = "cuda",
) -> dict:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        stem        = Path(textgrid_path).stem
        output_name = f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"

    # ------------------------------------------------------------------
    # 1. Parse TextGrid
    # ------------------------------------------------------------------
    logger.info("\n[1/4] Parsing TextGrid...")
    full_text, gt_duration, words = parse_textgrid(textgrid_path)

    # ------------------------------------------------------------------
    # 2. Synthesize with CosyVoice2
    # ------------------------------------------------------------------
    logger.info("\n[2/4] Synthesizing speech...")
    os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')
    cosyvoice   = CosyVoice2(cosyvoice_model_path, load_jit=False, load_trt=False, fp16=False)
    sample_rate = cosyvoice.sample_rate

    ref_path   = Path(audios_dir) / reference_audio
    ref_speech = load_wav(str(ref_path), 16000)
    max_samples = int(ref_duration * 16000)
    if ref_speech.shape[1] > max_samples:
        ref_speech = ref_speech[:, :max_samples]
        logger.info(f"  Reference truncated to {ref_duration}s")

    audio_list, token_list = [], []
    t_syn = time.time()

    # Split into short chunks so CosyVoice2 never truncates
    chunks = split_into_sentences(full_text, max_words=20)
    logger.info(f"  Text split into {len(chunks)} chunk(s)")
    for idx, c in enumerate(chunks):
        logger.info(f"    [{idx+1}] \"{c}\"")

    if prompt_text:
        logger.info(f"  Mode: zero-shot  |  prompt: \"{prompt_text[:70]}\"")
    else:
        instruction = EMOTION_INSTRUCTIONS.get(
            (emotion or 'neutral').lower(),
            EMOTION_INSTRUCTIONS['neutral']
        )
        logger.info(f"  Mode: instruct  |  emotion: {emotion or 'neutral'}")
        logger.info(f"  Instruction: \"{instruction}\"")

    for chunk in chunks:
        if prompt_text:
            generator = cosyvoice.inference_zero_shot(
                chunk, prompt_text, ref_speech, stream=False)
        else:
            generator = cosyvoice.inference_instruct2(
                chunk, instruction, ref_speech, stream=False)

        for result in generator:
            if 'tts_speech' in result: audio_list.append(result['tts_speech'])
            if 'tts_token'  in result: token_list.append(result['tts_token'])

    if not audio_list:
        raise RuntimeError("No speech generated")

    syn_tensor   = torch.cat(audio_list, dim=1)       # (1, N)
    syn_duration = syn_tensor.shape[1] / sample_rate
    synth_time   = time.time() - t_syn
    logger.info(f"  ✓ {syn_duration:.2f}s audio in {synth_time:.1f}s")

    # Save raw
    raw_path = output_dir / f"{output_name}_raw.wav"
    torchaudio.save(str(raw_path), syn_tensor, sample_rate)
    logger.info(f"  ✓ Raw audio saved: {raw_path}")

    # Save tokens
    tokens_path = None
    if token_list:
        tokens      = torch.cat(token_list, dim=1)
        tokens_path = output_dir / f"{output_name}_tokens.pt"
        # torch.save(tokens.cpu(), tokens_path)
        tokens_dict = {Path(textgrid_path).stem: tokens.squeeze(0).cpu()}
        torch.save(tokens_dict, tokens_path)        
        logger.info(f"  ✓ Tokens saved: {tokens_path}  shape={tokens.shape}")

    # ------------------------------------------------------------------
    # 3. DTW warp to GT duration
    # ------------------------------------------------------------------
    logger.info(f"\n[3/4] DTW warping to GT duration ({gt_duration:.2f}s)...")
    syn_mono = syn_tensor.squeeze(0).numpy()
    warped   = dtw_warp_audio(syn_mono, gt_duration, sample_rate)

    warped_tensor = torch.from_numpy(warped).unsqueeze(0)
    warped_path   = output_dir / f"{output_name}_warped.wav"
    torchaudio.save(str(warped_path), warped_tensor, sample_rate)
    logger.info(f"  ✓ Warped audio saved: {warped_path}  "
                f"({warped_tensor.shape[1]/sample_rate:.2f}s)")

    # ------------------------------------------------------------------
    # 4. Save metadata
    # ------------------------------------------------------------------
    logger.info("\n[4/4] Saving metadata...")
    metadata = {
        'output_name':       output_name,
        'textgrid':          textgrid_path,
        'reference':         reference_audio,
        'prompt_text':       prompt_text,
        'emotion':           emotion,
        'full_text':         full_text,
        'num_words':         len(words),
        'gt_duration_s':     gt_duration,
        'syn_duration_s':    syn_duration,
        'warped_duration_s': float(len(warped) / sample_rate),
        'sample_rate':       sample_rate,
        'raw_audio':         str(raw_path),
        'warped_audio':      str(warped_path),
        'tokens':            str(tokens_path) if tokens_path else None,
        'timing': {'synthesis_s': synth_time},
    }
    meta_path = output_dir / f"{output_name}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ Metadata saved: {meta_path}")

    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info(f"  GT duration:      {gt_duration:.2f}s")
    logger.info(f"  Raw synthetic:    {syn_duration:.2f}s")
    logger.info(f"  Warped synthetic: {len(warped)/sample_rate:.2f}s")
    logger.info(f"  Raw:    {raw_path}")
    logger.info(f"  Warped: {warped_path}")
    logger.info("=" * 60)

    return metadata


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Synthesize voice from a TextGrid and warp to GT duration via DTW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Zero-shot (voice cloning):
    python synth_from_textgrid.py \\
        --textgrid   data/speaker1.TextGrid \\
        --reference  ref_audios/chunk_0_audio.wav \\
        --prompt_text "The first thing I like to do on weekends is relaxing." \\
        --output_dir results/synth_aligned/

  Instruct mode:
    python synth_from_textgrid.py \\
        --textgrid   data/speaker1.TextGrid \\
        --reference  ref_audios/chunk_0_audio.wav \\
        --emotion    neutral \\
        --output_dir results/synth_aligned/
"""
    )

    parser.add_argument('--textgrid',  type=str, required=True,
                        help='Path to Praat .TextGrid file')
    parser.add_argument('--reference', type=str, required=True,
                        help='Reference audio filename (inside --audios_dir)')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prompt_text', type=str, default=None,
                      help='[Zero-shot] Transcription of the reference audio (~1 sentence)')
    mode.add_argument('--emotion',     type=str, default=None,
                      choices=list(EMOTION_INSTRUCTIONS.keys()),
                      help='[Instruct] Emotion (default: neutral)')

    parser.add_argument('--output_name',     type=str,   default=None)
    parser.add_argument('--cosyvoice_model', type=str,
                        default='../pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('--audios_dir',      type=str,   default='ref_audios')
    parser.add_argument('--output_dir',      type=str,   default='results/synth_aligned/')
    parser.add_argument('--ref_duration',    type=float, default=5.0,
                        help='Max reference audio duration in seconds')
    parser.add_argument('--device',          type=str,   default='cuda')

    args = parser.parse_args()

    synthesize_and_warp(
        textgrid_path        = args.textgrid,
        reference_audio      = args.reference,
        output_dir           = args.output_dir,
        cosyvoice_model_path = args.cosyvoice_model,
        audios_dir           = args.audios_dir,
        prompt_text          = args.prompt_text,
        emotion              = args.emotion,
        ref_duration         = args.ref_duration,
        output_name          = args.output_name,
        device               = args.device,
    )

'''

python full_tts_pipeline_v2.py \
    --textgrid   ../../BEAT2/beat_english_v2.0.0/textgrid/1_wayne_0_2_2.TextGrid \
    --reference chunk_0_audio.wav \
    --prompt_text "The first thing I like to do on weekends is relaxing, and." \
    --output_dir results/synth_aligned_v2/

'''