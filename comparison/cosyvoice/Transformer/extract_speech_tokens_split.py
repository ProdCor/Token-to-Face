#!/usr/bin/env python3
# Modified version to handle long audio files with chunking and train/eval splits
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_split_info(csv_path):
    """
    Load split information from CSV file
    
    Args:
        csv_path: path to the CSV file with id,type columns
    
    Returns:
        dict mapping utterance IDs to split types
    """
    df = pd.read_csv(csv_path)
    split_dict = dict(zip(df['id'], df['type']))
    return split_dict

def chunk_audio(audio, sample_rate, chunk_duration=30, overlap_duration=0):
    """
    Split audio into chunks with optional overlap between consecutive chunks
    
    Args:
        audio: tensor of shape (channels, samples)
        sample_rate: sample rate of audio
        chunk_duration: duration of each chunk in seconds
        overlap_duration: overlap between consecutive chunks in seconds
    
    Returns:
        list of audio chunks
    """
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    step_samples = chunk_samples - overlap_samples
    num_samples = audio.shape[1]
    
    # If audio is shorter than chunk duration, return as single chunk
    if num_samples <= chunk_samples:
        return [audio]
    
    chunks = []
    start = 0
    
    while start < num_samples:
        end = min(start + chunk_samples, num_samples)
        chunk = audio[:, start:end]
        chunks.append(chunk)
        
        # Move to next chunk with overlap
        start += step_samples
        
        # Stop if we've covered all audio
        if end >= num_samples:
            break
    
    return chunks

def single_job(utt, split_type):
    try:
        # Load audio
        audio, sample_rate = torchaudio.load(utt2wav[utt], backend='soundfile')
        logging.info(f"Processing {utt} ({split_type}): Original shape={audio.shape}, sample_rate={sample_rate}")
        
        # Resample if needed
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
            sample_rate = 16000
            logging.info(f"  Resampled to 16kHz: shape={audio.shape}")
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            logging.info(f"  Converted to mono: shape={audio.shape}")
        
        duration = audio.shape[1] / sample_rate
        logging.info(f"  Duration: {duration:.2f} seconds")
        
        # Split into chunks
        chunks = chunk_audio(
            audio, 
            sample_rate, 
            chunk_duration=args.chunk_duration,
            overlap_duration=args.overlap_duration
        )
        logging.info(f"  Split into {len(chunks)} chunks")
        
        # Store tokens for each chunk separately
        chunk_tokens = {}
        
        for i, chunk in enumerate(chunks):
            chunk_duration = chunk.shape[1] / sample_rate
            chunk_start_time = i * (args.chunk_duration - args.overlap_duration)
            chunk_end_time = chunk_start_time + chunk_duration
            
            logging.info(f"  Processing chunk {i}/{len(chunks)-1}: time={chunk_start_time:.1f}s-{chunk_end_time:.1f}s, duration={chunk_duration:.2f}s, shape={chunk.shape}")
            
            # Extract mel spectrogram
            feat = whisper.log_mel_spectrogram(chunk, n_mels=128)
            logging.info(f"    Mel spectrogram shape: {feat.shape}")
            
            # Run ONNX model
            onnx_input_feat = feat.detach().cpu().numpy()
            onnx_input_len = np.array([feat.shape[2]], dtype=np.int32)
            
            logging.info(f"    ONNX input feat shape: {onnx_input_feat.shape}")
            logging.info(f"    ONNX input len: {onnx_input_len}")
            
            speech_token = ort_session.run(
                None, 
                {
                    ort_session.get_inputs()[0].name: onnx_input_feat,
                    ort_session.get_inputs()[1].name: onnx_input_len
                }
            )[0]
            
            logging.info(f"    ONNX output shape: {speech_token.shape}")
            speech_token_list = speech_token.flatten().tolist()
            logging.info(f"    Number of tokens: {len(speech_token_list)}")
            logging.info(f"    Token range: [{min(speech_token_list)}, {max(speech_token_list)}]")
            logging.info(f"    First 10 tokens: {speech_token_list[:10]}")
            
            # Save chunk with index
            chunk_key = f"{utt}_chunk_{i}"
            chunk_tokens[chunk_key] = speech_token_list
        
        total_tokens = sum(len(tokens) for tokens in chunk_tokens.values())
        logging.info(f"  Total chunks for {utt}: {len(chunk_tokens)}, total tokens: {total_tokens}")
        return utt, chunk_tokens, split_type
        
    except Exception as e:
        logging.error(f"Error processing {utt}: {str(e)}")
        return utt, {}, split_type

def main(args):
    # Load split information
    split_dict = load_split_info(args.split_csv)
    logging.info(f"Loaded split information for {len(split_dict)} utterances")
    
    # Count splits
    split_counts = {}
    for split_type in split_dict.values():
        split_counts[split_type] = split_counts.get(split_type, 0) + 1
    logging.info(f"Split distribution: {split_counts}")
    
    # Filter utterances based on what's in wav.scp and split CSV
    valid_utts = set(utt2wav.keys()) & set(split_dict.keys())
    logging.info(f"Found {len(valid_utts)} valid utterances (in both wav.scp and split CSV)")
    
    logging.info(f"Processing {len(valid_utts)} audio files")
    logging.info(f"Using {args.num_thread} threads")
    
    # Create tasks with split information
    all_task = [executor.submit(single_job, utt, split_dict[utt]) for utt in valid_utts]
    
    # Separate dictionaries for each split
    split_data = {
        'train': {},
        'val': {},
        'test': {},
        'additional': {}
    }
    
    for future in tqdm(as_completed(all_task), total=len(all_task)):
        utt, chunk_tokens, split_type = future.result()
        if chunk_tokens:  # Only add if processing was successful
            split_data[split_type].update(chunk_tokens)
    
    # Save each split separately
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_tokens in split_data.items():
        if split_tokens:  # Only save if there's data
            output_path = output_dir / f'utt2speech_token_{split_name}.pt'
            torch.save(split_tokens, str(output_path))
            
            total_chunks = len(split_tokens)
            total_tokens = sum(len(tokens) for tokens in split_tokens.values())
            avg_tokens = total_tokens / total_chunks if total_chunks else 0
            
            logging.info(f"\n{split_name.upper()} Split Summary:")
            logging.info(f"  Total chunks: {total_chunks}")
            logging.info(f"  Total tokens: {total_tokens}")
            logging.info(f"  Average tokens per chunk: {avg_tokens:.2f}")
            logging.info(f"  Saved to: {output_path}")
    
    # Also save combined data for backward compatibility
    all_tokens = {}
    for split_tokens in split_data.values():
        all_tokens.update(split_tokens)
    
    combined_path = output_dir / 'utt2speech_token_all.pt'
    torch.save(all_tokens, str(combined_path))
    logging.info(f"\nCombined data saved to: {combined_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, default='all_speakers/all_speakers.scp', 
                        help="Path to wav.scp file")
    parser.add_argument("--split_csv", type=str, default='../BEAT2/beat_english_v2.0.0/train_test_split.csv',
                        help="Path to CSV file with split information (id,type columns)")
    parser.add_argument("--onnx_path", type=str, default="pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output (default: same as wav_dir)")
    parser.add_argument("--num_thread", type=int, default=8,
                        help="Number of threads for parallel processing")
    parser.add_argument("--chunk_duration", type=int, default=30,
                        help="Duration of each audio chunk in seconds")
    parser.add_argument("--overlap_duration", type=float, default=0.0,
                        help="Overlap between consecutive chunks in seconds")
    args = parser.parse_args()
    
    # Set output dir to wav_dir if not specified
    if args.output_dir is None:
        args.output_dir = str(Path(args.wav_dir).parent)
    
    # Load wav.scp
    utt2wav = {}
    wav_scp_path = Path(args.wav_dir)
    
    if not wav_scp_path.exists():
        logging.error(f"wav.scp not found at {wav_scp_path}")
        exit(1)
    
    with open(wav_scp_path) as f:
        for l in f:
            l = l.replace('\n', '').split()
            if len(l) >= 2:
                utt2wav[l[0]] = l[1]
    
    logging.info(f"Loaded {len(utt2wav)} utterances from {wav_scp_path}")
    
    # Setup ONNX Runtime
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    logging.info(f"Loading ONNX model from {args.onnx_path}")
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    
    # Print model info
    logging.info("ONNX Model Information:")
    for i, input_node in enumerate(ort_session.get_inputs()):
        logging.info(f"  Input {i}: name={input_node.name}, shape={input_node.shape}, dtype={input_node.type}")
    for i, output_node in enumerate(ort_session.get_outputs()):
        logging.info(f"  Output {i}: name={output_node.name}, shape={output_node.shape}, dtype={output_node.type}")
    
    executor = ThreadPoolExecutor(max_workers=args.num_thread)
    
    main(args)