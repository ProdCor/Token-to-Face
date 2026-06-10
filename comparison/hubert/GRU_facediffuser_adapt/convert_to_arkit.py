#!/usr/bin/env python3
"""Convert FLAME NPY files to ARKit"""
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# Import ARKit utilities
from utils_cosy import load_flame_to_arkit_transform, flame_to_arkit_blendshapes

# Configuration
INPUT_DIR = "data/beat2/vertices_npy"
OUTPUT_DIR = "data/beat2/vertices_npy_arkit"
TRANSFORM_PATH = "arkit_to_flame.npy"

print("="*60)
print("Converting FLAME (103) -> ARKit (51)")
print("="*60)

# Load transformation
print(f"\nLoading transformation from {TRANSFORM_PATH}...")
transform = load_flame_to_arkit_transform(TRANSFORM_PATH)
print("✓ Loaded")

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Find files
npy_files = list(Path(INPUT_DIR).glob("*.npy"))
print(f"\nFound {len(npy_files)} files to convert")

# Convert
converted = 0
for npy_file in tqdm(npy_files, desc="Converting"):
    # Load FLAME
    flame_params = np.load(npy_file)
    
    # Check dimension
    if flame_params.shape[1] != 103:
        print(f"⚠️  {npy_file.name} has {flame_params.shape[1]} dims, expected 103")
        continue
    
    # Transform to ARKit
    arkit_params = flame_to_arkit_blendshapes(flame_params, transform)
    
    # Save
    output_file = Path(OUTPUT_DIR) / npy_file.name
    np.save(output_file, arkit_params)
    converted += 1

# Create templates
templates = {'neutral': np.zeros((1, 51))}
with open('data/beat2/templates_arkit.pkl', 'wb') as f:
    pickle.dump(templates, f)

print("\n" + "="*60)
print(f"✅ Converted {converted} files")
print(f"Output: {OUTPUT_DIR}")
print(f"Templates: data/beat2/templates_arkit.pkl")
print("="*60)

# Verify
sample = np.load(list(Path(OUTPUT_DIR).glob("*.npy"))[0])
print(f"\nVerification:")
print(f"Sample file shape: {sample.shape}")
print(f"Dimensions: {sample.shape[1]}")
if sample.shape[1] == 51:
    print("✅ Conversion successful!")
else:
    print(f"❌ Expected 51, got {sample.shape[1]}")

print("\n📋 Training command:")
print("python main.py \\")
print("    --dataset beat2 \\")
print("    --train_subjects train \\")
print("    --val_subjects val \\")
print("    --test_subjects test \\")
print("    --vertice_dim 51 \\")
print("    --vertices_path vertices_npy_arkit \\")
print("    --template_file templates_arkit.pkl \\")
print("    --wav_path wav \\")
print("    --feature_dim 256 \\")
print("    --gru_dim 256 \\")
print("    --output_fps 30 \\")
print("    --max_epoch 100")