#!/usr/bin/env python3
"""
Main script for training blendshape decoder with mask presets
"""
import argparse
import yaml
import torch
from pathlib import Path
import logging

from train import create_trainer
# from train_analysis import create_trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_mask_categories(config, args):
    """
    Get mask categories from config or args
    
    Priority:
    1. Command-line --mask_categories
    2. Command-line --mask_preset
    3. Config file mask_preset
    4. Config file mask_categories
    5. Default: ['jaw', 'mouth']
    """
    # Check command-line categories (highest priority)
    if args.mask_categories:
        categories = args.mask_categories
        logging.info(f"Using mask categories from command line: {categories}")
        return categories
    
    # Check command-line preset
    if args.mask_preset:
        preset_name = args.mask_preset
        if 'mask_presets' in config and preset_name in config['mask_presets']:
            preset = config['mask_presets'][preset_name]
            categories = preset['categories']
            logging.info(f"Using mask preset '{preset_name}': {preset['description']}")
            logging.info(f"Categories: {categories}")
            return categories
        else:
            logging.error(f"Mask preset '{preset_name}' not found in config")
            raise ValueError(f"Unknown preset: {preset_name}")
    
    # Check config file preset
    train_config = config.get('training', {})
    if 'mask_preset' in train_config:
        preset_name = train_config['mask_preset']
        if 'mask_presets' in config and preset_name in config['mask_presets']:
            preset = config['mask_presets'][preset_name]
            categories = preset['categories']
            logging.info(f"Using mask preset from config '{preset_name}': {preset['description']}")
            logging.info(f"Categories: {categories}")
            return categories
    
    # Check config file categories
    if 'mask_categories' in train_config:
        categories = train_config['mask_categories']
        logging.info(f"Using mask categories from config: {categories}")
        return categories
    
    # Default
    categories = ['jaw', 'mouth']
    logging.info(f"Using default mask categories: {categories}")
    return categories


def get_checkpoint_prefix(config, args):
    """
    Get checkpoint prefix for naming files
    
    Priority:
    1. Command-line --checkpoint_prefix
    2. Auto-generate from mask preset/categories
    3. Config file checkpoint_prefix
    4. Default: 'model'
    """
    # Check command-line prefix (highest priority)
    if args.checkpoint_prefix:
        return args.checkpoint_prefix
    
    # Auto-generate from mask preset if using one
    if args.mask_preset:
        return f"{args.mask_preset}_model"
    
    # Auto-generate from categories if specified
    if args.mask_categories:
        cat_str = "_".join(args.mask_categories)
        return f"{cat_str}_model"
    
    # Check config file
    train_config = config.get('training', {})
    if 'checkpoint_prefix' in train_config:
        return train_config['checkpoint_prefix']
    
    # Auto-generate from config preset
    if 'mask_preset' in train_config:
        return f"{train_config['mask_preset']}_model"
    
    # Default
    return 'model'


def list_mask_presets(config):
    """Print available mask presets"""
    print("\n" + "="*70)
    print("Available Mask Presets:")
    print("="*70)
    
    if 'mask_presets' not in config:
        print("No presets defined in config")
        return
    
    for name, preset in config['mask_presets'].items():
        num_categories = len(preset['categories'])
        print(f"\n{name}:")
        print(f"  Description: {preset['description']}")
        print(f"  Categories ({num_categories}): {', '.join(preset['categories'])}")
    
    print("\n" + "="*70 + "\n")


def create_training_config(config, args):
    """Create training configuration with overrides"""
    train_config = config.get('training', {}).copy()
    
    # Get mask categories
    mask_categories = get_mask_categories(config, args)
    train_config['mask_categories'] = mask_categories
    
    # Get checkpoint prefix
    checkpoint_prefix = get_checkpoint_prefix(config, args)
    train_config['checkpoint_prefix'] = checkpoint_prefix
    logging.info(f"Checkpoint prefix: '{checkpoint_prefix}'")
    logging.info(f"Checkpoint files will be named: {checkpoint_prefix}_epoch_{{epoch}}.pth")
    
    # Remove mask_preset from final config (we've resolved it to categories)
    train_config.pop('mask_preset', None)
    
    # Override with command line arguments
    if args.train_tokens:
        train_config['train_tokens'] = args.train_tokens
    if args.val_tokens:
        train_config['val_tokens'] = args.val_tokens
    if args.beat2_dir:
        train_config['beat2_dir'] = args.beat2_dir
    if args.batch_size:
        train_config['batch_size'] = args.batch_size
    if args.num_epochs:
        train_config['num_epochs'] = args.num_epochs
    if args.lr:
        train_config['lr'] = args.lr
    if args.checkpoint_dir:
        train_config['checkpoint_dir'] = args.checkpoint_dir
    if args.log_dir:
        train_config['log_dir'] = args.log_dir
    if args.loss_type:
        train_config['loss_type'] = args.loss_type
    if args.save_every:
        train_config['save_every'] = args.save_every
    if args.validate_every:
        train_config['validate_every'] = args.validate_every
    
    return train_config


def save_config(config, path):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Saved training config to {path}")


def main(args):
    """Main training function"""
    
    # Load config
    if args.config:
        logging.info(f"Loading config from {args.config}")
        config = load_config(args.config)
    else:
        logging.error("Config file required! Use --config path/to/config.yaml")
        return
    
    # List presets and exit if requested
    if args.list_presets:
        list_mask_presets(config)
        return
    
    # Create training configuration
    train_config = create_training_config(config, args)
    
    # Create checkpoint directory
    checkpoint_dir = Path(train_config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save effective config
    config_save_path = checkpoint_dir / 'training_config2.yaml'
    save_config(train_config, config_save_path)
    
    # Print configuration
    logging.info("="*70)
    logging.info("Training Configuration:")
    logging.info("="*70)
    for key, value in train_config.items():
        if key == 'mask_categories':
            logging.info(f"  {key}: {value}")
            # Count active blendshapes
            from utils import get_arkit_mask
            mask = get_arkit_mask(value)
            logging.info(f"    → Active blendshapes: {mask.sum()}/51")
        else:
            logging.info(f"  {key}: {value}")
    logging.info("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logging.warning("No GPU available, using CPU")
    
    # Create trainer
    logging.info("Creating trainer...")
    trainer = create_trainer(train_config)
    
    # Load checkpoint if resuming
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logging.info("Starting training...")
    trainer.train(
        num_epochs=train_config['num_epochs'],
        save_every=train_config['save_every'],
        validate_every=train_config['validate_every']
    )
    
    logging.info("Training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train blendshape decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,

    )
    
    # Config
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--list_presets', action='store_true',
                        help='List available mask presets and exit')
    
    # Mask selection (mutually exclusive with preset)
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument('--mask_preset', type=str,
                           help='Mask preset name (e.g., mouth_jaw, full_face, eyes_only)')
    mask_group.add_argument('--mask_categories', nargs='+',
                           help='Custom mask categories (e.g., jaw mouth brow)')
    
    # Data
    parser.add_argument('--train_tokens', type=str,
                        help='Path to training speech tokens')
    parser.add_argument('--val_tokens', type=str,
                        help='Path to validation speech tokens')
    parser.add_argument('--beat2_dir', type=str,
                        help='Path to BEAT2 dataset directory')
    
    # Training
    parser.add_argument('--batch_size', type=int,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--loss_type', type=str, choices=['l1', 'l2', 'smooth_l1'],
                        help='Loss function type')
    
    # Checkpointing & Logging
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint_prefix', type=str,
                        help='Prefix for checkpoint filenames (default: auto-generated from mask)')
    parser.add_argument('--log_dir', type=str,
                        help='TensorBoard log directory')
    parser.add_argument('--save_every', type=int,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--validate_every', type=int,
                        help='Validate every N epochs')
    parser.add_argument('--resume', type=str,
                        help='Resume from checkpoint file')
    
    args = parser.parse_args()
    
    main(args)