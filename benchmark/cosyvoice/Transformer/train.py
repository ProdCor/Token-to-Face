#!/usr/bin/env python3
"""
Training script for blendshape decoder
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from model import BlendshapeDecoder, BlendshapeEncoderDecoder, BlendshapeDecoderConv1D
from loss import BlendshapeLoss, BlendshapeLossL1Pure
from dataloader import SpeechBlendshapeDataset, collate_fn
from utils import get_arkit_mask

logging.basicConfig(level=logging.INFO)

class Trainer:
    """Trainer for blendshape decoder"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        checkpoint_dir='checkpoints',
        log_dir='logs',
        blendshape_mask=None,
        checkpoint_prefix='model'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.blendshape_mask = blendshape_mask
        self.checkpoint_prefix = checkpoint_prefix
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Loss history for plotting
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_epochs = []
    
    def plot_training_progress(self):
        """Generate and save training progress plot"""
        if not self.train_loss_history:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot training loss
        epochs = list(range(len(self.train_loss_history)))
        ax.plot(epochs, self.train_loss_history, 'b-', linewidth=2, label='Train Loss', alpha=0.7)
        
        # Plot validation loss
        if self.val_loss_history:
            ax.plot(self.val_epochs, self.val_loss_history, 'r-', linewidth=2, 
                   label='Val Loss', marker='o', markersize=4)
        
        # Mark best validation loss
        if self.val_loss_history:
            best_val_idx = np.argmin(self.val_loss_history)
            best_val_epoch = self.val_epochs[best_val_idx]
            best_val_loss = self.val_loss_history[best_val_idx]
            ax.plot(best_val_epoch, best_val_loss, 'g*', markersize=15, 
                   label=f'Best Val (Epoch {best_val_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training Progress - {self.checkpoint_prefix}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add current stats as text
        current_train = self.train_loss_history[-1]
        current_val = self.val_loss_history[-1] if self.val_loss_history else None
        
        stats_text = f'Current Epoch: {self.epoch}\n'
        stats_text += f'Train Loss: {current_train:.4f}\n'
        if current_val is not None:
            stats_text += f'Val Loss: {current_val:.4f}\n'
        stats_text += f'Best Val: {self.best_val_loss:.4f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.checkpoint_dir / f'{self.checkpoint_prefix}_training_progress.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved training plot to {plot_path}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            # Move to device
            tokens = batch['speech_tokens'].to(self.device)
            target = batch['arkit_blendshapes'].to(self.device)
            token_lengths = batch['token_lengths'].to(self.device)
            frame_lengths = batch['frame_lengths'].to(self.device)
            
            # Forward pass
            pred = self.model(
                tokens,
                target_length=target.shape[1],
                token_lengths=token_lengths
            )
            
            # Compute loss
            loss = self.criterion(
                pred,
                target,
                frame_lengths=frame_lengths,
                blendshape_mask=self.blendshape_mask
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                tokens = batch['speech_tokens'].to(self.device)
                target = batch['arkit_blendshapes'].to(self.device)
                token_lengths = batch['token_lengths'].to(self.device)
                frame_lengths = batch['frame_lengths'].to(self.device)
                
                pred = self.model(
                    tokens,
                    target_length=target.shape[1],
                    token_lengths=token_lengths
                )
                
                loss = self.criterion(
                    pred,
                    target,
                    frame_lengths=frame_lengths,
                    blendshape_mask=self.blendshape_mask
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'val_epochs': self.val_epochs
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint: {path}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """Load checkpoint"""
        path = self.checkpoint_dir / filename
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load loss history
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        self.val_epochs = checkpoint.get('val_epochs', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logging.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs, save_every=25, validate_every=10):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
        """
        logging.info(f"Starting training for {num_epochs} epochs")
        logging.info(f"Device: {self.device}")
        logging.info(f"Train batches: {len(self.train_loader)}")
        logging.info(f"Val batches: {len(self.val_loader)}")
        logging.info(f"Checkpoint prefix: '{self.checkpoint_prefix}'")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_loss_history.append(train_loss)
            logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            
            # Validate
            if (epoch + 1) % validate_every == 0:
                val_loss = self.validate()
                self.val_loss_history.append(val_loss)
                self.val_epochs.append(epoch)
                logging.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    logging.info(f"New best model! Val Loss: {val_loss:.4f}")
                
                # Plot training progress after validation
                self.plot_training_progress()
            
            # Save checkpoint with custom prefix
            if (epoch + 1) % save_every == 0:
                filename = f'{self.checkpoint_prefix}_epoch_{epoch}.pth'
                self.save_checkpoint(filename=filename)
                # Also plot when saving checkpoint
                self.plot_training_progress()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/learning_rate', current_lr, epoch)
        
        # Final plot
        self.plot_training_progress()
        logging.info("Training complete!")
        self.writer.close()
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_trainer(config):
    """
    Create trainer from config
    
    Args:
        config: Dictionary with training configuration
    
    Returns:
        trainer: Trainer instance
    """
    # Create datasets
    train_dataset = SpeechBlendshapeDataset(
        speech_tokens_path=config['train_tokens'],
        beat2_dir=config['beat2_dir'],
        mask_categories=config.get('mask_categories', ['jaw', 'mouth'])
    )
    
    val_dataset = SpeechBlendshapeDataset(
        speech_tokens_path=config['val_tokens'],
        beat2_dir=config['beat2_dir'],
        mask_categories=config.get('mask_categories', ['jaw', 'mouth'])
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # # Create model
    # model = BlendshapeEncoderDecoder(
    #     vocab_size=config.get('vocab_size', 6561),
    #     d_model=config.get('d_model', 512),
    #     nhead=config.get('nhead', 8),
    #     num_encoder_layers=config.get('num_encoder_layers', 6),
    #     num_decoder_layers=config.get('num_decoder_layers', 6),
    #     dim_feedforward=config.get('dim_feedforward', 2048),
    #     dropout=config.get('dropout', 0.1),
    #     max_target_len=config.get('max_target_len', 2000)
    # )

    # # Create model
    model = BlendshapeDecoder(
        vocab_size=config.get('vocab_size', 8192),
        d_model=config.get('d_model', 512),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1)
    )

    # Create model
    # model = BlendshapeDecoderConv1D(
    #     vocab_size=config.get('vocab_size', 8192),
    #     d_model=config.get('d_model', 512),
    #     nhead=config.get('nhead', 8),
    #     num_layers=config.get('num_layers', 6),
    #     dim_feedforward=config.get('dim_feedforward', 2048),
    #     dropout=config.get('dropout', 0.1)
    # )

    print(f"Model parameters: {count_parameters(model):,}")

    if hasattr(model, 'frame_queries'):
        with torch.no_grad():
            # Sample 10 queries
            sample_queries = model.frame_queries.weight[:10]  # (10, 512)
            
            # Compute cosine similarity between all pairs
            from torch.nn.functional import cosine_similarity
            
            similarities = []
            for i in range(10):
                for j in range(i+1, 10):
                    sim = cosine_similarity(
                        sample_queries[i].unsqueeze(0), 
                        sample_queries[j].unsqueeze(0)
                    ).item()
                    similarities.append(sim)
            
            avg_sim = sum(similarities) / len(similarities)
            print(f"Average query similarity: {avg_sim:.4f}")
            print(f"  (Should be close to 0 for good diversity)")
            print(f"  (Close to 1 means queries are too similar!)")
        
    # Create loss
    # criterion = BlendshapeLoss(loss_type=config.get('loss_type', 'l1'))
    criterion = BlendshapeLossL1Pure()
    
    # Get blendshape mask
    blendshape_mask = get_arkit_mask(
        config.get('mask_categories', ['jaw', 'mouth']),
        as_tensor=True
    )
    # Move mask to GPU
    device = config.get('device', 'cuda')
    if torch.cuda.is_available() and device == 'cuda':
        blendshape_mask = blendshape_mask.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config.get('min_lr', 1e-6)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.get('device', 'cuda'),
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        log_dir=config.get('log_dir', 'logs'),
        blendshape_mask=blendshape_mask,
        checkpoint_prefix=config.get('checkpoint_prefix', 'model')
    )
    
    return trainer