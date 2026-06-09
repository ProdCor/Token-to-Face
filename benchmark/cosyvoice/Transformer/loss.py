#!/usr/bin/env python3
"""
Loss functions for blendshape prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlendshapeLossWithMask(nn.Module):
    def __init__(self, loss_type='l1', reduction='mean', 
                 velocity_weight=0.3, smoothness_weight=0.05):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.velocity_weight = velocity_weight
        self.smoothness_weight = smoothness_weight
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def compute_velocity_loss(self, pred, target, blendshape_mask=None):
        """Encourage matching velocity (first derivative)"""
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        
        if blendshape_mask is not None:
            # Apply mask to velocity differences
            mask = blendshape_mask.view(1, 1, -1)
            pred_vel = pred_vel * mask
            target_vel = target_vel * mask
            # Normalize by number of active blendshapes
            return F.mse_loss(pred_vel, target_vel, reduction='sum') / blendshape_mask.sum()
        else:
            return F.mse_loss(pred_vel, target_vel)
    
    def compute_smoothness_loss(self, pred, blendshape_mask=None):
        """Penalize jittery motion (second derivative)"""
        pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        
        if blendshape_mask is not None:
            # Apply mask to acceleration
            mask = blendshape_mask.view(1, 1, -1)
            pred_acc = pred_acc * mask
            # Normalize by number of active blendshapes
            return (pred_acc ** 2).sum() / blendshape_mask.sum()
        else:
            return torch.mean(pred_acc ** 2)
    
    def forward(self, pred, target, frame_lengths=None, blendshape_mask=None):
        """
        Compute loss with velocity and smoothness terms
        
        Args:
            pred: (batch, seq_len, 51) model predictions
            target: (batch, seq_len, 51) ground truth (already masked in dataloader)
            frame_lengths: (batch,) actual sequence lengths (for padding mask)
            blendshape_mask: (51,) boolean/float mask indicating active blendshapes
        """
        # ============================================================
        # STEP 1: Apply blendshape mask to predictions and targets
        # ============================================================
        if blendshape_mask is not None:
            # Reshape mask for broadcasting: (51,) -> (1, 1, 51)
            mask = blendshape_mask.view(1, 1, -1)
            
            # Mask both pred and target (target already masked, but be explicit)
            pred_masked = pred * mask
            target_masked = target * mask
            
            num_active_blendshapes = blendshape_mask.sum()
        else:
            pred_masked = pred
            target_masked = target
            num_active_blendshapes = pred.shape[-1]  # All 51
        
        # ============================================================
        # STEP 2: Compute reconstruction loss (only on active blendshapes)
        # ============================================================
        loss = self.loss_fn(pred_masked, target_masked)  # (batch, seq_len, 51)
        
        # ============================================================
        # STEP 3: Apply temporal mask (for variable-length sequences)
        # ============================================================
        if frame_lengths is not None:
            batch_size, seq_len, _ = loss.shape
            temporal_mask = torch.arange(seq_len, device=loss.device)[None, :] < frame_lengths[:, None]
            temporal_mask = temporal_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            loss = loss * temporal_mask
            
            # Normalize by number of valid timesteps AND active blendshapes
            if self.reduction == 'mean':
                # Total valid elements = (valid timesteps) * (active blendshapes)
                num_valid_elements = temporal_mask.sum() * num_active_blendshapes
                recon_loss = loss.sum() / num_valid_elements
            elif self.reduction == 'sum':
                recon_loss = loss.sum()
        else:
            if self.reduction == 'mean':
                # Mean over batch, time, and active blendshapes
                recon_loss = loss.sum() / (loss.shape[0] * loss.shape[1] * num_active_blendshapes)
            elif self.reduction == 'sum':
                recon_loss = loss.sum()
        
        # ============================================================
        # STEP 4: Velocity loss (with mask applied)
        # ============================================================
        vel_loss = 0
        if self.velocity_weight > 0:
            vel_loss = self.compute_velocity_loss(pred_masked, target_masked, blendshape_mask)
        
        # ============================================================
        # STEP 5: Smoothness loss (with mask applied)
        # ============================================================
        smooth_loss = 0
        if self.smoothness_weight > 0:
            smooth_loss = self.compute_smoothness_loss(pred_masked, blendshape_mask)
        
        # ============================================================
        # STEP 6: Combined loss
        # ============================================================
        total_loss = recon_loss + \
                     self.velocity_weight * vel_loss + \
                     self.smoothness_weight * smooth_loss
        
        return total_loss


class BlendshapeLoss(nn.Module):
    def __init__(self, loss_type='l1', reduction='mean', 
                 velocity_weight=0.3, smoothness_weight=0.05): # original was velocity_weight=0.3, smoothness_weight=0.05
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.velocity_weight = velocity_weight
        self.smoothness_weight = smoothness_weight
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def compute_velocity_loss(self, pred, target):
        """Encourage matching velocity (first derivative)"""
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        return F.mse_loss(pred_vel, target_vel)
    
    def compute_smoothness_loss(self, pred):
        """Penalize jittery motion (second derivative)"""
        pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        return torch.mean(pred_acc ** 2)
    
    def forward(self, pred, target, frame_lengths=None, blendshape_mask=None):
        """
        Compute loss with velocity and smoothness terms
        
        IMPORTANTE: Não aplicamos mask aqui porque target já tem zeros nos
        blendshapes não-ativos, então a loss vai penalizar automaticamente
        se o modelo prever não-zero para eles.
        """
        # Main reconstruction loss - PENALIZA TUDO
        loss = self.loss_fn(pred, target)  # (batch, seq_len, 51)
        
        # Apply frame length mask (ignora padding)
        if frame_lengths is not None:
            batch_size, seq_len, _ = loss.shape
            mask = torch.arange(seq_len, device=loss.device)[None, :] < frame_lengths[:, None]
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            loss = loss * mask
            
            if self.reduction == 'mean':
                recon_loss = loss.sum() / (mask.sum() * loss.shape[-1])
            elif self.reduction == 'sum':
                recon_loss = loss.sum()
        else:
            if self.reduction == 'mean':
                recon_loss = loss.mean()
            elif self.reduction == 'sum':
                recon_loss = loss.sum()
        
        # Velocity loss (encourages dynamic motion)
        vel_loss = 0
        if self.velocity_weight > 0:
            vel_loss = self.compute_velocity_loss(pred, target)
        
        # Smoothness loss (prevents jitter)
        smooth_loss = 0
        if self.smoothness_weight > 0:
            smooth_loss = self.compute_smoothness_loss(pred)
        
        # Combined loss
        total_loss = recon_loss + \
                     self.velocity_weight * vel_loss + \
                     self.smoothness_weight * smooth_loss
        
        return total_loss

class BlendshapeLossL1Pure(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.loss_fn = nn.L1Loss(reduction='none')
    
    
    def forward(self, pred, target, frame_lengths=None, blendshape_mask=None):
        """
        Compute loss with velocity and smoothness terms
        
        IMPORTANTE: Não aplicamos mask aqui porque target já tem zeros nos
        blendshapes não-ativos, então a loss vai penalizar automaticamente
        se o modelo prever não-zero para eles.
        """
        # Main reconstruction loss - PENALIZA TUDO
        loss = self.loss_fn(pred, target)  # (batch, seq_len, 51)
        
        # Apply frame length mask (ignora padding)
        if frame_lengths is not None:
            batch_size, seq_len, _ = loss.shape
            mask = torch.arange(seq_len, device=loss.device)[None, :] < frame_lengths[:, None]
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            loss = loss * mask
            
            if self.reduction == 'mean':
                recon_loss = loss.sum() / (mask.sum() * loss.shape[-1])
            elif self.reduction == 'sum':
                recon_loss = loss.sum()
        else:
            if self.reduction == 'mean':
                recon_loss = loss.mean()
            elif self.reduction == 'sum':
                recon_loss = loss.sum()
        
        return recon_loss