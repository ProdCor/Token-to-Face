#!/usr/bin/env python3
"""
Loss functions for blendshape prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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