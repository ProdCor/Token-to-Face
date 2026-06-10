#!/usr/bin/env python3
"""
Transformer Decoder for Speech Token to ARKit Blendshape prediction
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# from hubert.modeling_hubert import HubertModel
from transformers import HubertModel

def adjust_input_representation(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    """
    Brings audio embeddings and visual frames to the same frame rate.

    Args:
        audio_embedding_matrix: (B, T_audio, D) audio embeddings 
        vertex_matrix: (B, T_video, D) animation sequence
        ifps: Input frame rate (50 for HuBERT)
        ofps: Output frame rate (30 for blendshapes)
    """
    batch_size = audio_embedding_matrix.shape[0]
    
    if ifps % ofps == 0:
        factor = ifps // ofps  # factor = 2 for 50->30? No, that's wrong
        # Actually for 50->30, we need interpolation, not simple downsampling
        # Fall through to interpolation case
        pass
    
    if ifps > ofps:
        # Downsample: 50Hz -> 30Hz
        factor = ifps / ofps  # 50/30 = 1.666...
        audio_embedding_seq_len = int(vertex_matrix.shape[1] * factor)
        
        # Interpolate audio to match vertex length
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)  # (B, D, T)
        audio_embedding_matrix = F.interpolate(
            audio_embedding_matrix, 
            size=vertex_matrix.shape[1],  # Target length
            align_corners=True, 
            mode='linear'
        )
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)  # (B, T, D)
    else:
        # Upsample
        factor = ofps / ifps
        audio_embedding_seq_len = int(vertex_matrix.shape[1] * factor)
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(
            audio_embedding_matrix, 
            size=audio_embedding_seq_len, 
            align_corners=True, 
            mode='linear'
        )
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
    
    frame_num = vertex_matrix.shape[1]
    
    # Truncate/pad to match exactly
    if audio_embedding_matrix.shape[1] > frame_num:
        audio_embedding_matrix = audio_embedding_matrix[:, :frame_num, :]
    elif audio_embedding_matrix.shape[1] < frame_num:
        vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1], :]
        frame_num = audio_embedding_matrix.shape[1]
    
    return audio_embedding_matrix, vertex_matrix, frame_num


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BlendshapeDecoder(nn.Module):
    """
    Transformer decoder for speech token to blendshape prediction
    
    Architecture:
    1. Token embedding
    2. Positional encoding
    3. Transformer decoder layers
    4. Output projection to 51 blendshapes
    """
    
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_blendshapes=51
    ):
        """
        Args:
            vocab_size: Size of speech token vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            num_blendshapes: Number of output blendshapes (51 for ARKit)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_blendshapes = num_blendshapes
        
        # # Token embedding
        # self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Audio encoder
        self.audio_encoder = HubertModel.from_pretrained("pretrained_models/hubert/hubert-base-ls960", 
                                                         local_files_only = True)
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size  # 768
        self.audio_encoder.feature_extractor._freeze_parameters()

        # Freeze early layers
        frozen_layers = [0, 1]
        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False
        
        # Project HuBERT features to model dimension
        self.audio_projection = nn.Linear(self.audio_dim, d_model)  
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_blendshapes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.audio_projection.weight)
        if self.audio_projection.bias is not None:
            nn.init.zeros_(self.audio_projection.bias)
    
    def forward(self, audio_waveforms, target_length=None, audio_lengths=None):
        """
        Forward pass
        
        Args:
            audio_waveforms: (batch, samples) raw audio at 16kHz
            target_length: Desired output sequence length (for training with ground truth length)
            audio_lengths: (batch,) actual lengths of audio in samples (for masking)
        
        Returns:
            blendshapes: (batch, target_length, 51) predicted blendshapes
        """
        batch_size = audio_waveforms.shape[0]
        device = audio_waveforms.device
        
        # Extract HuBERT features (50Hz)
        x = self.audio_encoder(audio_waveforms).last_hidden_state  # (B, T_50hz, 768)
        x = self.audio_projection(x)  # (B, T_50hz, d_model)
        
        # Create dummy target for alignment
        if target_length is None:
            # Inference: estimate target length from HuBERT output
            target_length = int(x.shape[1] * 0.6)  # 50Hz -> 30Hz
        
        # Create dummy target tensor with correct shape for adjust_input_representation
        dummy_target = torch.zeros(batch_size, target_length, self.d_model, device=device)
        
        # Align HuBERT features (50Hz) to target fps (30Hz)
        x, dummy_target, frame_num = adjust_input_representation(
            x, dummy_target, ifps=50, ofps=30
        )
        x = x[:, :frame_num]  # (B, frame_num, d_model)
        
        # Apply positional encoding after alignment
        x = self.pos_encoder(x)
        
        # Create attention mask for padding (if audio_lengths provided)
        if audio_lengths is not None:
            # Convert audio sample lengths to HuBERT feature lengths (50Hz)
            feature_lengths_50hz = (audio_lengths / 16000 * 50).long()
            # Adjust to 30fps after alignment
            feature_lengths_30fps = (feature_lengths_50hz * 0.6).long()
            feature_lengths_30fps = torch.clamp(feature_lengths_30fps, max=frame_num)
            
            src_key_padding_mask = torch.arange(frame_num, device=device)[None, :] >= feature_lengths_30fps[:, None]
        else:
            src_key_padding_mask = None
        
        # Target queries (use frame_num as target length)
        tgt = self.pos_encoder.pe[:, :frame_num, :].expand(batch_size, -1, -1)  # (B, frame_num, d_model)
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt=tgt,
            memory=x,
            memory_key_padding_mask=src_key_padding_mask
        )  # (B, frame_num, d_model)
        
        # Project to blendshapes
        blendshapes = self.output_proj(output)  # (B, frame_num, 51)
        
        return blendshapes
    
    def predict(self, audio_waveforms, target_length):
        """
        Inference mode
        
        Args:
            audio_waveforms: (batch, samples) or (samples,)
            target_length: Desired output length
        
        Returns:
            blendshapes: (batch, target_length, 51) or (target_length, 51)
        """
        was_1d = False
        if audio_waveforms.dim() == 1:
            audio_waveforms = audio_waveforms.unsqueeze(0)
            was_1d = True
        
        with torch.no_grad():
            blendshapes = self.forward(audio_waveforms, target_length=target_length)
        
        if was_1d:
            blendshapes = blendshapes.squeeze(0)
        
        return blendshapes