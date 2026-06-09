#!/usr/bin/env python3
"""
Transformer Decoder for Speech Token to ARKit Blendshape prediction
"""
import torch
import torch.nn as nn
import math


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
        vocab_size=6561,
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
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
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
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        
        for p in self.output_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, speech_tokens, target_length=None, token_lengths=None):
        """
        Forward pass
        
        Args:
            speech_tokens: (batch, seq_len) speech token indices
            target_length: Desired output sequence length (for training with ground truth length)
            token_lengths: (batch,) actual lengths of sequences (for masking)
        
        Returns:
            blendshapes: (batch, target_length, 51) predicted blendshapes
        """
        batch_size, seq_len = speech_tokens.shape
        
        # Embed tokens
        x = self.token_embedding(speech_tokens) * math.sqrt(self.d_model)  # (B, T, D)
        x = self.pos_encoder(x)
        
        # Create attention mask for padding
        if token_lengths is not None:
            # Create mask: True for padding positions
            src_key_padding_mask = torch.arange(seq_len, device=speech_tokens.device)[None, :] >= token_lengths[:, None]
        else:
            src_key_padding_mask = None
        
        # Prepare target queries
        if target_length is None:
            # Inference: use heuristic (e.g., seq_len * 0.8 for 25Hz tokens -> 30Hz frames)
            target_length = int(seq_len * 1.2)  # Approximate 25Hz -> 30Hz conversion
        
        # Using positional encodings as queries
        tgt = self.pos_encoder.pe[:, :target_length, :].expand(batch_size, -1, -1)  # (B, T_tgt, D)

        # Transformer decoder
        # memory = encoder output (in this case, same as input embeddings)
        # tgt = target queries
        output = self.transformer_decoder(
            tgt=tgt,
            memory=x,
            memory_key_padding_mask=src_key_padding_mask
        )  # (B, T_tgt, D)
        
        # Project to blendshapes
        blendshapes = self.output_proj(output)  # (B, T_tgt, 51)
        
        return blendshapes
    
    def predict(self, speech_tokens, target_length):
        """
        Inference mode
        
        Args:
            speech_tokens: (batch, seq_len) or (seq_len,)
            target_length: Desired output length
        
        Returns:
            blendshapes: (batch, target_length, 51) or (target_length, 51)
        """
        was_1d = False
        if speech_tokens.dim() == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
            was_1d = True
        
        with torch.no_grad():
            blendshapes = self.forward(speech_tokens, target_length=target_length)
        
        if was_1d:
            blendshapes = blendshapes.squeeze(0)
        
        return blendshapes