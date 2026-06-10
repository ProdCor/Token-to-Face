#!/usr/bin/env python3
"""
Transformer Decoder for SpeechTokenizer Speech Token to ARKit Blendshape prediction
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SpeechTokenizerEmbedder(nn.Module):
    """
    RVQ embedder: one embedding table per quantizer level, summed.
    Standard RVQ reconstruction convention.
    """
    def __init__(self, n_q=8, vocab_size=1024, d_model=512):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(n_q)
        ])

    def forward(self, codes):
        """
        Args:
            codes: (batch, n_q, T) integer token indices
        Returns:
            (batch, T, d_model)
        """
        return sum(emb(codes[:, i, :]) for i, emb in enumerate(self.embeddings))


class BlendshapeDecoder(nn.Module):
    """
    Transformer decoder for SpeechTokenizer speech tokens to blendshape prediction.

    Architecture:
    1. Per-RVQ-layer embedding + sum (SpeechTokenizerEmbedder)
    2. Positional encoding
    3. Transformer decoder layers
    4. Output projection to 51 blendshapes
    """
    
    def __init__(
        self,
        vocab_size=1024,        # SpeechTokenizer vocab size per RVQ level
        n_q=8,                  # Number of RVQ quantizer levels
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_blendshapes=51,
        token_fps=50,           # SpeechTokenizer: 50Hz (16kHz / 320 stride)
        blendshape_fps=30,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_blendshapes = num_blendshapes
        self.time_ratio = blendshape_fps / token_fps  # 30/50 = 0.6

        # RVQ embedding (replaces single nn.Embedding)
        self.token_embedding = SpeechTokenizerEmbedder(n_q, vocab_size, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
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
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_blendshapes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for emb in self.token_embedding.embeddings:
            emb.weight.data.uniform_(-0.1, 0.1)
        for p in self.output_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, speech_tokens, target_length=None, token_lengths=None):
        """
        Args:
            speech_tokens: (batch, n_q, T) integer token indices
            target_length: Desired output sequence length
            token_lengths: (batch,) actual lengths for padding mask
        
        Returns:
            blendshapes: (batch, target_length, 51)
        """
        batch_size = speech_tokens.shape[0]
        seq_len = speech_tokens.shape[2]  # T is now dim 2, not dim 1
        
        # Embed all RVQ levels and sum -> (B, T, D)
        x = self.token_embedding(speech_tokens) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        if token_lengths is not None:
            src_key_padding_mask = torch.arange(seq_len, device=speech_tokens.device)[None, :] >= token_lengths[:, None]
        else:
            src_key_padding_mask = None
        
        if target_length is None:
            # SpeechTokenizer is 50Hz, blendshapes are 30Hz -> ratio 0.6
            target_length = int(seq_len * self.time_ratio)
        
        tgt = self.pos_encoder.pe[:, :target_length, :].expand(batch_size, -1, -1)

        output = self.transformer_decoder(
            tgt=tgt,
            memory=x,
            memory_key_padding_mask=src_key_padding_mask
        )  # (B, T_tgt, D)
        
        blendshapes = self.output_proj(output)  # (B, T_tgt, 51)
        
        return blendshapes
    
    def predict(self, speech_tokens, target_length):
        """
        Args:
            speech_tokens: (batch, n_q, T) or (n_q, T)
        Returns:
            blendshapes: (batch, target_length, 51) or (target_length, 51)
        """
        was_2d = False
        if speech_tokens.dim() == 2:  # (n_q, T) -> (1, n_q, T)
            speech_tokens = speech_tokens.unsqueeze(0)
            was_2d = True
        
        with torch.no_grad():
            blendshapes = self.forward(speech_tokens, target_length=target_length)
        
        if was_2d:
            blendshapes = blendshapes.squeeze(0)
        
        return blendshapes