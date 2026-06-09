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

class BlendshapeEncoderDecoder(nn.Module):
    """
    Encoder-Decoder Transformer for speech token to blendshape prediction
    
    Architecture:
    1. Encoder: Process speech tokens with bidirectional attention
    2. Decoder: Generate blendshapes with learnable frame queries
    3. Output projection to 51 blendshapes
    
    Key improvement over decoder-only: Learnable frame queries that can
    dynamically attend to relevant parts of the encoded token sequence.
    """
    
    def __init__(
        self,
        vocab_size=6561,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_blendshapes=51,
        max_target_len=2000
    ):
        """
        Args:
            vocab_size: Size of speech token vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            num_blendshapes: Number of output blendshapes (51 for ARKit)
            max_target_len: Maximum target sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_blendshapes = num_blendshapes
        self.max_target_len = max_target_len
        
        # ============ ENCODER ============
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding for encoder
        self.encoder_pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # ============ DECODER ============
        # Learnable frame queries (content embeddings)
        self.frame_queries = nn.Embedding(max_target_len, d_model)
        
        # Positional encoding for decoder
        self.decoder_pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
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
            num_layers=num_decoder_layers
        )
        
        # ============ OUTPUT ============
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_blendshapes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.frame_queries.weight.data.normal_(0, initrange)
        
        for p in self.output_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def _init_weights(self):
    #     """Initialize weights"""
    #     initrange = 0.1
    #     self.token_embedding.weight.data.uniform_(-initrange, initrange)
        
    #     # Initialize frame queries with orthogonal initialization
    #     nn.init.orthogonal_(self.frame_queries.weight)  # ← MUCH BETTER!
        
    #     for p in self.output_proj.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    #         elif p.dim() == 1:
    #             nn.init.zeros_(p)
    
    def forward(self, speech_tokens, target_length=None, token_lengths=None):
        """
        Forward pass
        
        Args:
            speech_tokens: (batch, seq_len) speech token indices
            target_length: Desired output sequence length
            token_lengths: (batch,) actual lengths of sequences (for masking)
        
        Returns:
            blendshapes: (batch, target_length, 51) predicted blendshapes
        """
        batch_size, seq_len = speech_tokens.shape
        device = speech_tokens.device
        
        # ============ ENCODER ============
        # 1. Embed and encode speech tokens
        x = self.token_embedding(speech_tokens) * math.sqrt(self.d_model)  # (B, T_token, D)
        x = self.encoder_pos_encoder(x)
        
        # 2. Create padding mask for encoder
        if token_lengths is not None:
            # True for padding positions
            src_key_padding_mask = torch.arange(seq_len, device=device)[None, :] >= token_lengths[:, None]
        else:
            src_key_padding_mask = None
        
        # 3. Encode tokens with bidirectional attention
        memory = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )  # (B, T_token, D)
        
        # ============ DECODER ============
        # 4. Determine target length
        if target_length is None:
            # Inference: approximate 25Hz -> 30Hz conversion
            target_length = int(seq_len * 1.2)
        
        if target_length > self.max_target_len:
            raise ValueError(f"target_length {target_length} exceeds max_target_len {self.max_target_len}")
        
        # 5. Create learnable queries for target frames
        frame_indices = torch.arange(target_length, device=device)
        tgt = self.frame_queries(frame_indices).unsqueeze(0).expand(batch_size, -1, -1)  # (B, T_target, D)
        #tgt = self.decoder_pos_encoder(tgt)
        
        # 6. Decode with cross-attention to encoded tokens
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_key_padding_mask
        )  # (B, T_target, D)
        
        # ============ OUTPUT ============
        # 7. Project to blendshapes
        blendshapes = self.output_proj(output)  # (B, T_target, 51)
        
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


class BlendshapeDecoderConv1D(nn.Module):
    """
    Decoder following ProbTalk3D's ACTUAL architecture from their code
    
    Key components from their TransformerDecoder:
    1. Conv1D (kernel=5, replicate padding)
    2. LeakyReLU + InstanceNorm
    3. Linear embedding
    4. Positional encoding
    5. Transformer
    6. Output projection
    """
    
    def __init__(
        self,
        vocab_size=6561,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_blendshapes=51,
        conv_kernel_size=5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_blendshapes = num_blendshapes
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # ===================================================================
        # ProbTalk3D's preprocessing module
        # From their code: Conv1d + LeakyReLU + InstanceNorm
        # ===================================================================
        self.expander = nn.Sequential(
            nn.Conv1d(
                d_model, 
                d_model, 
                kernel_size=conv_kernel_size,
                stride=1,
                padding=conv_kernel_size // 2,
                padding_mode='replicate'  # Important: ProbTalk3D use replicate mode
            ),
            nn.LeakyReLU(0.2, inplace=False),
            nn.InstanceNorm1d(d_model, affine=False)
        )
        
        # Linear embedding (they do this AFTER conv)
        self.linear_embedding = nn.Linear(d_model, d_model)
        
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
        
        # Output projection (they call it "feature_mapping_reverse")
        self.feature_mapping_reverse = nn.Linear(d_model, num_blendshapes, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        
        # Initialize linear embedding
        nn.init.xavier_uniform_(self.linear_embedding.weight)
        if self.linear_embedding.bias is not None:
            nn.init.constant_(self.linear_embedding.bias, 0)
        
        # Output projection
        nn.init.xavier_uniform_(self.feature_mapping_reverse.weight)
    
    def forward(self, speech_tokens, target_length=None, token_lengths=None):
        """
        Forward pass following ProbTalk3D's architecture
        
        Args:
            speech_tokens: (batch, seq_len) speech token indices
            target_length: Desired output sequence length
            token_lengths: (batch,) actual lengths of sequences (for masking)
        
        Returns:
            blendshapes: (batch, target_length, 51) predicted blendshapes
        """
        batch_size, seq_len = speech_tokens.shape
        
        # Embed tokens
        x = self.token_embedding(speech_tokens) * math.sqrt(self.d_model)  # (B, T, D)
        
        # ===================================================================
        # Apply ProbTalk3D's preprocessing module
        # Conv1D expects (batch, channels, time)
        # ===================================================================
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.expander(x)     # Conv1D + LeakyReLU + InstanceNorm
        x = x.permute(0, 2, 1)  # (B, T, D)
        
        # Linear embedding (they do this after conv)
        x = self.linear_embedding(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask for padding
        if token_lengths is not None:
            src_key_padding_mask = torch.arange(seq_len, device=speech_tokens.device)[None, :] >= token_lengths[:, None]
        else:
            src_key_padding_mask = None
        
        # Prepare target queries
        if target_length is None:
            target_length = int(seq_len * 1.2)  # 25Hz -> 30Hz
        
        # Target queries (using positional encodings)
        tgt = self.pos_encoder.pe[:, :target_length, :].expand(batch_size, -1, -1)
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt=tgt,
            memory=x,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Project to blendshapes (no bias, like ProbTalk3D)
        blendshapes = self.feature_mapping_reverse(output)
        
        return blendshapes
    
    def predict(self, speech_tokens, target_length):
        """Inference mode"""
        was_1d = False
        if speech_tokens.dim() == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
            was_1d = True
        
        with torch.no_grad():
            blendshapes = self.forward(speech_tokens, target_length=target_length)
        
        if was_1d:
            blendshapes = blendshapes.squeeze(0)
        
        return blendshapes