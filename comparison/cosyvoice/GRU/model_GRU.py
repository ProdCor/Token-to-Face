import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from hubert.modeling_hubert import HubertModel
from torch import Tensor


def adjust_input_representation(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    """
    Brings audio embeddings and visual frames to the same frame rate.

    Args:
        audio_embedding_matrix: The audio embeddings extracted by the audio encoder
        vertex_matrix: The animation sequence represented as a series of vertex positions (or blendshape controls)
        ifps: The input frame rate (it is 50 for the HuBERT encoder)
        ofps: The output frame rate
    """
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
            vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
    elif ifps > ofps:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
    else:
        factor = 1
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))
    return audio_embedding_matrix, vertex_matrix, frame_num


# dropout mask for speech conditioning
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

class FaceDiffBeatCosyVoice2(nn.Module):
    """
    FaceDiffuser adapted for BEAT2 dataset with CosyVoice2 tokens
    Uses pre-extracted discrete speech tokens instead of HuBERT
    WITH DROPOUT REGULARIZATION
    """
    def __init__(
            self,
            args,
            vertice_dim: int,  # 51 for ARKit
            latent_dim: int = 256,
            diffusion_steps: int = 1000,
            gru_latent_dim: int = 256,
            num_layers: int = 2,
            embedding_dim: int = 2*768,  # CosyVoice2 token embedding dimension
            dropout: float = 0.3,  # Dropout rate for regularization
    ) -> None:
        super().__init__()
        
        self.vertice_dim = vertice_dim
        self.i_fps = 25  # CosyVoice2 token rate (25 Hz)
        self.o_fps = args.output_fps  # 30 fps for BEAT2
        self.one_hot_timesteps = np.eye(args.diff_steps)
        self.dropout_rate = dropout
        
        # CosyVoice2 token embedding (replaces HuBERT)
        self.vocab_size = 6561  # CosyVoice2 codebook size
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0  # Use 0 as padding token
        )
        self.audio_dim = self.embedding_dim
        
        self.device = args.device
        
        # Calculate conditional feature dimension after frame rate adjustment
        cond_feature_dim = self.audio_dim 
        
        print(f"FaceDiffBeatCosyVoice2 Configuration:")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Token rate: {self.i_fps} fps")
        print(f"  Motion rate: {self.o_fps} fps")
        print(f"  Conditional feature dim: {cond_feature_dim}")
        print(f"  Vertice dim: {vertice_dim}")
        print(f"  Dropout rate: {self.dropout_rate}")
        
        # Dropout for embedding
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Timestep embedding with dropout
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm_cond = nn.LayerNorm(cond_feature_dim + vertice_dim + latent_dim)
        
        # Dropout before GRU
        self.pre_gru_dropout = nn.Dropout(dropout)
        
        # GRU decoder (dropout only active with num_layers > 1)
        self.gru = nn.GRU(
            cond_feature_dim + vertice_dim + latent_dim,
            gru_latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout before output layer
        self.output_dropout = nn.Dropout(dropout)
        
        # Output layer
        self.final_layer = nn.Linear(gru_latent_dim, vertice_dim)
        
        # Initialize output layer to predict small changes
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)
    
    def forward(
        self, 
        x: Tensor,  # Noised facial parameters (B, T, vertice_dim)
        times: Tensor,  # Diffusion timestep 
        cond_embed: Tensor,  # CosyVoice2 token indices (B, token_seq_len)
        one_hot=None,  # Not used for BEAT2
        template=None  # Not used for BEAT2
    ):
        batch_size, device = x.shape[0], x.device
        
        # Convert timestep tensor to one-hot encoding
        times_np = times.cpu().numpy()
        times_onehot = torch.from_numpy(self.one_hot_timesteps[times_np]).float()
        times_onehot = times_onehot.to(device=device)
        
        # Embed discrete tokens
        token_indices = cond_embed.long()
        hidden_states = self.token_embedding(token_indices)
        
        # Apply dropout to embeddings
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Get sequence lengths
        token_seq_len = hidden_states.shape[1]
        motion_seq_len = x.shape[1]
        
        # Calculate expected token length
        expected_token_len = int(motion_seq_len * self.i_fps / self.o_fps)
        
        # Trim or pad tokens to expected length
        if token_seq_len > expected_token_len:
            hidden_states = hidden_states[:, :expected_token_len, :]
        elif token_seq_len < expected_token_len:
            padding = torch.zeros(
                batch_size, 
                expected_token_len - token_seq_len, 
                self.embedding_dim,
                device=device
            )
            hidden_states = torch.cat([hidden_states, padding], dim=1)
        
        # Interpolate from 25fps to 30fps
        hidden_states = hidden_states.permute(0, 2, 1)  # (B, D, T)
        hidden_states = torch.nn.functional.interpolate(
            hidden_states,
            size=motion_seq_len,
            mode='linear',
            align_corners=True
        )
        hidden_states = hidden_states.permute(0, 2, 1)  # (B, T, D)
        
        # Match sequence lengths
        frame_num = min(hidden_states.shape[1], x.shape[1])
        cond_embed = hidden_states[:, :frame_num]
        x = x[:, :frame_num]
        
        # Create timestep embedding
        t_tokens = self.time_mlp(times_onehot)  # Dropout already applied in time_mlp
        t_tokens = t_tokens.unsqueeze(1).repeat(1, frame_num, 1)
        
        # Concatenate conditioning
        full_cond_tokens = torch.cat([cond_embed, x, t_tokens], dim=-1)
        full_cond_tokens = self.norm_cond(full_cond_tokens)
        
        # Apply dropout before GRU
        full_cond_tokens = self.pre_gru_dropout(full_cond_tokens)
        
        # GRU processing
        output, _ = self.gru(full_cond_tokens)
        
        # Apply dropout before final layer
        output = self.output_dropout(output)
        
        # Final output
        output = self.final_layer(output)
        
        return output