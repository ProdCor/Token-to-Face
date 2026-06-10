import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


def adjust_input_representation(audio_embedding_matrix, vertex_matrix, ifps, ofps):
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


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class SpeechTokenizerEmbedder(nn.Module):
    """
    RVQ embedder: one embedding table per quantizer level, summed.
    Input: (B, n_q, T) integer codes
    Output: (B, T, embedding_dim)
    """
    def __init__(self, n_q, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim) for _ in range(n_q)
        ])

    def forward(self, codes):  # codes: (B, n_q, T)
        return sum(emb(codes[:, i, :]) for i, emb in enumerate(self.embeddings))


class FaceDiffBeatSpeechTokenizer(nn.Module):
    """
    FaceDiffuser adapted for BEAT2 dataset with SpeechTokenizer tokens.
    Uses all RVQ levels (semantic + acoustic) via per-level embedding + sum.
    """
    def __init__(
            self,
            args,
            vertice_dim: int,       # 51 for ARKit
            latent_dim: int = 256,
            diffusion_steps: int = 1000,
            gru_latent_dim: int = 256,
            num_layers: int = 2,
            dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.vertice_dim = vertice_dim
        self.i_fps = 50             # SpeechTokenizer token rate (16kHz / 320 stride = 50 Hz)
        self.o_fps = args.output_fps
        self.one_hot_timesteps = np.eye(args.diff_steps)
        self.dropout_rate = dropout

        # SpeechTokenizer RVQ embedding
        self.vocab_size = args.codebook_size        # 1024 per RVQ level
        self.n_q = args.n_q                         # 8 RVQ levels
        self.embedding_dim = args.token_embedding_dim

        self.token_embedding = SpeechTokenizerEmbedder(
            n_q=self.n_q,
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim
        )
        self.audio_dim = self.embedding_dim

        self.device = args.device

        cond_feature_dim = self.audio_dim

        print(f"FaceDiffBeatSpeechTokenizer Configuration:")
        print(f"  Vocab size (per RVQ level): {self.vocab_size}")
        print(f"  RVQ levels (n_q): {self.n_q}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Token rate: {self.i_fps} fps")
        print(f"  Motion rate: {self.o_fps} fps")
        print(f"  Conditional feature dim: {cond_feature_dim}")
        print(f"  Vertice dim: {vertice_dim}")
        print(f"  Dropout rate: {self.dropout_rate}")

        self.embedding_dropout = nn.Dropout(dropout)

        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        self.norm_cond = nn.LayerNorm(cond_feature_dim + vertice_dim + latent_dim)
        self.pre_gru_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            cond_feature_dim + vertice_dim + latent_dim,
            gru_latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(gru_latent_dim, vertice_dim)

        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(
        self,
        x: Tensor,          # Noised facial parameters (B, T, vertice_dim)
        times: Tensor,      # Diffusion timestep
        cond_embed: Tensor, # SpeechTokenizer token indices (B, n_q, T_tokens)
        one_hot=None,
        template=None
    ):
        batch_size, device = x.shape[0], x.device

        # Timestep one-hot
        times_np = times.cpu().numpy()
        times_onehot = torch.from_numpy(self.one_hot_timesteps[times_np]).float().to(device)

        # Embed all RVQ levels and sum -> (B, T_tokens, embedding_dim)
        hidden_states = self.token_embedding(cond_embed)
        hidden_states = self.embedding_dropout(hidden_states)

        token_seq_len = hidden_states.shape[1]
        motion_seq_len = x.shape[1]

        # Expected token length at 50Hz for motion_seq_len frames at o_fps
        expected_token_len = int(motion_seq_len * self.i_fps / self.o_fps)

        # Trim or pad to expected length
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

        # Interpolate from 50Hz to o_fps (e.g. 30fps)
        hidden_states = hidden_states.permute(0, 2, 1)          # (B, D, T)
        hidden_states = F.interpolate(
            hidden_states,
            size=motion_seq_len,
            mode='linear',
            align_corners=True
        )
        hidden_states = hidden_states.permute(0, 2, 1)          # (B, T, D)

        # Match sequence lengths
        frame_num = min(hidden_states.shape[1], x.shape[1])
        cond_embed = hidden_states[:, :frame_num]
        x = x[:, :frame_num]

        # Timestep embedding
        t_tokens = self.time_mlp(times_onehot)
        t_tokens = t_tokens.unsqueeze(1).repeat(1, frame_num, 1)

        # Concatenate and normalize
        full_cond_tokens = torch.cat([cond_embed, x, t_tokens], dim=-1)
        full_cond_tokens = self.norm_cond(full_cond_tokens)
        full_cond_tokens = self.pre_gru_dropout(full_cond_tokens)

        # GRU
        output, _ = self.gru(full_cond_tokens)
        output = self.output_dropout(output)
        output = self.final_layer(output)

        return output