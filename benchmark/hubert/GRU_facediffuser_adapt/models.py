import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from hubert.modeling_hubert import HubertModel
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


class FaceDiffDamm(nn.Module):
    def __init__(self, args):
        super(FaceDiffDamm, self).__init__()
        self.num_blendshapes = args.vertice_dim
        self.latent_dim = args.feature_dim
        self.num_layers = 4
        self.dropout = 0.1

        self.i_fps = 50
        self.o_fps = 30

        # audio encoder
        self.audio_encoder = HubertModel.from_pretrained("pretrained_models/hubert/hubert-base-ls960", 
                                                         local_files_only=True
        )
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size
        self.audio_encoder.feature_extractor._freeze_parameters()

        frozen_layers = [0,1]
        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # rnn based
        self.rnn = nn.GRU(self.audio_dim * 2 + self.num_blendshapes + self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True, dropout=0.3)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # final output
        self.out = nn.Linear(self.latent_dim, self.num_blendshapes)
        nn.init.normal_(self.out.weight)
        nn.init.normal_(self.out.bias)

    def forward(self, noised_anim, timesteps, cond_embed, one_hot=None, template=None):
        _, n_frames, _ = noised_anim.shape
        emb = self.embed_timestep(timesteps)
        emb = emb.repeat(n_frames, 1, 1)
        emb = emb.permute(1, 0, 2)

        hidden_states = cond_embed
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        hidden_states, noised_anim, frame_num = adjust_input_representation(
            hidden_states, noised_anim, self.i_fps, self.o_fps
        )
        hidden_states = hidden_states[:, :frame_num]

        input_emb = torch.cat([hidden_states, emb, noised_anim], -1)
        output, _ = self.rnn(input_emb)
        y_pred = self.out(output)
        return y_pred


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class FaceDiffBeat(nn.Module):
    def __init__(
            self,
            args,
            vertice_dim: int,
            latent_dim: int = 256,
            cond_feature_dim: int = 1536,
            diffusion_steps: int = 1000,
            gru_latent_dim: int = 256,
            num_layers: int = 2,

    ) -> None:

        super().__init__()
        self.i_fps = args.input_fps # audio fps (input to the network)
        self.o_fps = args.output_fps # 4D Scan fps (output or target)
        self.one_hot_timesteps = np.eye(args.diff_steps)

        # Audio Encoder
        self.audio_encoder = HubertModel.from_pretrained("pretrained_models/hubert/hubert-base-ls960", 
        local_files_only=True)
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.device = args.device

        frozen_layers = [0,1]

        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False

        # timestep projection
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
        )

        self.norm_cond = nn.LayerNorm(cond_feature_dim + vertice_dim + latent_dim)

        # facial decoder
        self.gru = nn.GRU(
            cond_feature_dim + vertice_dim + latent_dim,
            gru_latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.final_layer = nn.Linear(gru_latent_dim, vertice_dim)

    def forward(
            self, x: Tensor,  times: Tensor, cond_embed: Tensor, one_hot=None, template=None
    ):
        batch_size, device = x.shape[0], x.device
        times = torch.FloatTensor(self.one_hot_timesteps[times])
        times = times.to(device=device)

        hidden_states = cond_embed
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        hidden_states, x, frame_num = adjust_input_representation(hidden_states, x, self.i_fps, self.o_fps)
        cond_embed = hidden_states[:, :frame_num]
        x = x[:, :frame_num]

        cond_tokens = cond_embed

        # create the diffusion timestep embedding
        t_tokens = self.time_mlp(times)
        t_tokens = t_tokens.repeat(frame_num, 1, 1)
        t_tokens = t_tokens.permute(1, 0, 2)

        # full conditioning tokens
        full_cond_tokens = torch.cat([cond_tokens, x, t_tokens], dim=-1)
        full_cond_tokens = self.norm_cond(full_cond_tokens)

        output, _ = self.gru(full_cond_tokens)
        output = self.final_layer(output)

        return output


class FaceDiff(nn.Module):
    def __init__(
            self,
            args,
            vertice_dim: int,
            latent_dim: int = 512,
            cond_feature_dim: int = 1536,
            diffusion_steps: int = 500,
            gru_latent_dim: int = 512,
            num_layers: int = 2,


    ) -> None:

        super().__init__()
        self.i_fps = args.input_fps # audio fps (input to the network)
        self.o_fps = args.output_fps # 4D Scan fps (output or target)
        self.one_hot_timesteps = np.eye(args.diff_steps)

        # Audio Encoder
        self.audio_encoder = HubertModel.from_pretrained("pretrained_models/hubert/hubert-base-ls960", 
        local_files_only=True)
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.device = args.device

        frozen_layers = [0,1]

        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False

        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)

        # noised animation projection
        self.input_projection = nn.Sequential(
            nn.Linear(vertice_dim, latent_dim * 2),
            nn.Conv1d(1, 1, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        # timestep projection
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
        )
        self.norm_cond = nn.LayerNorm(latent_dim * 3)

        # facial decoder
        self.gru = nn.GRU(latent_dim * 3, gru_latent_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.final_layer = nn.Linear(gru_latent_dim, vertice_dim)
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

        # Subject embedding, S
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), latent_dim, bias=False)

    def forward(
            self, x: Tensor,  times: Tensor, cond_embed: Tensor, template, one_hot,
    ):
        batch_size, device = x.shape[0], x.device
        times = torch.FloatTensor(self.one_hot_timesteps[times])
        times = times.to(device=device)

        template = template.unsqueeze(1)
        obj_embedding = self.obj_vector(one_hot)

        # project to latent space
        x = x.permute(1, 0, 2)
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)

        hidden_states = cond_embed
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        hidden_states, x, frame_num = adjust_input_representation(hidden_states, x, self.i_fps, self.o_fps)
        cond_embed = hidden_states[:, :frame_num]
        x = x[:, :frame_num]

        cond_tokens = self.cond_projection(cond_embed)

        # create the diffusion timestep embedding
        t_tokens = self.time_mlp(times)
        t_tokens = t_tokens.repeat(frame_num, 1, 1)
        t_tokens = t_tokens.permute(1, 0, 2)

        # full conditioning tokens
        full_cond_tokens = torch.cat([cond_tokens, x, t_tokens], dim=-1)
        full_cond_tokens = self.norm_cond(full_cond_tokens)

        output, _ = self.gru(full_cond_tokens)
        output = output * obj_embedding
        output = self.final_layer(output)
        output = output + template

        return output


'''class FaceDiffBeat2(nn.Module):
    """
    FaceDiffuser adapted for BEAT2 dataset
    Supports both FLAME (103) and ARKit (51) representations
    """
    def __init__(
            self,
            args,
            vertice_dim: int,  # 103 for FLAME, 51 for ARKit
            latent_dim: int = 256,
            diffusion_steps: int = 1000,
            gru_latent_dim: int = 256,
            num_layers: int = 2,
    ) -> None:
        super().__init__()
        
        self.vertice_dim = vertice_dim
        self.i_fps = args.input_fps
        self.o_fps = args.output_fps
        self.one_hot_timesteps = np.eye(args.diff_steps)
        
        # Audio Encoder (HuBERT)
        hubert_path = getattr(args, 'hubert_path', 'facebook/hubert-base-ls960')
        print(f"Loading HuBERT from: {hubert_path}")
        
        self.audio_encoder = HubertModel.from_pretrained("pretrained_models/hubert/hubert-base-ls960", 
        local_files_only=True)
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size  # 768
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.device = args.device
        
        # Freeze early layers
        frozen_layers = [0, 1]
        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False
        
        # Audio feature dimensionality after adjust_input_representation
        # HuBERT outputs 768-dim features at 50 fps
        # After adjustment to 30 fps, we get 768*2 = 1536 dim
        cond_feature_dim = self.audio_dim * (self.i_fps // self.o_fps)
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
        )
        
        # Layer normalization
        self.norm_cond = nn.LayerNorm(cond_feature_dim + vertice_dim + latent_dim)
        
        # GRU decoder
        self.gru = nn.GRU(
            cond_feature_dim + vertice_dim + latent_dim,
            gru_latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Output layer
        self.final_layer = nn.Linear(gru_latent_dim, vertice_dim)
        
        # Initialize output layer to predict small changes
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)
    
    def forward(
            self, 
            x: Tensor,  # Noised facial parameters
            times: Tensor,  # Diffusion timestep
            cond_embed: Tensor,  # Audio features
            one_hot=None,  # Not used for BEAT2
            template=None  # Not used for BEAT2
    ):
        batch_size, device = x.shape[0], x.device
        
        # One-hot encode timestep
        times = torch.FloatTensor(self.one_hot_timesteps[times])
        times = times.to(device=device)
        
        # Extract audio features
        hidden_states = cond_embed
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        
        # Adjust frame rates (50 fps audio -> 30 fps motion)
        hidden_states, x, frame_num = adjust_input_representation(
            hidden_states, x, self.i_fps, self.o_fps
        )
        
        cond_embed = hidden_states[:, :frame_num]
        x = x[:, :frame_num]
        
        # Create timestep embedding
        t_tokens = self.time_mlp(times)
        t_tokens = t_tokens.repeat(frame_num, 1, 1)
        t_tokens = t_tokens.permute(1, 0, 2)
        
        # Concatenate conditioning
        full_cond_tokens = torch.cat([cond_embed, x, t_tokens], dim=-1)
        full_cond_tokens = self.norm_cond(full_cond_tokens)
        
        # GRU processing
        output, _ = self.gru(full_cond_tokens)
        output = self.final_layer(output)
        
        return output'''

class FaceDiffBeat2(nn.Module):
    """
    FaceDiffuser adapted for BEAT2 dataset
    Supports both FLAME (103) and ARKit (51) representations
    """
    def __init__(
            self,
            args,
            vertice_dim: int,  # 103 for FLAME, 51 for ARKit
            latent_dim: int = 256,
            diffusion_steps: int = 1000,
            gru_latent_dim: int = 256,
            num_layers: int = 2,
    ) -> None:
        super().__init__()
        
        self.vertice_dim = vertice_dim
        self.i_fps = args.input_fps  # 50
        self.o_fps = args.output_fps  # 30
        self.one_hot_timesteps = np.eye(args.diff_steps)
        
        # Audio Encoder (HuBERT) - Load from local path
        hubert_path = getattr(args, 'hubert_path', 'facebook/hubert-base-ls960')
        print(f"Loading HuBERT from: {hubert_path}")
        
        try:
            self.audio_encoder = HubertModel.from_pretrained(
                hubert_path,
                local_files_only=True
            )
        except Exception as e:
            print(f"Failed to load from {hubert_path}, trying default...")
            self.audio_encoder = HubertModel.from_pretrained(
                "facebook/hubert-base-ls960"
            )
        
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size  # 768
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.device = args.device
        
        # Freeze early layers
        frozen_layers = [0, 1]
        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False
        
        # Audio feature dimensionality after adjust_input_representation
        # HuBERT outputs 768-dim features at 50 fps
        # After adjustment to 30 fps: we concatenate multiple frames
        # The actual dimension depends on how adjust_input_representation works
        # For 50->30 fps conversion with factor=2, we get 768*2 = 1536
        factor = self.i_fps // self.o_fps if self.i_fps % self.o_fps == 0 else 1
        cond_feature_dim = self.audio_dim * factor
        
        print(f"Model configuration:")
        print(f"  Audio encoder dim: {self.audio_dim}")
        print(f"  FPS conversion: {self.i_fps} -> {self.o_fps} (factor: {factor})")
        print(f"  Conditional feature dim: {cond_feature_dim}")
        print(f"  Vertice dim: {vertice_dim}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Total input dim: {cond_feature_dim + vertice_dim + latent_dim}")
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
        )
        
        # Layer normalization for concatenated input
        self.norm_cond = nn.LayerNorm(cond_feature_dim + vertice_dim + latent_dim)
        
        # GRU decoder
        self.gru = nn.GRU(
            cond_feature_dim + vertice_dim + latent_dim,
            gru_latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Output layer
        self.final_layer = nn.Linear(gru_latent_dim, vertice_dim)
        
        # Initialize output layer to predict small changes
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)
    
    def forward(
            self, 
            x: Tensor,  # Noised facial parameters (B, T, vertice_dim)
            times: Tensor,  # Diffusion timestep
            cond_embed: Tensor,  # Audio features (B, audio_len)
            one_hot=None,  # Not used for BEAT2
            template=None  # Not used for BEAT2
    ):
        batch_size, device = x.shape[0], x.device
        
        # One-hot encode timestep
        times = torch.FloatTensor(self.one_hot_timesteps[times])
        times = times.to(device=device)
        
        # Extract audio features
        hidden_states = cond_embed
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        
        # Adjust frame rates (50 fps audio -> 30 fps motion)
        hidden_states, x, frame_num = adjust_input_representation(
            hidden_states, x, self.i_fps, self.o_fps
        )
        
        # Ensure we have matching sequence lengths
        cond_embed = hidden_states[:, :frame_num]
        x = x[:, :frame_num]
        
        # Debug: print shapes
        if not hasattr(self, '_debug_printed'):
            print(f"\nForward pass shapes:")
            print(f"  Audio features after adjust: {cond_embed.shape}")
            print(f"  Facial params: {x.shape}")
            print(f"  Expected cond dim: {self.norm_cond.normalized_shape}")
            self._debug_printed = True
        
        # Create timestep embedding
        t_tokens = self.time_mlp(times)  # (B, latent_dim)
        t_tokens = t_tokens.unsqueeze(1).repeat(1, frame_num, 1)  # (B, T, latent_dim)
        
        # Concatenate conditioning
        full_cond_tokens = torch.cat([cond_embed, x, t_tokens], dim=-1)
        
        # Check dimension
        expected_dim = self.norm_cond.normalized_shape[0]
        actual_dim = full_cond_tokens.shape[-1]
        
        if expected_dim != actual_dim:
            raise RuntimeError(
                f"Dimension mismatch in forward pass:\n"
                f"  Expected: {expected_dim}\n"
                f"  Got: {actual_dim}\n"
                f"  cond_embed: {cond_embed.shape}\n"
                f"  x: {x.shape}\n"
                f"  t_tokens: {t_tokens.shape}\n"
                f"  Check if i_fps={self.i_fps} and o_fps={self.o_fps} are correct"
            )
        
        # Apply normalization
        full_cond_tokens = self.norm_cond(full_cond_tokens)
        
        # GRU processing
        output, _ = self.gru(full_cond_tokens)
        output = self.final_layer(output)
        
        return output