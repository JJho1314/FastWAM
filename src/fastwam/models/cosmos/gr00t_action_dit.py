"""GR00T-N1.7 flow-matching action DiT, vendored for the AGRA coupling.

This is the REAL NVIDIA GR00T action DiT (Isaac-GR00T `gr00t/model/modules/dit.py`,
Apache-2.0), copied verbatim so we can (a) load GR00T's pretrained `action_head.model.*`
weights and (b) keep it import-light — it depends ONLY on torch + diffusers, never on
the gr00t package, transformers, tyro, flash-attn, or the Eagle/Cosmos-Reason VLM.

The AGRA paper ("Making Foresight Actionable", arXiv 2606.12217) "employs the
flow-matching action DiT from GR00T-N1": the action DiT cross-attends to the world
model's foresight features instead of a VLM backbone. GR00T's native `DiT` feeds ONE
shared backbone context to every cross-attention block; AGRA's multi-layer bridge feeds
a DIFFERENT video-DiT layer to each action cross-attn block. `BridgeDiT` below is the
only change: its forward takes a per-(cross-)block context list.

GR00T-N1.7-3B `diffusion_model_cfg` (config.json): num_layers=32 (interleaved →
16 cross + 16 self), num_attention_heads=32, attention_head_dim=48 (inner_dim=1536),
output_dim=1024, norm_type="ada_norm", cross_attention_dim=2048 (backbone_embedding_dim).
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)


# --------------------------------------------------------------------------- #
# vendored verbatim from Isaac-GR00T gr00t/model/modules/dit.py (Apache-2.0)  #
# --------------------------------------------------------------------------- #
def _is_spark_sm121() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) == (12, 1)


def _should_force_math_sdpa() -> bool:
    override = os.environ.get("GR00T_DIT_SDPA_MODE")
    if override == "math":
        return True
    if override == "default":
        return False
    return _is_spark_sm121()


def _sdpa_context():
    if not _should_force_math_sdpa():
        return nullcontext()
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False, enable_cudnn=False
    )


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, norm_elementwise_affine: bool = False,
                 norm_eps: float = 1e-5, chunk_dim: int = 0):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim, dropout=dropout, activation_fn=activation_fn,
            final_dropout=final_dropout, inner_dim=ff_inner_dim, bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        with _sdpa_context():
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=(
                    encoder_attention_mask if encoder_hidden_states is not None else attention_mask
                ),
            )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=self.config.compute_dtype
        )

        all_blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None
            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(hidden_states, attention_mask=None,
                                      encoder_hidden_states=None, encoder_attention_mask=None, temb=temb)
            else:
                hidden_states = block(hidden_states, attention_mask=None,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=None, temb=temb)
            all_hidden_states.append(hidden_states)
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)


# --------------------------------------------------------------------------- #
# AGRA change: per-(cross-)block context (the multi-layer foresight bridge).   #
# --------------------------------------------------------------------------- #
class BridgeDiT(DiT):
    """GR00T DiT whose cross-attention blocks each read a SEPARATE context.

    The standard ``DiT.forward`` feeds one shared backbone context to every cross
    block; here ``forward`` takes ``encoder_hidden_states_list`` with one entry per
    cross-attention block (= the AGRA multi-layer bridge: video-DiT layer ell_j fed to
    action cross-block j). Self-attention blocks (odd idx under interleaving) get no
    context, exactly as GR00T. Param layout is identical to ``DiT`` so GR00T's
    pretrained ``action_head.model.*`` weights load unchanged.
    """

    def num_cross_blocks(self) -> int:
        if self.config.interleave_self_attention:
            return sum(1 for idx in range(self.config.num_layers) if idx % 2 == 0)
        return self.config.num_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states_list: List[torch.Tensor],
        timestep: Optional[torch.LongTensor] = None,
    ):
        n_cross = self.num_cross_blocks()
        if len(encoder_hidden_states_list) != n_cross:
            raise ValueError(
                f"BridgeDiT expects {n_cross} contexts (one per cross block), got "
                f"{len(encoder_hidden_states_list)}"
            )
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        ci = 0
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(hidden_states, attention_mask=None,
                                      encoder_hidden_states=None, encoder_attention_mask=None, temb=temb)
            else:
                ctx = encoder_hidden_states_list[ci].contiguous()
                ci += 1
                hidden_states = block(hidden_states, attention_mask=None,
                                      encoder_hidden_states=ctx, encoder_attention_mask=None, temb=temb)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


# GR00T-N1.7-3B diffusion_model_cfg (from config.json), the cfg the pretrained
# action_head.model.* weights were trained with.
GR00T_N1D7_DIT_CFG = dict(
    num_attention_heads=32,
    attention_head_dim=48,        # inner_dim = 32*48 = 1536
    output_dim=1024,
    num_layers=32,                # interleaved -> 16 cross + 16 self
    dropout=0.2,
    attention_bias=True,
    activation_fn="gelu-approximate",
    norm_type="ada_norm",
    norm_elementwise_affine=False,
    final_dropout=True,
    positional_embeddings=None,
    interleave_self_attention=True,
)


class Gr00tActionDiTHead(nn.Module):
    """AGRA action head built on GR00T's REAL flow-matching action DiT.

    Mirrors the ``ForesightActionHead`` API so the AGRA coupling/runtime/inference are
    unchanged: ``forward(noisy_action [B,K,A], t_a [B], proprio0 [B,P], contexts: list)
    -> velocity [B,K,A]``. ``contexts`` has ``num_contexts`` entries (= #cross blocks),
    each [B, Sv, crossattn_dim] (the per-layer Cosmos foresight, projected to 2048).

    The DiT (``self.dit``, a ``BridgeDiT``) is initialised from GR00T-N1.7-3B's
    pretrained ``action_head.model.*`` weights. The state/action encoders + decoder are
    fresh, simple Linears for the LIBERO 7-D action / 8-D proprio (we keep FastWAM's own
    normalisation, NOT GR00T's 132-D padded multi-embodiment scheme).

    Flow-matching: FastWAM's WanContinuousFlowMatchScheduler feeds t in [0, T] where
    large t = high noise; GR00T's DiT was trained with t where large t = clean. We feed
    ``timestep = T - t_a`` to the (loaded) timestep encoder so its AdaLN conditioning
    sees a GR00T-consistent noise level (the fresh readout learns the velocity sign).
    """

    def __init__(
        self,
        action_dim: int,
        proprio_dim: int,
        crossattn_dim: int = 2048,
        action_horizon: int | None = None,
        flow_t_max: float = 1000.0,
        dit_cfg: dict | None = None,
        model_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        cfg = dict(GR00T_N1D7_DIT_CFG if dit_cfg is None else dit_cfg)
        cfg["cross_attention_dim"] = int(crossattn_dim)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.crossattn_dim = int(crossattn_dim)
        self.action_horizon = None if action_horizon is None else int(action_horizon)
        self.flow_t_max = float(flow_t_max)

        self.dit = BridgeDiT(**cfg)
        self.inner_dim = self.dit.inner_dim          # 1536
        self.output_dim = int(cfg["output_dim"])     # 1024

        # fresh simple encoders/decoder (LIBERO 7-D action / 8-D proprio)
        self.action_encoder = nn.Linear(self.action_dim, self.inner_dim)
        self.proprio_encoder = nn.Linear(self.proprio_dim, self.inner_dim)
        max_tokens = 1 + (self.action_horizon or 64)
        self.position_embedding = nn.Embedding(max_tokens, self.inner_dim)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        # zero-init readout: start at zero velocity so the flow-matching loss begins at
        # ~||target||^2 and trains up smoothly (DiT/AdaLN-zero convention).
        self.action_decoder = nn.Linear(self.output_dim, self.action_dim)
        nn.init.zeros_(self.action_decoder.weight)
        nn.init.zeros_(self.action_decoder.bias)

        self.to(dtype=model_dtype)

    @property
    def num_contexts(self) -> int:
        return self.dit.num_cross_blocks()

    def load_gr00t_dit(self, state_dict, strict: bool = True):
        """Load GR00T's pretrained action DiT weights (keys stripped to the BridgeDiT
        namespace, i.e. ``transformer_blocks.* / timestep_encoder.* / proj_out_*``)."""
        missing, unexpected = self.dit.load_state_dict(state_dict, strict=strict)
        return missing, unexpected

    def forward(self, noisy_action, t_a, proprio0, contexts):
        if len(contexts) != self.num_contexts:
            raise ValueError(
                f"Gr00tActionDiTHead expects {self.num_contexts} contexts, got {len(contexts)}"
            )
        B, K, _ = noisy_action.shape
        a_tok = self.action_encoder(noisy_action)                 # [B,K,1536]
        p_tok = self.proprio_encoder(proprio0).unsqueeze(1)       # [B,1,1536]
        x = torch.cat([p_tok, a_tok], dim=1)                      # [B,1+K,1536] (proprio prepended)
        pos = self.position_embedding.weight[: x.shape[1]].unsqueeze(0).to(x.dtype)
        x = x + pos

        if t_a.ndim == 0:
            t_a = t_a.reshape(1)
        if t_a.shape[0] != B:
            t_a = t_a.expand(B)
        # map FastWAM (large t = noise) -> GR00T (large t = clean)
        timestep = (self.flow_t_max - t_a.float()).clamp(0.0, self.flow_t_max)

        ctxs = [c.to(x.dtype) for c in contexts]
        out = self.dit(x, ctxs, timestep=timestep)                # [B,1+K,1024]
        vel = self.action_decoder(out)                            # [B,1+K,A]
        return vel[:, 1:, :]                                      # drop proprio token


def bridge_video_layers(num_video_blocks: int, num_contexts: int) -> list[int]:
    """AGRA multi-layer bridge (Eq.14, generalised to any #contexts): action cross-block
    j reads video layer round(j*(M-1)/(N-1)). M=num_video_blocks, N=num_contexts."""
    if num_contexts == 1:
        return [num_video_blocks - 1]
    return [round(j * (num_video_blocks - 1) / (num_contexts - 1)) for j in range(num_contexts)]
