"""MoT (mixture-of-transformers) joint attention for Cosmos-Predict2.5 blocks.

Mirrors the original FastWAM MoT (``models/wan22/mot.py``) but targets the
Cosmos ``MiniTrainDIT`` block layout instead of Wan's. A Cosmos ``Block`` does
``self-attn -> cross-attn(text) -> MLP``, each with its own AdaLN
``(shift, scale, gate)`` modulation (minimal_v4_dit.py:1257-1382).

MoT changes ONLY the self-attention step: the video and action streams compute
their own Q/K/V (each with its own RoPE), then attend over the **concatenated**
K/V of both streams (so video can see action and vice-versa). Cross-attention
(to Qwen text) and the MLP stay per-stream, exactly as in the Cosmos block.

This works *around* Cosmos' pretrained self-attention (we reuse its
``compute_qkv`` + ``output_proj``; we only swap the attention op for a masked
SDPA over the joint K/V), so pretrained weights still load.

The video and action blocks MUST share ``n_heads`` and ``head_dim`` (so their
K/V concatenate). The action expert is built to mirror the Cosmos block dims.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange


def _sdpa_joint(q_B_S_H_D, k_B_L_H_D, v_B_L_H_D, attn_mask=None):
    """Masked scaled-dot-product attention. Inputs [B, S, H, D] -> out [B, S, H*D].

    Mirrors ``torch_attention_op`` (minimal_v4_dit.py:267) but lets the query and
    key/value lengths differ (S vs L) so a stream's queries can attend the joint
    K/V. ``attn_mask`` broadcasts to [B, H, S, L] (bool: True = attend).
    """
    q = rearrange(q_B_S_H_D, "b s h d -> b h s d")
    k = rearrange(k_B_L_H_D, "b l h d -> b h l d")
    v = rearrange(v_B_L_H_D, "b l h d -> b h l d")
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # [B,H,S,D]
    return rearrange(out, "b h s d -> b s (h d)")


def _adaln(block, emb_B_T_D, adaln_lora_B_T_3D=None):
    """Compute the 3x (shift, scale, gate) AdaLN modulations of a Cosmos block.

    Mirrors minimal_v4_dit.py:1271-1301 (without the [B,T,1,1,D] reshape — we keep
    [B,T,D] and reshape at the broadcast site since our tokens may be flat).
    """
    def split(mod):
        out = mod(emb_B_T_D)
        if adaln_lora_B_T_3D is not None:
            out = out + adaln_lora_B_T_3D
        return out.chunk(3, dim=-1)

    sa = split(block.adaln_modulation_self_attn)
    ca = split(block.adaln_modulation_cross_attn)
    mlp = split(block.adaln_modulation_mlp)
    return {"self_attn": sa, "cross_attn": ca, "mlp": mlp}


def _mod_flat(norm_layer, x_B_S_D, scale_B_Tm_D, shift_B_Tm_D, T, HW):
    """LayerNorm + AdaLN modulate on flat tokens [B, S=T*HW, D].

    The modulation is per-frame ``[B, Tm, D]`` where ``Tm`` is 1 (one diffusion
    timestep for the whole clip) or T (per-frame). Reshape tokens to [B,T,HW,D]
    and broadcast the modulation as [B,Tm,1,D] over HW (and over T when Tm==1) —
    matching the Cosmos Block's [B,T,1,1,D] broadcast over [B,T,H,W,D].
    """
    B, S, D = x_B_S_D.shape
    x = norm_layer(x_B_S_D).view(B, T, HW, D)
    scale = scale_B_Tm_D.unsqueeze(2).type_as(x)
    shift = shift_B_Tm_D.unsqueeze(2).type_as(x)
    return (x * (1 + scale) + shift).reshape(B, S, D)


def _gate_flat(gate_B_Tm_D, out_B_S_D, T, HW):
    B, S, D = out_B_S_D.shape
    gate = gate_B_Tm_D.unsqueeze(2).type_as(out_B_S_D)
    return (out_B_S_D.view(B, T, HW, D) * gate).reshape(B, S, D)


class CosmosMoTStream:
    """Per-stream state threaded through the MoT block loop.

    Holds a stream's flat tokens [B, S, D], its RoPE, per-frame time embedding,
    text cross-attn context, AdaLN-LoRA, and the (T, HW) split used to broadcast
    the per-frame modulation across spatial tokens.
    """

    def __init__(self, tokens_B_S_D, rope, t_emb_B_T_D, crossattn_emb, T, HW, adaln_lora_B_T_3D=None):
        self.x = tokens_B_S_D
        self.rope = rope
        self.t_emb = t_emb_B_T_D
        self.crossattn_emb = crossattn_emb
        self.T = T
        self.HW = HW
        self.adaln_lora = adaln_lora_B_T_3D


def mot_block_forward(
    video_block,
    action_block,
    video: CosmosMoTStream,
    action: CosmosMoTStream,
    bidirectional: bool = False,
    video_mask: Optional[torch.Tensor] = None,
    action_mask: Optional[torch.Tensor] = None,
):
    """One MoT layer: per-stream self/joint attention then per-stream cross-attn + MLP.

    Replicates ``Block.forward`` (minimal_v4_dit.py:1257-1382) for BOTH the video
    and action Cosmos blocks. Updates ``video.x`` / ``action.x`` in place.

    Coupling direction (matches the original FastWAM MoT, models/wan22/mot.py:
    ``prefill_video_cache`` -> "Video prefill uses only video self-attention mask"):
      - bidirectional=False (DEFAULT): the VIDEO attends to VIDEO only (independent
        of the action, so the world model stays usable standalone); the ACTION
        attends to the JOINT [video ; action] K/V.
      - bidirectional=True: both streams attend the joint K/V (video also sees action).
    Masks (optional): broadcastable to [B, H, Sq, Sk], bool True=attend.
    """
    mv = _adaln(video_block, video.t_emb, video.adaln_lora)
    ma = _adaln(action_block, action.t_emb, action.adaln_lora)

    # ---- joint self-attention -------------------------------------------------
    nv = _mod_flat(video_block.layer_norm_self_attn, video.x, mv["self_attn"][1], mv["self_attn"][0], video.T, video.HW)
    na = _mod_flat(action_block.layer_norm_self_attn, action.x, ma["self_attn"][1], ma["self_attn"][0], action.T, action.HW)

    qv, kv, vv = video_block.self_attn.compute_qkv(nv, None, rope_emb=video.rope)   # [B,Sv,H,D]
    qa, ka, va = action_block.self_attn.compute_qkv(na, None, rope_emb=action.rope)  # [B,Sa,H,D]

    k = torch.cat([kv, ka], dim=1)
    v = torch.cat([vv, va], dim=1)

    # video: self-attn only (independent of action) unless bidirectional; action: joint
    if bidirectional:
        ov = _sdpa_joint(qv, k, v, attn_mask=video_mask)       # video sees video+action
    else:
        ov = _sdpa_joint(qv, kv, vv, attn_mask=video_mask)     # video sees video only
    oa = _sdpa_joint(qa, k, v, attn_mask=action_mask)          # action sees video+action

    ov = video_block.self_attn.output_dropout(video_block.self_attn.output_proj(ov))
    oa = action_block.self_attn.output_dropout(action_block.self_attn.output_proj(oa))

    video.x = video.x + _gate_flat(mv["self_attn"][2], ov, video.T, video.HW)
    action.x = action.x + _gate_flat(ma["self_attn"][2], oa, action.T, action.HW)

    # ---- per-stream cross-attention (to Qwen text) ----------------------------
    for blk, st, mod in ((video_block, video, mv), (action_block, action, ma)):
        n = _mod_flat(blk.layer_norm_cross_attn, st.x, mod["cross_attn"][1], mod["cross_attn"][0], st.T, st.HW)
        out = blk.cross_attn(n, st.crossattn_emb, rope_emb=None)  # text K/V, no RoPE
        st.x = st.x + _gate_flat(mod["cross_attn"][2], out, st.T, st.HW)

    # ---- per-stream MLP -------------------------------------------------------
    for blk, st, mod in ((video_block, video, mv), (action_block, action, ma)):
        n = _mod_flat(blk.layer_norm_mlp, st.x, mod["mlp"][1], mod["mlp"][0], st.T, st.HW)
        out = blk.mlp(n)
        st.x = st.x + _gate_flat(mod["mlp"][2], out, st.T, st.HW)

    return video, action
