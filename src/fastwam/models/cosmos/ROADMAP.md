# FastWAM × Cosmos-Predict2.5 backbone (MoT coupling)

Goal: swap FastWAM's video backbone to **Cosmos-Predict2.5-2B** (`MiniTrainDIT`),
keeping the **original FastWAM MoT** (mixture-of-transformers joint masked
attention) coupling — feasible because Cosmos uses standard softmax attention
(unlike SANA's linear attention, which forced cross-attention coupling).

Official repo (fresh clone): `/data/LFT-W02_data/junjie/VLA_WM/cosmos-predict2.5`
Weights: `…/weights/Cosmos-Predict2.5-2B`, `Cosmos-Tokenizer-CI8x8` (Wan2.1 VAE)

## Backbone facts (from architecture map)
- Video DiT `MiniTrainDIT` (`cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py`):
  hidden 2048, 16 heads, head_dim 128, blocks = self-attn(RoPE) + cross-attn(text) + MLP,
  **AdaLN** timestep modulation. Input `x_B_C_T_H_W`, patch 8×8 (spatial)/? temporal.
- Tokenizer: Wan2.1 VAE — 8× spatial, 4× temporal, 16 latent channels, deterministic.
  (Same VAE family the SANA branch already wired via `diffusion.model.wan.vae.WanVAE`.)
- Text encoder: Qwen2.5-VL-7B (3584-dim) → projected to 1024 for cross-attn.
- Diffusion: flow-matching (velocity), shift=3 — matches FastWAM's scheduler family.

## Original MoT to replicate (`models/wan22/mot.py`, 556 lines)
- Joint self-attention over concat(video_tokens, action_tokens) with a block mask
  (`_mixed_attention`, `_build_expert_attention_io`).
- Per-expert modulation split (`_split_modulation`) + per-expert post-block
  (cross-attn + FFN) (`_apply_expert_post_block`).
- Video-KV-cache prefill for inference (`prefill_video_cache`) — denoise action
  against a fixed, once-computed video. (Training path doesn't need this.)
- `ActionDiT` (`models/wan22/action_dit.py`) mirrors WAN's block; init by
  interpolating the video DiT backbone.

## Integration plan (incremental, each step tested)
1. **[understand]** Read `MiniTrainDIT` Block/Attention internals end-to-end:
   exact modulation split (shift/scale/gate per sub-block), RoPE application
   point, Q/K/V layout, attention backend. This dictates the MoT rewrite.
2. **cosmos/video_expert.py** — load + wrap `MiniTrainDIT` (2B), expose blocks +
   per-layer attention I/O hooks (mirror how SANA's video_expert exposed features,
   but here we need the *Q/K/V* path for joint attention, not just block output).
3. **cosmos/action_dit.py** — an ActionDiT whose block matches `MiniTrainDIT`
   (16 heads, 128 head_dim, AdaLN, RoPE), init by interpolating the Cosmos video
   DiT backbone (mirror `ActionDiT.from_pretrained` interp-init).
4. **cosmos/mot.py** — MoT joint attention for Cosmos blocks: at each layer, concat
   video+action Q/K/V, joint masked softmax (RoPE applied per-stream first), split,
   then per-expert AdaLN post-block (cross-attn to Qwen text + MLP).
5. **cosmos/fastwam_cosmos.py** — top model: WanVAE encode → MoT(video,action) →
   flow-matching velocity loss on both streams (mirror `wan22/fastwam.py`).
6. **runtime.create_fastwam_cosmos** — factory: load Cosmos 2B (shape-filtered),
   WanVAE, build ActionDiT (interp init), wrap in MoT + FastWAMCosmos.
7. **text embeds**: precompute Qwen2.5-VL embeddings (new cache; SANA used gemma).
8. **configs**: `configs/model/fastwam_cosmos.yaml`, data/task configs, sbatch.
9. **train on LIBERO** (mirror the SANA run; bf16 + FSDP, the validated recipe).

## Key risks
- MoT requires video & action DiTs to have **identical block count + attn dims**.
  The ActionDiT must exactly mirror `MiniTrainDIT`'s block (incl. AdaLN modulation
  layout + RoPE), or joint attention will be wrong. Step 1 is the gate.
- Cosmos attention may use TransformerEngine kernels; the MoT joint path likely
  needs a plain `scaled_dot_product_attention` fallback that accepts the block mask.
- Reusing pretrained Cosmos weights ⇒ cannot change its attention op; MoT must wrap
  *around* Cosmos's existing self-attention (concat K/V from the other stream),
  not replace it.

---

## Step 1 findings — concrete MoT design (DONE: read `minimal_v4_dit.py`)

### Cosmos `Block.forward` (minimal_v4_dit.py:1257-1382), tokens `[B,T,H,W,D]`:
3 independent AdaLN modulations (`adaln_modulation_{self_attn,cross_attn,mlp}`),
each `SiLU→Linear(D,3D)` → chunk → (shift, scale, gate). Sequence:
1. self-attn:  `n = LN_sa(x)*(1+scale_sa)+shift_sa`;  flatten `b t h w d→b (t h w) d`;
   `self_attn(n, None, rope)`;  `x = x + gate_sa * attn_out`
2. cross-attn: `n = LN_ca(x)*(1+scale_ca)+shift_ca`;  `cross_attn(n, crossattn_emb, rope)`;
   `x = x + gate_ca * out`   (crossattn_emb = Qwen text, K/V from text; no RoPE on it)
3. mlp:        `n = LN_mlp(x)*(1+scale_mlp)+shift_mlp`;  `mlp(n)`;  `x = x + gate_mlp*out`

### `Attention` (minimal_v4_dit.py:388-575) — MoT-friendly:
`compute_qkv(x, context, rope)` → q,k,v `[B,S,H,D]`; RMSNorm(q),RMSNorm(k) per-head;
RoPE applied to q,k **only when self-attn**. `compute_attention` calls `self.attn_op`.
Use `backend="torch"` → `torch_attention_op` = SDPA with `attn_mask` (mask support!).

### MoT injection = the SELF-ATTENTION step only. Per layer L:
```
# video & action each have a Cosmos-style Block (matching n_heads=16, head_dim=128)
nv = modulate(LN_sa_v(xv), ...);  na = modulate(LN_sa_a(xa), ...)
qv,kv,vv = video.self_attn.compute_qkv(flatten(nv), None, rope_video)   # [B,Sv,16,128]
qa,ka,va = action.self_attn.compute_qkv(flatten(na), None, rope_action) # [B,Sa,16,128]
k = cat([kv,ka], dim=S); v = cat([vv,va], dim=S)                        # joint K/V
ov = torch_attention_op(qv, k, v, mask_v)  → video.self_attn.output_proj # video sees v+a
oa = torch_attention_op(qa, k, v, mask_a)  → action.self_attn.output_proj
xv = xv + gate_sa_v * unflatten(ov);  xa = xa + gate_sa_a * unflatten(oa)
# then EACH stream does its own cross-attn(text) + mlp, exactly as Cosmos Block
```
Masks: default full joint (video↔action both ways) like original `_mixed_attention`;
keep configurable (e.g. action sees video but video doesn't see action) per FastWAM.

### Action expert spec
Mirror ONE Cosmos `Block` per layer (n_heads=16, head_dim=128 → inner=2048 to match
joint K/V; hidden can be 2048 to start, smaller later), with its own AdaLN + RoPE
over action-token positions. Init by interpolating the Cosmos video DiT blocks
(mirror `wan22/action_dit.from_pretrained` interp-init).

### Still to read before coding step 2-4
`MiniTrainDIT.forward` (top level, ~1805): RoPE generation (`VideoRopePosition3DEmb`),
`PatchEmbed`, the block loop, `FinalLayer`, timestep→emb. Need the exact RoPE tensor
+ how blocks are iterated, to drive both streams layer-by-layer through the MoT.

---

## Step 1 DONE — MiniTrainDIT.forward + exact 2B net config

### `MiniTrainDIT.forward` (minimal_v4_dit.py:1712-1798):
```
x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos = prepare_embedded_sequence(x_B_C_T_H_W, fps)
  # x_embedder=PatchEmbed -> [B,T,H,W,D];  rope from self.pos_embedder (VideoRope3D)
if use_crossattn_projection: crossattn_emb = crossattn_proj(crossattn_emb)  # Qwen 3584->1024
t_emb_B_T_D, adaln_lora_B_T_3D = t_embedder(timesteps_B_T); t_emb = t_embedding_norm(t_emb)
for block in self.blocks:                  # the loop we intercept for MoT
    x = block(x, t_emb, crossattn, rope_emb, adaln_lora, extra_pos)
x = final_layer(x, t_emb, adaln_lora);  return unpatchify(x)   # -> [B,C,T,H,W]
```
=> MoT driver: run patch+rope+t_emb for video; build action tokens+rope+t_emb;
   loop layers calling `mot_block_forward(vblk, ablk, vstream, astream)`; then each
   stream's `final_layer`. adaln_lora is per-stream and MUST be passed to `_adaln`.

### Exact Predict2.5-2B net config (`COSMOS_V1_2B_NET_MININET`, text2world/defaults/net.py:49):
```
MiniTrainDIT(in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
  model_channels=2048, num_blocks=28, num_heads=16,   # head_dim = 2048/16 = 128
  concat_padding_mask=True, pos_emb_cls="rope3d", pos_emb_learnable=True,
  use_adaln_lora=True, adaln_lora_dim=256, extra_per_block_abs_pos_emb=False,
  rope_*_extrapolation_ratio=1.0, max_img_h/w=240, max_frames=128)
```
CORRECTIONS vs first map: patch is **2×2 spatial / 1 temporal** (NOT 8×8);
**use_adaln_lora=True**; default `atten_backend="minimal_a2a"` — we will build the
net with **atten_backend="torch"** (same weights, but `torch_attention_op`=SDPA so the
MoT joint masked attention + cross-attn work without context-parallel). Open: confirm
the Predict2.5-2B ckpt matches V1_2B vs `COSMOS_V2_2B_NET` (differs: extra_pos_emb,
no sac) — resolve by inspecting `base/pre-trained` checkpoint keys once env is up.

### Tokenizer / weights layout (HPC3 `weights/Cosmos-Predict2.5-2B/`)
`tokenizer.pth` (Wan2.1 VAE) + `base/{pre-trained,post-trained,distilled}/` (the DiT).
Examples to crib the load path: `examples/{inference,action_conditioned,robot_multiview}.py`.

## HPC3 debug env (in progress)
uv-based (NOT conda): fresh official clone at
`/data/user/jhe724/workspace/cosmos-predict2.5-fw`, `uv sync --extra=cu128` builds
`.venv` (torch2.7-cu128 + transformer_engine). Run tests via that venv +
`PYTHONPATH=<fastwam>/src`. FastWAM cosmos branch pushed to fork (clone/rsync to HPC3).
