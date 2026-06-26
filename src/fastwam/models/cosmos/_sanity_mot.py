"""Sanity: build the FastWAM-Cosmos MoT model (video DiT + copy-init action DiT)
and run one joint MoT forward on dummy data. Verifies the novel integration
(joint attention, action RoPE, finalize) before wiring data/training.

Usage: COSMOS_PY _sanity_mot.py <video_ckpt.pt>
"""
import sys
import torch


def main():
    ckpt = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from fastwam.models.cosmos.video_expert import CosmosVideoExpert
    from fastwam.models.cosmos.action_expert import CosmosActionExpert
    from fastwam.models.cosmos.fastwam_cosmos import FastWAMCosmos

    ve = CosmosVideoExpert.from_pretrained(ckpt_path=ckpt, atten_backend="torch",
                                           device="cpu", torch_dtype=torch.bfloat16)
    net = ve.net
    ae = CosmosActionExpert(action_dim=7, model_channels=net.model_channels,
                            num_blocks=len(net.blocks),
                            num_heads=int(net.blocks[0].self_attn.n_heads),
                            action_hidden_dim=1024,
                            action_ffn_dim=4096,
                            attention_head_dim=128)
    ae.copy_init_from_video(net)

    model = FastWAMCosmos(ve, ae, vae=None, vae_encode_fn=None, crossattn_dim=1024,
                          proprio_dim=None, device=dev, torch_dtype=torch.bfloat16)
    model = model.to(dev).to(torch.bfloat16).eval()
    n = sum(p.numel() for p in model.dit.parameters())
    print(f"FastWAMCosmos dit params: {n / 1e9:.3f}B")
    print(f"Action expert params: {sum(p.numel() for p in ae.parameters()) / 1e9:.3f}B")

    B, C, T, H, W, Ta = 1, 16, 4, 16, 16, 8
    noisy_v = torch.randn(B, C, T, H, W, device=dev, dtype=torch.bfloat16)
    noisy_a = torch.randn(B, Ta, 7, device=dev, dtype=torch.bfloat16)
    t_v = torch.full((B,), 500.0, device=dev, dtype=torch.bfloat16)
    t_a = torch.full((B,), 500.0, device=dev, dtype=torch.bfloat16)
    crossattn = torch.randn(B, 16, 1024, device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        pred_v, pred_a = model.couple_forward(noisy_v, t_v, noisy_a, t_a, crossattn)
    print("pred_v:", tuple(pred_v.shape), "pred_a:", tuple(pred_a.shape))
    assert pred_v.shape[1] == 16, pred_v.shape
    assert pred_a.shape[-1] == 7, pred_a.shape
    print("SANITY-MOT-PASSED")


if __name__ == "__main__":
    main()
