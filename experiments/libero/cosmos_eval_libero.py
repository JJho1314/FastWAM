"""Standalone LIBERO success-rate eval for FastWAM-Cosmos.

Reuses the tested rollout loop (experiments/libero/eval_libero_single.run_single_task)
but builds the Cosmos model directly and loads the CONSOLIDATED accelerate checkpoint
(pytorch_model_fsdp.bin) -- the per-step weights .pt has FSDP-flattened root params.

Run (per GPU, CUDA_VISIBLE_DEVICES pins the device):
  python cosmos_eval_libero.py --suites libero_spatial --task_ids 0 --num_trials 2
"""
import argparse, functools, json, logging, os, sys, time
import numpy as np
import torch

torch.load = functools.partial(torch.load, weights_only=False)  # LIBERO init files + bf16 ckpt are trusted

REPO = "/data/users/junjie/FastWAM_cosmos"
COSMOS = "/data/users/junjie/cosmos-predict2.5-fw"
LIBERO = "/data/users/junjie/LIBERO"
for p in (REPO + "/src", COSMOS, LIBERO, REPO + "/experiments/libero"):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("FASTWAM_TEXT_CACHE_DIR", REPO + "/data/text_embeds_cache/libero_qwen")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from omegaconf import OmegaConf
from hydra.utils import instantiate

from fastwam.models.cosmos.runtime import create_fastwam_cosmos
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from libero.libero import benchmark
import eval_libero_single as E
from libero.libero import get_libero_path  # noqa

W = "/data/users/junjie/weights/Cosmos-Predict2.5-2B"
BASE_CKPT = W + "/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
POSTTRAIN_CKPT = W + "/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
VAE_PTH = W + "/tokenizer.pth"
DEFAULT_RUN_DIR = REPO + "/runs/train/2026-06-15_11-29-58"   # mot (Wan-faithful) global128 run
DATA_CFG = REPO + "/configs/data/libero_2cam_cosmos.yaml"


def build_model(device, dtype, args, ckpt_path):
    model = create_fastwam_cosmos(
        video_dit_pretrained_path=args.base_ckpt,
        vae={"vae_pth": VAE_PTH},
        action_dim=7, proprio_dim=8, crossattn_dim=1024,
        coupling=args.coupling, feature_layer=-1,
        action_hidden_dim=args.action_hidden_dim,
        action_ffn_dim=args.action_ffn_dim,
        action_attention_head_dim=args.action_attention_head_dim,
        model_dtype=dtype, device=device,
    )
    model.load_checkpoint(ckpt_path)  # DeepSpeed weights .pt {dit_cosmos,text_proj,proprio_encoder}
    logging.info("loaded %s ckpt: %s", args.coupling, ckpt_path)
    return model.to(device).eval()


def build_cfg(args):
    data = OmegaConf.load(DATA_CFG)
    cfg = OmegaConf.create({"data": data})
    cfg.seed = args.seed
    cfg.eval_num_inference_steps = args.num_inference_steps
    cfg.gpu_id = 0
    cfg.EVALUATION = OmegaConf.create({
        "task_suite_name": None, "task_id": None,
        "num_trials": args.num_trials,
        "trial_start": args.trial_start,
        "env_num": 1,
        "num_steps_wait": 30,
        "replan_steps": 10,
        "binarize_gripper": True,
        "use_action_ensembler": False,
        "visualize_future_video": False,
        "action_horizon": None,
        "num_inference_steps": args.num_inference_steps,
        "sigma_shift": None,
        "text_cfg_scale": 1.0,
        "negative_prompt": "",
        "rand_device": "cpu",
        "tiled": False,
        "output_dir": args.out_dir,
        "save_rollout_video": args.save_videos,
    })
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    ap.add_argument("--task_ids", default="all")  # "all" or comma list
    ap.add_argument("--pairs", default="")  # explicit "suite:tid,suite:tid,..." (overrides suites/task_ids)
    ap.add_argument("--num_trials", type=int, default=20)
    ap.add_argument("--trial_start", type=int, default=0)
    ap.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num_inference_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=REPO + "/evaluate_results/cosmos_crossattn")
    ap.add_argument("--tag", default="")  # unique per-process id (for multi-proc-per-GPU; avoids result-file collision)
    ap.add_argument("--coupling", default="mot")
    ap.add_argument("--run_dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--step", type=int, default=21700)
    ap.add_argument("--base_ckpt", default=POSTTRAIN_CKPT)
    ap.add_argument("--action_hidden_dim", type=int, default=1024)
    ap.add_argument("--action_ffn_dim", type=int, default=4096)
    ap.add_argument("--action_attention_head_dim", type=int, default=128)
    args = ap.parse_args()

    device = "cuda:0"
    dtype = torch.bfloat16
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = build_cfg(args)

    ckpt_path = args.ckpt or os.path.join(args.run_dir, "checkpoints", "weights", f"step_{args.step:06d}.pt")
    model = build_model(device, dtype, args, ckpt_path)
    dataset_stats = load_dataset_stats_from_json(os.path.join(args.run_dir, "dataset_stats.json"))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon = int(cfg.data.train.num_frames) - 1
    vh, vw = cfg.data.train.video_size
    input_h, input_w = int(vh), int(vw)

    bd = benchmark.get_benchmark_dict()
    # build the (suite, task_id) work list
    if args.pairs:
        pairs = []
        for tok in args.pairs.split(","):
            if not tok:
                continue
            s, t = tok.split(":")
            pairs.append((s, int(t)))
    else:
        pairs = []
        for suite in args.suites.split(","):
            if not suite:
                continue
            n = bd[suite]().n_tasks
            ids = range(n) if args.task_ids == "all" else [int(x) for x in args.task_ids.split(",")]
            pairs += [(suite, tid) for tid in ids]

    _tag = args.tag or os.environ.get("CUDA_VISIBLE_DEVICES", "x")
    partial_name = "results_partial_%s.json" % _tag
    grand = {"by_task": [], "by_suite": {}, "overall": None}
    tot_succ = tot_eps = 0
    t_start = time.time()
    suite_cache = {}

    for suite, tid in pairs:
        ts = suite_cache.setdefault(suite, bd[suite]())
        cfg.EVALUATION.task_suite_name = suite
        cfg.EVALUATION.task_id = tid
        task = ts.get_task(tid)
        inits = ts.get_task_init_states(tid)
        trial_stop = int(args.trial_start) + int(args.num_trials)
        while len(inits) < trial_stop:
            inits.extend(inits[: trial_stop - len(inits)])
        inits = inits[int(args.trial_start):trial_stop]
        vdir = os.path.join(args.out_dir, suite, "videos")
        os.makedirs(vdir, exist_ok=True)
        t0 = time.time()
        res = E.run_single_task(
            task=task, initial_states=inits, model=model, processor=processor, cfg=cfg,
            video_dir=vdir, predicted_video_dir=vdir,
            action_horizon=action_horizon, input_w=input_w, input_h=input_h,
            model_device=device,
        )
        sc = int(res["successes"]); ep = int(args.num_trials)
        tot_succ += sc; tot_eps += ep
        rec = {"suite": suite, "task_id": tid, "successes": sc, "trials": ep,
               "trial_start": int(args.trial_start), "rate": sc / ep,
               "desc": res.get("task_description"), "sec": round(time.time() - t0, 1)}
        grand["by_task"].append(rec)
        bs = grand["by_suite"].setdefault(suite, {"successes": 0, "trials": 0})
        bs["successes"] += sc; bs["trials"] += ep; bs["rate"] = bs["successes"] / max(bs["trials"], 1)
        logging.info("[%s t%d] %d/%d = %.0f%% (%.0fs)  %s",
                     suite, tid, sc, ep, 100 * sc / ep, rec["sec"], rec["desc"])
        with open(os.path.join(args.out_dir, partial_name), "w") as f:
            json.dump(grand, f, indent=2)

    grand["overall"] = {"successes": tot_succ, "trials": tot_eps,
                        "rate": tot_succ / max(tot_eps, 1), "minutes": round((time.time() - t_start) / 60, 1)}
    with open(os.path.join(args.out_dir, "results_%s.json" % _tag), "w") as f:
        json.dump(grand, f, indent=2)
    print("OVERALL %d/%d = %.2f%%  (%.1f min)" % (
        tot_succ, tot_eps, 100 * tot_succ / max(tot_eps, 1), grand["overall"]["minutes"]))
    print("EVAL-DONE")


if __name__ == "__main__":
    main()
