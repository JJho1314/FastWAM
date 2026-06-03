#!/usr/bin/env bash
# One-shot: install SANA deps into the fastwam_sana env on HPC3 and point the
# editable `fastwam` install at the FastWAM_sana checkout. Run after the env clone.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam_sana

REPO=/data/user/jhe724/workspace/FastWAM_sana
OFF="-i https://pypi.org/simple"

echo "[1/5] fla-core + flash-linear-attention + xformers (torch 2.7.1+cu128)"
pip install --no-deps $OFF fla-core flash-linear-attention xformers==0.0.31

echo "[2/5] mmcv 1.7.2 (lite, pure-python) + addict/yapf"
pip install $OFF addict yapf
MMCV_WITH_OPS=0 pip install "mmcv==1.7.2" --no-build-isolation --no-deps $OFF

echo "[3/5] diffusers 0.38 + misc"
pip install -U "diffusers==0.38.0" --no-deps $OFF
pip install $OFF qwen_vl_utils ftfy termcolor easydict

echo "[4/5] point editable fastwam install at FastWAM_sana"
pip install -e "$REPO" --no-deps

echo "[5/5] verify SANA import"
python - <<'PY'
import sys; sys.path.insert(0, "/data/user/jhe724/workspace/Sana")
from diffusion.model.nets.sana_multi_scale_video import SanaMSVideo_2000M_P2_D20
import fastwam, os
from fastwam.models.sana import FastWAMSana
print("fastwam from:", os.path.dirname(fastwam.__file__))
print("SANA import OK")
PY
echo "DONE"
