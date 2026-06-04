"""SANA-Video variant of FastWAM.

A parallel implementation of FastWAM that swaps the Wan2.2 MoT backbone for a
SANA-Video linear-attention DiT. Instead of MoT joint attention between video
and action tokens, the action expert is coupled to the video world model via
**cross-attention** (the action expert reads the video DiT's intermediate token
features). The data pipeline, flow-matching scheduler, and trainer are reused
unchanged from the Wan22 path.
"""

from .action_expert import SanaActionExpert
from .action_expert_dit import SanaActionExpertDiT
from .video_expert import SanaVideoExpert
from .fastwam_sana import FastWAMSana

__all__ = ["SanaActionExpert", "SanaActionExpertDiT", "SanaVideoExpert", "FastWAMSana"]
