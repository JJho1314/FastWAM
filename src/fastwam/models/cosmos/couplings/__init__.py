"""Pluggable video<->action coupling strategies for FastWAM-Cosmos.

Each coupling registers under a name and provides:
  - setup(model):   create any coupling-specific submodules on the FastWAMCosmos
  - forward(model, noisy_latents, t_v, noisy_action, t_a, crossattn_emb)
        -> (pred_v, pred_a)

A new coupling is added by dropping a file in this package and decorating its class
with ``@register_coupling("name")`` (and importing it below for side effects). No
edits to ``runtime.py`` / ``fastwam_cosmos.py`` are required to dispatch it.
"""
from __future__ import annotations

COUPLINGS: dict = {}


def register_coupling(name: str):
    """Class decorator: register a Coupling subclass under ``name``."""
    def deco(cls):
        COUPLINGS[str(name)] = cls
        return cls
    return deco


def get_coupling(name: str):
    if name not in COUPLINGS:
        raise KeyError(f"unknown coupling '{name}'; registered: {sorted(COUPLINGS)}")
    return COUPLINGS[name]


def build_coupling(name: str):
    """Instantiate the registered coupling for ``name``."""
    return get_coupling(name)()


# Register the built-in couplings (import for side effects).
from . import mot as _mot            # noqa: E402,F401
from . import cross_attn as _xattn   # noqa: E402,F401
from . import agra as _agra          # noqa: E402,F401
