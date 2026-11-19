"""Exact parameter counts from HF config.json files.

Two conventions matter for scaling-law fits (Pearce & Song 2024 show the choice
explains much of the Kaplan-vs-Chinchilla gap):

* ``total``     - every trainable weight, incl. input embedding and (untied) LM head.
                  This is what Hoffmann et al. (2022) count.
* ``nonembed``  - transformer blocks + final norm only (Kaplan et al. 2020).

Both suites used here have untied embeddings (Pythia: GPT-NeoX, DataDecide: OLMo).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParamCount:
    total: int
    nonembed: int
    embed: int  # input embedding + LM head


def gpt_neox(cfg: dict) -> ParamCount:
    """Pythia. Per layer: qkv 3d^2+3d, out d^2+d, fc1 4d^2+4d, fc2 4d^2+d, 2 LN 4d."""
    d, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    ff = cfg.get("intermediate_size", 4 * d)
    per_layer = (3 * d * d + 3 * d) + (d * d + d) + (d * ff + ff) + (ff * d + d) + 4 * d
    nonembed = L * per_layer + 2 * d  # final LayerNorm (weight+bias)
    embed = 2 * V * d if not cfg.get("tie_word_embeddings", False) else V * d
    return ParamCount(nonembed + embed, nonembed, embed)


def olmo(cfg: dict) -> ParamCount:
    """DataDecide (OLMo-1 codebase): no biases, RMSNorm weight-only, SwiGLU.

    ff_proj: d -> mlp_ratio*d (split in two halves for SwiGLU), ff_out: (mlp_ratio*d/2) -> d.
    """
    d, L, V = cfg["d_model"], cfg["n_layers"], cfg["embedding_size"]
    assert not cfg["include_bias"] and cfg["activation_type"] == "swiglu"
    hidden = cfg["mlp_hidden_size"] or cfg["mlp_ratio"] * d
    attn = 4 * d * d
    mlp = d * hidden + (hidden // 2) * d
    norms = 2 * d if cfg["layer_norm_with_affine"] else 0
    nonembed = L * (attn + mlp + norms) + (d if cfg["layer_norm_with_affine"] else 0)
    embed = V * d * (1 if cfg["weight_tying"] else 2)
    return ParamCount(nonembed + embed, nonembed, embed)


def from_config_file(path: str | Path) -> ParamCount:
    cfg = json.loads(Path(path).read_text())
    if "d_model" in cfg:
        return olmo(cfg)
    return gpt_neox(cfg)
