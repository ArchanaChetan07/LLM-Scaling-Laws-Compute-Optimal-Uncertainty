"""A fitting *specification*: every analyst choice that sits between raw checkpoint
losses and a tokens-per-parameter number, made explicit so it can be varied.

``prepare(spec)`` turns a Spec into the arrays a fit consumes. Nothing else in the
pipeline makes data-selection decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from . import data
from .law import FitConfig, lr_multiplier

# Final LR as a fraction of peak: Pythia (GPT-NeoX config, min_lr = 0.1 lr),
# DataDecide (OLMo ladder: cosine_with_warmup, alpha_f = 0.01). Both warm up for ~1%.
FINAL_LR = {"pythia": 0.1, "datadecide": 0.01}

# Size subsets. "small" thresholds are by nominal size label.
PYTHIA_SUBSETS = {
    "all": data.PYTHIA_SIZES,
    "drop_small2": data.PYTHIA_SIZES[2:],  # drop 70m, 160m (late-training degradation)
    "drop_largest": data.PYTHIA_SIZES[:-1],
    "ge_1.4b": data.PYTHIA_SIZES[3:],
}
DD_SIZES = list(data.DD_BATCH)
DD_SUBSETS = {
    "all": DD_SIZES,
    "drop_small6": DD_SIZES[6:],  # drop the 8-layer 4M-16M models (embedding-dominated)
    "drop_largest": DD_SIZES[:-1],
    "ge_60M": DD_SIZES[7:],
}


@dataclass(frozen=True)
class Spec:
    suite: str = "pythia"  # pythia | datadecide | chinchilla
    param: str = "total"  # total | nonembed
    sizes: str = "all"
    start: float = 0.1  # keep checkpoints with frac-of-run >= start
    tail: str = "truncate_min"  # keep | truncate_min | first_epoch (pythia)
    weight: str = "uniform"  # uniform | run | logd
    loss: str = "huber"
    delta: float = 1e-3
    tie: bool = False
    schedule: bool = False  # LR-schedule-aware law (see law.py)
    corpus: str = "Dolma1.7"  # datadecide only
    eval: str = "pile"  # datadecide only (Pythia is always Pile-val)

    def fitcfg(self, n_starts=32) -> FitConfig:
        return FitConfig(loss=self.loss, delta=self.delta, tie_exponents=self.tie,
                         schedule=self.schedule, n_starts=n_starts)

    def label(self) -> str:
        base = BASELINES[self.suite]
        diff = {k: v for k, v in asdict(self).items() if getattr(base, k) != v}
        return ", ".join(f"{k}={v}" for k, v in diff.items()) or "baseline"

    def with_(self, **kw) -> "Spec":
        return replace(self, **kw)


BASELINES = {
    "pythia": Spec("pythia"),
    # schedule=True: the suite's authors note intermediate checkpoints need the LR
    # schedule modelled; out-of-sample final-loss prediction confirms (validate.py).
    # eval: in-distribution held-out loss (as Hoffmann et al.); CC is the bulk of Dolma 1.7.
    "datadecide": Spec("datadecide", tail="keep", schedule=True, eval="dolma_common-crawl"),
    "chinchilla": Spec("chinchilla", start=0.0, tail="keep"),
}

# Axes of the multiverse: values each choice can take (first = baseline).
AXES = {
    "pythia": {
        "loss": [("huber", 1e-3), ("huber", 1e-5), ("huber", 1e-2), ("mse_log", 0), ("mse_lin", 0)],
        "param": ["total", "nonembed"],
        "sizes": list(PYTHIA_SUBSETS),
        "start": [0.1, 0.0, 0.25, 0.5],
        "tail": ["truncate_min", "keep", "first_epoch"],
        "weight": ["uniform", "run", "logd"],
        "tie": [False, True],
        # no "schedule" axis: every Pythia run ends at the same 300B tokens, so the
        # run fraction f = D / 300B is a function of D alone and K [h(f) - h(1)] is
        # exactly confounded with B D^-beta (not identifiable).
    },
    "datadecide": {
        "loss": [("huber", 1e-3), ("huber", 1e-5), ("huber", 1e-2), ("mse_log", 0), ("mse_lin", 0)],
        "param": ["total", "nonembed"],
        "sizes": list(DD_SUBSETS),
        "start": [0.1, 0.0, 0.25, 0.5],
        "tail": ["keep", "truncate_min"],
        "weight": ["uniform", "run", "logd"],
        "tie": [False, True],
        "schedule": [True, False],
        "eval": ["dolma_common-crawl", "pile", "c4_en", "wikitext_103"],
    },
}


# --------------------------------------------------------------------------- tables
_CACHE: dict = {}


def _tables():
    if not _CACHE:
        _CACHE["pythia"] = data.load_pythia()
        _CACHE["cube"] = data.load_pythia_cube()
        _CACHE["datadecide"] = data.load_datadecide()
        _CACHE["chinchilla"] = data.load_chinchilla()
    return _CACHE


def pythia_table(seq_idx: np.ndarray | None = None) -> pd.DataFrame:
    """Pythia table, optionally with losses recomputed on a resample of the val text."""
    t = _tables()
    df = t["pythia"]
    if seq_idx is None:
        return df
    cube = t["cube"]
    means = cube["loss"][:, :, seq_idx].mean(axis=2)  # (size, step)
    df = df.copy()
    pos = {s: i for i, s in enumerate(cube["sizes"])}
    step_pos = {s: j for j, s in enumerate(cube["steps"])}
    df["loss"] = means[df["size"].map(pos).to_numpy(), df["step"].map(step_pos).to_numpy()]
    return df


def _truncate_at_min(df: pd.DataFrame) -> pd.DataFrame:
    """Drop checkpoints after each run's (smoothed) loss minimum.

    The parametric law is monotone decreasing in D; post-minimum checkpoints
    (late-training degradation, epoch-2 effects) violate it by construction.
    A rolling median (window 5) keeps a single noisy checkpoint from moving the cut.
    """
    out = []
    for _, g in df.groupby("run", sort=False):
        g = g.sort_values("step")
        sm = g["loss"].rolling(5, center=True, min_periods=1).median()
        out.append(g[g["step"] <= g["step"].iloc[int(np.argmin(sm.to_numpy()))]])
    return pd.concat(out)


def select(spec: Spec, df: pd.DataFrame | None = None) -> pd.DataFrame:
    t = _tables()
    if spec.suite == "chinchilla":
        df = t["chinchilla"]
        return df[df.loss < np.sort(df.loss.to_numpy())[-5]]  # Besiroglu et al.: drop 5 worst
    if spec.suite == "pythia":
        df = pythia_table() if df is None else df
        df = df[df["size"].isin(PYTHIA_SUBSETS[spec.sizes]) & (df.step >= 1000)]
    else:
        df = t["datadecide"] if df is None else df
        df = df[(df.corpus == spec.corpus) & (df["eval"] == spec.eval)
                & df["size"].isin(DD_SUBSETS[spec.sizes])]
    # Tail rules are decided on the whole run, *then* the fit-start cut is applied.
    if spec.tail == "truncate_min":
        df = _truncate_at_min(df)
    elif spec.tail == "first_epoch":
        df = df[df.tokens <= data.PYTHIA_DEDUP_EPOCH_TOKENS]
    return df[df.frac >= spec.start]


def weights(spec: Spec, df: pd.DataFrame) -> np.ndarray:
    if spec.weight == "uniform":
        return np.ones(len(df))
    if spec.weight == "run":  # every run contributes equally regardless of checkpoint count
        return (1.0 / df.groupby("run")["run"].transform("size")).to_numpy()
    if spec.weight == "logd":  # equal weight per unit log-token within a run
        w = np.empty(len(df))
        lD = np.log(df["tokens"].to_numpy())
        for _, idx in df.groupby("run").indices.items():
            order = idx[np.argsort(lD[idx])]
            w[order] = np.gradient(lD[order]) if len(order) > 1 else 1.0
        return w
    raise ValueError(spec.weight)


def prepare(spec: Spec, df: pd.DataFrame | None = None) -> dict:
    df = select(spec, df)
    col = "N_total" if spec.param == "total" else "N_nonembed"
    if spec.suite in FINAL_LR:
        # Absolute LR above its end-of-run value, in units of 1e-3. Peak LR shrinks
        # ~N^(-1/3) in both suites, so a size-agnostic multiplier would force equal
        # schedule excess on a 4M and a 1B model.
        fl = FINAL_LR[spec.suite]
        peak = df["size"].map(data.PYTHIA_PEAK_LR if spec.suite == "pythia" else data.DD_PEAK_LR)
        h = (lr_multiplier(df["frac"].to_numpy(), 0.01, fl) - fl) * peak.to_numpy() / 1e-3
    else:
        h = np.zeros(len(df))
    return {
        "h": h,
        "N": df[col].to_numpy(float), "D": df["tokens"].to_numpy(float),
        "L": df["loss"].to_numpy(float), "w": weights(spec, df),
        "size": df["size"].to_numpy(), "run": df["run"].to_numpy(),
        "D_max": float(df["tokens"].max()),
        "C_max": float((6.0 * df[col].astype(float) * df["tokens"].astype(float)).max()),
    }
