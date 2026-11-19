"""Bootstrap schemes for scaling-law fits.

The resampling unit is the whole point. Checkpoints of one training run share
initialisation, data order and LR schedule, so their residuals are strongly
autocorrelated; resampling them i.i.d. (what most published scaling-law CIs do)
treats ~1,000 correlated points as ~1,000 independent ones and under-covers.

Schemes (all implemented as integer multiplicity weights on the prepared data):

* ``iid``          - resample checkpoints (Besiroglu et al. 2024 style).
* ``cluster``      - resample model sizes, keep each size's checkpoints together.
* ``seed``         - DataDecide: keep sizes, resample seed-runs within each size
                     (pure run-to-run training noise).
* ``cluster_seed`` - DataDecide: two-stage, sizes then seeds within size.
* ``eval``         - Pythia: resample the 1,700 validation sequences (paired across
                     sizes) and recompute every checkpoint loss (measurement noise).
* ``cluster_eval`` - Pythia: sizes and validation text jointly.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import law
from .spec import Spec, prepare, pythia_table, _tables

# 1e19: inside every suite's IsoFLOP range; 1e21: inside Pythia/Chinchilla;
# 5.76e23: Chinchilla-70B compute (an extrapolation for both checkpoint suites).
C_REFS = {"C=1e19": 1e19, "C=1e21": 1e21, "C=5.76e23": law.CHINCHILLA_C}

_W: dict = {}


def _init(spec: Spec, theta0: np.ndarray, scheme: str):
    _W.update(spec=spec, theta0=theta0, scheme=scheme, base=prepare(spec))


def _multiplicity(rng, base, scheme):
    n = len(base["L"])
    if scheme == "iid":
        return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    sizes = np.unique(base["size"])
    m = np.zeros(n)
    if scheme in ("cluster", "cluster_eval"):
        for s in rng.choice(sizes, len(sizes)):
            m[base["size"] == s] += 1
        return m
    if scheme in ("seed", "cluster_seed"):
        chosen = rng.choice(sizes, len(sizes)) if scheme == "cluster_seed" else sizes
        for s in chosen:
            runs = np.unique(base["run"][base["size"] == s])
            for r in rng.choice(runs, len(runs)):
                m[base["run"] == r] += 1
        return m
    if scheme == "eval":
        return np.ones(n)
    raise ValueError(scheme)


def _one(seed: int):
    spec, scheme, base = _W["spec"], _W["scheme"], _W["base"]
    rng = np.random.default_rng(seed)
    if scheme in ("eval", "cluster_eval"):
        n_seq = _tables()["cube"]["loss"].shape[2]
        base = prepare(spec, pythia_table(rng.integers(0, n_seq, n_seq)))
    m = _multiplicity(rng, base, scheme)
    keep = m > 0
    w = (base["w"] * m)[keep]
    f = law.fit(base["N"][keep], base["D"][keep], base["L"][keep], spec.fitcfg(n_starts=4),
                starts=_W["theta0"], weights=w, h=base["h"][keep])
    return np.r_[f.theta, [law.tokens_per_param(f.theta, c) for c in C_REFS.values()],
                 len(np.unique(base["size"][keep])), f.at_bound]


def bootstrap(spec: Spec, scheme: str, B: int = 1000, seed: int = 0,
              theta0: np.ndarray | None = None, workers: int | None = None) -> np.ndarray:
    """Return (B, 5 + len(C_REFS) + 2): theta, ratios, n_distinct_sizes, at_bound."""
    if theta0 is None:
        b = prepare(spec)
        f0 = law.fit(b["N"], b["D"], b["L"], spec.fitcfg(), weights=b["w"], h=b["h"])
        theta0 = np.r_[f0.theta, f0.K]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    seeds = np.random.SeedSequence(seed).generate_state(B)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(spec, theta0, scheme)) as ex:
        return np.array(list(ex.map(_one, seeds, chunksize=max(1, B // (4 * workers)))))


def summarize(boot: np.ndarray, point: dict, level=0.95) -> dict:
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = {}
    names = list(law.NAMES) + list(C_REFS)
    for j, k in enumerate(names):
        v = boot[:, j]
        if k in ("a", "b", "e"):
            v = np.exp(v); k = k.upper()
        out[k] = {"point": point[k], "lo": float(np.percentile(v, lo)),
                  "hi": float(np.percentile(v, hi)), "median": float(np.median(v)),
                  # log-scale SD is the natural spread measure for a ratio
                  "sd_log": float(np.std(np.log(v))) if (v > 0).all() else float("nan")}
    out["frac_at_bound"] = float(boot[:, -1].mean())
    out["min_distinct_sizes"] = int(boot[:, -2].min())
    return out


def point_fit(spec: Spec, n_starts: int = 32) -> tuple[law.Fit, dict]:
    b = prepare(spec)
    f = law.fit(b["N"], b["D"], b["L"], spec.fitcfg(n_starts), weights=b["w"], h=b["h"])
    d = f.as_dict()
    d.update({k: float(law.tokens_per_param(f.theta, c)) for k, c in C_REFS.items()})
    d.update(n_points=len(b["L"]), n_sizes=int(len(np.unique(b["size"]))), C_max=b["C_max"])
    return f, d


def jackknife_sizes(spec: Spec) -> list[dict]:
    """Leave-one-size-out refits: how much does any single model size drive the answer?"""
    b = prepare(spec)
    f0 = law.fit(b["N"], b["D"], b["L"], spec.fitcfg(), weights=b["w"], h=b["h"])
    full = np.r_[f0.theta, f0.K]
    rows = []
    for s in np.unique(b["size"]):
        k = b["size"] != s
        f = law.fit(b["N"][k], b["D"][k], b["L"][k], spec.fitcfg(8), starts=full,
                    weights=b["w"][k], h=b["h"][k])
        rows.append({"dropped": s, **f.as_dict(),
                     **{k_: float(law.tokens_per_param(f.theta, c)) for k_, c in C_REFS.items()}})
    return rows


# ----------------------------------------------------------------------------- Approach 2
def _corrected_table(spec: Spec, df=None, theta0=None):
    """Selected checkpoints with the fitted schedule excess K h removed (if spec.schedule)."""
    from .spec import select
    sel = select(spec, df).copy()
    if spec.schedule:
        b = prepare(spec, df)
        f = law.fit(b["N"], b["D"], b["L"], spec.fitcfg(4 if theta0 is not None else 32),
                    starts=theta0, weights=b["w"], h=b["h"])
        sel["loss"] = sel["loss"].to_numpy() - f.K * b["h"]
    return sel


def isoflop_point(spec: Spec, df=None, theta0=None) -> tuple[dict, "pd.DataFrame"]:
    from . import isoflop
    sel = _corrected_table(spec, df, theta0)
    col = "N_total" if spec.param == "total" else "N_nonembed"
    sl = isoflop.isoflop_slices(sel, n_col=col, min_sizes=3)
    fr = isoflop.fit_frontier(sl)
    fr.update({k: isoflop.tokens_per_param(fr, c) for k, c in C_REFS.items()})
    return fr, sl


def _one_iso(seed: int):
    import pandas as pd
    spec, scheme = _W["spec"], _W["scheme"]
    rng = np.random.default_rng(seed)
    if scheme in ("eval", "cluster_eval"):
        n_seq = _tables()["cube"]["loss"].shape[2]
        src = pythia_table(rng.integers(0, n_seq, n_seq))
    else:
        src = _tables()["pythia"] if spec.suite == "pythia" else _tables()["datadecide"]
    if spec.suite == "datadecide":
        src = src[(src.corpus == spec.corpus) & (src["eval"] == spec.eval)]
    sizes = np.unique(src["size"])
    if scheme in ("cluster", "cluster_eval", "cluster_seed"):
        sizes = np.unique(rng.choice(sizes, len(sizes)))  # a duplicated size adds no information
    if scheme in ("seed", "cluster_seed"):
        by_run = {r: g for r, g in src[src["size"].isin(sizes)].groupby("run")}
        parts = []
        for s_ in sizes:
            runs = [r for r in by_run if by_run[r]["size"].iloc[0] == s_]
            parts += [by_run[r] for r in rng.choice(runs, len(runs))]
        frame = pd.concat(parts)
    else:
        frame = src[src["size"].isin(sizes)]
    fr, _ = isoflop_point(spec, frame, _W["theta0"])
    return np.array([fr["a"], fr["c"], fr["n_slices"]] + [fr[k] for k in C_REFS])


def isoflop_bootstrap(spec: Spec, scheme: str, B: int = 1000, seed: int = 0, workers=None):
    """Return (B, 3 + len(C_REFS)): a, c, n_slices, ratios."""
    theta0 = None
    if spec.schedule:
        b = prepare(spec)
        f0 = law.fit(b["N"], b["D"], b["L"], spec.fitcfg(), weights=b["w"], h=b["h"])
        theta0 = np.r_[f0.theta, f0.K]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    seeds = np.random.SeedSequence(seed).generate_state(B)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(spec, theta0, scheme)) as ex:
        return np.array(list(ex.map(_one_iso, seeds, chunksize=max(1, B // (4 * workers)))))
