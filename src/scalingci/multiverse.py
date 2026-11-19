"""Specification ("multiverse") analysis: how far does the compute-optimal answer
move when each defensible analyst choice is changed?

* one-at-a-time (OAT): vary a single axis from the baseline -> "this choice moves it by Y%".
* factorial: every combination of axis values -> distribution of answers and
  main effects (marginal median of log r per level), which captures interactions
  OAT misses.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .spec import AXES, BASELINES, Spec
from .uncertainty import C_REFS, point_fit

REF = "C=1e21"


def _apply(spec: Spec, axis: str, value) -> Spec:
    if axis == "loss":
        loss, delta = value
        return spec.with_(loss=loss, delta=delta if loss == "huber" else 0.0)
    return spec.with_(**{axis: value})


def axis_value(spec: Spec, axis: str):
    return (spec.loss, spec.delta) if axis == "loss" else getattr(spec, axis)


def fmt_value(axis, v):
    if axis == "loss":
        return f"huber delta={v[1]:g}" if v[0] == "huber" else v[0]
    return str(v)


def oat_specs(suite: str) -> list[tuple[str, object, Spec]]:
    base = BASELINES[suite]
    out = [("baseline", None, base)]
    for ax, vals in AXES[suite].items():
        for v in vals:
            if v != axis_value(base, ax):
                out.append((ax, v, _apply(base, ax, v)))
    return out


# DataDecide factorial is thinned (tail is a no-op there: no run degrades; 15,360 -> 1,536 specs).
FACTORIAL_AXES = {
    "datadecide": {**AXES["datadecide"], "tail": ["keep"],
                   "loss": [("huber", 1e-3), ("mse_log", 0)], "eval": ["dolma_common-crawl", "pile"]},
}


def factorial_specs(suite: str) -> list[Spec]:
    axes = FACTORIAL_AXES.get(suite, AXES[suite])
    specs = []
    for combo in itertools.product(*axes.values()):
        s = BASELINES[suite]
        for ax, v in zip(axes, combo):
            s = _apply(s, ax, v)
        specs.append(s)
    return specs


def _fit_row(spec: Spec, n_starts: int):
    try:
        _, d = point_fit(spec, n_starts)
    except Exception as e:  # a spec can leave too few points; record, don't crash
        d = {"error": repr(e)}
    return d


def run_points(specs: list[Spec], n_starts=16, workers=None) -> pd.DataFrame:
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_fit_row, specs, [n_starts] * len(specs),
                           chunksize=max(1, len(specs) // (8 * workers))))
    meta = pd.DataFrame([s.__dict__ for s in specs])
    return pd.concat([meta, pd.DataFrame(rows)], axis=1)


def main_effects(df: pd.DataFrame, suite: str, ref: str | None = None) -> pd.DataFrame:
    ref = ref or {"pythia": "C=1e21", "datadecide": "C=1e19"}[suite]
    ok = df[df["converged"].astype(bool) & ~df["at_bound"].astype(bool)]
    if "error" in df:
        ok = ok[ok["error"].isna()]
    lr = np.log(ok[ref])
    rows = []
    for ax in FACTORIAL_AXES.get(suite, AXES[suite]):
        key = ok.apply(lambda r: axis_value(Spec(**{k: r[k] for k in Spec.__dataclass_fields__}), ax), axis=1)
        med = lr.groupby(key.map(lambda v: fmt_value(ax, v))).median()
        rows.append({"axis": ax, "levels": {k: float(np.exp(v)) for k, v in med.items()},
                     "range_factor": float(np.exp(med.max() - med.min()))})
    return pd.DataFrame(rows).sort_values("range_factor", ascending=False)
