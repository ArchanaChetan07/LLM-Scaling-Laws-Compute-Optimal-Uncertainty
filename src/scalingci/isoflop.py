"""Chinchilla Approach 2 (IsoFLOP profiles) built from checkpoint suites.

For each compute level C, every run whose checkpoints span C contributes the loss
interpolated (linearly in log C) at C = 6 N D. A parabola in log N through the
lowest-loss size and its neighbours gives N_opt(C). We keep slices whose minimum is
*interior* (not the smallest or largest available size) and fit
log N_opt = a log C + c, so r(C) = C / (6 N_opt(C)^2).

This estimator does not assume the parametric form of Approach 3, which makes it a
useful check on whether disagreements come from the functional form or from the data.
It inherits the LR-schedule bias of intermediate checkpoints (pass ``loss_col`` with a
schedule-corrected loss to remove the fitted K h term).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def isoflop_slices(df: pd.DataFrame, n_col="N_total", loss_col="loss", n_grid=40,
                   min_sizes=4, window=2) -> pd.DataFrame:
    # Average seeds at identical (size, step) first: the slice is a function of size.
    g = (df.assign(C=6.0 * df[n_col].astype(float) * df["tokens"].astype(float))
           .groupby(["size", "step"], as_index=False)
           .agg(N=(n_col, "first"), C=("C", "first"), L=(loss_col, "mean")))
    runs = {s: r.sort_values("C") for s, r in g.groupby("size")}
    lo = np.log(min(r.C.min() for r in runs.values()))
    hi = np.log(max(r.C.max() for r in runs.values()))
    rows = []
    for lc in np.linspace(lo, hi, n_grid):
        pts = [(np.log(r.N.iloc[0]), np.interp(lc, np.log(r.C), r.L))
               for r in runs.values() if np.log(r.C.min()) <= lc <= np.log(r.C.max())]
        if len(pts) < min_sizes:
            continue
        pts.sort()
        lN, L = map(np.array, zip(*pts))
        i = int(np.argmin(L))
        interior = 0 < i < len(L) - 1
        j0, j1 = max(0, i - window), min(len(L), i + window + 1)
        if j1 - j0 < 3:
            continue
        c2, c1, c0 = np.polyfit(lN[j0:j1], L[j0:j1], 2)
        lN_opt = -c1 / (2 * c2) if c2 > 0 else np.nan
        ok = interior and c2 > 0 and lN[j0] <= lN_opt <= lN[j1 - 1]
        rows.append({"logC": lc, "logN_opt": lN_opt, "interior": ok, "n_sizes": len(L),
                     "L_min": c0 - c1 * c1 / (4 * c2) if c2 > 0 else np.nan})
    return pd.DataFrame(rows)


def fit_frontier(slices: pd.DataFrame) -> dict:
    s = slices[slices.interior] if len(slices) else slices
    if len(s) < 3:
        return {"a": np.nan, "c": np.nan, "n_slices": len(s), "logC_min": np.nan, "logC_max": np.nan}
    a, c = np.polyfit(s.logC, s.logN_opt, 1)
    return {"a": float(a), "c": float(c), "n_slices": int(len(s)),
            "logC_min": float(s.logC.min()), "logC_max": float(s.logC.max())}


def tokens_per_param(front: dict, C: float) -> float:
    lN = front["a"] * np.log(C) + front["c"]
    return float(C / (6.0 * np.exp(2 * lN)))
