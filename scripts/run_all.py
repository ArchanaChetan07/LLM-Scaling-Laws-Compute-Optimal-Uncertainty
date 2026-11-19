"""Full analysis. Writes results/metrics.json + results/tables/*.csv.

    python scripts/run_all.py            # B=1000 headline, B=300 per OAT spec (~40 min, 24 cores)
    python scripts/run_all.py --quick    # B=60/30 smoke run
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from scalingci import multiverse as mv
from scalingci.data import ROOT
from scalingci.spec import AXES, BASELINES
from scalingci.uncertainty import (C_REFS, bootstrap, isoflop_bootstrap, isoflop_point,
                                   jackknife_sizes, point_fit, summarize)

TAB = ROOT / "results" / "tables"
SCHEMES = {"chinchilla": ["iid", "cluster"], "pythia": ["iid", "eval", "cluster", "cluster_eval"],
           "datadecide": ["iid", "seed", "cluster_seed"]}
# Headline compute per suite = inside the range its own data covers.
HEADLINE_C = {"chinchilla": "C=1e21", "pythia": "C=1e21", "datadecide": "C=1e19"}
HEADLINE_SCHEME = {"chinchilla": "iid", "pythia": "cluster_eval", "datadecide": "cluster_seed"}
ISO_SCHEMES = {"pythia": ["eval", "cluster", "cluster_eval"], "datadecide": ["seed", "cluster_seed"]}
ISO_START = 0.01  # IsoFLOP slices need overlapping C ranges: post-warmup checkpoints


def pct(v, lo=2.5, hi=97.5):
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if len(v) else [np.nan, np.nan]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def approach3(B):
    out = {}
    for suite in ["chinchilla", "pythia", "datadecide"]:
        spec = BASELINES[suite]
        f, point = point_fit(spec)
        res = {"spec": spec.__dict__, "point": point, "boot": {}}
        for sc in SCHEMES[suite]:
            b = bootstrap(spec, sc, B=B, theta0=np.r_[f.theta, f.K])
            res["boot"][sc] = summarize(b, point)
            np.save(TAB / f"boot_a3_{suite}_{sc}.npy", b)
            log(suite, sc, {k: [round(res["boot"][sc][k]["lo"], 2), round(res["boot"][sc][k]["hi"], 2)]
                            for k in C_REFS})
        if suite != "chinchilla":
            pd.DataFrame(jackknife_sizes(spec)).to_csv(TAB / f"jackknife_{suite}.csv", index=False)
        out[suite] = res
    # Besiroglu et al. (2024) reproduction: their scaled-Huber likelihood ~ Huber delta=1e-5
    f5, p5 = point_fit(BASELINES["chinchilla"].with_(delta=1e-5))
    out["chinchilla"]["besiroglu_repro_delta1e-5"] = p5
    return out


def approach2(B):
    out = {}
    for suite in ["pythia", "datadecide"]:
        spec = BASELINES[suite].with_(start=ISO_START)
        fr, sl = isoflop_point(spec)
        sl.to_csv(TAB / f"isoflop_slices_{suite}.csv", index=False)
        res = {"spec": spec.__dict__, "point": fr, "boot": {}}
        for sc in ISO_SCHEMES[suite]:
            b = isoflop_bootstrap(spec, sc, B=B)
            np.save(TAB / f"boot_a2_{suite}_{sc}.npy", b)
            valid = np.isfinite(b[:, 3])
            res["boot"][sc] = {"valid_frac": float(valid.mean()),
                               **{k: {"lo": pct(b[:, 3 + i])[0], "hi": pct(b[:, 3 + i])[1],
                                      "median": float(np.nanmedian(b[:, 3 + i]))}
                                  for i, k in enumerate(C_REFS)},
                               "a": {"lo": pct(b[:, 0])[0], "hi": pct(b[:, 0])[1]}}
            log("A2", suite, sc, res["boot"][sc][HEADLINE_C[suite]], "valid", round(valid.mean(), 2))
        out[suite] = res
    return out


def oat(B):
    rows = []
    for suite in ["pythia", "datadecide"]:
        sc = HEADLINE_SCHEME[suite]
        for axis, value, spec in mv.oat_specs(suite):
            f, p = point_fit(spec)
            row = {"suite": suite, "estimator": "A3", "axis": axis,
                   "value": "baseline" if axis == "baseline" else mv.fmt_value(axis, value),
                   **{k: p[k] for k in ["E", "alpha", "beta", "K", "at_bound", "n_points", "n_sizes", "C_max"]},
                   **{k: p[k] for k in C_REFS}}
            if B:
                b = bootstrap(spec, sc, B=B, theta0=np.r_[f.theta, f.K])
                for i, k in enumerate(C_REFS):
                    row[f"{k}_lo"], row[f"{k}_hi"] = pct(b[:, 5 + i])
                row["boot_frac_at_bound"] = float(b[:, -1].mean())
            rows.append(row)
            hc = HEADLINE_C[suite]
            log("OAT A3", suite, row["axis"], row["value"], round(row[hc], 2),
                [round(row.get(f"{hc}_lo", np.nan), 1), round(row.get(f"{hc}_hi", np.nan), 1)])
        # Approach 2 OAT (data-selection axes only; loss/tie/weight do not apply)
        base2 = BASELINES[suite].with_(start=ISO_START)
        for axis, value, spec in mv.oat_specs(suite):
            if axis not in ("baseline", "param", "sizes", "start", "tail", "schedule", "eval"):
                continue
            if axis == "start":
                spec = spec  # keep the A3 start values as-is (0.0 / 0.25 / 0.5 / 0.1)
            else:
                spec = mv._apply(base2, axis, value) if axis != "baseline" else base2
            fr, _ = isoflop_point(spec)
            row = {"suite": suite, "estimator": "A2", "axis": axis,
                   "value": "baseline" if axis == "baseline" else mv.fmt_value(axis, value),
                   "a_iso": fr["a"], "n_slices": fr["n_slices"], **{k: fr[k] for k in C_REFS}}
            if B and np.isfinite(fr["C=1e19"]):
                b = isoflop_bootstrap(spec, "cluster_eval" if suite == "pythia" else "cluster_seed", B=B)
                for i, k in enumerate(C_REFS):
                    row[f"{k}_lo"], row[f"{k}_hi"] = pct(b[:, 3 + i])
                row["boot_valid"] = float(np.isfinite(b[:, 3]).mean())
            rows.append(row)
            log("OAT A2", suite, row["axis"], row["value"], round(row[HEADLINE_C[suite]], 2))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "oat.csv", index=False)
    return df


def corpora(B):
    from scalingci.data import load_datadecide
    rows = []
    for c in sorted(load_datadecide()["corpus"].unique()):
        s3 = BASELINES["datadecide"].with_(corpus=c)
        try:
            _, p = point_fit(s3)
        except Exception as e:  # noqa: BLE001
            p = {k: np.nan for k in C_REFS}; p["error"] = repr(e)
        fr, _ = isoflop_point(s3.with_(start=ISO_START))
        row = {"corpus": c, **{f"A3_{k}": p.get(k) for k in C_REFS}, "A3_at_bound": p.get("at_bound"),
               **{f"A2_{k}": fr[k] for k in C_REFS}, "A2_a": fr["a"], "A2_n_slices": fr["n_slices"]}
        if B:
            b = isoflop_bootstrap(s3.with_(start=ISO_START), "seed", B=B)
            for i, k in enumerate(C_REFS):
                row[f"A2_{k}_lo"], row[f"A2_{k}_hi"] = pct(b[:, 3 + i])
        rows.append(row)
        log("corpus", c, row["A3_C=1e19"], row["A2_C=1e19"])
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "corpora.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    B, Bo = (60, 30) if a.quick else (1000, 200)
    TAB.mkdir(parents=True, exist_ok=True)
    mpath = ROOT / "results" / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
    steps = a.only or ["a3", "a2", "oat", "corpora"]
    if "a3" in steps:
        metrics["approach3"] = approach3(B)
    if "a2" in steps:
        metrics["approach2"] = approach2(B)
    if "oat" in steps:
        oat(Bo)
    if "corpora" in steps:
        corpora(Bo // 3)
    metrics["config"] = {"B": B, "B_oat": Bo, "C_refs": C_REFS, "quick": a.quick,
                         "axes": {k: {ax: [str(v) for v in vals] for ax, vals in ax_.items()}
                                  for k, ax_ in AXES.items()}}
    mpath.write_text(json.dumps(metrics, indent=2, default=float))
    log("done")


if __name__ == "__main__":
    main()
