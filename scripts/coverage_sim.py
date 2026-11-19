"""Do the bootstrap CIs mean what they say? Coverage on synthetic checkpoint suites.

Ground truth: Besiroglu et al.'s Chinchilla law. A synthetic "suite" mirrors Pythia's
design (7 sizes = Pythia N, 40 checkpoints per run over frac 0.1-1 of a 300B-token run)
with a realistic error structure: a per-run offset (seed / hyperparameter luck) plus
AR(1) noise along the run (consecutive checkpoints share most of their state).
We record how often each scheme's 95% percentile CI for r(1e21) contains the truth.

Writes results/tables/coverage.csv.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scalingci import law
from scalingci.data import ROOT

TRUE = np.array([np.log(482.01), np.log(2085.43), np.log(1.8172), 0.3478, 0.3658])
N_SIZES = np.array([70426624, 162322944, 405334016, 1414647808, 2775208960, 6857302016, 11846072320.0])
C_REF = 1e21


def simulate(rng, sd_run, sd_ar, rho, n_ck=40):
    f = np.linspace(0.1, 1.0, n_ck)
    N, D, L, run = [], [], [], []
    for k, n in enumerate(N_SIZES):
        e = np.empty(n_ck)
        e[0] = rng.normal(0, sd_ar)
        for t in range(1, n_ck):
            e[t] = rho * e[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sd_ar)
        d = f * 300e9
        N.append(np.full(n_ck, n)); D.append(d); run.append(np.full(n_ck, k))
        L.append(law.predict(TRUE, np.full(n_ck, n), d) * np.exp(rng.normal(0, sd_run) + e))
    return [np.concatenate(x) for x in (N, D, L, run)]


def one(args):
    seed, B, sd_run, sd_ar, rho = args
    rng = np.random.default_rng(seed)
    N, D, L, run = simulate(rng, sd_run, sd_ar, rho)
    cfg = law.FitConfig(n_starts=16)
    f0 = law.fit(N, D, L, cfg)
    cfg_b = law.FitConfig(n_starts=0)  # warm start from the full-data optimum (as Besiroglu et al.)
    out = {"seed": seed, "point": f0.tokens_per_param(C_REF)}
    for scheme in ["iid", "cluster"]:
        rs = []
        for _ in range(B):
            if scheme == "iid":
                m = np.bincount(rng.integers(0, len(L), len(L)), minlength=len(L)).astype(float)
            else:
                m = np.zeros(len(L))
                for k in rng.integers(0, len(N_SIZES), len(N_SIZES)):
                    m[run == k] += 1
            keep = m > 0
            fb = law.fit(N[keep], D[keep], L[keep], cfg_b, starts=f0.theta, weights=m[keep])
            rs.append(fb.tokens_per_param(C_REF))
        lo, hi = np.percentile(rs, [2.5, 97.5])
        out[f"{scheme}_lo"], out[f"{scheme}_hi"] = lo, hi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--B", type=int, default=200)
    a = ap.parse_args()
    r_true = law.tokens_per_param(TRUE, C_REF)
    rows = []
    # (per-run offset sd, AR sd, AR rho): independent noise vs realistic correlated noise
    for sd_run, sd_ar, rho in [(0.0, 0.01, 0.0), (0.0, 0.01, 0.95), (0.01, 0.005, 0.95)]:
        jobs = [(s, a.B, sd_run, sd_ar, rho) for s in range(a.reps)]
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as ex:
            res = pd.DataFrame(list(ex.map(one, jobs)))
        for sc in ["iid", "cluster"]:
            cov = ((res[f"{sc}_lo"] <= r_true) & (r_true <= res[f"{sc}_hi"])).mean()
            width = np.median(np.log(res[f"{sc}_hi"] / res[f"{sc}_lo"]))
            rows.append({"sd_run": sd_run, "sd_ar": sd_ar, "rho": rho, "scheme": sc,
                         "coverage": cov, "median_log_width": width, "reps": a.reps, "B": a.B,
                         "r_true": r_true, "median_point": res["point"].median()})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "coverage.csv", index=False)


if __name__ == "__main__":
    main()
