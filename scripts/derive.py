"""Summaries computed from results/tables/*.csv -> results/derived.json (input to the README)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scalingci.data import ROOT

TAB = ROOT / "results" / "tables"
HC = {"chinchilla": "C=1e21", "pythia": "C=1e21", "datadecide": "C=1e19"}


def oat_swings():
    df = pd.read_csv(TAB / "oat.csv")
    out = {}
    for (suite, est), d in df.groupby(["suite", "estimator"]):
        c = HC[suite]
        base = float(d[d.axis == "baseline"][c].iloc[0])
        axes = {}
        for ax, g in d[d.axis != "baseline"].groupby("axis"):
            g = g[np.isfinite(g[c])]
            n_fail = int((~np.isfinite(d[d.axis == ax][c])).sum())
            if len(g) == 0:
                axes[ax] = {"max_pct": None, "value": None, "n_failed": n_fail}
                continue
            lr = np.log(g[c] / base)
            i = int(np.argmax(np.abs(lr.to_numpy())))
            axes[ax] = {"max_pct": float((np.exp(lr.iloc[i]) - 1) * 100), "value": str(g["value"].iloc[i]),
                        "r": float(g[c].iloc[i]), "factor": float(np.exp(np.abs(lr.iloc[i]))),
                        "n_failed": n_fail}
        order = sorted(axes, key=lambda a: -(axes[a]["factor"] if axes[a].get("factor") else 0))
        out[f"{suite}_{est}"] = {"baseline": base, "C": c, "axes": axes, "ranked": order}
    return out


def factorial():
    out = {}
    for suite in ["pythia", "datadecide"]:
        p = TAB / f"factorial_{suite}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        ok = df[df.converged.astype(bool) & ~df.at_bound.astype(bool)]
        c = HC[suite]
        q = np.percentile(ok[c], [5, 25, 50, 75, 95])
        out[suite] = {"n_specs": len(df), "n_interior": len(ok), "frac_at_bound": float(df.at_bound.mean()),
                      "q05": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q95": q[4],
                      "frac_within_2x_of_20": float(((ok[c] > 10) & (ok[c] < 40)).mean())}
    me = TAB / "main_effects.csv"
    if me.exists():
        m = pd.read_csv(me)
        out["main_effects"] = {s: dict(zip(g.axis, g.range_factor.astype(float))) for s, g in m.groupby("suite")}
    return out


def misc():
    out = {}
    if (TAB / "coverage.csv").exists():
        cv = pd.read_csv(TAB / "coverage.csv")
        out["coverage"] = [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in r.items()} for r in cv.to_dict("records")]
    if (TAB / "pythia_loss_verification.json").exists():
        out["verification"] = json.loads((TAB / "pythia_loss_verification.json").read_text())
        p2 = TAB / "pythia_loss_verification_2.json"
        if p2.exists():
            out["verification"] += json.loads(p2.read_text())
    if (TAB / "schedule_oos.csv").exists():
        out["schedule_oos"] = pd.read_csv(TAB / "schedule_oos.csv").to_dict("records")
    if (TAB / "corpora.csv").exists():
        co = pd.read_csv(TAB / "corpora.csv")
        v = co["A2_C=1e19"].dropna()
        out["corpora"] = {"n": len(co), "A2_min": float(v.min()), "A2_max": float(v.max()),
                          "A2_median": float(v.median()), "A2_argmin": co.loc[v.idxmin(), "corpus"],
                          "A2_argmax": co.loc[v.idxmax(), "corpus"],
                          "A3_frac_at_bound": float(co["A3_at_bound"].astype(float).mean())}
    for suite in ["pythia", "datadecide"]:
        p = TAB / f"jackknife_{suite}.csv"
        if p.exists():
            j = pd.read_csv(p)
            c = HC[suite]
            out[f"jackknife_{suite}"] = {"min": float(j[c].min()), "max": float(j[c].max()),
                                         "argmax": str(j.loc[j[c].idxmax(), "dropped"]),
                                         "argmin": str(j.loc[j[c].idxmin(), "dropped"])}
    return out


if __name__ == "__main__":
    d = {"oat": oat_swings(), "factorial": factorial(), **misc()}
    (ROOT / "results" / "derived.json").write_text(json.dumps(d, indent=2, default=float))
    print(json.dumps(d["oat"], indent=1, default=float)[:3000])
