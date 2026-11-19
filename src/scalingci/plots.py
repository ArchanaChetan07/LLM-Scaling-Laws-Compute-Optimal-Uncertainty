"""Static figures for the README (the interactive report lives in web/report.html).

Palette: reference data-viz palette. Model size is an ordered magnitude -> one-hue
blue ramp (ordinal, steps 250..700). Suites are identities -> fixed categorical
slots: Chinchilla = aqua, Pythia = blue, DataDecide = orange. Naive (i.i.d.)
intervals are drawn in muted gray so the honest interval carries the colour.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import law, spec as sp  # noqa: E402
from .data import ROOT  # noqa: E402

FIG = ROOT / "results" / "figures"
HEADLINE_C = {"chinchilla": "C=1e21", "pythia": "C=1e21", "datadecide": "C=1e19"}
TAB = ROOT / "results" / "tables"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SUITE = {"chinchilla": "#1baf7a", "pythia": "#2a78d6", "datadecide": "#eb6834"}
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
        "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _ramp(n):
    idx = np.linspace(0, len(RAMP) - 1, n).round().astype(int)
    return [RAMP[i] for i in idx]


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, suite in zip(axes, ["pythia", "datadecide"]):
        base = sp.BASELINES[suite]
        df = sp.pythia_table() if suite == "pythia" else sp._tables()["datadecide"]
        if suite == "datadecide":
            df = df[(df.corpus == base.corpus) & (df["eval"] == base.eval) & (df.seed == "default")]
        kept = sp.select(base.with_(sizes="all"))
        kept_idx = set(zip(kept["run"], kept["step"]))
        sizes = sp.PYTHIA_SUBSETS["all"] if suite == "pythia" else sp.DD_SUBSETS["all"]
        for s, c in zip(sizes, _ramp(len(sizes))):
            g = df[df["size"] == s].sort_values("tokens")
            g = g[g.step > 0]
            ax.plot(g.tokens, g.loss, color=c, lw=1.0, alpha=0.35)
            m = np.array([(r, st) in kept_idx for r, st in zip(g["run"], g["step"])])
            ax.plot(g.tokens[m], g.loss[m], color=c, lw=2, label=s)
        ax.set_xscale("log")
        ax.set_xlabel("training tokens D")
        ax.set_ylabel("held-out loss (nats/token)")
        ax.set_title({"pythia": "Pythia-deduped (Pile val, 1,700 seqs)",
                      "datadecide": "DataDecide / Dolma 1.7 (Dolma-CC val)"}[suite])
        ax.legend(fontsize=7, ncol=2, title="size (bold = used in baseline fit)", title_fontsize=7)
        if suite == "pythia":
            ax.set_ylim(2.4, 4.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_loss_curves.png", dpi=160)
    plt.close(fig)


def fig_schemes(metrics):
    """The headline figure: same data, same fit, different resampling unit."""
    rows = []
    a3 = metrics["approach3"]
    label = {"iid": "i.i.d. checkpoints", "eval": "eval text only", "cluster": "model sizes",
             "cluster_eval": "sizes + eval text", "seed": "seeds within size",
             "cluster_seed": "sizes + seeds"}
    for suite in ["chinchilla", "pythia", "datadecide"]:
        c = HEADLINE_C[suite]
        for sc, s in a3[suite]["boot"].items():
            rows.append((f"{suite} @{c[2:]}", "A3 · " + label[sc], a3[suite]["point"][c],
                         s[c]["lo"], s[c]["hi"], sc))
    for suite in ["pythia", "datadecide"]:
        c = HEADLINE_C[suite]
        a2 = metrics["approach2"][suite]
        for sc, s in a2["boot"].items():
            rows.append((f"{suite} @{c[2:]}", "A2 · " + label[sc], a2["point"][c],
                         s[c]["lo"], s[c]["hi"], sc))
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    for yi, (suite, lab, p, lo, hi, sc) in zip(y, rows):
        naive = sc in ("iid", "eval", "seed")
        col = MUTED if naive else SUITE[suite.split(" ")[0]]
        ax.plot([lo, hi], [yi, yi], color=col, lw=4 if not naive else 2.5, solid_capstyle="round")
        ax.plot([p], [yi], "o", ms=8, color=SURF, mec=col, mew=2)
        ax.text(1.4, yi, f"{lo:.0f}–{hi:.0f}" if hi < 1e4 else f"{lo:.0f}–{hi:.1e}",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}  ·  {r[1]}" for r in rows], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 2e3)
    ax.axvline(20, color=MUTED, lw=1, ls=":")
    ax.text(21, -0.9, "20 tok/param", fontsize=7, color=INK2)
    ax.set_ylim(-1.2, y.max() + 0.6)
    ax.set_xlabel("compute-optimal tokens per parameter at an in-range compute (95% bootstrap CI)")
    ax.set_title("What you resample decides how certain you look")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_bootstrap_schemes.png", dpi=160)
    plt.close(fig)


def fig_ratio_vs_C(metrics):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    Cs = np.logspace(18, 25, 80)
    scheme = {"chinchilla": "iid", "pythia": "cluster_eval", "datadecide": "cluster_seed"}
    for suite in ["chinchilla", "pythia", "datadecide"]:
        b = np.load(TAB / f"boot_a3_{suite}_{scheme[suite]}.npy")
        R = np.array([law.tokens_per_param(th, Cs) for th in b[:, :5]])
        lo, hi = np.nanpercentile(R, [2.5, 97.5], axis=0)
        p = metrics["approach3"][suite]["point"]
        th = [np.log(p["A"]), np.log(p["B"]), np.log(p["E"]), p["alpha"], p["beta"]]
        cmax = p["C_max"]
        ax.fill_between(Cs, lo, hi, color=SUITE[suite], alpha=0.15, lw=0)
        r = law.tokens_per_param(th, Cs)
        lab = f"{suite} (A3)" + ("  - E at bound: not identified" if p.get("at_bound") else "")
        ax.plot(Cs[Cs <= cmax], r[Cs <= cmax], color=SUITE[suite], label=lab)
        ax.plot(Cs[Cs >= cmax], r[Cs >= cmax], color=SUITE[suite], ls="--", lw=1.5)
    ax.axvline(law.CHINCHILLA_C, color=MUTED, lw=1, ls=":")
    ax.text(law.CHINCHILLA_C, 2e3, " Chinchilla-70B\n compute", fontsize=7, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.3, 1e4)
    ax.set_xlabel("training compute C (FLOPs)")
    ax.set_ylabel("compute-optimal tokens / parameter")
    ax.set_title("Solid = inside each suite's compute range; dashed = extrapolation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ratio_vs_compute.png", dpi=160)
    plt.close(fig)


def fig_oat():
    df = pd.read_csv(TAB / "oat.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharex=True)
    for ax, suite in zip(axes, ["pythia", "datadecide"]):
        d = df[df.suite == suite].reset_index(drop=True)
        c = HEADLINE_C[suite]
        y = np.arange(len(d))[::-1]
        base = {e: d[(d.estimator == e) & (d.axis == "baseline")][c].iloc[0] for e in ["A3", "A2"]}
        for yi, (_, r) in zip(y, d.iterrows()):
            col = SUITE[suite] if r.estimator == "A3" else "#6250d6"
            v = r[c]
            if np.isfinite(r.get(f"{c}_lo", np.nan)):
                ax.plot([max(r[f"{c}_lo"], 0.11), min(r[f"{c}_hi"], 9e3)], [yi, yi], color=col, lw=2.5, alpha=0.5)
            if np.isfinite(v):
                ax.plot([min(max(v, 0.11), 9e3)], [yi], "o", ms=7, color=col,
                        mfc=SURF if r.get("at_bound", False) is True else col)
            else:
                ax.text(0.12, yi, "no interior IsoFLOP minimum", fontsize=7, va="center", color=INK2)
        ax.axvline(base["A3"], color=SUITE[suite], lw=1, ls=":")
        ax.axvline(base["A2"], color="#6250d6", lw=1, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.estimator}  {r.axis}: {r.value}" for _, r in d.iterrows()], fontsize=7)
        ax.set_xscale("log"); ax.set_xlim(0.1, 1e4)
        ax.set_title(f"{suite}: one choice changed at a time")
        ax.set_xlabel(f"tokens/param at {c} (bar = 95% CI, sizes resampled)")
    fig.text(0.01, 0.005, "A3 = parametric fit (suite colour), A2 = IsoFLOP profiles (violet). "
             "Hollow marker = fit hit a parameter bound.", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_one_at_a_time.png", dpi=160)
    plt.close(fig)


def fig_factorial():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, suite in zip(axes, ["pythia", "datadecide"]):
        p = TAB / f"factorial_{suite}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        ok = df[df["converged"].astype(bool) & ~df["at_bound"].astype(bool)] if "converged" in df else df
        c = HEADLINE_C[suite]
        v = np.log10(ok[c].clip(1e-2, 1e6))
        ax.hist(v, bins=60, color=SUITE[suite], alpha=0.85, edgecolor=SURF, linewidth=0.5)
        q = np.percentile(ok[c], [5, 50, 95])
        for x in q:
            ax.axvline(np.log10(x), color=INK2, lw=1, ls=":")
        ax.set_title(f"{suite}: {len(ok)} interior fits of {len(df)} specs\n"
                     f"5/50/95th pct = {q[0]:.1f} / {q[1]:.1f} / {q[2]:.0f}")
        ax.set_xlabel(f"log10 tokens/param at {c}")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_factorial.png", dpi=160)
    plt.close(fig)


def fig_corpora():
    p = TAB / "corpora.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("A2_C=1e19")
    fig, ax = plt.subplots(figsize=(7, 7))
    y = np.arange(len(df))
    if "A2_C=1e19_lo" in df:
        ax.hlines(y, df["A2_C=1e19_lo"], df["A2_C=1e19_hi"], color="#6250d6", alpha=0.5, lw=2.5)
    ax.plot(df["A2_C=1e19"], y, "o", color="#6250d6", ms=6, label="A2 IsoFLOP (seed CI)")
    a3 = df["A3_C=1e19"].where(~df["A3_at_bound"].astype(bool))
    ax.plot(a3.clip(0.1, 1e4), y, "D", color=SUITE["datadecide"], ms=5, mfc=SURF, label="A3 parametric")
    ax.set_yticks(y); ax.set_yticklabels(df.corpus, fontsize=7)
    ax.set_xscale("log"); ax.set_xlim(1, 1e4)
    ax.axvline(20, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("tokens/param at C=1e19 (inside DataDecide's range)")
    ax.set_title("Same models, 25 pretraining corpora")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_corpora.png", dpi=160)
    plt.close(fig)


def all_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    fig_curves()
    fig_schemes(metrics)
    fig_ratio_vs_C(metrics)
    fig_oat()
    fig_factorial()
    fig_corpora()


if __name__ == "__main__":
    all_figures()
