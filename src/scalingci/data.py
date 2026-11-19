"""Build and load tidy (N, D, L) tables for the three sources.

Raw inputs (see ``scripts/download_data.sh``):

* Pythia-deduped: per-sequence mean NLL for 7 sizes x 144 checkpoints from
  Lesci et al. (2024) ``pietrolesci/pythia-deduped-stats``. We keep only the
  Pile *validation* sequences (negative ``seq_idx``), restricted to the 1,700
  ids shared by every size so that all losses are measured on identical text
  (paired design). Training-set sequences are dropped: deduped Pythia runs ~1.5
  epochs, so "not yet seen" training sequences stop being held-out after epoch 1.
* DataDecide (AI2, OLMo codebase): checkpoint perplexities for 14 sizes x 25
  corpora x up to 3 seeds (``allenai/DataDecide-ppl-results``).
* Chinchilla: the 245 (N, C, L) points Besiroglu et al. (2024) extracted from
  Hoffmann et al. Fig. 4 (``epoch-research/analyzing-chinchilla``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .params import from_config_file

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

PYTHIA_SIZES = ["70m", "160m", "410m", "1.4b", "2.8b", "6.9b", "12b"]
PYTHIA_TOKENS_PER_STEP = 1024 * 2048  # batch 1024 seqs x 2048 ctx (Biderman et al. 2023)
PYTHIA_TOTAL_STEPS = 143_000
PYTHIA_WARMUP_STEPS = 1_430  # 1% warmup
PYTHIA_DEDUP_EPOCH_TOKENS = 207e9  # deduplicated Pile ~207B tokens -> epoch 2 starts ~98.7k steps

# DataDecide model table (model card): batch size in sequences of 2048 tokens.
DD_BATCH = {"4M": 32, "6M": 32, "8M": 32, "10M": 32, "14M": 32, "16M": 32, "20M": 64,
            "60M": 96, "90M": 160, "150M": 192, "300M": 320, "530M": 448, "750M": 576,
            "1B": 704}
DD_FINAL_STEPS = {"4M": 5725, "6M": 9182, "8M": 13039, "10M": 15117, "14M": 21953,
                  "16M": 24432, "20M": 14584, "60M": 29042, "90M": 29901, "150M": 38157,
                  "300M": 45787, "530M": 57786, "750M": 63589, "1B": 69369}
# Peak learning rates. Pythia: Biderman et al. (2023) Table / GPT-NeoX configs.
PYTHIA_PEAK_LR = {"70m": 1e-3, "160m": 6e-4, "410m": 3e-4, "1b": 3e-4, "1.4b": 2e-4,
                  "2.8b": 1.6e-4, "6.9b": 1.2e-4, "12b": 1.2e-4}
# DataDecide: OLMo ladder lr = 0.0047 (N / 108M)^(-1/3), N = model-card size (matches card LRs).
DD_PEAK_LR = {s: 0.0047 * (m / 108.0) ** (-1 / 3) for s, m in {
    "4M": 3.7, "6M": 6.0, "8M": 8.5, "10M": 9.9, "14M": 14.4, "16M": 16.0, "20M": 19.1,
    "60M": 57.1, "90M": 97.9, "150M": 151.9, "300M": 320.0, "530M": 530.1, "750M": 681.3,
    "1B": 1176.8}.items()}

DD_EVALS = ["pile", "c4_en", "dolma_common-crawl", "wikitext_103", "dolma_books",
            "dolma_wiki", "dolma_pes2o", "dolma_reddit", "dolma_stack", "m2d2_s2orc", "ice"]


# --------------------------------------------------------------------------- Pythia
def build_pythia() -> None:
    frames, mats = [], {}
    common: set[int] | None = None
    for m in PYTHIA_SIZES:
        d = pd.read_parquet(RAW / "lesci" / f"{m}.parquet", columns=["seq_idx", "step", "sup_seq"])
        v = d[d.seq_idx < 0]
        ids = set(v.seq_idx.unique())
        common = ids if common is None else common & ids
        mats[m] = v.pivot(index="step", columns="seq_idx", values="sup_seq")
    cols = sorted(common)
    steps = mats[PYTHIA_SIZES[0]].index.to_numpy().astype(np.int64)
    cube = np.stack([mats[m].loc[steps, cols].to_numpy(np.float32) for m in PYTHIA_SIZES])
    assert np.isfinite(cube).all(), "missing (size, step, seq) cells"
    np.savez_compressed(PROC / "pythia_val_cube.npz", loss=cube, steps=steps,
                        seq_ids=np.array(cols), sizes=np.array(PYTHIA_SIZES))
    for i, m in enumerate(PYTHIA_SIZES):
        pc = from_config_file(RAW / "configs" / f"pythia_{m}.json")
        frames.append(pd.DataFrame({
            "suite": "pythia", "size": m, "run": f"pythia-{m}", "seed": "default",
            "step": steps, "tokens": steps.astype(float) * PYTHIA_TOKENS_PER_STEP,
            "frac": steps / PYTHIA_TOTAL_STEPS,
            "N_total": pc.total, "N_nonembed": pc.nonembed,
            "loss": cube[i].mean(axis=1).astype(float),
            "loss_se": cube[i].std(axis=1, ddof=1) / np.sqrt(cube.shape[2]),
        }))
    pd.concat(frames).to_csv(PROC / "pythia.csv", index=False)


def load_pythia() -> pd.DataFrame:
    return pd.read_csv(PROC / "pythia.csv")


def load_pythia_cube() -> dict:
    z = np.load(PROC / "pythia_val_cube.npz")
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- DataDecide
def build_datadecide() -> None:
    d = pd.read_parquet(RAW / "datadecide" / "ppl.parquet")
    pcs = {s: from_config_file(RAW / "configs" / f"dd_{s}.json") for s in DD_BATCH}
    rows = []
    for ev in DD_EVALS:
        col = f"eval/{ev}-validation/Perplexity"
        sub = d[["data", "params", "seed", "step", col]].dropna()
        sub = sub.rename(columns={"params": "size", "data": "corpus"})
        sub["eval"] = ev
        sub["loss"] = np.log(sub.pop(col))
        rows.append(sub)
    t = pd.concat(rows, ignore_index=True)
    t["suite"] = "datadecide"
    t["step"] = t["step"].astype(int)
    t["tokens"] = t["step"].astype(float) * t["size"].map(DD_BATCH) * 2048.0
    t["frac"] = t["step"] / t["size"].map(DD_FINAL_STEPS)
    t["N_total"] = t["size"].map(lambda s: pcs[s].total)
    t["N_nonembed"] = t["size"].map(lambda s: pcs[s].nonembed)
    t["run"] = t["corpus"] + "|" + t["size"] + "|" + t["seed"]
    # Duplicate (run, step, eval) rows exist in the release; average them.
    t = (t.groupby(["suite", "corpus", "eval", "size", "seed", "run", "step"], as_index=False)
          .agg(tokens=("tokens", "first"), frac=("frac", "first"), N_total=("N_total", "first"),
               N_nonembed=("N_nonembed", "first"), loss=("loss", "mean"), n_dup=("loss", "size")))
    t.to_csv(PROC / "datadecide.csv", index=False)


def load_datadecide() -> pd.DataFrame:
    return pd.read_csv(PROC / "datadecide.csv")


# --------------------------------------------------------------------------- Chinchilla
def load_chinchilla() -> pd.DataFrame:
    e = pd.read_csv(RAW / "epoch" / "svg_extracted_data.csv")
    out = pd.DataFrame({"N_total": e["Model Size"], "C": e["Training FLOP"], "loss": e["loss"]})
    out["tokens"] = out["C"] / (6 * out["N_total"])
    out["N_nonembed"] = np.nan
    out["suite"], out["size"] = "chinchilla", out["N_total"].round(-6).astype("int64").astype(str)
    out["run"] = out["size"]
    return out
