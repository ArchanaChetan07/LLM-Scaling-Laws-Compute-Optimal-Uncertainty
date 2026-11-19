"""raw/ -> processed/ tidy tables. Idempotent."""
from scalingci import data

if __name__ == "__main__":
    data.PROC.mkdir(parents=True, exist_ok=True)
    data.build_pythia()
    data.build_datadecide()
    p, d, c = data.load_pythia(), data.load_datadecide(), data.load_chinchilla()
    print(f"pythia: {len(p)} rows, {p['size'].nunique()} sizes")
    print(f"datadecide: {len(d)} rows, {d['run'].nunique()} runs, "
          f"{d['corpus'].nunique()} corpora, dup rows averaged: {(d.n_dup > 1).sum()}")
    print(f"chinchilla: {len(c)} points")
    print(p.groupby("size")[["N_total", "N_nonembed"]].first())
