"""Full-factorial multiverse of Approach-3 point fits. Writes results/tables/factorial_<suite>.csv
and results/tables/main_effects.csv."""
import time

import pandas as pd

from scalingci import multiverse as mv
from scalingci.data import ROOT

TAB = ROOT / "results" / "tables"

if __name__ == "__main__":
    eff = []
    import sys
    for suite in [a for a in sys.argv[1:] if not a.startswith("--")] or ["pythia", "datadecide"]:
        p = TAB / f"factorial_{suite}.csv"
        t = time.time()
        if p.exists() and "--reuse" in sys.argv:
            df = pd.read_csv(p)
        else:
            df = mv.run_points(mv.factorial_specs(suite), n_starts=12)
            df.to_csv(p, index=False)
        e = mv.main_effects(df, suite)
        e.insert(0, "suite", suite)
        eff.append(e)
        ok = df[df.converged.astype(bool) & ~df.at_bound.astype(bool)]
        print(suite, len(df), "specs", f"{time.time()-t:.0f}s", "valid", len(ok), flush=True)
        print(e[["axis", "range_factor"]].to_string(index=False), flush=True)
    out = TAB / "main_effects.csv"
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    allm = pd.concat([prev[~prev.suite.isin([e.suite.iloc[0] for e in eff])] if len(prev) else prev] + eff)
    allm.to_csv(out, index=False)
