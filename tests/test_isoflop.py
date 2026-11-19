import numpy as np
import pandas as pd

from scalingci import isoflop, law

TRUE = np.array([np.log(400.0), np.log(1800.0), np.log(1.8), 0.34, 0.28])


def test_isoflop_recovers_parametric_frontier():
    sizes = np.geomspace(2e7, 2e10, 16)
    rows = []
    for k, N in enumerate(sizes):
        D = np.geomspace(1e8, 2e12, 120)
        rows += [{"size": str(k), "step": j, "N_total": N, "tokens": d,
                  "loss": float(law.predict(TRUE, N, d))} for j, d in enumerate(D)]
    df = pd.DataFrame(rows)
    sl = isoflop.isoflop_slices(df, n_grid=30, window=1)
    fr = isoflop.fit_frontier(sl)
    assert fr["n_slices"] >= 5
    a_true = TRUE[4] / (TRUE[3] + TRUE[4])
    assert abs(fr["a"] - a_true) < 0.03
    C = np.exp(np.median(sl[sl.interior].logC))
    r_true = law.tokens_per_param(TRUE, C)
    assert abs(np.log(isoflop.tokens_per_param(fr, C) / r_true)) < 0.25
