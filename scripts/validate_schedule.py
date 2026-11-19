"""Out-of-sample test of the schedule term: fit on checkpoints with frac <= cut,
predict every run's end-of-run losses (frac >= 0.95). Writes results/tables/schedule_oos.csv."""
import numpy as np
import pandas as pd

from scalingci import law, spec as sp
from scalingci.spec import BASELINES


def run():
    rows = []
    for suite in ["datadecide", "pythia"]:
        for cut in [0.3, 0.5, 0.7]:
            for sched in [False, True]:
                s = BASELINES[suite].with_(schedule=sched)
                b, df = sp.prepare(s), sp.select(s)
                fr = df["frac"].to_numpy()
                tr, te = fr <= cut, fr >= 0.95
                f = law.fit(b["N"][tr], b["D"][tr], b["L"][tr], s.fitcfg(), h=b["h"][tr], weights=b["w"][tr])
                pred = f.predict(b["N"][te], b["D"][te], b["h"][te] if sched else None)
                err = np.log(b["L"][te]) - np.log(pred)
                rows.append({"suite": suite, "train_frac_max": cut, "schedule": sched, "K": f.K,
                             "final_mae_log": np.abs(err).mean(), "final_bias_log": err.mean(),
                             "alpha": f.alpha, "beta": f.beta,
                             "r_1e21": f.tokens_per_param(1e21), "r_chinchilla": f.tokens_per_param()})
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(sp.data.ROOT / "results" / "tables" / "schedule_oos.csv", index=False)
    return out


if __name__ == "__main__":
    run()
