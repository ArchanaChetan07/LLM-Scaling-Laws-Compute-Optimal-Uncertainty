"""Independent check of the Lesci et al. Pythia losses on our own Pile-val sample.

The late-training loss *increase* in small Pythia-deduped models is the single most
consequential data feature for the fits, so we re-measure it rather than trust it:
evaluate selected checkpoints on the first ``--chunks`` x 2048-token chunks of the
Pile validation split and compare the (min-step -> final-step) change to the
change in the Lesci data. Absolute values differ (different text sample), deltas
should agree.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request

import numpy as np
import torch
import zstandard
from transformers import AutoModelForCausalLM, AutoTokenizer

from scalingci import data

VAL_URL = "https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
CHECKS = {"70m": [21000, 143000], "160m": [55000, 143000], "410m": [68000, 143000]}


def pile_val_chunks(tok, n_chunks: int, ctx: int = 2048) -> torch.Tensor:
    ids: list[int] = []
    with urllib.request.urlopen(VAL_URL) as r:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(r), encoding="utf-8")
        for line in reader:
            ids.extend(tok(json.loads(line)["text"])["input_ids"] + [tok.eos_token_id])
            if len(ids) >= n_chunks * ctx:
                break
    return torch.tensor(ids[: n_chunks * ctx]).view(n_chunks, ctx)


@torch.no_grad()
def mean_nll(model, chunks, bs=4) -> np.ndarray:
    out = []
    for i in range(0, len(chunks), bs):
        x = chunks[i:i + bs].cuda()
        logits = model(x).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none")
        out.append(nll.view(x.size(0), -1).mean(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--checks", default=None, help='e.g. "410m:33000,68000;1.4b:83000,143000"')
    ap.add_argument("--out", default="pythia_loss_verification.json")
    args = ap.parse_args()
    checks = CHECKS if not args.checks else {
        k: [int(x) for x in v.split(",")] for k, v in (c.split(":") for c in args.checks.split(";"))}
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    chunks = pile_val_chunks(tok, args.chunks)
    lesci = data.load_pythia().set_index(["size", "step"]).loss
    rows = []
    for size, steps in checks.items():
        vals = {}
        for st in steps:
            m = AutoModelForCausalLM.from_pretrained(
                f"EleutherAI/pythia-{size}-deduped", revision=f"step{st}",
                torch_dtype=torch.float16).cuda().eval()
            vals[st] = mean_nll(m, chunks)
            del m; torch.cuda.empty_cache()
        a, b = steps
        diff = vals[b] - vals[a]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append({"size": size, "step_a": a, "step_b": b,
                     "ours_a": float(vals[a].mean()), "ours_b": float(vals[b].mean()),
                     "ours_delta": float(diff.mean()), "ours_delta_se": float(se),
                     "lesci_delta": float(lesci[(size, b)] - lesci[(size, a)])})
        print(rows[-1], flush=True)
    out = data.ROOT / "results" / "tables" / args.out
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
