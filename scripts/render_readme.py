"""README.md = README.template.md with {{path:fmt}} placeholders filled from
results/metrics.json (prefix m.) and results/derived.json (prefix d.).

    python scripts/render_readme.py          # write README.md
    python scripts/render_readme.py --check  # exit 1 if README.md is stale (CI / make check)
"""
from __future__ import annotations

import json
import re
import sys

from scalingci.data import ROOT

PAT = re.compile(r"\{\{\s*([^}:]+?)\s*(?::\s*([^}]+?))?\s*\}\}")


def lookup(src: dict, path: str):
    cur = src
    for part in path.split("|"):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def render() -> str:
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    d = json.loads((ROOT / "results" / "derived.json").read_text())
    tpl = (ROOT / "README.template.md").read_text(encoding="utf-8")

    def sub(mo):
        path, fmt = mo.group(1), mo.group(2) or "g"
        root, rest = path.split(".", 1)
        v = lookup(m if root == "m" else d, rest.replace(".", "|").replace("~", "."))
        if fmt == "sig2":
            return f"{v:.2g}" if abs(v) < 100 else f"{v:,.0f}"
        if fmt == "pct":
            return f"{v:+.0f}%"
        if fmt == "x":
            return f"{v:.1f}×"
        return format(v, fmt) if not isinstance(v, str) else v

    return PAT.sub(sub, tpl)


if __name__ == "__main__":
    out = render()
    p = ROOT / "README.md"
    if "--check" in sys.argv:
        ok = p.exists() and p.read_text(encoding="utf-8") == out
        print("README numbers match results/" if ok else "README.md is STALE vs results/ - rerun render")
        sys.exit(0 if ok else 1)
    p.write_text(out, encoding="utf-8")
    print("wrote", p)
