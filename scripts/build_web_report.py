"""web/report.html = web/report.template.html with result placeholders filled (same
renderer as the README) and figures embedded as data URIs ({{img:name.png}})."""
import base64
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from render_readme import PAT, lookup  # noqa: E402

from scalingci.data import ROOT  # noqa: E402


def fmt(v, f):
    if f == "sig2":
        return f"{v:.2g}" if abs(v) < 100 else f"{v:,.0f}"
    if f == "pct":
        return f"{v:+.0f}%"
    return format(v, f) if not isinstance(v, str) else v


def main():
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    d = json.loads((ROOT / "results" / "derived.json").read_text())
    tpl = (ROOT / "web" / "report.template.html").read_text(encoding="utf-8")

    def img(mo):
        b = (ROOT / "results" / "figures" / mo.group(1)).read_bytes()
        return "data:image/png;base64," + base64.b64encode(b).decode()

    tpl = re.sub(r"\{\{img:([^}]+)\}\}", img, tpl)

    def sub(mo):
        path, f = mo.group(1), mo.group(2) or "g"
        root, rest = path.split(".", 1)
        return fmt(lookup(m if root == "m" else d, rest.replace(".", "|").replace("~", ".")), f)

    out = PAT.sub(sub, tpl)
    (ROOT / "web" / "report.html").write_text(out, encoding="utf-8")
    print("wrote web/report.html", f"{len(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
