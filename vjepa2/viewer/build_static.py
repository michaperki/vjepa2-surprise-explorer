#!/usr/bin/env python3
"""Build a static, GitHub-Pages-ready copy of the viewer into docs/.

GitHub Pages serves no Python, so this bakes the serve.py API into plain files:

    docs/
      index.html, app.js, style.css     # relative asset paths + VIEWER_STATIC flag
      .nojekyll                          # serve every path verbatim (no Jekyll)
      data/
        runs.json                        # <- GET /api/runs
        <run>/manifest.json              # <- GET /api/manifest?run=<run> (compact)
        <run>/inventory.json             # <- GET /api/inventory?run=<run>
        <run>/assets/...                 # <- /assets/<run>/...  (the MP4s)

The manifest is re-emitted compact (no indentation): ~128 MB pretty -> ~28 MB,
which GitHub Pages then gzips to ~4 MB over the wire. Curation save is a no-op on
the static site (app.js hides the button when VIEWER_STATIC is set).

Pages publishes from main /docs, which is the *repo root* docs/, so point --out
there (one level up from this package):

    PYTHONPATH=. python3 -m viewer.build_static \
        --run runs/violation_review_all --out ../docs
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"


def build_index(src_html: str) -> str:
    """Rewrite the dev shell for a project Pages site: absolute /asset paths break
    under https://user.github.io/<repo>/, so make them relative, and inject the
    static-mode flag the viewer reads at startup."""
    html = src_html.replace('href="/style.css"', 'href="style.css"')
    html = html.replace('src="/app.js"', 'src="app.js"')
    flag = '<script>window.VIEWER_STATIC = "data";</script>\n    '
    return html.replace('<script src="app.js"></script>', flag + '<script src="app.js"></script>')


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True,
                    help="run directory with manifest.json (repeatable)")
    ap.add_argument("--out", default="../docs", help="output dir (Pages: repo-root docs/)")
    args = ap.parse_args()

    out = Path(args.out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)

    (out / "index.html").write_text(build_index((STATIC / "index.html").read_text()))
    shutil.copy(STATIC / "app.js", out / "app.js")
    shutil.copy(STATIC / "style.css", out / "style.css")
    (out / ".nojekyll").write_text("")

    runs = []
    for raw in args.run:
        run_dir = Path(raw)
        if not (run_dir / "manifest.json").exists():
            raise SystemExit(f"no manifest.json in {run_dir}")
        rid = run_dir.name
        runs.append({"id": rid})
        dst = data / rid
        dst.mkdir(parents=True, exist_ok=True)

        manifest = json.loads((run_dir / "manifest.json").read_text())
        (dst / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))

        inv = run_dir / "inventory.json"
        if inv.exists():
            inventory = json.loads(inv.read_text())
            (dst / "inventory.json").write_text(json.dumps(inventory, separators=(",", ":")))

        src_assets = run_dir / "assets"
        if src_assets.exists():
            if (dst / "assets").exists():
                shutil.rmtree(dst / "assets")
            shutil.copytree(src_assets, dst / "assets")

        mb = (dst / "manifest.json").stat().st_size / 1e6
        amb = dir_size_mb(dst / "assets") if (dst / "assets").exists() else 0.0
        print(f"  {rid}: manifest {mb:.1f} MB, assets {amb:.1f} MB")

    (data / "runs.json").write_text(json.dumps({"runs": runs}, separators=(",", ":")))
    print(f"Built {out}/ ({dir_size_mb(out):.1f} MB total).")
    print("Enable: GitHub -> Settings -> Pages -> Source = main / docs.")


if __name__ == "__main__":
    main()
