"""Copies the rendered example pages (examples/NN-topic/NN-page/) into docs/examples/ before a build."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def on_pre_build(config):
    target = ROOT / "docs" / "examples"
    shutil.rmtree(target, ignore_errors=True)
    for page in sorted((ROOT / "examples").glob("[0-9][0-9]-*/[0-9][0-9]-*/README.md")):
        out = target / page.parent.relative_to(ROOT / "examples")
        out.mkdir(parents=True, exist_ok=True)
        for f in page.parent.iterdir():
            if f.suffix in {".md", ".png", ".py"}:
                shutil.copy2(f, out / f.name)
