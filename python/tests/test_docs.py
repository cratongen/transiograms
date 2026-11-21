import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted(
    p.relative_to(ROOT).as_posix()
    for p in [
        ROOT / "README.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "docs" / "guide").glob("*.md"),
        *(ROOT / "docs" / "api").glob("*.md"),
        *(ROOT / "examples").glob("[0-9][0-9]-*/[0-9][0-9]-*/README.md"),
    ]
    if "```python" in p.read_text(encoding="utf-8")
)
EXAMPLE_PAGES = [p for p in PAGES if p.startswith("examples/")]
BLOCK = re.compile(r"```(python|text)\n(.*?)\n```", re.DOTALL)
RUNNER = """
import contextlib, io, json, sys, traceback
import matplotlib
matplotlib.use("Agg")
blocks = json.load(sys.stdin)
ns = {"__name__": "__main__"}
results = []
for code in blocks:
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(code, "<doc>", "exec"), ns)
    except BaseException:
        results.append({"stdout": out.getvalue(), "error": traceback.format_exc(limit=3)})
        break
    results.append({"stdout": out.getvalue(), "error": None})
print(json.dumps(results))
"""


def documented(page: str) -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    for kind, body in BLOCK.findall((ROOT / page).read_text(encoding="utf-8")):
        if kind == "python":
            pairs.append((body, None))
        elif pairs and pairs[-1][1] is None:
            pairs[-1] = (pairs[-1][0], body)
    return pairs


def normalise(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


@pytest.mark.parametrize("page", PAGES)
def test_documented_code_runs_and_prints_what_the_page_says(page, tmp_path):
    check_page(page, tmp_path)


@pytest.mark.parametrize("page", EXAMPLE_PAGES)
def test_full_script_download_runs_on_its_own(page, tmp_path):
    script = next((ROOT / page).parent.glob("example_*.py"))
    alone = tmp_path / script.name
    alone.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
    python = os.environ.get("TG_DOCS_PYTHON", sys.executable)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"MPLBACKEND": "Agg"}
    run = subprocess.run(
        [python, "-I", alone.name],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=1800,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    figures = set(re.findall(r"!\[[^]]*\]\(([^)]+\.png)\)", (ROOT / page).read_text(encoding="utf-8")))
    assert figures <= {f.name for f in tmp_path.glob("*.png")}


def check_page(page: str, tmp_path: Path):
    pairs = documented(page)
    python = os.environ.get("TG_DOCS_PYTHON", sys.executable)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"MPLBACKEND": "Agg"}
    run = subprocess.run(
        [python, "-I", "-c", RUNNER],
        check=False,
        input=json.dumps([code for code, _ in pairs]),
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=1800,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    results = json.loads(run.stdout.splitlines()[-1])
    for (code, expected), got in zip(pairs, results, strict=False):
        assert got["error"] is None, f"{page}: block\n{code}\nraised\n{got['error']}"
        if expected is not None:
            assert normalise(got["stdout"]) == normalise(expected), f"{page}: block\n{code}"


def test_api_pages_render_docstrings_with_stub_types_and_defaults(tmp_path):
    pytest.importorskip("mkdocstrings_handlers.python")
    site = tmp_path / "site"
    run = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-q", "-d", str(site)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=1800,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    page = re.sub(r"<[^>]+>", " ", (site / "api" / "hmm" / "index.html").read_text(encoding="utf-8"))
    page = re.sub(r"\s+", " ", page)
    assert "Baum-Welch iterations" in page, "runtime docstrings are missing"
    assert re.search(r"tol float Baum-Welch iterations.*? 1e-06 ", page), (
        "grouped entries lost their own default"
    )
    assert "Sequences" in page and "ArrayLike" in page, "stub types are missing"
