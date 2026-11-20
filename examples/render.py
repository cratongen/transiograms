"""Runs each page's script (examples/NN-topic/NN-page/example_NN_NN.py) cell by cell and writes its README.md.

The module docstring is the opening text. Cells start with `# %%` (code), `# %% [markdown]` (text written as
`# ` comments) or `# %% [hidden]` (run, not shown). Each script runs in its own folder, so the
figures it saves with `fig.savefig("name.png")` land next to it; `common.py` sets the figure style
here, so the scripts themselves need nothing beyond transiograms, NumPy and matplotlib. `render.py 01-markov-chains/01-fit` renders one page; no
argument renders every page.
"""

import ast
import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import common  # noqa: F401  (applies the shared figure style)

SAVED: list[Path] = []
_savefig = matplotlib.figure.Figure.savefig


def _recording_savefig(self, fname, *args, **kwargs):
    SAVED.append(Path(fname).resolve())
    return _savefig(self, fname, *args, **kwargs)


matplotlib.figure.Figure.savefig = _recording_savefig


def cells(source: str):
    kind, lines = None, []
    for line in source.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                yield kind, "\n".join(lines).strip("\n")
            kind = "markdown" if "[markdown]" in line else "hidden" if "[hidden]" in line else "code"
            lines = []
        elif kind is not None:
            lines.append(line)
    if kind is not None:
        yield kind, "\n".join(lines).strip("\n")


def render(script: Path) -> str:
    source = script.read_text(encoding="utf-8")
    namespace = {"__file__": str(script), "__name__": "__main__"}
    parts = [ast.get_docstring(ast.parse(source)) or ""]
    for kind, body in cells(source):
        if kind == "markdown":
            parts.append(
                "\n".join(
                    line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines()
                )
            )
            continue
        SAVED.clear()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.chdir(script.parent):
            exec(compile(body, str(script), "exec"), namespace)  # noqa: S102
        if kind == "hidden":
            continue
        block = f"```python\n{body}\n```"
        if printed := out.getvalue().rstrip():
            block += f"\n\n```text\n{printed}\n```"
        for figure in SAVED:
            block += f"\n\n![{figure.stem}]({figure.name})"
        parts.append(block)
    parts.append(f"Full script: [`{script.name}`]({script.name})")
    return "\n\n".join(parts) + "\n"


def pages() -> list[Path]:
    return sorted(ROOT.glob("[0-9][0-9]-*/[0-9][0-9]-*/example_*.py"))


if __name__ == "__main__":
    chosen = [next((ROOT / p).glob("example_*.py")) for p in sys.argv[1:]] or pages()
    for script in chosen:
        (script.parent / "README.md").write_text(render(script), encoding="utf-8", newline="\n")
        print(f"rendered {script.parent.relative_to(ROOT).as_posix()}")
    if not chosen:
        print("no example pages yet")
