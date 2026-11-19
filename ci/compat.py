"""Install the transiograms wheel into fresh environments and run a smoke check in each.

Usage: python ci/compat.py [all|venv|uv|pixi|conda]

The wheel is built once (abi3, Python >= 3.11) and installed with pip into a venv, with uv into a
Python 3.11 venv, into a pixi environment and into a micromamba (conda-forge) environment.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("venv", "uv", "pixi", "conda")
SMOKE = """
import pickle
import numpy as np
import transiograms as tg
c = tg.MarkovChain.from_matrix([[0.9, 0.1], [0.2, 0.8]])
assert np.allclose(c.transition_matrix.sum(axis=1), 1.0)
assert np.allclose(c.proportions, [2 / 3, 1 / 3])
assert pickle.loads(pickle.dumps(c)) == c
try:
    tg.MarkovChain.from_matrix([[0.5, 0.6], [0.5, 0.5]])
    raise SystemExit("bad matrix accepted")
except tg.InvalidInput:
    pass
print("transiograms", tg.__version__, "ok")
"""


def run(*cmd, cwd=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def build(out):
    run(sys.executable, "-m", "maturin", "build", "--release", "-q", "-o", out, cwd=ROOT)
    return next(Path(out).glob("transiograms-*.whl"))


def venv(tmp, wheel):
    env = tmp / "venv"
    run(sys.executable, "-m", "venv", env)
    run(env / "bin" / "python", "-m", "pip", "install", "-q", wheel)
    run(env / "bin" / "python", "-c", SMOKE)


def uv(tmp, wheel):
    env = tmp / "uv"
    run("uv", "venv", "-q", "--python", "3.11", env)
    run("uv", "pip", "install", "-q", "--python", env / "bin" / "python", wheel)
    run(env / "bin" / "python", "-c", SMOKE)


def pixi(tmp, wheel):
    manifest = tmp / "pixi" / "pixi.toml"
    run("pixi", "init", "-q", manifest.parent)
    run("pixi", "add", "--manifest-path", manifest, "python=3.12", "numpy", "pip")
    run("pixi", "run", "--manifest-path", manifest, "python", "-m", "pip", "install", "-q", wheel)
    run("pixi", "run", "--manifest-path", manifest, "python", "-c", SMOKE)


def conda(tmp, wheel):
    env = tmp / "conda"
    run("micromamba", "create", "-y", "-q", "-p", env, "-c", "conda-forge", "python=3.13", "numpy", "pip")
    run("micromamba", "run", "-p", env, "python", "-m", "pip", "install", "-q", wheel)
    run("micromamba", "run", "-p", env, "python", "-c", SMOKE)


def main(argv):
    mode = argv[1] if len(argv) > 1 else "all"
    if mode not in ("all", *MODES):
        raise SystemExit(f"mode must be one of all, {', '.join(MODES)}; got {mode!r}")
    with tempfile.TemporaryDirectory(prefix="transiograms-compat-") as d:
        tmp = Path(d)
        wheel = build(tmp / "wheel")
        failed = []
        for m in MODES if mode == "all" else (mode,):
            try:
                globals()[m](tmp, wheel)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"{m}: FAILED ({e})", flush=True)
                failed.append(m)
        if failed:
            raise SystemExit(f"failed: {', '.join(failed)}")
        print("all requested installs passed")


if __name__ == "__main__":
    main(sys.argv)
