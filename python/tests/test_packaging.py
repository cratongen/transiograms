"""Release metadata checks on the source checkout: licence, URLs, classifiers, crates and tasks."""

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LICENSE = "MIT OR Apache-2.0"

pytestmark = pytest.mark.skipif(
    not (ROOT / "pyproject.toml").is_file() or not (ROOT / "Cargo.toml").is_file(),
    reason="needs the source checkout",
)


def _toml(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text(encoding="utf-8"))


def _crates() -> dict[str, dict]:
    return {p.parent.name: _toml(p.relative_to(ROOT).as_posix()) for p in (ROOT / "crates").glob("*/Cargo.toml")}


def test_licence_is_mit_or_apache_everywhere():
    project = _toml("pyproject.toml")["project"]
    assert project["license"] == LICENSE
    assert project["license-files"] == ["LICENSE-MIT", "LICENSE-APACHE"]
    for name in project["license-files"]:
        assert (ROOT / name).is_file(), name
    assert not (ROOT / "LICENSE").exists()
    assert "Apache License" in (ROOT / "LICENSE-APACHE").read_text(encoding="utf-8")
    assert _toml("Cargo.toml")["workspace"]["package"]["license"] == LICENSE
    for crate, manifest in _crates().items():
        assert manifest["package"]["license"] == {"workspace": True}, crate
