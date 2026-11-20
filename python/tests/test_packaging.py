"""Release metadata checks on the source checkout: licence, URLs, classifiers, crates and tasks."""

import os
import re
import tarfile
import tomllib
import zipfile
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
    return {
        p.parent.name: _toml(p.relative_to(ROOT).as_posix()) for p in (ROOT / "crates").glob("*/Cargo.toml")
    }


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


SITE = "https://cratongen.github.io/transiograms/"
GITHUB = "https://github.com/cratongen/transiograms"


def test_project_urls_and_authors_point_at_the_organization():
    project = _toml("pyproject.toml")["project"]
    assert project["authors"]
    urls = project["urls"]
    assert {"Homepage", "Documentation", "Repository", "Issues", "Changelog"} <= set(urls)
    for name, url in urls.items():
        assert url.startswith((SITE, GITHUB)), (name, url)
    assert urls["Documentation"] == SITE
    assert urls["Repository"] == GITHUB


def test_mkdocs_site_and_repo_urls():
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert f"site_url: {SITE}\n" in text
    assert f"repo_url: {GITHUB}\n" in text
    assert "repo_name: cratongen/transiograms\n" in text


def test_readme_names_the_docs_and_links_work_on_pypi():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert SITE in text and GITHUB in text
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:src|srcset)="([^"]+)"', text)
    assert not [t for t in targets if not t.startswith(("https://", "#"))]


def _python_classifiers() -> list[str]:
    prefix = "Programming Language :: Python :: 3."
    return [
        c.removeprefix("Programming Language :: Python :: ") for c in _classifiers() if c.startswith(prefix)
    ]


def _classifiers() -> list[str]:
    return _toml("pyproject.toml")["project"]["classifiers"]


def test_classifiers_match_a_first_public_release_of_binary_wheels():
    classifiers = _classifiers()
    assert [c for c in classifiers if c.startswith("Development Status")] == [
        "Development Status :: 3 - Alpha"
    ]
    assert "Operating System :: OS Independent" not in classifiers
    assert {
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: Implementation :: CPython",
    } <= set(classifiers)


def test_python_floor_matches_requires_python_and_the_abi3_feature():
    floor = _toml("pyproject.toml")["project"]["requires-python"].removeprefix(">=")
    versions = _python_classifiers()
    assert versions[0] == floor
    minors = [int(v.split(".")[1]) for v in versions]
    assert minors == list(range(minors[0], minors[-1] + 1))
    pyo3 = _toml("Cargo.toml")["workspace"]["dependencies"]["pyo3"]
    assert f"abi3-py{floor.replace('.', '')}" in pyo3["features"]


def test_workspace_crates_can_be_packaged_later_and_the_binding_never_is():
    workspace = _toml("Cargo.toml")["workspace"]
    version = workspace["package"]["version"]
    assert workspace["package"]["repository"] == GITHUB
    assert workspace["package"]["homepage"] == SITE
    local = {k: v for k, v in workspace["dependencies"].items() if isinstance(v, dict) and "path" in v}
    assert set(local) == {f"transiograms-{c}" for c in _crates() if c != "py"}
    for name, dep in local.items():
        assert dep["version"] == version, name
    for crate, manifest in _crates().items():
        package = manifest["package"]
        assert package["repository"] == {"workspace": True}, crate
        assert package["homepage"] == {"workspace": True}, crate
        for name, dep in manifest.get("dependencies", {}).items():
            if name.startswith("transiograms-"):
                assert dep == {"workspace": True}, (crate, name)
    assert _crates()["py"]["package"]["publish"] is False


def test_mise_pins_the_python_floor_and_the_newest_classifier():
    pins = _toml("mise.toml")["tools"]["python"]
    minors = sorted(int(v.split(".")[1]) for v in pins)
    versions = _python_classifiers()
    assert f"3.{minors[0]}" == versions[0]
    assert f"3.{minors[-1]}" == versions[-1]


def test_rust_toolchain_pin_meets_the_declared_rust_version():
    channel = _toml("rust-toolchain.toml")["toolchain"]["channel"]
    msrv = _toml("Cargo.toml")["workspace"]["package"]["rust-version"]
    assert tuple(map(int, channel.split("."))) >= tuple(map(int, msrv.split(".")))
    assert msrv in (ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    assert _toml("mise.toml")["tasks"]["test-sdist"]["env"]["RUSTUP_TOOLCHAIN"] == channel


def test_release_tasks_build_and_check_without_uploading():
    tasks = _toml("mise.toml")["tasks"]
    assert {"wheel", "sdist", "check-dist", "test-wheel", "test-sdist"} <= set(tasks)
    assert "--interpreter python3.11" in " ".join(tasks["wheel"]["run"])
    commands = " ".join(
        " ".join(t["run"]) if isinstance(t.get("run"), list) else t.get("run", "") for t in tasks.values()
    )
    assert not re.search(r"\b(upload|publish)\b", commands)


def _dist(pattern: str) -> Path | None:
    found = sorted((ROOT / "dist").glob(pattern))
    if not found:
        if os.environ.get("TG_REQUIRE_DIST"):
            pytest.fail(f"no dist/{pattern}: run mise run wheel")
        pytest.skip(f"no dist/{pattern} (mise run wheel, mise run sdist)")
    return found[-1]


def test_built_wheel_metadata_names_urls_licence_and_classifiers():
    wheel = _dist("transiograms-*.whl")
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        metadata = z.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
    assert "-cp311-abi3-" in wheel.name
    assert f"License-Expression: {LICENSE}\n" in metadata
    assert f"Project-URL: Repository, {GITHUB}\n" in metadata
    assert f"Project-URL: Documentation, {SITE}\n" in metadata
    assert "Author: " in metadata
    assert "Classifier: Programming Language :: Python :: 3.11\n" in metadata
    assert "Classifier: Operating System :: OS Independent" not in metadata
    for name in ("LICENSE-MIT", "LICENSE-APACHE"):
        assert any(n.endswith(f".dist-info/licenses/{name}") for n in names), name


def test_built_sdist_carries_the_licences_and_the_lockfile():
    with tarfile.open(_dist("transiograms-*.tar.gz")) as t:
        names = {n.split("/", 1)[1] for n in t.getnames() if "/" in n}
    assert {"LICENSE-MIT", "LICENSE-APACHE", "Cargo.lock", "pyproject.toml", "crates/py/Cargo.toml"} <= names


def test_release_checklist_names_the_minetypes_gate_and_the_types_it_replaces():
    text = (ROOT / "docs" / "release-checklist.md").read_text(encoding="utf-8")
    assert "The one remaining gate: minetypes adoption" in text
    for name in (
        "`tg.Categories`",
        "`tg.Grid`",
        "`transiograms_core::Categories`",
        "`transiograms_core::Grid`",
    ):
        assert name in text, name
    assert "release-checklist.md" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "minetypes" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "minetypes" not in (ROOT / "Cargo.toml").read_text(encoding="utf-8")
