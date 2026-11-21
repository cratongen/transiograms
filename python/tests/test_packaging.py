"""Release metadata checks on the source checkout: licence, URLs, classifiers, crates and tasks."""

import os
import re
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LICENSE = "MIT OR Apache-2.0"
THIRD_PARTY = "THIRD-PARTY-LICENSES.txt"
LICENSE_FILES = ["LICENSE-MIT", "LICENSE-APACHE", THIRD_PARTY]

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
    assert project["license-files"] == LICENSE_FILES
    for name in project["license-files"]:
        assert (ROOT / name).is_file(), name
    assert not (ROOT / "LICENSE").exists()
    assert "Apache License" in (ROOT / "LICENSE-APACHE").read_text(encoding="utf-8")
    assert _toml("Cargo.toml")["workspace"]["package"]["license"] == LICENSE
    for crate, manifest in _crates().items():
        assert manifest["package"]["license"] == {"workspace": True}, crate


RELEASE_TARGETS = [
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
]


def test_cargo_about_covers_the_release_targets_and_build_dependencies():
    about = _toml("about.toml")
    assert sorted(about["targets"]) == sorted(RELEASE_TARGETS)
    assert about["ignore-build-dependencies"] is False
    assert about["ignore-dev-dependencies"] is True
    assert (
        "cargo-about generate --locked -m crates/py/Cargo.toml"
        in _toml("mise.toml")["tasks"]["third-party-licenses"]["run"]
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo (mise install)")
def test_third_party_licences_name_every_crate_the_extension_links():
    """The extension links its Rust dependencies statically, so MIT, BSD and Apache notices must ship
    beside it; rerun `mise run third-party-licenses` after a Cargo.lock change."""
    text = (ROOT / THIRD_PARTY).read_text(encoding="utf-8")
    targets = [arg for t in RELEASE_TARGETS for arg in ("--target", t)]
    out = subprocess.run(
        ["cargo", "tree", "-p", "transiograms-py", "-e", "normal,build", *targets]
        + ["--prefix", "none", "--offline", "--locked"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    linked = {tuple(line.split(" ")[:2]) for line in out.stdout.splitlines() if line.strip()}
    assert len(linked) > 40
    missing = sorted(f"{n} {v}" for n, v in linked if f"\n  {n} {v.lstrip('v')}\n" not in text)
    assert not missing, f"{THIRD_PARTY} lacks {missing[:10]}"
    for spdx in ("MIT", "Apache-2.0", "BSD-2-Clause", "Unicode-3.0"):
        assert f"({spdx})" in text, spdx
    assert str(Path.home()) not in text


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


def test_readme_opens_with_install_and_a_quick_start_not_a_roadmap():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    install, quick = text.find("pip install transiograms"), text.find("```python")
    assert 0 < install < quick, "the PyPI page shows the install command, then a runnable quick start"
    for stale in ("## Roadmap", "| Milestone |", "sedimentary rock", "proglacial", "varvite"):
        assert stale not in text, stale


def test_no_shipped_or_documented_text_keeps_the_old_name():
    files = [
        ROOT / n for n in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md", "pyproject.toml", "mkdocs.yml")
    ]
    for folder in ("docs", "python", "examples", "crates", ".github"):
        files += [
            f
            for f in (ROOT / folder).rglob("*")
            if f.suffix in {".md", ".py", ".pyi", ".rs", ".toml", ".yml"}
        ]
    generated = ROOT / "docs" / "examples"
    hits = [
        f.relative_to(ROOT).as_posix()
        for f in files
        if f.is_file()
        and f != Path(__file__).resolve()
        and generated not in f.parents
        and "varvite" in f.read_text(encoding="utf-8").lower()
    ]
    assert hits == []


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
    for name in LICENSE_FILES:
        assert any(n.endswith(f".dist-info/licenses/{name}") for n in names), name
        assert f"License-File: {name}\n" in metadata, name
    with zipfile.ZipFile(wheel) as z:
        shipped = z.read(next(n for n in names if n.endswith(f".dist-info/licenses/{THIRD_PARTY}")))
    assert shipped == (ROOT / THIRD_PARTY).read_bytes()


def test_built_sdist_carries_the_licences_and_the_lockfile():
    with tarfile.open(_dist("transiograms-*.tar.gz")) as t:
        names = {n.split("/", 1)[1] for n in t.getnames() if "/" in n}
    assert {*LICENSE_FILES, "Cargo.lock", "pyproject.toml", "crates/py/Cargo.toml"} <= names


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


def test_contributing_and_issue_templates_say_code_starts_with_an_issue():
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "## A code contribution starts with an issue" in text
    assert "MIT OR Apache-2.0" in text
    yaml = pytest.importorskip("yaml")
    templates = ROOT / ".github" / "ISSUE_TEMPLATE"
    forms = {f.name: yaml.safe_load(f.read_text(encoding="utf-8")) for f in templates.glob("*.yml")}
    assert {"config.yml", "bug-report.yml", "feature-request.yml"} <= set(forms)
    assert forms["config.yml"]["blank_issues_enabled"] is False
    for name, form in forms.items():
        if name != "config.yml":
            assert form["name"] and form["description"] and form["body"], name
    assert "CONTRIBUTING.md" in str(forms["feature-request.yml"]["body"])


def test_docs_site_uses_the_family_theme_in_both_schemes():
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "extra_css:\n  - stylesheets/transiograms.css\n" in text
    assert "scheme: default" in text and "scheme: slate" in text
    css = (ROOT / "docs" / "stylesheets" / "transiograms.css").read_text(encoding="utf-8")
    assert '[data-md-color-scheme="default"]' in css and '[data-md-color-scheme="slate"]' in css
    urls = re.findall(r"url\(\"?([^\")]+)\"?\)", css)
    assert "@import" not in css and "//" not in "".join(urls), "the theme makes no request to another server"
    for url in urls:
        assert (ROOT / "docs" / "stylesheets" / url).resolve().is_file(), url


def test_changelog_has_a_section_for_this_version():
    version = _toml("Cargo.toml")["workspace"]["package"]["version"]
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(
        rf"^## {re.escape(version)} \((unreleased|\d{{4}}-\d{{2}}-\d{{2}})\)$", text, re.MULTILINE
    )
    assert text.isascii(), "no emoji or other non-ASCII headings"
    assert (ROOT / "cliff.toml").is_file()
