import json
import re
import tomllib
from pathlib import Path

import pytest
import transiograms as tg

ROOT = Path(__file__).parents[2]
PLUGIN = ROOT / "packaging" / "plugin"
SKILL = PLUGIN / "skills" / "transiograms" / "SKILL.md"
CLAUDE = PLUGIN / ".claude-plugin" / "plugin.json"
PORTABLE = PLUGIN / "plugin.json"
CLAUDE_MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
CODEX_MARKETPLACE = ROOT / ".agents" / "plugins" / "marketplace.json"

pytestmark = pytest.mark.skipif(not PLUGIN.is_dir(), reason="packaging/plugin not present")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def frontmatter():
    head = SKILL.read_text(encoding="utf-8").split("---\n")[1]
    return dict(line.split(": ", 1) for line in head.splitlines())


def test_manifests_carry_the_library_version():
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    assert {load(CLAUDE)["version"], load(PORTABLE)["version"]} == {version}


def test_manifests_agree_on_what_the_plugin_is():
    claude, portable = load(CLAUDE), load(PORTABLE)
    for field in ("name", "description", "author", "homepage", "repository", "license"):
        assert claude[field] == portable[field], field
    assert claude["name"] == "transiograms"


def test_both_marketplaces_point_at_the_plugin_folder():
    claude = load(CLAUDE_MARKETPLACE)["plugins"]
    codex = load(CODEX_MARKETPLACE)["plugins"]
    assert [entry["name"] for entry in claude] == [entry["name"] for entry in codex] == ["transiograms"]
    assert (ROOT / claude[0]["source"]).resolve() == PLUGIN
    assert (ROOT / codex[0]["source"]["path"]).resolve() == PLUGIN
    assert {"installation", "authentication"} <= set(codex[0]["policy"])
    assert codex[0]["category"]


def test_portable_manifest_stays_within_the_listing_limits():
    portable = load(PORTABLE)
    interface = portable["extensions"]["com.openai"]["interface"]
    assert len(interface["displayName"]) <= 30
    assert len(interface["shortDescription"]) <= 30
    assert len(portable["suggestedPrompts"]) <= 3


def test_skill_frontmatter_is_the_part_every_agent_accepts():
    fields = frontmatter()
    assert set(fields) == {"name", "description"}
    assert fields["name"] == SKILL.parent.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"])
    assert 0 < len(fields["description"]) <= 1024
    assert not set("<>") & set(fields["description"])


def test_skill_names_only_what_the_library_exports():
    text = SKILL.read_text(encoding="utf-8")
    named = set(re.findall(r"\btg\.([A-Za-z_]\w*)", text))
    missing = {name for name in named if not hasattr(tg, name)}
    assert missing == set()
    exported = set(tg.__all__) | {"datasets", "plot", "checks"}
    invented = {name for name in named if name not in exported and not name.startswith("_")}
    assert invented == set()
