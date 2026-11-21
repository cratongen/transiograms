"""The docs theme stays black, white and grey in both colour schemes, and self-contained.

The static checks read docs/stylesheets/transiograms.css, which holds the palette, the fonts and
the mapping onto Material for MkDocs. The checks against Material read the CSS of a built site
(`mise run docs-build` runs this module with TG_REQUIRE_SITE=1); without a site they are skipped.
The theme and the checks are ported from segyrs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
THEME = REPO / "docs" / "stylesheets" / "transiograms.css"
BRAND = REPO / "docs" / "brand"
SITE_CSS = REPO / "site" / "assets" / "stylesheets"
COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgb|hsl)a?\([^)]*\)")

pytestmark = pytest.mark.skipif(not THEME.is_file(), reason="needs the source checkout")


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _rules(css: str) -> list[tuple[str, str]]:
    """Leaf rules as (selector, body), with @media and @supports wrappers opened."""
    css, out, stack, start = _strip_comments(css), [], [], 0
    for i, ch in enumerate(css):
        if ch == "{":
            stack.append(css[start:i].strip())
            start = i + 1
        elif ch == "}":
            head = stack.pop() if stack else ""
            body = css[start:i]
            if "{" not in body and not head.startswith("@"):
                out.append((head, body))
            start = i + 1
    return out


def _declarations(body: str) -> list[tuple[str, str]]:
    return [
        (name.strip(), value.strip())
        for name, _, value in (d.partition(":") for d in body.split(";"))
        if value
    ]


def _split(selector_list: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in selector_list:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    return [*parts, cur.strip()]


def _specificity(sel: str) -> tuple[int, int, int]:
    a = b = c = 0
    i = 0
    while i < len(sel):
        ch = sel[i]
        if ch == "#":
            m = re.match(r"#[\w-]+", sel[i:])
            a += 1
            i += len(m.group(0))
        elif ch == ".":
            m = re.match(r"\.[\w-]+", sel[i:])
            b += 1
            i += len(m.group(0))
        elif ch == "[":
            j = sel.index("]", i)
            b += 1
            i = j + 1
        elif ch == ":":
            m = re.match(r"::?([\w-]+)", sel[i:])
            name, i = m.group(1), i + len(m.group(0))
            args = ""
            if i < len(sel) and sel[i] == "(":
                depth, j = 0, i
                while True:
                    depth += sel[j] == "("
                    depth -= sel[j] == ")"
                    if depth == 0:
                        break
                    j += 1
                args, i = sel[i + 1 : j], j + 1
            if name in ("is", "not", "has"):
                best = max(_specificity(s) for s in _split(args))
                a, b, c = a + best[0], b + best[1], c + best[2]
            elif name == "where":
                pass
            elif m.group(0).startswith("::") or name in ("before", "after"):
                c += 1
            else:
                b += 1
        elif re.match(r"[a-zA-Z]", ch):
            m = re.match(r"[a-zA-Z][\w-]*", sel[i:])
            c += 1
            i += len(m.group(0))
        else:
            i += 1
    return a, b, c


def _chromatic(value: str) -> bool:
    if "var(--md-hue)" in value:
        return True
    for lit in COLOUR.findall(value):
        if lit.startswith("#"):
            h = lit[1:]
            h = "".join(x * 2 for x in h) if len(h) in (3, 4) else h
            r, g, b = h[0:2], h[2:4], h[4:6]
            if not r == g == b:
                return True
        elif lit.startswith("hsl"):
            sat = re.findall(r"([\d.]+)%", lit)
            if sat and float(sat[0]) != 0:
                return True
        else:
            r, g, b = (float(x) for x in re.findall(r"[\d.]+", lit)[:3])
            if not r == g == b:
                return True
    return False


def _part(sel: str) -> str:
    last = re.split(r"[\s>+~]+", sel.strip())[-1]
    if re.search(r":before\b", last):
        return "before"
    if re.search(r":after\b", last):
        return "after"
    if "admonition-title" in last or last.startswith("summary"):
        return "title"
    return "box"


SHORTHANDS = {
    "background-color": ("background-color", "background"),
    "border-color": ("border-color", "border", "border-top"),
    "border": ("border",),
    "box-shadow": ("box-shadow",),
    "color": ("color",),
}


def _material() -> list[tuple[str, str]]:
    sheets = sorted(SITE_CSS.glob("*.css")) if SITE_CSS.is_dir() else []
    if not sheets:
        if os.environ.get("TG_REQUIRE_SITE") == "1":
            pytest.fail("no built site: run mise run docs-build")
        pytest.skip("no built site (mise run docs-build)")
    return [r for s in sheets for r in _rules(s.read_text(encoding="utf-8"))]


def _palette_and_rest() -> tuple[str, str]:
    css = _strip_comments(THEME.read_text(encoding="utf-8"))
    match = re.search(r":root\s*{[^}]*--tg-black[^}]*}", css)
    assert match, "transiograms.css must open with a :root palette block"
    return match.group(0), css[: match.start()] + css[match.end() :]


def test_palette_is_grey_and_the_rest_of_the_theme_uses_only_tokens():
    palette, rest = _palette_and_rest()
    literals = COLOUR.findall(palette)
    assert literals and not any(_chromatic(v) for v in literals)
    assert not COLOUR.findall(rest), "use the --tg-* tokens outside the :root palette block"
    defined = set(re.findall(r"(--tg-[\w-]+)\s*:", palette + rest))
    used = set(re.findall(r"var\((--tg-[\w-]+)\)", palette + rest))
    assert used <= defined, used - defined
    greys = set(re.findall(r"(--tg-(?:grey-\d+|black|white))\s*:", palette))
    assert greys <= used, f"tokens no page uses: {sorted(greys - used)}"
    count = {7: "seven"}[len(greys - {"--tg-black", "--tg-white"})]
    assert f"black, white and {count} greys" in (REPO / "docs" / "brand.md").read_text(encoding="utf-8")


def test_the_theme_makes_no_outside_request():
    css = _strip_comments(THEME.read_text(encoding="utf-8"))
    assert "@import" not in css and "http" not in css
    urls = re.findall(r"url\(\"?([^\")]+)\"?\)", css)
    assert urls
    for url in urls:
        assert (THEME.parent / url).resolve().is_file(), url
    assert (BRAND / "fonts" / "OFL.txt").is_file()
    assert "font: false" in (REPO / "mkdocs.yml").read_text(encoding="utf-8")


def test_brand_svgs_are_monotone_and_need_no_font():
    files = sorted(BRAND.glob("*.svg"))
    assert {f.name for f in files} >= {
        "favicon.svg",
        "mark-black.svg",
        "mark-white.svg",
        "wordmark-black.svg",
        "wordmark-white.svg",
    }
    for f in files:
        svg = f.read_text(encoding="utf-8")
        assert (
            "<text" not in svg and "font-family" not in svg and "Gradient" not in svg and "filter" not in svg
        ), f
        assert not any(_chromatic(v) for v in COLOUR.findall(svg)), f
    for stem in ("wordmark", "mark"):
        black = (BRAND / f"{stem}-black.svg").read_text(encoding="utf-8")
        assert (BRAND / f"{stem}-white.svg").read_text(encoding="utf-8") == black.replace("#000", "#fff")


def test_muted_text_meets_four_and_a_half_to_one_on_every_surface():
    def lum(hexcode: str) -> float:
        rgb = [int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    palette, theme = _palette_and_rest()
    tokens = dict(re.findall(r"(--tg-[\w-]+):\s*(#[0-9a-f]{6})", palette))
    for scheme in ("default", "slate"):
        block = re.search(rf'\[data-md-color-scheme="{scheme}"\]\s*{{([^}}]*)}}', theme).group(1)
        role = dict(re.findall(r"(--tg-[\w-]+):\s*var\((--tg-[\w-]+)\)", block))
        for text in ("--tg-fg", "--tg-muted"):
            for ground in ("--tg-bg", "--tg-surface"):
                hi, lo = sorted((lum(tokens[role[text]]), lum(tokens[role[ground]])), reverse=True)
                assert (hi + 0.05) / (lo + 0.05) >= 4.5, (scheme, text, ground)
    text_vars = re.findall(r"(--md-[\w-]*(?:fg|comment)[\w-]*):\s*var\((--tg-grey-500)\)", theme)
    assert not text_vars, f"--tg-grey-500 is below 4.5:1 on black; keep it off text: {text_vars}"


def test_every_material_colour_variable_is_set_by_the_theme():
    scheme_vars = {
        name
        for selector, body in _material()
        if ":root" in selector or "data-md-color-scheme" in selector
        for name, value in _declarations(body)
        if name.startswith("--md-") and _chromatic(value)
    }
    theme = _strip_comments(THEME.read_text(encoding="utf-8"))
    block = re.search(r"\[data-md-color-scheme\]\s*{([^}]*)}", theme).group(1)
    ours = set(re.findall(r"(--md-[\w-]+)\s*:", block))
    assert scheme_vars and not scheme_vars - ours, sorted(scheme_vars - ours)


def test_theme_admonition_rules_outrank_material_hued_ones():
    ours: dict[tuple[str, str], tuple[int, int, int]] = {}
    for selector, body in _rules(THEME.read_text(encoding="utf-8")):
        props = {name for name, _ in _declarations(body)}
        for sel in _split(selector):
            if not re.search(r"admonition|details|summary", sel):
                continue
            for prop, covers in SHORTHANDS.items():
                if props & set(covers):
                    key = (_part(sel), prop)
                    ours[key] = max(ours.get(key, (0, 0, 0)), _specificity(sel))
    beaten = []
    for selector, body in _material():
        for name, value in _declarations(body):
            if name not in SHORTHANDS or not _chromatic(value):
                continue
            for sel in _split(selector):
                if not re.search(r"admonition|details|summary", sel):
                    continue
                theirs = _specificity(sel)
                if ours.get((_part(sel), name), (-1, 0, 0)) < theirs:
                    beaten.append(f"{sel} {{{name}: {value}}} {theirs}")
    assert not beaten, "\n".join(sorted(set(beaten))[:20])
