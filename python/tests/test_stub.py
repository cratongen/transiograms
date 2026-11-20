import ast
import inspect
import re
from pathlib import Path

import pytest
import transiograms as tg

STUB = ast.parse(Path(tg._transiograms.__file__).with_name("_transiograms.pyi").read_text(encoding="utf-8"))
STUB_ONLY = {"Distribution", "DurationLaw", "Rows", "Sequences"}


def stub_callables():
    for node in STUB.body:
        if isinstance(node, ast.FunctionDef):
            yield node.name, getattr(tg, node.name), node
        elif isinstance(node, ast.ClassDef):
            cls = getattr(tg, node.name)
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and not any(
                    isinstance(d, ast.Name) and d.id == "property" for d in m.decorator_list
                ):
                    obj = cls if m.name == "__init__" else getattr(cls, m.name)
                    yield f"{node.name}.{m.name}", obj, m


def stub_signature(fn):
    a = fn.args
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + [ast.literal_eval(d) for d in a.defaults]
    out = [(x.arg, "positional", d) for x, d in zip(pos, defaults) if x.arg not in ("self", "cls")]
    out += [
        (x.arg, "keyword", None if d is None else ast.literal_eval(d))
        for x, d in zip(a.kwonlyargs, a.kw_defaults)
    ]
    return out


def runtime_signature(obj):
    out = []
    for p in inspect.signature(obj).parameters.values():
        if p.name in ("self", "cls"):
            continue
        kind = "keyword" if p.kind == p.KEYWORD_ONLY else "positional"
        out.append((p.name, kind, None if p.default is p.empty else p.default))
    return out


CALLABLES = [c for c in stub_callables() if not c[0].endswith(("__eq__", "__len__", "__repr__"))]


@pytest.mark.parametrize("name,obj,node", CALLABLES, ids=[c[0] for c in CALLABLES])
def test_stub_names_order_and_defaults_match_runtime(name, obj, node):
    assert stub_signature(node) == runtime_signature(obj)


def test_no_runtime_default_shows_as_ellipsis():
    shown = [name for name, obj, _ in CALLABLES if "..." in str(inspect.signature(obj))]
    assert shown == []


def test_stub_and_runtime_export_the_same_names():
    stubbed = {n.name for n in STUB.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    stubbed |= {
        t.id
        for n in STUB.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        if isinstance(t, ast.Name)
    }
    runtime = {n for n in dir(tg._transiograms) if not n.startswith("_")}
    assert {n for n in stubbed if not n.startswith("_")} - STUB_ONLY == runtime
    for node in STUB.body:
        if isinstance(node, ast.ClassDef):
            members = {
                m.name for m in node.body if isinstance(m, ast.FunctionDef) and not m.name.startswith("_")
            }
            public = {a for a in vars(getattr(tg, node.name)) if not a.startswith("_")}
            assert members == public, node.name


def api_page_nodes():
    root = Path(__file__).resolve().parents[2] / "docs" / "api"
    for page in sorted(root.glob("*.md")):
        for block in re.findall(r"```python\n(.*?)\n```", page.read_text(encoding="utf-8"), re.DOTALL):
            for node in ast.parse(block).body:
                if isinstance(node, ast.FunctionDef):
                    yield page.name, node.name, node
                elif isinstance(node, ast.ClassDef):
                    for m in node.body:
                        if isinstance(m, ast.FunctionDef):
                            yield page.name, f"{node.name}.{m.name}", m


def stub_nodes():
    out = {}
    for node in STUB.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            out.update({f"{node.name}.{m.name}": m for m in node.body if isinstance(m, ast.FunctionDef)})
    return out


def test_api_pages_copy_the_stub_signatures():
    stub = stub_nodes()
    differ = [
        f"{page}: {name}"
        for page, name, node in api_page_nodes()
        if name not in stub or ast.dump(node.args) != ast.dump(stub[name].args)
    ]
    assert differ == []
