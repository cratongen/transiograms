"""Every public object documents itself, and every docstring example runs."""

import doctest
import inspect
import re

import matplotlib
import numpy as np
import pytest
import transiograms as tg

matplotlib.use("Agg")

MODULES = (tg.plot, tg.datasets, tg.checks)


def public():
    for name in sorted(tg.__all__):
        obj = getattr(tg, name)
        if name == "__version__" or not (callable(obj) or isinstance(obj, type)):
            continue
        yield f"tg.{name}", obj, "class" if isinstance(obj, type) else "function"
        if isinstance(obj, type) and not issubclass(obj, BaseException):
            for attr, raw in sorted(vars(obj).items()):
                if attr.startswith("_"):
                    continue
                kind = "property" if inspect.isdatadescriptor(raw) else "method"
                yield f"tg.{name}.{attr}", getattr(obj, attr) if kind == "method" else raw, kind
    for mod in MODULES:
        for name, f in sorted(vars(mod).items()):
            if not name.startswith("_") and inspect.isfunction(f) and f.__module__ == mod.__name__:
                yield f"{mod.__name__}.{name}", f, "function"


OBJECTS = list(public())
IDS = [q for q, _, _ in OBJECTS]


def params(obj, kind):
    if kind == "property" or (kind == "class" and issubclass(obj, BaseException)):
        return []
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    return [p.name for p in sig.parameters.values() if p.name not in ("self", "cls")]


def section(doc, title):
    m = re.search(rf"^{title}\n-+\n(.*?)(?=^\S[^\n]*\n-+\n|\Z)", doc, re.DOTALL | re.MULTILINE)
    return None if m is None else m.group(1)


def documented(doc):
    body = section(doc, "Parameters") or ""
    names = set()
    for line in body.splitlines():
        if line and not line.startswith(" "):
            head = line.split(" : ")[0]
            names |= {n.strip().lstrip("*") for n in head.split(",")}
    return names


@pytest.mark.parametrize("qual,obj,kind", OBJECTS, ids=IDS)
def test_every_public_object_has_a_docstring(qual, obj, kind):
    doc = inspect.getdoc(obj) if kind != "property" else obj.__doc__
    assert doc and doc.strip(), f"{qual} has no docstring"


@pytest.mark.parametrize("qual,obj,kind", OBJECTS, ids=IDS)
def test_every_parameter_is_documented(qual, obj, kind):
    names = params(obj, kind)
    if not names:
        return
    doc = inspect.getdoc(obj)
    missing = [n for n in names if n not in documented(doc)]
    assert not missing, f"{qual} does not document {missing}"


CALLABLES = [o for o in OBJECTS if o[2] != "property"]


@pytest.mark.parametrize("qual,obj,kind", CALLABLES, ids=[o[0] for o in CALLABLES])
def test_sections_returns_raises_examples(qual, obj, kind):
    if kind == "class" and issubclass(obj, BaseException):
        return
    doc = inspect.getdoc(obj)
    if kind in ("function", "method"):
        assert section(doc, "Returns") is not None, f"{qual} has no Returns section"
    if params(obj, kind):
        assert section(doc, "Raises") is not None, f"{qual} has no Raises section"
    if kind in ("class", "function"):
        assert ">>>" in (section(doc, "Examples") or ""), f"{qual} has no Examples section"


def test_every_public_method_is_exercised_by_a_docstring_example():
    def examples(obj):
        return section(inspect.getdoc(obj) or "", "Examples") or ""

    missing = []
    for qual, obj, kind in OBJECTS:
        if kind != "method":
            continue
        owner, attr = qual.rsplit(".", 1)
        text = examples(obj) + examples(getattr(tg, owner.split(".", 1)[1]))
        if not re.search(rf"\.{attr}\(", text):
            missing.append(qual)
    assert not missing, f"no example in the method's or its class's docstring calls {missing}"


DOCTESTS = [
    (q, inspect.getdoc(o)) for q, o, k in OBJECTS if k != "property" and ">>>" in (inspect.getdoc(o) or "")
]


@pytest.mark.parametrize("qual,doc", DOCTESTS, ids=[d[0] for d in DOCTESTS])
def test_docstring_examples_run(qual, doc):
    import matplotlib.pyplot as plt

    test = doctest.DocTestParser().get_doctest(doc, {"tg": tg, "np": np}, qual, None, 0)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    out = []
    try:
        result = runner.run(test, out=out.append)
    finally:
        plt.close("all")
    assert result.failed == 0, "".join(out)
