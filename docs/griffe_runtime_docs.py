"""Griffe extension: the API pages read signatures from the stub and docstrings from the extension module.

`_transiograms.pyi` carries the types but no docstrings; the compiled module carries the NumPy
docstrings. This copies each runtime docstring onto the stub object of the same path, and splits
a grouped entry (``a, b : int``) into one entry per name so each shows its own default.
"""

import importlib
import inspect
import re

import griffe


def _runtime(path: str):
    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for name in parts[cut:]:
            obj = getattr(obj, name, None)
            if obj is None:
                return None
        return obj
    return None


GROUPED = re.compile(r"^(\w+(?:, \w+)+)( : .*)?$")


def _split_groups(doc: str) -> str:
    lines, out, i = inspect.cleandoc(doc).splitlines(), [], 0
    while i < len(lines):
        m = GROUPED.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and lines[j].startswith(" "):
            j += 1
        for name in m.group(1).split(", "):
            out += [name + (m.group(2) or ""), *lines[i + 1 : j]]
        i = j
    return "\n".join(out)


class RuntimeDocstrings(griffe.Extension):
    def on_package(self, *, pkg, **kwargs):
        stack = [pkg]
        while stack:
            obj = stack.pop()
            if obj.is_alias:
                continue
            stack.extend(obj.members.values())
            if obj.docstring is None and (doc := getattr(_runtime(obj.path), "__doc__", None)):
                if obj.is_attribute and type(doc) is str and doc.startswith(("str(object", "int(")):
                    continue
                obj.docstring = griffe.Docstring(_split_groups(doc), parent=obj)
