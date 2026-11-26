"""Import direction between packages (plan/benchmark-code.md §2.1). Runs on source, not imports."""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "writing_assistant"

ALLOWED = {
    "__init__": set(),
    "episode": set(),
    "llm": {"episode"},
    "search": {"episode"},
    "judges": {"episode", "llm"},
    "corpus": {"episode", "llm", "search", "judges"},
    "suites": {"episode", "llm", "search", "judges", "corpus"},
    "drivers": {"episode", "llm"},
    "run": {"episode", "llm", "search", "judges", "corpus", "suites", "drivers"},
    "cli": {"episode", "llm", "search", "judges", "corpus", "suites", "drivers", "run"},
    "label_ui": set(),
}
# Leaves inside a layer: a module only its layer's entry point imports. `cli/view.py` reads
# runs through the run loader and `report`'s aggregation (one reading of a run, not two),
# so it sits in `cli`; nothing may build on the page's server.
LEAVES = {("cli", "view"): ("cli", "__init__")}


def _imports(path: Path) -> list[tuple[str, ...]]:
    """Modules under writing_assistant imported by one file, relative imports resolved."""
    package = path.relative_to(PKG.parent).with_suffix("").parts[:-1]
    out = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            out += [tuple(a.name.split(".")) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else ()
            mod = base + tuple(node.module.split(".") if node.module else ())
            out.append(mod)
            # `from writing_assistant.suites import world` imports the submodule `world`.
            for a in node.names:
                sub = PKG.parent.joinpath(*mod, a.name)
                if sub.is_dir() or sub.with_suffix(".py").exists():
                    out.append(mod + (a.name,))
    return [m[1:] for m in out if m[:1] == ("writing_assistant",) and len(m) > 1]


def test_imports_point_down():
    bad = []
    for path in sorted(PKG.rglob("*.py")):
        src = path.relative_to(PKG).with_suffix("").parts
        layer = src[0]
        assert layer in ALLOWED, f"{path}: unknown layer {layer!r}; add it to ALLOWED"
        for target in _imports(path):
            if target[0] == layer:
                # Within a layer: suites/<a> never imports suites/<b>; no provider imports gateway.
                if layer == "suites" and len(src) > 2 and len(target) > 1 and target[1] != src[1]:
                    bad.append(f"{'/'.join(src)} -> {'.'.join(target)}")
                if src[:2] == ("search", "providers") and target[:2] == ("search", "gateway"):
                    bad.append(f"{'/'.join(src)} -> {'.'.join(target)}")
            elif target[0] not in ALLOWED[layer]:
                bad.append(f"{'/'.join(src)} -> {'.'.join(target)}")
            if target[:2] in LEAVES and src != LEAVES[target[:2]] and src[:2] != target[:2]:
                bad.append(f"{'/'.join(src)} -> {'.'.join(target)} (a leaf)")
    assert not bad, "upward or sideways imports:\n" + "\n".join(bad)
