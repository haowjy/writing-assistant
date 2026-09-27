"""Keep new task-graph seam modules outside runtime import cycles."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "writing_agent"
NEW_SEAM_MODULES = {
    "writing_agent.task_graph_errors",
    "writing_agent.task_graph_calls",
    "writing_agent.task_graph_records",
    "writing_agent.task_graph_wire",
    "writing_agent.task_graph_record_contracts",
    "writing_agent.task_graph_payloads",
    "writing_agent.task_graph_operation",
    "writing_agent.task_graph_transition",
    "writing_agent.task_graph_derive_entry",
    "writing_agent.task_graph_derive_context",
    "writing_agent.task_graph_derive_outcome",
    "writing_agent.task_graph_derive_author",
    "writing_agent.task_graph_derive_writer",
    "writing_agent.task_graph_controller",
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_PACKAGE).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("writing_agent", *parts))


def _add_edge(graph: dict[str, set[str]], source: str, target: str) -> None:
    """Include imported modules and package initializers Python loads on the way."""
    parts = target.split(".")
    for end in range(1, len(parts)):
        package = ".".join(parts[:end])
        if package in graph:
            graph[source].add(package)
    if target in graph:
        graph[source].add(target)


class RuntimeImports(ast.NodeVisitor):
    """Collect imports Python executes, excluding annotations under TYPE_CHECKING."""

    def __init__(self) -> None:
        self.nodes: list[ast.Import | ast.ImportFrom] = []

    def visit_If(self, node: ast.If) -> None:
        test = node.test
        type_checking = isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
        type_checking |= (
            isinstance(test, ast.Attribute)
            and test.attr == "TYPE_CHECKING"
            and isinstance(test.value, ast.Name)
            and test.value.id == "typing"
        )
        if type_checking:
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.nodes.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.nodes.append(node)


def build_import_graph() -> dict[str, set[str]]:
    """Read every Python module in writing_agent, including nested imports."""
    paths = sorted(SOURCE_PACKAGE.rglob("*.py"))
    modules_by_path = {path: _module_name(path) for path in paths}
    modules = set(modules_by_path.values())
    graph = {module: set() for module in modules}

    for path in paths:
        source = modules_by_path[path]
        package = source if path.name == "__init__.py" else source.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = RuntimeImports()
        imports.visit(tree)
        for node in imports.nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "writing_agent" or alias.name.startswith("writing_agent."):
                        _add_edge(graph, source, alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = package.split(".")
                    parent_parts = package_parts[: len(package_parts) - node.level + 1]
                    base_parts = [*parent_parts]
                    if node.module:
                        base_parts.extend(node.module.split("."))
                    base = ".".join(base_parts)
                else:
                    base = node.module or ""

                if base == "writing_agent" or base.startswith("writing_agent."):
                    _add_edge(graph, source, base)
                    for alias in node.names:
                        imported = f"{base}.{alias.name}"
                        if imported in graph:
                            _add_edge(graph, source, imported)
                elif node.level and base in graph:
                    _add_edge(graph, source, base)
                    if node.module is None:
                        for alias in node.names:
                            imported = f"{base}.{alias.name}"
                            if imported in graph:
                                _add_edge(graph, source, imported)
    return graph


def strongly_connected_components(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    """Return Tarjan SCCs for all graph nodes in stable order."""
    next_index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal next_index
        indices[node] = lowlinks[node] = next_index
        next_index += 1
        stack.append(node)
        on_stack.add(node)

        for target in sorted(graph[node]):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])

        if lowlinks[node] == indices[node]:
            component = []
            while True:
                target = stack.pop()
                on_stack.remove(target)
                component.append(target)
                if target == node:
                    break
            components.append(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return sorted(components)


class TaskGraphImportTests(unittest.TestCase):
    def test_new_seam_modules_are_not_in_import_cycles(self) -> None:
        graph = build_import_graph()
        components = strongly_connected_components(graph)
        cyclic_components = [
            component
            for component in components
            if len(component) > 1 or component[0] in graph[component[0]]
        ]
        cycles_through_other_modules = [
            component
            for component in cyclic_components
            if any(not module.rsplit(".", 1)[-1].startswith("task_graph") for module in component)
        ]
        edge_count = sum(len(targets) for targets in graph.values())
        print(
            "writing_agent import graph: "
            f"{len(graph)} modules, {edge_count} edges, {len(components)} SCCs, "
            f"{len(cyclic_components)} cyclic SCCs: {cyclic_components}; "
            f"cycles through non-task_graph modules: {cycles_through_other_modules}"
        )

        error_module = SOURCE_PACKAGE / "task_graph_errors.py"
        error_tree = ast.parse(error_module.read_text(encoding="utf-8"))
        self.assertFalse(
            any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(error_tree))
        )
        self.assertIn("writing_agent.task_graph_errors", graph)
        self.assertLessEqual(
            graph["writing_agent.task_graph_calls"],
            {
                "writing_agent",
                "writing_agent.task_graph",
                "writing_agent.task_graph_accounting",
                "writing_agent.task_graph_errors",
                "writing_agent.task_graph_records",
            },
        )
        self.assertLessEqual(
            graph["writing_agent.task_graph_records"],
            {
                "writing_agent",
                "writing_agent.task_graph",
                "writing_agent.task_graph_contracts",
                "writing_agent.task_graph_errors",
                "writing_agent.task_graph_wire",
                "writing_agent.task_graph_record_contracts",
                "writing_agent.task_graph_payloads",
            },
        )
        self.assertLessEqual(
            graph["writing_agent.task_graph_transition"],
            {
                "writing_agent",
                "writing_agent.task_graph",
                "writing_agent.task_graph_admission",
                "writing_agent.task_graph_contracts",
                "writing_agent.task_graph_group_contract",
                "writing_agent.task_graph_records",
            },
        )
        self.assertLessEqual(
            graph["writing_agent.task_graph_derive_entry"],
            {
                "writing_agent",
                "writing_agent.task_graph",
                "writing_agent.task_graph_admission",
                "writing_agent.task_graph_contracts",
                "writing_agent.task_graph_records",
                "writing_agent.task_graph_transition",
            },
        )
        for forbidden in {
            "writing_agent.task_graph_store",
            "writing_agent.task_graph_ports",
            "writing_agent.task_graph_local",
            "writing_agent.task_graph_writer",
            "writing_agent.task_graph_environment",
            "writing_agent.task_graph_replay",
        }:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, graph["writing_agent.task_graph_derive_entry"])
        for module in NEW_SEAM_MODULES:
            with self.subTest(module=module):
                self.assertIn(module, graph)
                self.assertFalse(
                    any(module in component for component in cyclic_components),
                    f"{module} appears in cyclic components {cyclic_components}",
                )

    def test_entry_derivation_has_no_store_port_or_filesystem_imports(self) -> None:
        path = SOURCE_PACKAGE / "task_graph_derive_entry.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        forbidden = {
            "os",
            "pathlib",
            "writing_agent.task_graph_store",
            "writing_agent.task_graph_ports",
            "writing_agent.task_graph_local",
            "writing_agent.workspace",
        }
        self.assertFalse(imported & forbidden)


if __name__ == "__main__":
    unittest.main()
