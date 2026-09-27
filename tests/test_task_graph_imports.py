"""Keep new task-graph seam modules outside runtime import cycles."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "writing_agent"
NEW_SEAM_MODULES = {"task_graph_errors", "task_graph_calls", "task_graph_records"}


def build_import_graph() -> dict[str, set[str]]:
    """Read every task-graph module, including imports nested in functions."""
    paths = sorted(SOURCE_PACKAGE.glob("task_graph*.py"))
    modules = {path.stem for path in paths}
    graph = {module: set() for module in modules}

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if len(parts) > 1 and parts[0] == "writing_agent":
                        target = parts[1]
                        if target in modules:
                            graph[path.stem].add(target)
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    targets = (alias.name for alias in node.names)
                    if node.level:
                        graph[path.stem].update(target for target in targets if target in modules)
                else:
                    module_name = node.module
                    if node.level:
                        module_name = f"writing_agent.{module_name}"
                    parts = module_name.split(".")
                    if len(parts) > 1 and parts[0] == "writing_agent":
                        target = parts[1]
                        if target in modules:
                            graph[path.stem].add(target)
                    elif module_name == "writing_agent":
                        graph[path.stem].update(
                            alias.name for alias in node.names if alias.name in modules
                        )
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
        edge_count = sum(len(targets) for targets in graph.values())
        print(
            "task_graph import graph: "
            f"{len(graph)} modules, {edge_count} edges, {len(components)} SCCs, "
            f"{len(cyclic_components)} cyclic SCCs: {cyclic_components}"
        )

        self.assertIn("task_graph_errors", graph)
        self.assertEqual(graph["task_graph_errors"], set())
        self.assertLessEqual(
            graph["task_graph_calls"],
            {"task_graph", "task_graph_accounting", "task_graph_errors"},
        )
        self.assertLessEqual(
            graph["task_graph_records"],
            {"task_graph", "task_graph_contracts"},
        )
        for module in NEW_SEAM_MODULES:
            with self.subTest(module=module):
                self.assertIn(module, graph)
                self.assertFalse(
                    any(module in component for component in cyclic_components),
                    f"{module} appears in cyclic components {cyclic_components}",
                )


if __name__ == "__main__":
    unittest.main()
