"""Run one Phase 8 task-graph probe phase."""

from smoke_task_graph_grpo_cpu import ScriptedNativeBackend

from writing_agent.grpo_task_graph_probe import main, register_scripted_native_backend

register_scripted_native_backend(ScriptedNativeBackend)

if __name__ == "__main__":
    raise SystemExit(main())
