"""Low-level forged events for rejection tests, deliberately outside production APIs."""

from writing_agent.task_graph import EventV1


def forged_effect(state, *, changes, history=(), delta=()):
    return {
        "artifact_type": "Phase2RecordedEffectV1",
        "before_state_ref": state.identity(),
        "file_delta": dict(delta),
        "set": dict(changes),
        "history_set": dict(history),
    }


def forged_event(writer, state, kind, payload_ref, *, actor, audience):
    return EventV1(
        previous=state.history["head"],
        seq=state.history["seq"] + 1,
        lineage_id=state.position["lineage_id"],
        rollout_id=writer.rollout_id,
        node_visit_id=state.position["visit_id"],
        kind=kind,
        actor=actor,
        audience=audience,
        payload_ref=payload_ref,
        versions_ref=state.versions_ref,
        provenance_ref=state.provenance_ref,
    )


def forged_reduced(writer, state, event_record, effect_body):
    return writer.store._apply_recorded_effect_body(state, event_record, effect_body)


def forged_log(writer, state, kind, record_ref, message_ref=None):
    entries = writer.environment.runtime_log(state)
    entry = {"seq": state.history["seq"] + 1, "kind": kind, "record_ref": record_ref}
    if message_ref is not None:
        entry["message_ref"] = message_ref
    entries.append(entry)
    return writer.store.put_artifact(
        {"record_type": "WriterRuntimeLogV1", "rollout_id": writer.rollout_id, "entries": entries}
    )
