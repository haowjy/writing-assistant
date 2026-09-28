"""Test-only storage verifiers for structural store tests."""


class PatchVerifier:
    """Accept structurally valid store test commits without semantic derives."""

    def view(self, store, checkpoint_id):
        return store.load_checkpoint(checkpoint_id)

    def verify_commit(self, store, base_checkpoint_id, events, next_state):
        del store, base_checkpoint_id, events, next_state
