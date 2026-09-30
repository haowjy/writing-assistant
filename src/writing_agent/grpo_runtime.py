"""Explicit public TRL implementations; verification performs no model/RNG mutation.

Tree digests bind every Python source path and byte to the qualification archives,
not an editable development version string or a local archive filename. Hash input
is compact sorted JSON mapping package-relative paths to SHA-256 content digests.
"""

import hashlib
import json
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path

STREAMING = "trl-6c5f135-streaming"
TRL_COMMIT = "6c5f1350488e9bba9a71242c47db45f2869796fa"
SOURCE_PINS = {
    "trl": (
        "trl",
        "1.14.0.dev0",
        "756ea0d4d7ca1e76704a2d07acb57a8759af359b97ed91c0561f51e3143b7482",
    ),
    "liger-kernel": (
        "liger_kernel",
        "0.8.3",
        "14abbe187dc87bb1c20f026c9caa5f0384ceff340536c1c3f849dee743dbbb84",
    ),
}


def implementation_plan(implementation):
    if implementation != STREAMING:
        raise ValueError("Unknown GRPO implementation")
    return {
        "id": STREAMING,
        "trl_commit": TRL_COMMIT,
        "numerics": "upstream-fp32-streaming-softcap; native-BF16-parity-not-claimed",
        "sources": {
            package: {"version": release, "python_tree_sha256": digest}
            for package, (_, release, digest) in SOURCE_PINS.items()
        },
        "config": {
            "use_liger_kernel": True,
            "cast_lm_head_to_fp32": False,
            "liger_kernel_config": dict.fromkeys(
                (
                    "rope",
                    "cross_entropy",
                    "fused_linear_cross_entropy",
                    "layer_norm",
                    "rms_norm",
                    "geglu",
                    "swiglu",
                ),
                False,
            ),
        },
    }


def python_tree_hash(root):
    files = {
        f"{root.name}/{path.relative_to(root).as_posix()}": hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_runtime(implementation):
    """Reject unsupported versions, shadowed imports and changed sources before loading."""
    plan = implementation_plan(implementation)
    for package, (module, release, expected) in SOURCE_PINS.items():
        dist = distribution(package)
        if dist.version != release:
            raise ValueError(f"Pinned GRPO requires {package} {release}")
        root = Path(dist.locate_file(module)).resolve()
        spec = find_spec(module)
        if spec is None or not spec.origin or Path(spec.origin).resolve() != root / "__init__.py":
            raise ValueError(f"Pinned GRPO import/source mismatch: {module}")
        if python_tree_hash(root) != expected:
            raise ValueError(f"Pinned GRPO source mismatch: {package}")
    return plan


def validate_streaming_model(config):
    # The all-disabled dispatch has been qualified for Gemma4 only. New model
    # families need their own public replacement-flag audit before admission.
    if config.model_type not in ("gemma4", "gemma4_text"):
        raise ValueError("Pinned streaming GRPO is qualified only for Gemma4")
    if getattr(config.get_text_config(), "enable_moe_block", False):
        raise ValueError("Pinned streaming GRPO requires dense Gemma4 (no MoE fallback)")
