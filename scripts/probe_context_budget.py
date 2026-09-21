"""Measure the real context budget of the local model instead of guessing at it.

Answers three separate questions that are easy to confuse:

1. **Inference KV cache** at a given context, computed from the model's own config.
   Gemma-4-E2B uses a 512-token sliding window on 28 of its 35 layers, single-head KV,
   and only 7 full-attention layers, so the cache is far smaller than an all-layers
   estimate. This is arithmetic, not a measurement.
2. **Attention step cost**, measured on the real layer shapes with a forward and a
   backward. This is the term that decides whether a long training step is affordable.
3. **Projected training cost** at a context, from the measured per-layer figure.

Reads the cached checkpoint's config only. Loads no weights, downloads nothing, runs no
training, and writes an inspectable artifact:

    python scripts/probe_context_budget.py
"""

import json
import sys
import time
from pathlib import Path

from writing_agent.inference import kv_cache_bytes

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "google/gemma-4-E2B-it"
REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
OUTPUT = ROOT / "data/processed/context-budget.json"
CONTEXTS = (8192, 32768, 65536, 131072)
PROBE_CONTEXTS = (8192, 32768, 65536)
# Full-attention layers dominate at length; a batch of one sequence.
OPTIMIZER_STEPS = 20
GRADIENT_ACCUMULATION = 8

# NF4 weights plus double-quantization overhead, as a fraction of bf16 parameters.
NF4_BYTES_PER_PARAM = 0.53
DEVICE_BYTES = 24_576 * 1024**2  # RTX 3090
ACTIVATION_BYTES = 2  # bf16
RUNTIME_OVERHEAD_BYTES = 2 * 1024**3  # CUDA context, fragmentation, LoRA grads and optimiser

# Named targets that are not cached locally. Layer and hidden counts are estimates
# from published configs; they are labelled as such wherever they are reported.
UNCACHED = {
    "gemma-4-31B": {"layers": 72, "hidden": 5376, "params": 31e9, "full_layers": 12},
    "qwen3.8-27B": {"layers": 64, "hidden": 5120, "params": 27e9, "full_layers": 64},
}
LADDER_CONTEXTS = (8192, 32768, 65536)


def cached_config() -> dict:
    """The pinned checkpoint's text config, from the local cache only."""
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            MODEL_ID, revision=REVISION, local_files_only=True, allow_patterns=["config.json"]
        )
    )
    return json.loads((root / "config.json").read_text())["text_config"]


def kv_bytes(text: dict, context: int) -> int:
    """Inference cache in bf16, computed from the checkpoint config."""
    return kv_cache_bytes(text, context)


def measure_layer(seq: int, *, heads: int, kv_heads: int, dim: int, dtype) -> dict:
    """Forward and backward one full-attention layer at its real shape."""
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    query, key, value = (
        torch.randn(1, n, seq, dim, device="cuda", dtype=dtype, requires_grad=True)
        for n in (heads, kv_heads, kv_heads)
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value, enable_gqa=True)
    out.sum().backward()
    torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - started,
        "peak_bytes": torch.cuda.max_memory_allocated(),
    }


def training_envelope(layers: int, hidden: int, params: float, context: int) -> dict:
    """QLoRA memory floor for one sequence, and the headroom left for attention transients.

    The floor is quantised weights plus one checkpointed activation boundary per layer.
    That boundary is what gradient checkpointing retains, so it scales with layers and
    context and does not shrink with batch size. The remaining headroom has to cover the
    recomputed attention block, whose measured cost is the layer probe above, plus runtime
    overhead. A model whose floor already exceeds the device cannot be trained at that
    length by any configuration.
    """
    weights = params * NF4_BYTES_PER_PARAM
    checkpoints = layers * context * hidden * ACTIVATION_BYTES
    floor = weights + checkpoints
    return {
        "context": context,
        "weights_bytes": int(weights),
        "checkpointed_bytes": int(checkpoints),
        "floor_bytes": int(floor),
        "headroom_bytes": int(DEVICE_BYTES - floor - RUNTIME_OVERHEAD_BYTES),
        "fits": floor + RUNTIME_OVERHEAD_BYTES < DEVICE_BYTES,
    }


def ladder() -> int:
    """Report the training envelope for every model we could plausibly use."""
    rows = {}
    hub = Path.home() / ".cache/huggingface/hub"
    for directory in sorted(hub.glob("models--google--gemma-4-*")):
        configs = list(directory.glob("snapshots/*/config.json"))
        if not configs:
            continue
        text = json.loads(configs[0].read_text()).get("text_config", {})
        weights = sum(f.stat().st_size for f in directory.glob("snapshots/*/*.safetensors"))
        rows[directory.name.split("models--google--")[-1]] = {
            "layers": len(text.get("layer_types", [])),
            "hidden": text.get("hidden_size"),
            "params": (weights / 2) or None,
            "measured": weights > 0,
            "max_context": text.get("max_position_embeddings"),
        }
        if rows[directory.name.split("models--google--")[-1]]["params"] is None:
            del rows[directory.name.split("models--google--")[-1]]
    rows.update({k: {**v, "measured": False, "max_context": None} for k, v in UNCACHED.items()})

    report = []
    for name, row in rows.items():
        for context in LADDER_CONTEXTS:
            envelope = training_envelope(row["layers"], row["hidden"], row["params"], context)
            report.append(
                {
                    "model": name,
                    "params": round(row["params"] / 1e9, 1),
                    "measured": row["measured"],
                    **{
                        k: (round(v / 1024**3, 1) if k.endswith("bytes") else v)
                        for k, v in envelope.items()
                        if k != "context"
                    },
                    "context": context,
                }
            )
    print(f"device: {DEVICE_BYTES / 1024**3:.0f} GiB, floor excludes the attention transient")
    print(
        f"{'model':<24}{'ctx':>7}{'params':>8}{'wt':>7}{'ckpt':>7}{'floor':>7}{'head':>7}  verdict"
    )
    for row in report:
        verdict = "fits" if row["fits"] else "EXCEEDS"
        label = row["model"] if row["measured"] else row["model"] + " (est.)"
        size = (
            f"{row['weights_bytes']:>7}{row['checkpointed_bytes']:>7}"
            f"{row['floor_bytes']:>7}{row['headroom_bytes']:>7}"
        )
        print(f"{label:<22}{row['context']:>7}{row['params']:>8}{size}  {verdict}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT.parent / "training-ladder.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwritten: {(OUTPUT.parent / 'training-ladder.json').relative_to(ROOT)}")
    return 0


def main() -> int:
    text = cached_config()
    heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    full_dim = text["global_head_dim"]

    report = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "max_position_embeddings": text["max_position_embeddings"],
        "layers": {
            "total": len(text["layer_types"]),
            "full_attention": text["layer_types"].count("full_attention"),
            "sliding_attention": text["layer_types"].count("sliding_attention"),
            "sliding_window": text["sliding_window"],
            "kv_sharing_layers": text["num_kv_shared_layers"],
        },
        "flash_attention_usable": False,
        "flash_attention_reason": (
            f"full-attention head_dim is {text['global_head_dim']}, above the 256 limit "
            "for the FlashAttention kernels available through SDPA on this stack"
        ),
        "kv_cache_bytes": {str(c): kv_bytes(text, c) for c in CONTEXTS},
    }

    if sys.platform == "linux" and _cuda_available():
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        dtype = torch.bfloat16
        report["layer_probe"] = {}
        for seq in PROBE_CONTEXTS:
            result = measure_layer(seq, heads=heads, kv_heads=kv_heads, dim=full_dim, dtype=dtype)
            full_layers = report["layers"]["full_attention"]
            report["layer_probe"][str(seq)] = {
                **result,
                "full_attention_layers": full_layers,
                "projected_step_seconds": result["seconds"] * full_layers,
            }
        report["projected"] = {
            "optimizer_steps": OPTIMIZER_STEPS,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "note": (
                "Projections multiply the measured per-layer figure by the full-attention "
                "layer count. They are lower bounds: they exclude the sliding layers, the "
                "MLP, embeddings, the optimizer step, and activation-checkpoint "
                "recomputation, and they are a layer-shape probe rather than a full "
                "training measurement."
            ),
        }
        for seq, row in report["layer_probe"].items():
            report["projected"][seq] = {
                "optimizer_step_minutes": round(
                    row["projected_step_seconds"] * GRADIENT_ACCUMULATION / 60, 1
                ),
                "run_hours": round(
                    row["projected_step_seconds"] * GRADIENT_ACCUMULATION * OPTIMIZER_STEPS / 3600,
                    1,
                ),
            }
    else:
        report["layer_probe"] = "unavailable: CUDA is required for the measured half"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "layer_probe"}, indent=1))
    print(f"\nwritten: {OUTPUT.relative_to(ROOT)}")
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(ladder() if "--ladder" in sys.argv else main())
