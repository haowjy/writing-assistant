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
    sys.exit(main())
