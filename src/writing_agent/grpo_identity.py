"""Non-mutating base and adapter tensor identities for task-graph training."""

import hashlib
import json

from writing_agent.task_graph import canonical_bytes


def base_tensor_identity(model):
    """Stream actual named parameters AND buffers, including nonpersistent buffers.

    Reads bounded CPU byte chunks without changing tensors, device, dtype or RNG.
    Dense contiguous tensors are required to avoid a hidden full-model copy. Alias
    names are retained; metadata and raw bytes are length-framed in sorted order.
    """
    import torch

    digest = hashlib.sha256()
    count = 0
    tensors = [
        *(
            ("parameter", name, value)
            for name, value in model.named_parameters(remove_duplicate=False)
        ),
        *(("buffer", name, value) for name, value in model.named_buffers(remove_duplicate=False)),
    ]
    for kind, name, tensor in sorted(tensors, key=lambda item: (item[0], item[1])):
        if (
            tensor.device.type == "meta"
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
        ):
            raise ValueError("Caller base tensors must be materialized, dense and contiguous")
        header = json.dumps(
            [kind, name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        data = tensor.detach().reshape(-1).view(torch.uint8)
        digest.update(data.numel().to_bytes(8, "big"))
        for offset in range(0, data.numel(), 1024 * 1024):
            digest.update(data[offset : offset + 1024 * 1024].cpu().numpy().tobytes())
        count += 1
    if not count:
        raise ValueError("Caller base model has no tensors")
    return {
        "trust": "actual-caller-base-tensors-v1",
        "sha256": digest.hexdigest(),
        "named_tensors": count,
    }


def adapter_tensor_hash(model, adapter_name: str = "default") -> str:
    """Hash the selected PEFT adapter's named tensor state with stable framing."""
    import torch
    from peft import get_peft_model_state_dict

    try:
        state = get_peft_model_state_dict(model, adapter_name=adapter_name)
    except TypeError:
        state = get_peft_model_state_dict(model)
    if not state:
        raise ValueError("expected PEFT adapter has no tensor state")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().contiguous().cpu()
        header = canonical_bytes([name, str(value.dtype), list(value.shape)])
        raw = value.view(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()
