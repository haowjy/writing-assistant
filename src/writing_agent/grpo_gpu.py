"""Complete NVML inventory admission for task-graph GPU execution."""

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from writing_agent.catalog import save_json

DISPLAY_POLICY = {
    "names": [
        "cosmic-comp",
        "cosmic-panel",
        "cosmic-bg",
        "cosmic-app-library",
        "cosmic-edit",
        "cosmic-settings",
        "cosmic-files",
        "xdg-desktop-portal-cosmic",
        "xwayland",
        "ghostty",
        "chrome",
        "cursor",
    ],
    "per_process_mib": 256,
    "total_mib": 768,
    "minimum_free_mib": 22000,
}

CUDA_ALLOCATOR_CONF = {
    "environment": ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"],
    "value": "expandable_segments:True",
}


def configure_cuda_allocator():
    """Bind expandable segments before Torch can initialize the CUDA allocator."""
    if "torch" in sys.modules:
        raise RuntimeError("Configure the CUDA allocator before importing torch")
    for name in CUDA_ALLOCATOR_CONF["environment"]:
        configured = os.environ.get(name)
        if configured not in (None, CUDA_ALLOCATOR_CONF["value"]):
            raise RuntimeError(f"Conflicting CUDA allocator setting: {name}={configured}")
        os.environ[name] = CUDA_ALLOCATOR_CONF["value"]
    return CUDA_ALLOCATOR_CONF


def inventory(xml):
    """Reject missing/unknown inventory fields; retain graphics and compute consumers."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError("Malformed GPU inventory XML") from exc
    gpus = root.findall("gpu")
    if len(gpus) != 1:
        raise ValueError("Production requires exactly one physical GPU")
    gpu = gpus[0]

    def required(node, key):
        value = node.findtext(key)
        if not value or value.strip() in {"N/A", "Unknown"}:
            raise ValueError(f"Missing GPU inventory field: {key}")
        return value.strip()

    def mib(node, key):
        value = required(node, key)
        if not re.fullmatch(r"\d+ MiB", value):
            raise ValueError(f"Unknown GPU memory accounting: {key}={value}")
        return int(value.split()[0])

    processes = gpu.find("processes")
    if processes is None or (processes.text or "").strip():
        raise ValueError("Unavailable complete GPU process inventory")
    consumers = []
    for process in processes:
        if process.tag != "process_info":
            raise ValueError("Unknown GPU process inventory entry")
        pid = int(required(process, "pid"))
        if pid <= 0:
            raise ValueError("Invalid GPU consumer PID")
        consumers.append(
            {
                "pid": pid,
                "name": required(process, "process_name"),
                "type": required(process, "type"),
                "memory_mib": mib(process, "used_memory"),
            }
        )
    return {
        "uuid": required(gpu, "uuid"),
        "name": required(gpu, "product_name"),
        "total_mib": mib(gpu, "fb_memory_usage/total"),
        "free_mib": mib(gpu, "fb_memory_usage/free"),
        "used_mib": mib(gpu, "fb_memory_usage/used"),
        "consumers": consumers,
        "process_total_mib": sum(p["memory_mib"] for p in consumers),
    }


def ownership_report(xml, *, policy=DISPLAY_POLICY):
    result = inventory(xml)
    errors = []
    for process in result["consumers"]:
        executable = Path(process["name"].split(maxsplit=1)[0]).name.lower()
        if executable not in policy["names"]:
            errors.append(f"Unknown consumer: {process}")
        if process["memory_mib"] > policy["per_process_mib"]:
            errors.append(f"Oversized consumer: {process}")
    if result["process_total_mib"] > policy["total_mib"]:
        errors.append("GPU process allocation exceeds total cap")
    if result["free_mib"] < policy["minimum_free_mib"]:
        errors.append("Insufficient free GPU memory")
    return {**result, "policy": policy, "errors": errors, "admitted": not errors}


def capture_inventory(path):
    """Always preserve raw stdout/stderr, including driver/query failures."""
    path = Path(path)
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True)
    path.write_text(result.stdout)
    path.with_suffix(".stderr").write_text(result.stderr)
    save_json(
        path.with_suffix(".command.json"), {"command": result.args, "returncode": result.returncode}
    )
    result.check_returncode()
    return result.stdout


def admit_gpu(directory, *, policy=DISPLAY_POLICY):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        report = ownership_report(capture_inventory(directory / "nvml.xml"), policy=policy)
        save_json(directory / "ownership.json", report)
        if not report["admitted"]:
            raise RuntimeError("GPU ownership rejected: " + "; ".join(report["errors"]))
        return report
    except BaseException as exc:
        save_json(directory / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
