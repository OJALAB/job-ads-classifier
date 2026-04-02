import os


def normalize_threads(threads):
    cpu_count = os.cpu_count() or 1

    if threads is None or threads == 0:
        return cpu_count
    if threads < 0:
        return max(1, cpu_count + threads + 1)
    return threads


def _load_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _available_cuda_devices(torch_module):
    if torch_module is None or not hasattr(torch_module, "cuda"):
        return 0

    try:
        if not torch_module.cuda.is_available():
            return 0
        return int(torch_module.cuda.device_count())
    except Exception:
        return 0


def normalize_precision(precision, accelerator, gpu_count):
    if gpu_count == 0 or accelerator == "cpu":
        return "32-true"

    aliases = {
        "16": "16-mixed",
        "16-mixed": "16-mixed",
        "bf16": "bf16-mixed",
        "bf16-mixed": "bf16-mixed",
        "32": "32-true",
        "32-true": "32-true",
        "64": "64-true",
        "64-true": "64-true",
    }

    return aliases.get(str(precision).lower(), precision)


def resolve_hardware(accelerator="auto", devices=1, precision=32):
    torch_module = _load_torch()
    requested_accelerator = accelerator
    normalized_accelerator = accelerator.lower() if isinstance(accelerator, str) else "auto"

    if normalized_accelerator == "cuda":
        normalized_accelerator = "gpu"

    gpu_count = 0 if normalized_accelerator == "cpu" else _available_cuda_devices(torch_module)

    if gpu_count == 0:
        return {
            "requested_accelerator": requested_accelerator,
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "gpu_count": 0,
            "torch_available": torch_module is not None,
        }

    try:
        devices = int(devices)
    except (TypeError, ValueError):
        devices = 1

    return {
        "requested_accelerator": requested_accelerator,
        "accelerator": normalized_accelerator,
        "devices": max(1, min(devices, gpu_count)),
        "precision": normalize_precision(precision, normalized_accelerator, gpu_count),
        "gpu_count": gpu_count,
        "torch_available": True,
    }
