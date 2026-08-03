"""
neural_env/configs/config.py

Minimal stand-in for the original RVC-Project's configs/config.py. Only the
two attributes actually imported by infer/rmvpe.py and infer/audio.py are
provided: infer_device and infer_dtype. The original file runs GPU
auto-detection/VRAM-based half-precision benchmarking logic that has no
purpose here -- this sidecar is pinned to CPU (see synthesis.neural.device
in the main app's config.yaml) -- so both are simply hardcoded.
"""
import torch

infer_device = torch.device("cpu")
infer_dtype = torch.float32


def get_device_dtype_sm(device_index):
    # Only ever reached on a CUDA code path, which this CPU-only sidecar
    # never takes. Kept so the import doesn't fail if it's ever referenced.
    return torch.device("cpu"), torch.float32, None, None
