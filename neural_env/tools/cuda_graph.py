"""
neural_env/tools/cuda_graph.py

Minimal, CPU-safe stand-in for the original RVC-Project's tools/cuda_graph.py.

The real implementation captures a CUDA graph per (module, key, arg-shape)
combination so repeated calls with identical tensor shapes skip Python/
kernel-launch overhead on the GPU. This sidecar runs on CPU (see
synthesis.neural.device: cpu in config.yaml) -- CUDA graphs don't apply
there at all -- so run_cuda_graph here is just a plain call-through.

It exists purely so infer/hubert.py, infer/rmvpe.py, infer/rtrvc.py, and
infer/fcpe.py can be used completely unmodified, import-for-import, exactly
as shipped from the upstream project. Nothing in those files needed to be
edited to run here.
"""


def run_cuda_graph(module, key, fn, *args):
    return fn(*args)


def cuda_graph_enabled(device) -> bool:
    return False
