"""Backend resolver: picks compute device/dtype/autocast for cuda, rocm, directml, cpu.

DirectML and CPU cannot use CUDA-specific paths (autocast("cuda", ...), CUDA-graph
compile, FlashInfer). This module is the single place that decides what each
backend can do, so demo.py never hardcodes "cuda" again.
"""

import contextlib

import torch


class BackendContext:
    def __init__(self, name, device, dtype, force_sdpa, allow_compile, allow_autocast):
        self.name = name
        self.device = device
        self.dtype = dtype
        self.force_sdpa = force_sdpa
        self.allow_compile = allow_compile
        self._allow_autocast = allow_autocast

    def autocast(self):
        if not self._allow_autocast:
            return contextlib.nullcontext()
        return torch.amp.autocast("cuda", dtype=self.dtype)

    def describe(self):
        return (
            f"backend={self.name} device={self.device} dtype={self.dtype} "
            f"sdpa={'forced' if self.force_sdpa else 'flashinfer-ok'} "
            f"compile={'on' if self.allow_compile else 'off'}"
        )


def _rocm_available():
    # ROCm builds of torch report themselves as "cuda" (torch.version.hip is set).
    return torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None


def _directml_available():
    try:
        import torch_directml  # noqa: F401
        return torch_directml.is_available()
    except ImportError:
        return False


def resolve_backend(name="auto"):
    name = name.lower()

    if name == "auto":
        if torch.cuda.is_available() and not _rocm_available():
            name = "cuda"
        elif _rocm_available():
            name = "rocm"
        elif _directml_available():
            name = "directml"
        else:
            name = "cpu"

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--backend cuda requested but torch.cuda.is_available() is False.")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        return BackendContext("cuda", torch.device("cuda"), dtype,
                               force_sdpa=False, allow_compile=True, allow_autocast=True)

    if name == "rocm":
        if not _rocm_available():
            raise RuntimeError("--backend rocm requested but no ROCm-enabled torch.cuda device found.")
        # FlashInfer is CUDA-only (no ROCm kernels) -> force SDPA. CUDA-graph
        # capture (torch.compile reduce-overhead) is not validated on ROCm here,
        # so keep it off until someone benchmarks it.
        return BackendContext("rocm", torch.device("cuda"), torch.float16,
                               force_sdpa=True, allow_compile=False, allow_autocast=True)

    if name == "directml":
        if not _directml_available():
            raise RuntimeError(
                "--backend directml requested but torch_directml is not installed "
                "or reports no device. Install with: pip install torch-directml"
            )
        import torch_directml
        device = torch_directml.device()
        # DirectML has no autocast/AMP support and no CUDA-graph compile path;
        # fp32 throughout keeps numerics simple until proven otherwise.
        return BackendContext("directml", device, torch.float32,
                               force_sdpa=True, allow_compile=False, allow_autocast=False)

    if name == "cpu":
        return BackendContext("cpu", torch.device("cpu"), torch.float32,
                               force_sdpa=True, allow_compile=False, allow_autocast=False)

    raise ValueError(f"Unknown backend {name!r}. Choose from: auto, cuda, rocm, directml, cpu.")
