"""Sanity-check that PyTorch can actually use this GPU.

Run with:  uv run scripts/check_env.py
"""

import torch

print(f"torch version      : {torch.__version__}")
print(f"built for CUDA     : {torch.version.cuda}")
print(f"CUDA available     : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"GPU                : {torch.cuda.get_device_name()}")
    print(f"compute capability : sm_{major}{minor}")
    # The list of GPU architectures this torch build ships compiled kernels for.
    # If our GPU's sm_XX isn't in here, torch will fail (or be very slow) on it.
    print(f"compiled archs     : {torch.cuda.get_arch_list()}")
    print(f"bf16 supported     : {torch.cuda.is_bf16_supported()}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free / total  : {free / 1e9:.1f} / {total / 1e9:.1f} GB")

    # Actually run a kernel: a bf16 matmul is exactly the kind of op training will do.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    print(f"bf16 matmul ok     : {(a @ b).shape}")
