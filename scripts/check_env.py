"""Sanity-check a teammate's environment: torch version and which accelerator is usable.

Usage: uv run python scripts/check_env.py
"""

import platform

import torch


def main() -> None:
    print(f"python  {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"torch   {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"device  cuda: {props.name}, {props.total_memory / 2**30:.1f} GiB, "
              f"compute capability {props.major}.{props.minor}")
        device = "cuda"
    elif torch.backends.mps.is_available():
        print("device  mps (Apple Silicon)")
        device = "mps"
    else:
        print("device  cpu only (no CUDA or MPS found)")
        device = "cpu"

    x = torch.randn(1024, 1024, device=device)
    (x @ x).sum().item()
    print(f"matmul on {device}: ok")


if __name__ == "__main__":
    main()
