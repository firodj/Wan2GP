import torch
import os
import contextlib

def is_mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

def is_cuda_available():
    return torch.cuda.is_available()

def get_device_string():
    if is_cuda_available():
        return "cuda"
    elif is_mps_available():
        return "mps"
    return "cpu"

def get_torch_device(device_index=None):
    device_str = get_device_string()
    if device_index is not None and device_str == "cuda":
        return torch.device(f"{device_str}:{device_index}")
    return torch.device(device_str)

def empty_cache():
    if is_cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    elif is_mps_available():
        torch.mps.empty_cache()

def synchronize():
    if is_cuda_available():
        torch.cuda.synchronize()
    elif is_mps_available():
        torch.mps.synchronize()

@contextlib.contextmanager
def device_autocast(device_type=None, dtype=None, enabled=True, cache_enabled=None):
    if device_type is None:
        device_type = get_device_string()

    if device_type == "mps":
        # MPS autocast support might vary by PyTorch version, but generally strictly requires 'mps' or 'cpu'
        # torch.cuda.amp.autocast is legacy. torch.amp.autocast is the new standard.
        if dtype is None:
            dtype = torch.float16 # Default for mixed precision
        with torch.amp.autocast(device_type="mps", dtype=dtype, enabled=enabled):
            yield
    elif device_type == "cuda":
        with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled, cache_enabled=cache_enabled):
            yield
    else:
        # CPU autocast
        if dtype is None:
            dtype = torch.bfloat16
        with torch.amp.autocast(device_type="cpu", dtype=dtype, enabled=enabled):
            yield

@contextlib.contextmanager
def maybe_sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
    if is_cuda_available():
        from torch.backends.cuda import sdp_kernel
        with sdp_kernel(enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem_efficient):
            yield
    else:
        # MPS uses SDPA by default if available, no explicit context manager needed/available
        yield