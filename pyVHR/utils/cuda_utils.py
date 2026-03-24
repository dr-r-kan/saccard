from numba import cuda
import os
import importlib
import importlib.util

if importlib.util.find_spec("torch") is None:
    torch = None
else:
    torch = importlib.import_module("torch")


def cuda_info():
    if torch is None:
        print("torch is not installed; CUDA info via torch is unavailable")
        return
    if torch.cuda.is_available():
        print("# CUDA devices: ", torch.cuda.device_count())
        for e in range(torch.cuda.device_count()):
            print("# device number ", e, ": ", torch.cuda.get_device_name(e))


def select_cuda_device(n):
    if torch is None:
        raise ModuleNotFoundError("torch is required to select a CUDA device")
    torch.cuda.device(n)
    cuda.select_device(n)
