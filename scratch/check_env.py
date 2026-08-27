import sys
import os
import platform
import psutil

print("=== SYSTEM & HARDWARE INFO ===")
print("Python:", sys.version)
print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("Processor:", platform.processor())
print(f"Physical CPU cores: {psutil.cpu_count(logical=False)}")
print(f"Logical CPU cores: {psutil.cpu_count(logical=True)}")
mem = psutil.virtual_memory()
print(f"Total System RAM: {mem.total / (1024**3):.2f} GB (Available: {mem.available / (1024**3):.2f} GB)")

try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"PyTorch CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
except Exception as e:
    print(f"PyTorch Error: {e}")

try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
except Exception as e:
    print(f"Transformers Error: {e}")

try:
    import peft
    print(f"PEFT Version: {peft.__version__}")
except Exception as e:
    print(f"PEFT Error: {e}")

try:
    import trl
    print(f"TRL Version: {trl.__version__}")
except Exception as e:
    print(f"TRL Error: {e}")

try:
    import bitsandbytes
    print(f"bitsandbytes Version: {bitsandbytes.__version__}")
except Exception as e:
    print(f"bitsandbytes Error: {e}")

try:
    import datasets
    print(f"datasets Version: {datasets.__version__}")
except Exception as e:
    print(f"datasets Error: {e}")
