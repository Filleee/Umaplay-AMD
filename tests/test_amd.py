import torch
import paddle
import subprocess
import sys

def check_pytorch():
    print("\n[PyTorch Check]")
    try:
        print(f"PyTorch Version: {torch.__version__}")
        is_available = torch.cuda.is_available()
        print(f"CUDA Available: {is_available}")
        
        if is_available:
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Device Count: {torch.cuda.device_count()}")
            
        try:
            print(f"ROCm (HIP) Version: {torch.version.hip}")
        except AttributeError:
            print("ROCm (HIP) Version: Not Found (Standard CUDA build?)")
            
        x = torch.rand(5, 3)
        if is_available:
            x = x.to("cuda")
            print("Tensor moved to GPU successfully.")
        else:
            print("Running on CPU.")
            
    except Exception as e:
        print(f"PyTorch Error: {e}")

def check_paddle():
    print("\n[PaddlePaddle Check]")
    try:
        print(f"Paddle Version: {paddle.__version__}")
        
        has_cuda = False
        if hasattr(paddle.device, "is_compiled_with_cuda") and paddle.device.is_compiled_with_cuda():
            print("Compiled with CUDA: Yes")
            has_cuda = True
        else:
            print("Compiled with CUDA: No")
            
        if hasattr(paddle.device, "is_compiled_with_rocm") and paddle.device.is_compiled_with_rocm():
            print("Compiled with ROCm: Yes")
            has_cuda = True
        else:
            print("Compiled with ROCm: No")
            
        device = paddle.get_device()
        print(f"Current Device: {device}")
        
    except Exception as e:
        print(f"Paddle Error: {e}")

def check_hip_sdk():
    print("\n[HIP SDK Check]")
    try:
        res = subprocess.run(["hipconfig", "--version"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            print(f"hipconfig Output:\n{res.stdout.strip()}")
        else:
            print("hipconfig not found or failed (env var issue or SDK not installed?)")
    except FileNotFoundError:
        print("hipconfig command not found in PATH.")

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    check_hip_sdk()
    check_pytorch()
    check_paddle()
