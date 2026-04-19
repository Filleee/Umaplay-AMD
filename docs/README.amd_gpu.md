# AMD GPU Support (ROCm) - Experimental

> [!WARNING]
> **Experimental Support**
> This setup has been tested primarily on the **AMD Radeon RX 7900 GRE**.
> Succes with other AMD cards is **not guaranteed** and may vary depending on your specific hardware and driver version.
> Use at your own risk.

This fork adds experimental support for AMD GPUs using **ROCm** on Windows. This allows the heaviest workload—**YOLO object detection**—to run on your GPU, significantly reducing CPU usage.

## Architecture

| Component | Runs On | libraries | Impact |
| :--- | :--- | :--- | :--- |
| **YOLO** (Detection) | 🎮 **GPU** | `ultralytics` + `torch` (ROCm) | **Massive Gain**: Offloads ~55% CPU load. 2-3x faster detection. |
| **OCR** (Text) | 🖥️ **CPU** | `paddlepaddle` + `paddleocr` | **Minor Trade-off**: Runs on CPU (standard Windows build). Disabling oneDNN (see below) costs ~10-20% speed but ensures stability. |

## Prerequisites

1.  **AMD GPU**: A modern AMD Radeon GPU (e.g., RX 6000/7000/9000 series).
2.  **OS**: Windows 10/11.
3.  **Drivers**: **AMD Software: Adrenalin Edition 26.1.1** (or newer).
    *   *Note*: This specific driver is required for ROCm 7.2 support on Windows.
4.  **Python**: **Python 3.12** is strictly required. The ROCm 7.2 wheels provided by AMD are built for this specific version.

## Installation Guide

### 1. Set Up Python Environment

Create a clean environment specifically for AMD ROCm (Python 3.12).

```bash
conda create -n env_uma python=3.12
conda activate env_uma
```

### 2. Install ROCm 7.2 SDK

Install the core ROCm SDK components required for PyTorch to interface with your AMD GPU.

```cmd
pip install --no-cache-dir --pre ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz
```
*(If using PowerShell, replace `^` with `` ` ``)*

### 3. Install PyTorch with ROCm 7.2 Support

Install the PyTorch wheels built for ROCm.

```cmd
pip install --no-cache-dir --pre ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl
```
*(If using PowerShell, replace `^` with `` ` ``)*

### 4. Install Remaining Dependencies

Install the rest of the bot's requirements (like PaddleOCR for CPU) using the AMD-specific requirements file.

```bash
pip install -r requirements-amd.txt
```

---

## Verification

To verify that your environment is correctly set up, run the included test script:

```bash
python tests/test_amd.py
```

**What this script checks:**
1.  **PyTorch ROCm**: Use `torch.cuda.is_available()` to verify the GPU is visible. It should print your AMD GPU name and the ROCm version.
2.  **PaddlePaddle**: Checks compilation status.
3.  **HIP SDK**: Checks if `hipconfig` is available.

## Troubleshooting & Technical Details

### PaddleOCR Crashes & Environment Variables
If you inspect `core/perception/ocr/ocr_local.py`, you'll see logic that automatically sets:

```python
if is_rocm:
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"
```

**What this does:**
*   It detects if you are running on an AMD ROCm system (by checking `torch.version.hip`).
*   If yes, it disables **oneDNN** (Intel MKLDNN) and the new **PIR API**.
*   **Reason**: PaddlePaddle 3.x often crashes on Windows/ROCm systems with `ConvertPirAttribute2RuntimeAttribute` errors due to conflicts with oneDNN. Disabling it is the only stable fix, even though it might slightly reduce CPU OCR speed.
*   **Note**: This is applied *automatically* only for AMD users. NVIDIA/Intel users retain full CPU optimization.

### "hipconfig" not found
If `tests/test_amd.py` says `hipconfig` is missing, ensure you have installed the HIP SDK if you plan to do any development.

### VS C++ Redistributable
If you get "DLL load failed" errors when importing torch, ensure you have the latest **Microsoft Visual C++ Redistributable** installed.
