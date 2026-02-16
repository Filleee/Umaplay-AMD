# AMD GPU Support (ROCm)

Umaplay supports AMD GPUs via the **ROCm** platform on Windows. This allows you to use your AMD graphics card for object detection (YOLO) and OCR (PaddleOCR), offering significantly better performance than CPU-only mode.

## Prerequisites

1.  **AMD GPU**: A compatible AMD Radeon graphics card (e.g., RX 6000, RX 7000 series, or newer APUs).
    *   **Note for RX 7900 GRE Users**: While not explicitly listed in the exact 7.2 release notes text, it is supported by ROCm 6.0+. If it doesn't work out-of-the-box, try setting the environment variable `HSA_OVERRIDE_GFX_VERSION=11.0.0` (matching the 7900 series architecture).
2.  **Drivers**: You need the specific **AMD Software: Adrenalin Edition™ Preview Driver for AMD Software: PyTorch on Windows**.

    *   **Download**: Visit the [AMD PyTorch on Windows](https://www.amd.com/en/developer/resources/rocm-hub/pytorch-on-windows.html) page.
    *   **Install**: Remove existing drivers if recommended, then install this preview driver.
3.  **HIP SDK**: Install the HIP SDK for Windows.
    *   Download from [AMD Infinity Hub](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html).
    *   Verify: `hipconfig --version`

## Installation

### 1. PyTorch (YOLO)

Official PyTorch builds for ROCm on Windows are not yet standard. You must follow the AMD Preview guide:

1.  **Create Environment**:
    ```bash
    conda create -n uma_amd python=3.10 -y
    conda activate uma_amd
    ```
2.  **Install PyTorch**:
    AMD provides a specific index URL for their preview builds. Try the following (verify current command on the AMD page):
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.2
    ```
    *If the above command fails or is not found, please check the [AMD PyTorch on Windows](https://www.amd.com/en/developer/resources/rocm-hub/pytorch-on-windows.html) page for the direct wheel links.*

    *Alternative*: Some users report success with:
    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.2
    ```

### 2. PaddlePaddle (OCR)

Native Windows ROCm support for PaddlePaddle is experimental.
-   If you cannot find a specific "rocm" version for Windows via pip, installing the standard version will result in CPU execution.
-   **Workaround**: Use the standard installation for now. The OCR will run on CPU (which is fast enough for many Umaplay tasks), while YOLO (the heavy lifter) runs on your AMD GPU.

    ```bash
    pip install paddlepaddle paddleocr
    ```

### 3. Dependencies

Install the rest of the project dependencies:
```bash
pip install -r requirements.txt
# Ensure you don't overwrite the specific torch version!
```

## Verification

Run the test script:
```bash
python tests/test_amd.py
```
-   **Success**: `torch.version.hip` should be detected. `torch.cuda.is_available()` should be True.
