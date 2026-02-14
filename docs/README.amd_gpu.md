# AMD GPU Setup Guide (DirectML)

If you have an **AMD Radeon GPU** (RX 7000/9000 series) and want GPU-accelerated inference, follow this guide.

> **Note**: All AI components (YOLO detection, digit/spirit classifiers, **and OCR**) benefit from GPU acceleration via DirectML. OCR uses RapidOCR (ONNX Runtime) instead of PaddleOCR for GPU support.

---

## 1. Install DirectML

With your `env_uma` environment activated:

```bash
conda activate env_uma
pip install -r requirements-amd.txt
```

This installs `torch-directml`, `onnxruntime-directml`, and `rapidocr_onnxruntime`.

---

## 2. Verify Installation

Run this quick test:

```python
import torch
import torch_directml

device = torch_directml.device()
print(f"DirectML device: {device}")

# Quick tensor test
x = torch.randn(2, 3).to(device)
print(f"Tensor on DirectML: {x}")
```

You should see output like `DirectML device: privateuseone:0`.

---

## 3. Configure Umaplay

### Option A: Environment Variable

Set the environment variable before running:

```bash
set Umaplay_GPU_BACKEND=dml
python main.py
```

### Option B: Auto-detect (Default)

By default, `GPU_BACKEND=auto` tries CUDA first, then DirectML, then CPU. If you only have an AMD GPU and DirectML is installed, it will be selected automatically.

---

## 4. Verify It's Working

When Umaplay starts (`python main.py`), look for these log lines in the console:

```
[GPU] Backend=auto → device=privateuseone:0  (CUDA=False, DirectML=True)
```

For YOLO specifically, you should also see:

```
[DirectML] ONNX session using ['DmlExecutionProvider', 'CPUExecutionProvider']
```

> **Note**: If you're running the separate inference server (`uvicorn server.main_inference:app --port 8001`), you can also check `GET http://localhost:8001/health` — the response will include `"directml": true`. This only applies to client-only mode, not when running `python main.py` directly.

---

## Valid GPU Backend Values

| Value  | Behaviour                                       |
|--------|-------------------------------------------------|
| `auto` | Try CUDA → DirectML → CPU (default)             |
| `cuda` | Force NVIDIA CUDA, fallback to CPU if unavailable|
| `dml`  | Force DirectML, fallback to CPU if unavailable   |
| `cpu`  | Always use CPU (disable GPU acceleration)        |

---

## Known Limitations

- **PaddleOCR**: Always runs on CPU. PaddlePaddle does not support DirectML or ROCm on Windows.
- **PyTorch version**: `torch-directml` currently supports up to PyTorch 2.3.1.
- **Performance**: DirectML is generally faster than CPU but slower than native CUDA on equivalent NVIDIA hardware.
- **Training**: DirectML support is inference-only for most workloads. This is fine for Umaplay since it only runs inference.

---

## Troubleshooting

### `torch_directml` import error
Make sure you installed it: `pip install torch-directml`

### DirectML detected but slow
- Ensure your AMD GPU drivers are up to date: [AMD Drivers](https://www.amd.com/en/support)
- Close other GPU-intensive applications

### Falls back to CPU despite AMD GPU
- Verify your GPU supports DirectX 12 (all RX 7000/9000 series do)
- Try setting `Umaplay_GPU_BACKEND=dml` explicitly instead of `auto`
