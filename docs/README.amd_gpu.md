# AMD GPU Setup Guide (DirectML)

If you have an **AMD Radeon GPU** (RX 7000/9000 series) and want GPU-accelerated inference, follow this guide.

> **Note**: All AI components benefit from GPU acceleration via DirectML:
> - **YOLO detection** — via ONNX Runtime + DirectML
> - **Digit & spirit classifiers** — via `torch-directml`
> - **OCR** — via [RapidOCR](https://github.com/RapidAI/RapidOCR) (PP-OCRv4 ONNX models) + DirectML

---

## 1. Install DirectML

With your `env_uma` environment activated:

```bash
conda activate env_uma
pip install -r requirements-amd.txt
```

This installs:
- `torch-directml` — DirectML backend for PyTorch (classifiers)
- `onnxruntime-directml` — DirectML provider for ONNX Runtime (YOLO + OCR)
- `rapidocr_onnxruntime` — ONNX-based OCR engine wrapping PP-OCRv4 models

> **Important**: `requirements-amd.txt` must be installed **after** `requirements.txt`, because `torch-directml` needs to overwrite the standard `torch` package. The `run_uma.bat` launcher handles this automatically.

---

## 2. Export YOLO Models to ONNX (One-Time)

DirectML requires ONNX format models. Export them once:

```bash
python -c "from ultralytics import YOLO; [YOLO(f'models/{m}').export(format='onnx', dynamic=True, simplify=True) for m in ['uma_ura.pt','uma_unity_cup.pt','uma_nav.pt']]"
```

This creates `.onnx` files next to the `.pt` weights in the `models/` folder.

---

## 3. Verify Installation

### PyTorch + DirectML

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

### ONNX Runtime + DirectML

```python
import onnxruntime
providers = onnxruntime.get_available_providers()
print(f"Providers: {providers}")
assert "DmlExecutionProvider" in providers, "DirectML not available!"
```

### OCR (RapidOCR)

```python
from rapidocr_onnxruntime import RapidOCR
import numpy as np, cv2

engine = RapidOCR(det_use_dml=True, rec_use_dml=True, cls_use_dml=True)
img = np.ones((80, 200, 3), dtype=np.uint8) * 255
cv2.putText(img, "847", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
result, _ = engine(img)
print(f"OCR result: {result}")
```

You should see `Windows 10 or above detected, try to use DirectML as primary provider` in the logs.

---

## 4. Configure Umaplay

### Option A: Auto-detect (Default)

By default, `GPU_BACKEND=auto` tries CUDA first, then DirectML, then CPU. If you only have an AMD GPU and DirectML is installed, it will be selected automatically — **no configuration needed**.

### Option B: Force DirectML

```bash
set Umaplay_GPU_BACKEND=dml
python main.py
```

---

## 5. Verify It's Working

When Umaplay starts (`python main.py`), look for these log lines:

**GPU device detection:**
```
[GPU] Backend=auto → device=privateuseone:0  (CUDA=False, DirectML=True)
```

**YOLO on DirectML:**
```
[DirectML] ONNX session using ['DmlExecutionProvider', 'CPUExecutionProvider']
```

**OCR on DirectML:**
```
[PERCEPTION] Using ONNX OCR engine (DirectML GPU)
Windows 10 or above detected, try to use DirectML as primary provider
```

> The `CPUExecutionProvider` listed alongside `DmlExecutionProvider` is normal — it's ONNX Runtime's required fallback. DirectML is the **primary** provider (listed first).

---

## Valid GPU Backend Values

| Value  | Behaviour                                        |
|--------|--------------------------------------------------|
| `auto` | Try CUDA → DirectML → CPU (default)              |
| `cuda` | Force NVIDIA CUDA, fallback to CPU if unavailable|
| `dml`  | Force DirectML, fallback to CPU if unavailable   |
| `cpu`  | Always use CPU (disable GPU acceleration)        |

---

## How OCR Engine Selection Works

The bot **automatically** selects the best OCR engine:

| Condition | OCR Engine Used |
|-----------|-----------------|
| `onnxruntime-directml` + `rapidocr_onnxruntime` installed | **RapidOCR** (PP-OCRv4 ONNX, GPU via DirectML) |
| Only standard packages installed | **PaddleOCR** (CPU) |

No manual configuration is needed — it's automatic based on installed packages.

---

## Known Limitations

- **PyTorch version**: `torch-directml` currently supports up to PyTorch 2.4.x.
- **Performance**: DirectML is generally faster than CPU but slower than native CUDA on equivalent NVIDIA hardware.
- **Training**: DirectML support is inference-only. This is fine for Umaplay since it only runs inference.
- **torch DLL conflict**: `torch-directml` conflicts with `modelscope` (a PaddleOCR dependency). This is why the bot uses RapidOCR instead of PaddleOCR on AMD systems.

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

### `DLL load failed while importing _C`
This happens when `requirements.txt` overwrites `torch-directml` with standard `torch`. Fix by reinstalling:
```bash
pip install -r requirements-amd.txt --force-reinstall --no-deps
```

---

## Credits

| Project | Role | Link |
|---------|------|------|
| [Umaplay](https://github.com/Magody/Umaplay) | Original bot by Magody | All game logic, UI, NVIDIA support |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) | ONNX OCR engine (PP-OCRv4) | Apache 2.0 |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Original OCR models by Baidu | Apache 2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Inference runtime by Microsoft | MIT |
| [DirectML](https://github.com/microsoft/DirectML) | GPU acceleration by Microsoft | MIT |
