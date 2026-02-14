"""
Export PaddleOCR detection and recognition models to ONNX format.

Usage:
    python scripts/export_ocr_onnx.py

Prerequisites:
    pip install paddle2onnx==1.3.1

Output:
    models/ocr_det.onnx   - Text detection model
    models/ocr_rec.onnx   - Text recognition model
    models/ocr_keys.txt   - Character dictionary for CTC decoding
"""
import os
import sys
import json
import shutil

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def find_paddlex_model_dir(model_name: str) -> str:
    """Find cached PaddleX model directory."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".paddlex", "official_models", model_name)
    if os.path.isdir(cache_dir):
        return cache_dir
    raise FileNotFoundError(
        f"PaddleOCR model '{model_name}' not found at {cache_dir}.\n"
        "Run 'python main.py' once to let PaddleOCR download models automatically,\n"
        "then re-run this script."
    )


def export_model(model_dir: str, output_path: str, opset_version: int = 14):
    """Export a PaddlePaddle inference model to ONNX."""
    import paddle2onnx

    model_file = os.path.join(model_dir, "inference.json")
    params_file = os.path.join(model_dir, "inference.pdiparams")

    if not os.path.exists(model_file):
        # Fallback to .pdmodel format
        model_file = os.path.join(model_dir, "inference.pdmodel")

    print(f"  Model:  {model_file}")
    print(f"  Params: {params_file}")

    with open(model_file, "rb") as f:
        model_content = f.read()
    with open(params_file, "rb") as f:
        params_content = f.read()

    onnx_bytes = paddle2onnx.export(model_content, params_content, opset_version=opset_version)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_bytes)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported: {output_path} ({size_mb:.1f} MB)")


def extract_char_dict(rec_model_dir: str, output_path: str):
    """Extract the character dictionary from the recognition model config."""
    config_file = os.path.join(rec_model_dir, "config.json")
    if not os.path.exists(config_file):
        print(f"  WARNING: config.json not found at {config_file}")
        return

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Look for character_dict in PostProcess config
    postprocess = config.get("PostProcess", {})
    char_dict_path = postprocess.get("character_dict_path", "")

    if char_dict_path and os.path.isabs(char_dict_path) and os.path.exists(char_dict_path):
        shutil.copy2(char_dict_path, output_path)
        print(f"  Char dict copied: {output_path}")
    else:
        # Try to find it relative to the model dir or in paddleocr package
        import paddleocr
        pkg_dir = os.path.dirname(paddleocr.__file__)
        # Common locations for English char dict
        candidates = [
            os.path.join(rec_model_dir, "en_dict.txt"),
            os.path.join(rec_model_dir, "ppocr_keys_v1.txt"),
            os.path.join(pkg_dir, "ppocr", "utils", "en_dict.txt"),
            os.path.join(pkg_dir, "ppocr", "utils", "ppocr_keys_v1.txt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                shutil.copy2(c, output_path)
                print(f"  Char dict copied from {c}: {output_path}")
                return

        # If still not found, extract from config
        char_list = postprocess.get("character_dict", None)
        if char_list:
            with open(output_path, "w", encoding="utf-8") as f:
                for ch in char_list:
                    f.write(ch + "\n")
            print(f"  Char dict extracted from config: {output_path}")
        else:
            print("  WARNING: Could not find character dictionary!")
            print("  You may need to manually provide models/ocr_keys.txt")


def main():
    det_model_name = "PP-OCRv5_mobile_det"
    rec_model_name = "en_PP-OCRv5_mobile_rec"

    models_dir = os.path.join(PROJECT_ROOT, "models")

    print("=" * 60)
    print("PaddleOCR -> ONNX Export")
    print("=" * 60)

    # 1. Export detection model
    print(f"\n[1/3] Exporting detection model ({det_model_name})...")
    det_dir = find_paddlex_model_dir(det_model_name)
    export_model(det_dir, os.path.join(models_dir, "ocr_det.onnx"))

    # 2. Export recognition model
    print(f"\n[2/3] Exporting recognition model ({rec_model_name})...")
    rec_dir = find_paddlex_model_dir(rec_model_name)
    export_model(rec_dir, os.path.join(models_dir, "ocr_rec.onnx"))

    # 3. Extract character dictionary
    print("\n[3/3] Extracting character dictionary...")
    extract_char_dict(rec_dir, os.path.join(models_dir, "ocr_keys.txt"))

    print("\n" + "=" * 60)
    print("Export complete! ONNX models saved to models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
