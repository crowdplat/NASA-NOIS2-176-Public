# Fire Boundary Detection Demo

A lightweight, **pure‑Python** pipeline for running a fire-boundary detection model (exported to ONNX) on still images.  
Everything is self‑contained: no Torch in the hot path, only **OpenCV + NumPy + ONNX Runtime**.

---

## 1  Quick start

| Requirement | Version |
|-------------|---------|
| **Python**  | **3.10.16 or newer** |
| **pip**     | 22 + (recommended) |

```bash
# 1 — clone / copy the repo and open a terminal in that folder
cd path/to/deployement_baseline_tests

# 2 — (optional) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3 — install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4 — prepare assets
#   sample_images/    ← put any .jpg/.png here
#   fastscnn.onnx     ← exported model (add manually)

# 5 — run
python main_onnx_modular.py

# 6- run world cordinate conversion script with example data inside it
python Pixel_to_World_Cordinate_Conversion.py
```

Example output:

```
Processed 25 images in 1.12 s
Total FPS: 22.3
Average Latency per image: 0.050 s
Done!
```

Overlay images appear in `data/test_result/`.

> **Tip:** The script treats relative paths as relative to the shell’s current directory. The simplest workflow is to `cd` into the project folder before running. Otherwise, pass absolute paths with `--images_dir`, `--outdir`, etc.

---

## 2  Project structure

```

deployement_baseline_tests/
├─ main_onnx_modular.py   ← main, optimised script (IO‑Binding, zero‑copy)
├─ fastscnn.onnx           ← model weights (you supply)
├─ Pixel_to_World_Cordinate_Conversion.py        ← pixel to world cordinates conversion script
├─ requirements.txt        ← OpenCV, NumPy, onnxruntime‑cpu
├─ sample_images/          ← input images (add more here)
└─ data/
   └─ test_result/         ← output overlays (auto‑created)
```

---

## 3  Command‑line flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--images_dir PATH` | Input image folder | `sample_images` |
| `--outdir PATH` | Output folder for overlays | `data/test_result` |
| `--model PATH` | `.onnx` or `.ort` model file | `fastscnn.onnx` |
| `--post_process {0,1}` | 1 = draw contours/hull, 0 = skip | 1 |
| `--no_save` | Do **not** write images to disk | off |

Examples

```bash
# Latency benchmark (model‑only)
python main_onnx_modular.py --post_process 0 --no_save

# Custom folders
python main_onnx_modular.py \
       --images_dir "/abs/path/in" \
       --outdir     "/abs/path/out"
```

---

## 4  How it works

1. **Pre‑process** with OpenCV (`cv2.imread`, `cv2.resize`, normalise).  
2. **Zero‑copy inference** using ONNX Runtime IO‑Binding—no Python↔C++ copies.  
3. **Post‑process** (optional) – resize mask, draw contours & convex hull.  
4. **Save** (or skip) based on flags.

Inline comments explain every micro‑optimisation.

---

## 5  Performance tips

| Tip | Gain |
|-----|------|
| Run with `--post_process 0 --no_save` for pure inference latency. | Removes OpenCV & disk I/O |
| Using an NVIDIA GPU? Install `onnxruntime-gpu` and it will pick the CUDA provider automatically. | 4–8× throughput |

---

## 6  Troubleshooting

| Issue | Fix |
|-------|-----|
| `Images directory not found` | `cd` into the repo folder or pass an absolute `--images_dir`. |
| `ModuleNotFoundError: onnxruntime` | `pip install -r requirements.txt` |

---


**Happy fire boundari detecting!**
