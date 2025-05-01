import os
import time
import cv2
import numpy as np
from pathlib import Path
import onnxruntime
import argparse, textwrap


# -------------------------------------------------------------------
# Constants / Config
# -------------------------------------------------------------------
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MEAN = 0.2555          # scalar because you are using 1‑channel images
STD  = 0.1917
RESIZE_HW = (512, 512) # (H, W)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def load_model(onnx_model_path: str) -> onnxruntime.InferenceSession:
    """
    Loads an ONNX model using onnxruntime and returns the inference session.
    """
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model file not found at: {onnx_model_path}")
    
    session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"]  # or "CUDAExecutionProvider" if installed
    )
    return session

def load_images(images_dir: str) -> list[str]:
    """
    Collects all image filenames from a directory that match the given extensions.
    """
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    ]
    image_files.sort()
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    
    return image_files

def prepare_input_cv(image_path: str) -> np.ndarray:
    """
    Open image with OpenCV in GRAY mode, resize, normalise, and return
    (1, 1, H, W) float32 array ready for ONNX Runtime – **no extra copies**.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   # Horig x Worig, uint8
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")
    img = cv2.resize(img, RESIZE_HW[::-1], interpolation=cv2.INTER_AREA)  # W,H order for cv2
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img[None, None, ...]          # -> (1, 1, H, W) batch + channel
    return img                          # already float32


# -------------------------------------------------------------------
# Zero‑copy ONNX inference via IO Binding
# -------------------------------------------------------------------
def onnx_predict_fast(img_np: np.ndarray,
                      session: onnxruntime.InferenceSession) -> np.ndarray:
    """
    Uses IO Binding so the input NumPy buffer is given directly to ORT and the
    output buffer is allocated by us – avoids two mem‑copies.
    """
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    io_binding  = session.io_binding()
    io_binding.bind_input(name=input_name,
                          device_type='cpu', device_id=0,
                          element_type=np.float32,
                          shape=img_np.shape,
                          buffer_ptr=img_np.ctypes.data)
    # allocate output
    n, c, h, w   = img_np.shape
    out_np = np.empty((n, h, w), dtype=np.uint8)  # we know argmax→uint8
    io_binding.bind_output(name=output_name,
                           device_type='cpu', device_id=0,
                           element_type=np.float32,     # ORT will write logits
                           shape=(n, 2, h, w))          # num_classes=2 for Fast‑SCNN
    session.run_with_iobinding(io_binding)

    # argmax over channel dim in‑place (logits buffer is now in io_binding)
    logits = io_binding.copy_outputs_to_cpu()[0]         # (1, C, H, W) float32
    np.argmax(logits, axis=1, out=out_np)                # write into uint8 array
    return out_np[0]                                     # (H, W) uint8

def post_process_mask(
    original_image: np.ndarray, 
    pred_mask: np.ndarray
) -> np.ndarray:
    """
    Resizes the predicted mask to match the original image size, then draws
    contours and an optional convex hull on a copy of the original image.
    """
    # Resize predicted mask to original image size
    h, w = original_image.shape[:2]
    mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Draw contours
    contours, hierarchy = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = original_image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # Green boundary

    # Draw single hull around all contours if desired
    if len(contours) > 0:
        all_points = np.vstack(contours)
        big_hull = cv2.convexHull(all_points)
        cv2.drawContours(overlay, [big_hull], -1, (255, 255, 255), 2)

    return overlay

def save_result(output_path: str, processed_image: np.ndarray) -> None:
    """
    Saves the processed image (with drawn contours/hulls) to disk.
    """
    bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_image)

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main(images_dir      ="sample_images",
         outdir          ="data/test_result",
         onnx_model_path ="fastscnn.onnx",
         save_images     =True,
         post_process    =True):
    """
    Orchestrates preprocessing → ONNX inference → (optional) post‑processing.

    Parameters
    ----------
    images_dir : str
        Folder containing input images.
    outdir : str
        Destination folder for overlay images.
    onnx_model_path : str
        Path to .onnx or .ort model file.
    save_images : bool
        If False the script skips cv2.imwrite (good for FPS benchmarking).
    post_process : bool
        If False the script skips contour / hull drawing entirely.
    """
    
    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    # ----------------------------------------------------------------
    # 1. Load ONNX model
    # ----------------------------------------------------------------
    print("Loading ONNX model into onnxruntime...")
    try:
        session = load_model(onnx_model_path)
    except FileNotFoundError as e:
        print(e)
        return
    print("ONNX model loaded successfully!")

    # ----------------------------------------------------------------
    # 2. Load image filenames
    # ----------------------------------------------------------------
    try:
        image_files = load_images(images_dir)
    except FileNotFoundError as e:
        print(e)
        return

    num_frames = len(image_files)
    print(f"Found {num_frames} images in '{images_dir}'...")

    # ----------------------------------------------------------------
    # 3. Process images (timing starts)
    # ----------------------------------------------------------------
    start_time = time.time()

    for filename in image_files:
        image_path = os.path.join(images_dir, filename)

        # 3a. Prepare input
        input_image_np = prepare_input_cv(image_path)

        # 3b. Model inference
        pred_mask = onnx_predict_fast(input_image_np, session)

        
        # --- 4c. OPTIONAL post‑processing --------------------------------
        if post_process:
            original = cv2.imread(image_path)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            h, w = original.shape[:2]
            mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay = original.copy()
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

            if contours:
                # The hull below demonstartes the pixel coordinates of the fire boundary
                # These can be coverted to World Cordinates using the Pixel_to_World_Cordinate_Conversion.py script
                hull = cv2.convexHull(np.vstack(contours))
                cv2.drawContours(overlay, [hull], -1, (255, 255, 255), 2)
        else:
            overlay = None  # nothing to save
        # -----------------------------------------------------------------

        # 4d. OPTIONAL write result to disk
        if save_images and overlay is not None:
            out_name = f"{Path(filename).stem}_onnx_output.jpg"
            out_path = Path(outdir) / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    # ----------------------------------------------------------------
    # 5. Timing and performance stats
    # ----------------------------------------------------------------
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time if total_time else 0
    latency = total_time / num_frames if num_frames else 0

    print(f"\nProcessed {num_frames} images in {total_time:.4f} seconds.")
    print(f"Total FPS: {fps:.2f}")
    print(f"Average Latency per image: {latency:.4f} seconds")
    print("Done!")

# ---------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------
def build_cli():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Fast‑SCNN ONNX inference demo.
        Examples
        --------
        # baseline run
        python maint_onnx_modular.py

        # latency benchmark, no contours, no disk I/O
        python maint_onnx_modular.py --post_process 0 --no_save
        """))
    p.add_argument("--images_dir", default="sample_images")
    p.add_argument("--outdir", default="data/test_result")
    p.add_argument("--model", default="fastscnn.onnx")
    p.add_argument("--post_process", type=int, choices=[0, 1], default=1,
                   help="1=draw contours/hull, 0=skip")
    p.add_argument("--no_save", action="store_true",
                   help="do not write overlay images to disk")
    return p

if __name__ == "__main__":
    args = build_cli().parse_args()
    main(images_dir=args.images_dir,
         outdir=args.outdir,
         onnx_model_path=args.model,
         save_images=not args.no_save,
         post_process=bool(args.post_process))
