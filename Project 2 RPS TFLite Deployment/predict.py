"""
predict_folder.py
─────────────────
Loads every image from INPUT_FOLDER, applies CGI-like preprocessing to better
match the training distribution, runs Rock-Paper-Scissors TFLite inference,
overlays the prediction label and per-class score bars, then saves to OUTPUT_FOLDER.

Usage:
    python predict_folder.py                        # uses defaults below
    python predict_folder.py --input captures/ --output results/

Requirements: pip install numpy opencv-python
  + either tflite-runtime  OR  tensorflow (for the interpreter)
"""

import argparse
import os
import sys
import numpy as np
import cv2

# ── Config ────────────────────────────────────────────────────
MODEL_PATH    = "RPS_full_int8-final.tflite"
INPUT_FOLDER  = "captures"      # folder with source images
OUTPUT_FOLDER = "output_images" # folder for labelled results
INPUT_SIZE    = 32              # model input width/height
CLASS_NAMES   = ["rock", "paper", "scissors"]
BAR_COLORS    = [(100, 100, 255), (100, 255, 100), (255, 100, 100)]  # BGR
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
# ──────────────────────────────────────────────────────────────


def load_interpreter(model_path: str):
    """Load TFLite model using TensorFlow's built-in interpreter."""
    try:
        import tensorflow as tf
    except ImportError:
        print("[ERROR] TensorFlow is required to run this script on Windows.")
        print("        pip install tensorflow")
        sys.exit(1)

    Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a camera capture into a more CGI-like input:
        BGR → grayscale → contrast enhancement → hand mask cleanup
        → white background composite → resize 32×32 → uint8 tensor
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    hand_only = cv2.bitwise_and(gray, gray, mask=mask)
    white_bg = np.full_like(gray, 255)
    white_bg[mask > 0] = hand_only[mask > 0]

    resized = cv2.resize(white_bg, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0).reshape(1, INPUT_SIZE, INPUT_SIZE, 1)


def run_inference(interp, input_data: np.ndarray) -> list:
    """Return list of 3 uint8 scores [rock, paper, scissors]."""
    in_details  = interp.get_input_details()
    out_details = interp.get_output_details()

    input_dtype = in_details[0]["dtype"]
    if input_dtype in (np.int8, np.uint8):
        input_scale, input_zero_point = in_details[0]["quantization"]
        q = np.round(input_data / input_scale + input_zero_point)
        info = np.iinfo(input_dtype)
        input_data = np.clip(q, info.min, info.max).astype(input_dtype)
    else:
        input_data = input_data.astype(input_dtype)

    interp.set_tensor(in_details[0]['index'], input_data)
    interp.invoke()

    scores = interp.get_tensor(out_details[0]['index'])[0]  # shape (3,)
    return [int(s) for s in scores]


def draw_overlay(img_bgr: np.ndarray, scores: list, class_idx: int) -> np.ndarray:
    """Draw prediction label at top and score bars at the bottom."""
    out  = img_bgr.copy()
    h, w = out.shape[:2]

    # ── Score bars ────────────────────────────────────────────
    bar_max_w = min(250, w - 20)
    bar_h     = max(22, h // 18)
    x0        = 10
    pad       = 5
    total_bar_h = len(CLASS_NAMES) * (bar_h + pad) + pad
    y_start   = h - total_bar_h

    # semi-transparent background panel behind bars
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y_start - 5), (bar_max_w + 20, h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    for i, (name, color) in enumerate(zip(CLASS_NAMES, BAR_COLORS)):
        score = scores[i]
        y     = y_start + pad + i * (bar_h + pad)

        # track
        cv2.rectangle(out, (x0, y), (x0 + bar_max_w, y + bar_h),
                      (60, 60, 60), -1)
        # filled portion
        filled = int(bar_max_w * score / 255)
        if filled > 0:
            cv2.rectangle(out, (x0, y), (x0 + filled, y + bar_h), color, -1)

        # highlight winning class bar with a border
        if i == class_idx:
            cv2.rectangle(out, (x0, y), (x0 + bar_max_w, y + bar_h),
                          (255, 255, 255), 1)

        # text label
        font_scale = max(0.45, bar_h / 50)
        label = f"{name}: {score}"
        cv2.putText(out, label, (x0 + 4, y + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x0 + 4, y + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # ── Prediction label at top ───────────────────────────────
    pred_text  = f"Prediction: {CLASS_NAMES[class_idx]} ({scores[class_idx]})"
    font_scale = max(0.7, w / 500)
    thickness  = max(2, w // 250)

    # measure text to size the background box
    (tw, th), baseline = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness)
    overlay2 = out.copy()
    cv2.rectangle(overlay2, (0, 0), (tw + 20, th + baseline + 14),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, out, 0.45, 0, out)

    cv2.putText(out, pred_text, (10, th + 8),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, pred_text, (10, th + 8),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 255, 0), thickness, cv2.LINE_AA)

    return out


def process_folder(input_folder: str, output_folder: str, model_path: str):
    if not os.path.isdir(input_folder):
        print(f"[ERROR] Input folder not found: {input_folder}")
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading model: {model_path}")
    interp = load_interpreter(model_path)

    image_files = sorted([
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])

    if not image_files:
        print(f"[WARN] No images found in '{input_folder}'")
        return

    print(f"Found {len(image_files)} image(s) in '{input_folder}'\n")

    for fname in image_files:
        in_path = os.path.join(input_folder, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"  [skip] Cannot read {fname}")
            continue

        # inference
        input_data = preprocess(img)
        scores     = run_inference(interp, input_data)
        class_idx  = int(np.argmax(scores))

        # draw
        result = draw_overlay(img, scores, class_idx)

        # save
        stem, ext = os.path.splitext(fname)
        out_name  = f"{stem}_pred_{CLASS_NAMES[class_idx]}{ext}"
        out_path  = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, result)

        print(f"  {fname:40s}  →  {CLASS_NAMES[class_idx]:8s}  "
              f"rock={scores[0]:3d}  paper={scores[1]:3d}  scissors={scores[2]:3d}"
              f"  →  saved: {out_name}")

    print(f"\nDone. Results saved to '{output_folder}/'")


def parse_args():
    p = argparse.ArgumentParser(description="Batch RPS inference on a folder of images")
    p.add_argument("--input",  default=INPUT_FOLDER,  help="Folder with input images")
    p.add_argument("--output", default=OUTPUT_FOLDER, help="Folder to save labelled images")
    p.add_argument("--model",  default=MODEL_PATH,    help="Path to .tflite model")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_folder(args.input, args.output, args.model)
