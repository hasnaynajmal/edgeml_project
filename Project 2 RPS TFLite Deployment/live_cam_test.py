"""
live_cam_test.py
────────────────
Sends 'c' to the Arduino over Serial, reads the grayscale frame
and the inference result, then displays the captured image with
the prediction overlaid. Press any key to capture again, Q/ESC to quit.

Requirements: pip install pyserial numpy opencv-python
"""

import serial
import serial.tools.list_ports
import numpy as np
import cv2
import time
from typing import List, Tuple, Optional

# ── Config ────────────────────────────────────────────────────
PORT          = 'COM5'    # ← change to your Arduino's port
BAUD          = 115200
CAM_WIDTH     = 160
CAM_HEIGHT    = 120
DISPLAY_SCALE = 4
READ_TIMEOUT  = 10
# ──────────────────────────────────────────────────────────────


def connect(port: str, baud: int) -> serial.Serial:  # type: ignore[return]
    try:
        ser = serial.Serial(port, baud, timeout=READ_TIMEOUT)
    except serial.SerialException as e:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        print(f"[ERROR] Cannot open {port}: {e}")
        if ports:
            print(f"  Available ports: {', '.join(ports)}")
            print("  Update PORT at the top of this script.")
        else:
            print("  No serial ports found")
        raise SystemExit(1)
    time.sleep(2)
    ser.reset_input_buffer()
    print(f"[OK] Connected to {port} @ {baud} baud")
    return ser


def request_frame(ser: serial.Serial) -> Tuple[Optional[np.ndarray], str, List[int]]:
    """Send 'c', read hex frame + scores line + prediction line."""
    ser.reset_input_buffer()
    ser.write(b'c')

    expected_chars = CAM_WIDTH * CAM_HEIGHT * 2
    hex_line = ""

    while len(hex_line) < expected_chars:
        chunk = ser.readline().decode('latin-1', errors='ignore').strip()
        if not chunk:
            continue
        try:
            bytes.fromhex(chunk)
            hex_line += chunk
        except ValueError:
            print(f"[skip] {chunk}")

    if len(hex_line) < expected_chars:
        print(f"[WARN] Short frame: {len(hex_line)//2}/{CAM_WIDTH*CAM_HEIGHT} bytes")
        return None, "", []

    raw   = bytes.fromhex(hex_line[:expected_chars])
    frame = np.frombuffer(raw, dtype=np.uint8).reshape((CAM_HEIGHT, CAM_WIDTH))

    scores     = []
    prediction = ""
    for _ in range(15):
        line = ser.readline().decode('latin-1', errors='ignore').strip()
        if line.startswith("Scores:"):
            # "Scores: 12,5,94"
            try:
                scores = [int(v) for v in line.split(":", 1)[1].strip().split(",")]
            except ValueError:
                pass
        elif line.startswith("Prediction"):
            prediction = line
            break

    return frame, prediction, scores


def overlay_text(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


CLASS_NAMES = ["rock", "paper", "scissors"]
BAR_COLORS  = [(100, 100, 255), (100, 255, 100), (255, 100, 100)]  # BGR


def draw_scores(img_bgr: np.ndarray, scores: List[int], prediction: str) -> np.ndarray:
    """Overlay score bars and prediction label on the image."""
    out     = img_bgr.copy()
    h, w    = out.shape[:2]
    bar_w   = 180   # max bar width in pixels
    bar_h   = 22
    x0      = 10
    y_start = h - (len(CLASS_NAMES) * (bar_h + 6)) - 10

    for i, (name, color) in enumerate(zip(CLASS_NAMES, BAR_COLORS)):
        score = scores[i] if i < len(scores) else 0
        y     = y_start + i * (bar_h + 6)

        # background track
        cv2.rectangle(out, (x0, y), (x0 + bar_w, y + bar_h), (40, 40, 40), -1)
        # filled bar (score 0-255 → 0-bar_w)
        filled = int(bar_w * score / 255)
        cv2.rectangle(out, (x0, y), (x0 + filled, y + bar_h), color, -1)
        # label
        label = f"{name}: {score}"
        cv2.putText(out, label, (x0 + 4, y + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x0 + 4, y + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Final prediction at top
    if prediction:
        cv2.putText(out, prediction, (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, prediction, (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return out


def main():
    ser = connect(PORT, BAUD)
    print("Press any key in the window to capture. Q or ESC to quit.\n")

    deadline = time.time() + 5
    while time.time() < deadline:
        line = ser.readline().decode('latin-1', errors='ignore').strip()
        if line:
            print(f"[board] {line}")
        if "Ready" in line:
            break

    while True:
        print("Capturing…")
        frame, prediction, scores = request_frame(ser)

        if frame is None:
            print("[WARN] No frame received, retrying…")
            continue

        print(f"  {prediction}  |  scores: {scores}")

        h, w    = frame.shape
        display = cv2.resize(frame, (w * DISPLAY_SCALE, h * DISPLAY_SCALE),
                             interpolation=cv2.INTER_NEAREST)
        display_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        display_bgr = draw_scores(display_bgr, scores, prediction)

        cv2.imshow("Capture (any key = next, Q = quit)", display_bgr)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()
    ser.close()
    print("Done.")


if __name__ == "__main__":
    main()
