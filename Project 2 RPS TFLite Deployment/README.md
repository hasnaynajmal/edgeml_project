# Project 2: Rock-Paper-Scissors TFLite Edge Deployment

**Course:** IT00CS34 Edge Computing for ML, Spring 2026

A CNN trained to classify Rock-Paper-Scissors hand gestures, compressed using pruning and post-training quantization, and deployed on an **Arduino Nano 33 BLE Sense** with an **OV767X camera** using TensorFlow Lite Micro.

---

## Project Structure

```
Project 2/
├── model_training.ipynb          # Full training, pruning, and quantization pipeline (Colab)
├── predict.py                    # Batch inference script with CGI-like preprocessing
├── live_cam_test.py              # Serial communication + live camera display via OpenCV
├── DisplayRawImage.py            # Decodes raw hex frame dumps from Serial monitor
├── RPS_full_int8-final.tflite    # Final deployed model (Full Int8, 16.70 KB)
├── requirements.txt              # Python dependencies
├── report_project2.md            # Project report
├── images/
│   ├── paper/                    # Test images: paper_1.png to paper_5.png
│   ├── rock/                     # Test images: rock_1.png to rock_4.png
│   ├── scissors/                 # Test images: scissor_1.png to scissor_5.png
│   └── predictions/              # Prediction output images with overlaid labels
└── inference/
    ├── inference.ino             # Arduino TFLite Micro sketch
    ├── model.h                   # TFLite model as C byte array
    └── image_data.h              # Sample image data for testing
```

---

## Model

| Property | Value |
|----------|-------|
| Architecture | 3x Conv2D + MaxPool, Dense(32), Dense(3) |
| Input | 32x32 grayscale |
| Parameters | 10,115 |
| Quantization | Full Int8 (2C) |
| Model size | 16.70 KB |
| Baseline accuracy | 87.37% |

---

## Setup

> **Windows note:** `tflite-runtime` is not available for Windows. Use a full TensorFlow installation in a Python 3.12 virtual environment.

```bash
python -m venv E:\tf312
E:\tf312\Scripts\activate
pip install tensorflow opencv-python
```

---

## Usage

### Batch inference on a folder of images

```bash
E:\tf312\Scripts\python.exe predict.py --input images/rock --output images/predictions
```

### Live camera inference (requires Arduino connected over Serial)

```bash
# Update PORT in live_cam_test.py to match your Arduino's COM port, then:
E:\tf312\Scripts\python.exe live_cam_test.py
# Press any key to capture, Q or ESC to quit
```

---

## Arduino Deployment

1. Open `inference/inference.ino` in the Arduino IDE.
2. Ensure `model.h` and `image_data.h` are in the same folder.
3. Install the **Arduino_OV767X** and **TensorFlowLite** libraries via the Library Manager.
4. Upload to the **Arduino Nano 33 BLE Sense**.
5. Open Serial Monitor at **115200 baud** and send `c` to trigger a capture and inference.

---

## Key Findings

- 25% whole-model pruning improved accuracy from 87.37% to **92.74%** (mild regularization effect).
- All three quantization methods reduced model size by over 60% with a 3-4% accuracy penalty.
- Pruning did not reduce model size after quantization because TFLite does not use sparse storage.
- A CGI domain gap caused paper and rock images to be misclassified as scissors. A preprocessing pipeline (histogram equalization + Otsu segmentation + white background compositing) was implemented to partially address this.

---

## Dependencies

See `requirements.txt`. Key packages:

- `tensorflow >= 2.18.0`
- `tensorflow-model-optimization == 0.8.0`
- `opencv-python >= 4.11.0`
- `tensorflow-datasets >= 4.9.8`
