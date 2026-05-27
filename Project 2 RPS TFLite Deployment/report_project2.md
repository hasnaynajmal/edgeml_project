# Project 2 Report: Rock-Paper-Scissors TFLite Edge Deployment

**Course:** IT00CS34 Edge Computing for ML, Spring 2026

| Name | Student Number |
|------|----------------|
| [Your Full Name] | [Your Student Number] |

**GitHub Repository:** [Insert your GitHub Classroom repository URL here]

---

## 1. Introduction

This report covers Project 2 of the Edge Computing for ML course. The objective was to train a convolutional neural network (CNN) to classify Rock-Paper-Scissors hand gestures, apply model compression (structured pruning and post-training quantization), and deploy the final compressed model on an Arduino Nano 33 BLE Sense microcontroller using TensorFlow Lite Micro.

Training was carried out in Google Colab using Python 3.12 and TensorFlow 2.x. The dataset was Laurence Moroney's Rock-Paper-Scissors dataset from TensorFlow Datasets (TFDS). After training and compression, the model was converted to a C byte-array header file (`model.h`) and embedded into an Arduino sketch for on-device inference paired with the OV767X camera module.

A significant practical challenge encountered during the project was the CGI domain gap: the training images are computer-generated (CGI) with clean white backgrounds, while real camera images contain backgrounds, shadows, and varying lighting. This gap caused consistent misclassification patterns that are documented throughout this report.

---

## 2. Performed Tasks

### 2.1 Implementation Environment

| Component | Details |
|-----------|---------|
| Training platform | Google Colab (Python 3.12, TensorFlow 2.x) |
| Inference hardware | Arduino Nano 33 BLE Sense |
| Camera module | OV767X (via Arduino_OV767X library) |
| Model compression library | tensorflow-model-optimization (tfmot) |
| Local inference | Python 3.12 venv at `E:\tf312\`, TensorFlow 2.21.0, OpenCV 4.x |

> **Note on local inference setup:** The `tflite-runtime` package is not distributed for Windows or Python versions above 3.12. A dedicated virtual environment was created at `E:\tf312\` with a full TensorFlow 2.21.0 installation to run the inference script locally. An additional complication was that Windows path length limits caused installation failures when using the default Python installation path. Creating the venv at a short root path resolved this.

### 2.2 Task Division

[Describe task division here, e.g., who was responsible for training, pruning, quantization, Arduino deployment]

---

### 2.3 Task 0: Baseline Model Training

**Dataset:** Laurence Moroney's Rock-Paper-Scissors TFDS dataset. 2520 training samples, split 90/10 into train and validation. A separate held-out test set was used for all accuracy evaluations.

**Model architecture:**

| Layer | Configuration |
|-------|--------------|
| Input | 32x32x1 grayscale |
| Conv2D + MaxPool2D | 8 filters, 3x3 kernel |
| Conv2D + MaxPool2D | 16 filters, 3x3 kernel |
| Conv2D + MaxPool2D | 32 filters, 3x3 kernel |
| Flatten | |
| Dense + Dropout | 32 units, dropout 0.3 |
| Dense (output) | 3 units, softmax |
| **Total parameters** | **10,115** |

Training was run for 10 epochs with a batch size of 32.

**Baseline results:**

| Metric | Value |
|--------|-------|
| Test accuracy | 87.37% |
| Test loss | 0.3441 |
| Model size (float32) | 43.43 KB |

---

### 2.4 Task 1A: Whole-Model Pruning

Whole-model pruning was applied using `tfmot.sparsity.keras.prune_low_magnitude` with a PolynomialDecay schedule. The `end_step` was set to 790, calculated from 10 epochs, batch size 32, and 2520 training samples. All layers were pruned.

**Results across sparsity levels:**

| Sparsity | Test Accuracy | Change vs. Baseline |
|----------|--------------|---------------------|
| Baseline | 0.8737 | |
| 25% | **0.9274** | +5.37% |
| 50% | 0.8253 | -4.84% |
| 75% | 0.7661 | -10.76% |
| 90% | 0.8441 | -2.96% |

**Key finding:** At 25% sparsity the model exceeded baseline accuracy by over 5 percentage points. This is consistent with pruning acting as a mild regularizer on a slightly overfitting model. Accuracy dropped significantly at 75% sparsity. The partial recovery at 90% compared to 75% suggests the optimizer found a sparse subnetwork with reasonable generalization, a known phenomenon sometimes called the "lottery ticket" effect.

---

### 2.5 Task 1B: Conv2D-Only (Layerwise) Pruning

In Task 1B, pruning was applied only to the three Conv2D layers, leaving the Dense layers untouched.

**Results across sparsity levels:**

| Sparsity | Test Accuracy | Change vs. Baseline |
|----------|--------------|---------------------|
| Baseline | 0.8737 | |
| 25% | 0.8387 | -3.50% |
| 50% | 0.8844 | +1.07% |
| 75% | 0.8387 | -3.50% |
| 90% | 0.7957 | -8.80% |

**Key finding:** Conv2D-only pruning was more stable at mid-range sparsities compared to whole-model pruning. The 50% result slightly exceeded the baseline. The identical accuracy at 25% and 75% suggests a plateau in the pruning effect on the convolutional layers at those levels. The large accuracy drop only appears at 90%, which is consistent with the convolutional feature extractors becoming too sparse to capture useful patterns.

---

### 2.6 Comparison: Task 1A vs Task 1B

| Sparsity | 1A (Whole Model) | 1B (Conv2D Only) |
|----------|-----------------|-----------------|
| Baseline | 0.8737 | 0.8737 |
| 25% | **0.9274** | 0.8387 |
| 50% | 0.8253 | **0.8844** |
| 75% | 0.7661 | **0.8387** |
| 90% | **0.8441** | 0.7957 |

At low sparsity (25%), whole-model pruning is clearly better. At higher sparsities (50%, 75%), Conv2D-only pruning preserves accuracy better because the Dense layers, which hold more representational capacity per parameter, are not pruned.

---

### 2.7 Task 2: Post-Training Quantization

Three quantization methods were applied to both the base model and the best pruned model. The pruned model selected was the 50% whole-model sparsity checkpoint from Task 1A.

**Results:**

| Method | Input Type | Model Size | Accuracy | Size vs. Float32 |
|--------|-----------|------------|----------|-----------------|
| Float32 (no quantization) | float32 | 43.43 KB | 0.8737 | baseline |
| 2A: Dynamic Range | float32 | 15.48 KB | 0.8360 | -64.4% |
| 2B: Int8 with float I/O | float32 | 17.04 KB | 0.8387 | -60.8% |
| 2C: Full Int8 | int8 | 16.70 KB | 0.8414 | -61.6% |
| Pruned + 2A | float32 | 15.48 KB | 0.8360 | -64.4% |
| Pruned + 2B | float32 | 17.04 KB | 0.8387 | -60.8% |
| Pruned + 2C | int8 | 16.70 KB | 0.8414 | -61.6% |

**Key finding:** All three methods achieved over 60% size reduction with only a 3-4% accuracy drop. A notable observation is that the pruned and unpruned models produced identical sizes after quantization. The reason is that weight pruning sets values to zero but does not remove them structurally. TFLite's quantization pipeline does not apply sparse tensor storage, so it compresses both models by the same amount. The memory benefit of pruning is not realized without explicit sparse format support.

**Full Int8 (2C) was selected for deployment** because it is the only format that is fully compatible with TFLite Micro for hardware inference without floating-point operations. The I/O tensors in this format are `int8`, which requires quantization-aware input preparation:

```python
input_scale, input_zero_point = in_details[0]["quantization"]
q = np.round(input_data / input_scale + input_zero_point)
input_data = np.clip(q, -128, 127).astype(np.int8)
```

---

### 2.8 Edge Deployment: Arduino Nano 33 BLE Sense

**Model conversion to C header:**

The final `RPS_full_int8-final.tflite` (16.70 KB) was converted to a C byte array using `xxd` and saved as `model.h`. The uncompressed header file is 105,510 bytes and is included directly in the Arduino sketch with `#include "model.h"`.

**TFLite Micro operation resolver:**

TFLite Micro requires every operation used by the model to be explicitly registered. The model uses 6 operations:

```cpp
static tflite::MicroMutableOpResolver<6> resolver;
resolver.AddQuantize();
resolver.AddConv2D();
resolver.AddMaxPool2D();
resolver.AddReshape();
resolver.AddFullyConnected();
resolver.AddSoftmax();
```

A tensor arena of 100 KB was allocated for inference buffers:

```cpp
constexpr int kTensorArenaSize = 100 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
```

**Capture and preprocessing on device:**

The OV767X camera captures a 160x120 QQVGA grayscale frame. This is downsampled to the 32x32 model input using nearest-neighbor interpolation:

```cpp
void preprocessFrame(byte* frame, uint8_t* out_buf) {
  for (int y = 0; y < INPUT_HEIGHT; y++) {
    for (int x = 0; x < INPUT_WIDTH; x++) {
      int src_x = x * CAMERA_WIDTH  / INPUT_WIDTH;
      int src_y = y * CAMERA_HEIGHT / INPUT_HEIGHT;
      out_buf[y * INPUT_WIDTH + x] = frame[src_y * CAMERA_WIDTH + src_x];
    }
  }
}
```

**Inference trigger and output:**

The sketch waits for a `'c'` character over Serial (115200 baud) to trigger a capture. After inference it prints the three class scores and the predicted label:

```
Scores: 12,5,238
Prediction: scissors (238)
```

A Python companion script (`live_cam_test.py`) sends the trigger, reads back the raw hex-encoded frame, and displays the captured image with score bars overlaid using OpenCV.

**Camera stability issues:**

During testing with the OV767X on the Arduino Nano 33 BLE Sense, dropped and corrupted frames were encountered. Some captures returned incomplete hex data, requiring the capture to be retried. A separate utility (`DisplayRawImage.py`) was used to decode raw hex frame dumps pasted from the Serial monitor, allowing individual frames to be inspected independently of the live script.

Because of these hardware stability issues, the test images used for the prediction evaluation in Section 3 were generated using an image generation tool rather than actual OV767X captures. This also meant the test images were visually closer to the CGI training distribution in some cases, which affected the prediction results.

---

### 2.9 CGI Domain Gap

**The problem:**

The Rock-Paper-Scissors TFDS dataset consists entirely of CGI-rendered hand images on a pure white background. Real camera images contain cluttered backgrounds, uneven lighting, shadows, and different skin tones. This domain mismatch caused the trained model to perform poorly on real-world images.

This was confirmed when testing the instructor-provided model (`random_uint8_model-new.tflite`) on real and mixed images: all predictions returned near-random confidence scores around 0.35 per class regardless of the gesture shown. The same effect was observed with the trained model on non-CGI images.

The main misclassification pattern observed was the model predicting **scissors** when it was uncertain. This happened with both paper and rock images:

- `paper_4.png` (true: paper) was predicted as **scissors**
- `rock_3.png` (true: rock) was predicted as **scissors**

A likely explanation is that the scissors class in the training data has spread-out finger silhouettes with dark gaps between them, which visually resembles the kind of edge noise and complex shapes present in real-world images. The model uses scissors as a default fallback for ambiguous inputs.

**What was done to address it:**

A preprocessing pipeline was implemented in `predict.py` to convert real camera images into a more CGI-like form before passing them to the model:

```python
def preprocess(img_bgr):
    # Step 1: Grayscale + contrast normalization
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Step 2: Blur + Otsu threshold to isolate hand
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: Composite hand on white background
    hand_only = cv2.bitwise_and(gray, gray, mask=mask)
    white_bg = np.full_like(gray, 255)
    white_bg[mask > 0] = hand_only[mask > 0]

    # Step 5: Resize to 32x32 and normalize
    resized = cv2.resize(white_bg, (32, 32), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0).reshape(1, 32, 32, 1)
```

The pipeline isolates the hand from the background and places it on a white background matching the training distribution. Despite this improvement, some errors remained. Otsu thresholding relies on a bimodal histogram, which is not guaranteed under all lighting conditions. A more robust fix would require training on a dataset that includes real camera images.

---

## 3. Results

### 3.1 Model Inference on Training-Distribution (CGI) Images

When the model was tested on images matching the CGI training distribution, it performed with high confidence:

| Image | True Label | Predicted | Confidence |
|-------|------------|-----------|------------|
| scissors (CGI) | scissors | scissors | 0.9961 |
| rock (CGI) | rock | rock | 0.9961 |
| paper (CGI) | paper | paper | 0.9961 |

### 3.2 Instructor Model Comparison

The instructor-provided model (`random_uint8_model-new.tflite`) was tested on three images to provide a reference point:

| Image | Rock Score | Paper Score | Scissors Score | Predicted |
|-------|-----------|-------------|----------------|-----------|
| test_image.png | 0.3477 | 0.2891 | 0.3672 | scissors |
| rock-cgi.jpg | 0.3516 | 0.2930 | 0.3516 | rock (tied) |
| paper-cgi.jpg | 0.3477 | 0.2891 | 0.3633 | scissors (wrong) |

All class scores are close to 0.33, confirming this model returns near-random predictions. It serves as a lower-bound comparison showing what an untrained model looks like.

### 3.3 Real-Image Prediction Results

Test images are in `images/paper/`, `images/rock/`, and `images/scissors/`. Prediction output images are in `images/predictions/`.

**Paper predictions:**

> **INSERT IMAGE:** `images/paper/paper_1.png` and `images/predictions/paper_1_pred_paper.png` (correct: paper)

> **INSERT IMAGE:** `images/paper/paper_4.png` and `images/predictions/paper_4_pred_scissors.png` (incorrect: paper predicted as scissors, CGI domain gap)

**Rock predictions:**

> **INSERT IMAGE:** `images/rock/rock_1.png` and `images/predictions/rock_1_pred_rock.png` (correct: rock)

> **INSERT IMAGE:** `images/rock/rock_3.png` and `images/predictions/rock_3_pred_scossors.png` (incorrect: rock predicted as scissors, CGI domain gap)

**Scissors predictions:**

> **INSERT IMAGE:** `images/scissors/scissor_1.png` and `images/predictions/scissor_1_pred_scissors.png` (correct: scissors)

**Full prediction summary:**

| Image File | True Label | Predicted Label | Correct |
|------------|------------|-----------------|---------|
| paper_1.png | paper | paper | Yes |
| paper_2.png | paper | paper | Yes |
| paper_3.png | paper | paper | Yes |
| paper_4.png | paper | **scissors** | **No** |
| paper_5.png | paper | paper | Yes |
| rock_1.png | rock | rock | Yes |
| rock_2.png | rock | rock | Yes |
| rock_3.png | rock | **scissors** | **No** |
| rock_4.png | rock | rock | Yes |
| scissor_1.png | scissors | scissors | Yes |
| scissor_2.png | scissors | scissors | Yes |
| scissor_3.png | scissors | scissors | Yes |
| scissor_4.png | scissors | scissors | Yes |
| scissor_5.png | scissors | scissors | Yes |

Overall accuracy: **12 / 14 correct (85.7%)**. Both errors were misclassifications as "scissors", which matches the expected pattern from the CGI domain gap.

---

## 4. Discussion and Conclusions

### 4.1 Pruning

The 25% whole-model sparsity result (0.9274) exceeding the baseline (0.8737) was the most interesting finding in the pruning experiments. It indicates the model was mildly overfitting and that removing low-magnitude weights acted as implicit regularization. The improvement disappeared beyond 25%, where accuracy degraded as expected.

Comparing 1A and 1B confirms that Dense layers are more sensitive to pruning than convolutional layers. At 50% and 75% sparsity, leaving the Dense layers intact (1B) produced noticeably better accuracy than pruning everything (1A).

A practical limitation is that pruning did not reduce model size after quantization. Without sparse storage support in TFLite, zero-valued weights are compressed the same as non-zero weights during quantization, making the memory footprint identical for pruned and unpruned models in this pipeline.

### 4.2 Quantization

All three quantization methods achieved a similar compression ratio (about 60-65% reduction). The accuracy penalty was consistent at around 3-4 percentage points across all methods. Full Int8 was chosen for deployment because TFLite Micro requires it for full compatibility. The difference in accuracy between quantization methods (0.8360 to 0.8414) was small enough to be practically irrelevant.

### 4.3 CGI Domain Gap

This was the dominant real-world challenge of the project. The model was nearly perfect on its training distribution but unreliable on real images. The preprocessing fix (Otsu segmentation + white background compositing) improved results but did not fully close the gap. Training on a dataset that includes real-world images would be the most reliable solution. An alternative would be data augmentation during training to simulate background noise and variable lighting.

The consistent misclassification pattern (both errors pointing to "scissors") is a useful diagnostic. It suggests the model is not randomly confused but has a systematic bias in the direction of the scissors class for out-of-distribution inputs.

### 4.4 Edge Deployment

The full pipeline from Colab training to on-device inference on the Arduino Nano 33 BLE Sense was completed successfully. The 16.70 KB Full Int8 model fit comfortably in flash, and a 100 KB tensor arena was sufficient for inference. The main reliability issue was the OV767X camera dropping frames. This did not affect the model itself but limited the ability to do extensive live camera testing.

---

## 5. Reflection

**Have you learned anything new?**

The CGI domain gap became very tangible through this project. Reading about distribution shift is different from watching a model that achieves near-perfect accuracy on training data immediately fail on real camera images. The preprocessing pipeline built to address it introduced practical use of histogram equalization, Otsu thresholding, and morphological operations as tools for domain adaptation.

The full int8 quantization pipeline and TFLite Micro integration were also new. Learning how to manually handle the int8 input quantization using the model's scale and zero point, and understanding why the Arduino sketch needs an explicit operation resolver with every op listed, gave a clearer picture of how much TFLite Micro differs from the full TFLite runtime.

**Did anything surprise you?**

The 25% sparsity result was surprising. Pruning is typically described as a compression technique that trades some accuracy for smaller model size. Gaining accuracy from pruning was not expected. The explanation in hindsight is regularization, but the magnitude of the improvement (+5.37%) was larger than anticipated.

The fact that pruned and base models were identical in size after quantization was also unexpected. The assumption going in was that zero weights would compress more efficiently. This turned out to be wrong: TFLite's quantization treats each weight as a dense value regardless of whether it is zero, and does not apply run-length or sparse encoding.

**Did you find anything challenging?**

The local inference environment setup was more difficult than expected. The `tflite-runtime` package does not have Windows builds for Python 3.12 or above. Creating a dedicated virtual environment with full TensorFlow at a short path to avoid Windows long-path failures added setup overhead that was not related to the core task.

Camera stability on the Arduino board was also a challenge. Dropped frames during capture required retries and made live testing inconsistent.

**Did you find anything satisfying?**

Completing the full end-to-end pipeline was satisfying. Training a model in Colab, compressing it to 16 KB, embedding it as a C array, and seeing the Arduino print a class prediction over Serial after pressing `c` confirmed that every step from data to hardware worked. The whole pipeline from dataset to microcontroller inference is a practical demonstration of what edge ML looks like in practice.

---

*End of Project 2 Report*
