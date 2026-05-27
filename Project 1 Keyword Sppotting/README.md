# Project 1: Keyword Spotting for Weather Queries

**Course:** IT00CS34 Edge Computing for ML, Spring 2026

A keyword spotting model trained on Edge Impulse to recognize city names ("Tokyo" / "London") spoken in the phrase *"What is the weather in ..."*, running on an **Arduino Nano 33 BLE Sense Rev2**. The sketch also includes a WiFi weather fetch via the OpenWeatherMap API, but this feature could not be completed on the hardware as the Arduino Nano 33 BLE Sense Rev2 does not include a WiFi chip.

---

## Project Structure

```
Project 1 Keyword Sppotting/
├── weather_inference.ino                    # Arduino sketch: inference + WiFi weather fetch
├── model.h                                  # Edge Impulse model as C header
├── convert_and_augment.py                   # Audio conversion (MP3→WAV) and augmentation
└── hasnaynajmal-project-1_v2_inferencing/   # Edge Impulse Arduino library (auto-generated)
    ├── library.properties
    └── src/
        ├── hasnaynajmal-project-1_v2_inferencing.h
        ├── edge-impulse-sdk/
        ├── model-parameters/
        └── tflite-model/
```

---

## Model

| Property | Value |
|----------|-------|
| Platform | Edge Impulse |
| Keywords | `tokyo`, `london` |
| Hardware | Arduino Nano 33 BLE Sense Rev2 |
| Confidence threshold | 80% |
| Input | PDM microphone audio |

---

## Dataset and Augmentation

Audio samples for "Tokyo" and "London" were collected and augmented using `convert_and_augment.py`. The script converts MP3 files to WAV and generates 9 augmented variants per sample:

| Augmentation | Description |
|-------------|-------------|
| `pitch_up` / `pitch_down` | Shift pitch by +2 / -2 semitones |
| `speed_up` / `speed_down` | Time-stretch by 1.15x / 0.85x |
| `noise` | Add low-level Gaussian noise |
| `vol_up` / `vol_down` | Scale volume up / down |
| `reverb` | Simulate room echo (6 echoes, 50 ms delay) |
| `pitch_speed` | Combine pitch shift and time stretch |

---

## Arduino Deployment

1. Install the Edge Impulse library by adding the `hasnaynajmal-project-1_v2_inferencing/` folder via **Sketch > Include Library > Add .ZIP Library** (or place it in your Arduino libraries folder).
2. Install **WiFiNINA** and **ArduinoJson** via the Library Manager.
3. Update `WIFI_SSID` and `WIFI_PASSWORD` in `weather_inference.ino`.
4. Upload to the **Arduino Nano 33 BLE Sense Rev2**.
5. Open Serial Monitor at **115200 baud**.
6. Say *"What is the weather in Tokyo"* or *"What is the weather in London"*.

When inference confidence exceeds 80%, the sketch attempts to fetch and print current weather conditions for the detected city via the OpenWeatherMap API. Note: this part of the sketch was written but could not be tested because the Arduino Nano 33 BLE Sense Rev2 does not have a WiFi chip. A board such as the MKR WiFi 1010 would be required to run the full weather fetch feature.

---

## Dependencies

- [Edge Impulse Arduino library](https://docs.edgeimpulse.com/docs/run-inference/arduino-library) (included in repo)
- **WiFiNINA** and **ArduinoJson** (Arduino Library Manager, required by the sketch but untestable on Nano 33 BLE Sense due to no WiFi hardware)
- Python: `librosa`, `soundfile`, `numpy` (for `convert_and_augment.py`)
