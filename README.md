# EdgeML Projects

**Course:** IT00CS34 Edge Computing for ML, Spring 2026

---

## Project 1: Keyword Spotting

**Folder:** `Project 1 Keyword Sppotting/`

Keyword spotting model trained on Edge Impulse to recognize city names ("Tokyo" / "London") in speech, deployed on an **Arduino Nano 33 BLE Sense Rev2** using the PDM microphone. The sketch also includes a WiFi weather fetch via OpenWeatherMap, but could not be completed as the Nano 33 BLE Sense has no WiFi chip.

See [`Project 1 Keyword Sppotting/README.md`](Project%201%20Keyword%20Sppotting/README.md) for details.

---

## Project 2: Rock-Paper-Scissors TFLite Edge Deployment

**Folder:** `Project 2/`

CNN trained to classify Rock-Paper-Scissors gestures, compressed using pruning and post-training quantization (Full Int8, 16.70 KB), and deployed on an **Arduino Nano 33 BLE Sense** with an **OV767X camera**. Includes investigation of the CGI domain gap and a preprocessing pipeline to partially address it.

See [`Project 2/README.md`](Project%202/README.md) for details.
