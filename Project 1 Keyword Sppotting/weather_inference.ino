/* Edge Impulse + WiFi Weather API
 * Arduino Nano 33 BLE Sense Rev2
 * Recognizes "Tokyo" and "London" then fetches weather
 */

#include <PDM.h>
#include <hasnaynajmal-project-1_v2_inferencing.h>
#include <WiFiNINA.h>
#include <ArduinoJson.h>

// ── WiFi & API credentials ────────────────────────────────────────────────────
const char* WIFI_SSID     = "TP-Link_5754";
const char* WIFI_PASSWORD = "54813144";
const char* API_KEY       = "b80deb65ce0ce56354dcfcafeba652e8";
const float CONFIDENCE    = 0.80;          // 80 % threshold to trigger API call

// ── Audio buffer ──────────────────────────────────────────────────────────────
typedef struct {
    int16_t *buffer;
    uint8_t  buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false;

// ── Forward declarations ──────────────────────────────────────────────────────
static void pdm_data_ready_inference_callback(void);
static bool microphone_inference_start(uint32_t n_samples);
static bool microphone_inference_record(void);
static int  microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr);
static void microphone_inference_end(void);
void        fetchWeather(const char* city);

// =============================================================================
void setup() {
    Serial.begin(115200);
    while (!Serial);
    Serial.println("Edge Impulse Inferencing + Weather Demo");

    // Connect to WiFi once at startup
    Serial.print("Connecting to WiFi: ");
    Serial.println(WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected!");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nWiFi connection FAILED — inference will still run.");
    }

    // Inference settings
    Serial.println("Inferencing settings:");
    Serial.print("  Interval: "); Serial.print((float)EI_CLASSIFIER_INTERVAL_MS); Serial.println(" ms.");
    Serial.print("  Frame size: "); Serial.println(EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    Serial.print("  Sample length: "); Serial.print(EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16); Serial.println(" ms.");
    Serial.print("  No. of classes: "); Serial.println(sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        Serial.println("ERR: Could not allocate audio buffer");
        return;
    }
    Serial.println("\nListening... Say 'What is the weather in Tokyo' or 'What is the weather in London'");
}

// =============================================================================
void loop() {
    bool m = microphone_inference_record();
    if (!m) {
        Serial.println("ERR: Failed to record audio");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data     = &microphone_audio_signal_get_data;

    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        Serial.print("ERR: Failed to run classifier ("); Serial.print(r); Serial.println(")");
        return;
    }

    // Print all predictions
    Serial.print("Predictions (DSP: ");
    Serial.print(result.timing.dsp);
    Serial.print(" ms., Classification: ");
    Serial.print(result.timing.classification);
    Serial.println(" ms.):");

    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        Serial.print("  ");
        Serial.print(result.classification[ix].label);
        Serial.print(": ");
        Serial.println(result.classification[ix].value);
    }

    // Check for high-confidence city detection
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        const char* label = result.classification[ix].label;
        float       value = result.classification[ix].value;

        if (value >= CONFIDENCE) {
            if (strcmp(label, "london") == 0) {
                Serial.println("\n>>> LONDON detected! Fetching weather...");
                fetchWeather("London");
                delay(5000);   // wait 5 s before listening again
                break;
            } else if (strcmp(label, "tokyo") == 0) {
                Serial.println("\n>>> TOKYO detected! Fetching weather...");
                fetchWeather("Tokyo");
                delay(5000);
                break;
            }
        }
    }
}

// =============================================================================
void fetchWeather(const char* city) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi not connected — cannot fetch weather.");
        return;
    }

    WiFiClient client;
    const char* host = "api.openweathermap.org";

    if (!client.connect(host, 80)) {
        Serial.println("Connection to weather API failed.");
        return;
    }

    // Build GET request
    String url = "/data/2.5/weather?q=";
    url += city;
    url += "&units=metric&appid=";
    url += API_KEY;

    client.print(String("GET ") + url + " HTTP/1.1\r\n" +
                 "Host: " + host + "\r\n" +
                 "Connection: close\r\n\r\n");

    // Wait for response
    unsigned long timeout = millis();
    while (client.available() == 0) {
        if (millis() - timeout > 5000) {
            Serial.println("Request timed out.");
            client.stop();
            return;
        }
    }

    // Skip HTTP headers
    String response = "";
    while (client.available()) {
        String line = client.readStringUntil('\n');
        if (line == "\r") break;          // end of headers
    }
    // Read JSON body
    while (client.available()) {
        response += client.readString();
    }
    client.stop();

    // Parse JSON
    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, response);
    if (error) {
        Serial.print("JSON parse failed: ");
        Serial.println(error.c_str());
        return;
    }

    // Extract fields
    const char* cityName    = doc["name"];
    float       temp        = doc["main"]["temp"];
    float       feels_like  = doc["main"]["feels_like"];
    int         humidity    = doc["main"]["humidity"];
    const char* description = doc["weather"][0]["description"];

    Serial.println("============================");
    Serial.print("Weather in "); Serial.println(cityName);
    Serial.print("Temperature : "); Serial.print(temp);     Serial.println(" °C");
    Serial.print("Feels like  : "); Serial.print(feels_like); Serial.println(" °C");
    Serial.print("Humidity    : "); Serial.print(humidity);  Serial.println(" %");
    Serial.print("Condition   : "); Serial.println(description);
    Serial.println("============================\n");
}

// =============================================================================
// Audio helper functions
// =============================================================================
static void pdm_data_ready_inference_callback(void) {
    int bytesAvailable = PDM.available();
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);
    if (inference.buf_ready == 0) {
        for (int i = 0; i < bytesRead >> 1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];
            if (inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}

static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer   = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (inference.buffer == NULL) return false;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;
    PDM.onReceive(&pdm_data_ready_inference_callback);
    PDM.setBufferSize(4096);
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        Serial.println("Failed to start PDM!");
        microphone_inference_end();
        return false;
    }
    PDM.setGain(127);
    return true;
}

static bool microphone_inference_record(void) {
    inference.buf_ready = 0;
    inference.buf_count = 0;
    while (inference.buf_ready == 0) { delay(10); }
    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}

static void microphone_inference_end(void) {
    PDM.end();
    free(inference.buffer);
}
