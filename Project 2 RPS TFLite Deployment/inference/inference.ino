#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino_OV767X.h>
#include "model.h"

#define CAMERA_WIDTH  160
#define CAMERA_HEIGHT 120
#define INPUT_WIDTH   32
#define INPUT_HEIGHT  32

// Reduce arena to leave room for everything else
constexpr int kTensorArenaSize = 100 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

byte frame_buffer[CAMERA_WIDTH * CAMERA_HEIGHT];

const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

const char* class_names[] = {"rock", "paper", "scissors"};

void preprocessFrame(byte* frame, uint8_t* out_buf) {
  for (int y = 0; y < INPUT_HEIGHT; y++) {
    for (int x = 0; x < INPUT_WIDTH; x++) {
      int src_x = x * CAMERA_WIDTH  / INPUT_WIDTH;
      int src_y = y * CAMERA_HEIGHT / INPUT_HEIGHT;
      out_buf[y * INPUT_WIDTH + x] = frame[src_y * CAMERA_WIDTH + src_x];
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!Camera.begin(QQVGA, GRAYSCALE, 1)) {
    Serial.println("Camera init failed!");
    while (1);
  }
  Serial.println("Camera ready!");

  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model mismatch!");
    while (1);
  }

  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddQuantize();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Ready! Send c to capture.");
}

void loop() {
  if (Serial.available() && Serial.read() == 'c') {

    Camera.readFrame(frame_buffer);

    // Print raw frame as hex
    int numPixels = CAMERA_WIDTH * CAMERA_HEIGHT;
    for (int i = 0; i < numPixels; i++) {
      byte p = frame_buffer[i];
      if (p < 0x10) Serial.print('0');
      Serial.print(p, HEX);
    }
    Serial.println();
    Serial.println();

    uint8_t processed[INPUT_WIDTH * INPUT_HEIGHT];
    preprocessFrame(frame_buffer, processed);

    for (int i = 0; i < INPUT_WIDTH * INPUT_HEIGHT; i++) {
      input->data.uint8[i] = processed[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }

    uint8_t max_score   = output->data.uint8[0];
    int predicted_class = 0;
    for (int i = 1; i < 3; i++) {
      if (output->data.uint8[i] > max_score) {
        max_score       = output->data.uint8[i];
        predicted_class = i;
      }
    }

    // Send all three scores then the final prediction
    Serial.print("Scores: ");
    Serial.print(output->data.uint8[0]); Serial.print(",");
    Serial.print(output->data.uint8[1]); Serial.print(",");
    Serial.println(output->data.uint8[2]);

    Serial.print("Prediction: ");
    Serial.print(class_names[predicted_class]);
    Serial.print(" (");
    Serial.print(max_score);
    Serial.println(")");
  }
}