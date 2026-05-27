import numpy as np
from PIL import Image
import os

# Configuration
image_width = 160
image_height = 120
color_mode = 'L'  # Grayscale
hex_file_path = './hex_data.txt'
output_folder = 'output_images'
output_filename = 'test_image.png'

def read_image_from_hex_file(file_path, width, height, mode):
    with open(file_path, 'r') as file:
        hex_data = file.read().replace(' ', '').replace('\n', '')

    print("Image data received. Decoding...")

    raw_data = bytes.fromhex(hex_data)
    print(f"Raw bytes: {len(raw_data)}, expected: {width * height}")

    if len(raw_data) != width * height:
        raise ValueError(
            f"Size mismatch: got {len(raw_data)} bytes, "
            f"expected {width * height} for {width}x{height} grayscale"
        )

    gray = np.frombuffer(raw_data, dtype=np.uint8)
    return Image.fromarray(gray.reshape((height, width)), mode)


def save_image(image, folder, filename):
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)
    image.save(save_path)
    assert os.path.isfile(save_path)
    print(f"Image saved to: {save_path}")


# Main execution
image = read_image_from_hex_file(hex_file_path, image_width, image_height, color_mode)
save_image(image, output_folder, output_filename)
image.show()