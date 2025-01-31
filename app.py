from flask import Flask, request, send_file
from PIL import Image
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import io
import os
import requests
import numpy as np

app = Flask(__name__)

# Define the model path and the external URL
model_path = 'weights/RealESRGAN_x4plus.pth'
model_url = 'https://your-file-hosting-service.com/path/to/RealESRGAN_x4plus.pth'

# Ensure the weights directory exists
os.makedirs('weights', exist_ok=True)

# Download the model if it's not already present
if not os.path.exists(model_path):
    print("Model not found locally. Downloading...")
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully.")

# Initialize the Real-ESRGAN model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True
)

@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'image' not in request.files:
        return 'No image file provided', 400

    file = request.files['image']
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return f'Invalid image file: {e}', 400

    # Process the image with Real-ESRGAN
    img_np = np.array(img)
    output, _ = upsampler.enhance(img_np, outscale=4)

    # Convert the output to a PIL image
    output_img = Image.fromarray(output)

    # Save the output image to a BytesIO object
    img_io = io.BytesIO()
    output_img.save(img_io, format='PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
