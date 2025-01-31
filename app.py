from flask import Flask, request, jsonify, send_file
import os
import torch
import numpy as np
import requests
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import uuid

app = Flask(__name__)

# Model settings
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = "RealESRGAN_x4plus.pth"

# Check if model exists, if not, download it
if not os.path.exists(MODEL_PATH):
    print("Downloading Real-ESRGAN model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully!")

# Load the Real-ESRGAN model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(scale=4, model_path=MODEL_PATH, model=model, tile=0, tile_pad=10, pre_pad=0, device=device)

@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    
    # Generate a unique filename
    file_ext = os.path.splitext(image_file.filename)[-1].lower()
    if file_ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return jsonify({"error": "Invalid file format"}), 400

    unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
    input_path = f"input_{unique_id}{file_ext}"
    output_path = f"output_{unique_id}.png"

    # Save input image
    image_file.save(input_path)

    # Load and process image
    img = Image.open(input_path).convert('RGB')
    img = np.array(img)

    try:
        # Perform upscaling
        output, _ = upsampler.enhance(img, outscale=4)
        output_img = Image.fromarray(output)
        output_img.save(output_path)
        return send_file(output_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup
        os.remove(input_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
