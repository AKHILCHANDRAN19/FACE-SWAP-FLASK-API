import os
import sys
import io
import cv2
import torch
import types
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string, send_file

# Fix for torchvision.transforms.functional_tensor
if "torchvision.transforms.functional_tensor" not in sys.modules:
    import torchvision.transforms.functional as F
    dummy = types.ModuleType("torchvision.transforms.functional_tensor")
    dummy.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = dummy

# Import RealESRGANer and RRDBNet after fixing the issue
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = Flask(__name__)

# Constants
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = "weights/RealESRGAN_x4.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def download_model():
    """Download the model if it doesn't exist"""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        os.makedirs("weights", exist_ok=True)
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully!")

def initialize_model():
    """Initialize the Real-ESRGAN model"""
    download_model()
    
    return RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32),
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True if DEVICE == "cuda" else False
    )

# Initialize model
model = initialize_model()
model.model.to(DEVICE)

# Global variable to hold the upscaled image
upscaled_image = None

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Real-ESRGAN Image Upscaler</title>
  <style>
    body { font-family: Arial, sans-serif; text-align: center; background: #f5f5f5; margin-top: 50px; }
    .container { margin: auto; padding: 20px; width: 500px; background: white; border-radius: 8px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
    input { margin: 10px 0; }
    button { padding: 10px 20px; background: #4285f4; color: white; border: none; border-radius: 5px; cursor: pointer; }
    img { margin-top: 20px; max-width: 100%; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Real-ESRGAN Image Upscaler</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <br>
      <button type="submit">Upload &amp; Upscale</button>
    </form>
    {% if upscaled %}
      <h2>Upscaled Image</h2>
      <img src="{{ url_for('get_upscaled_image') }}" alt="Upscaled Image">
      <br>
      <a href="{{ url_for('get_upscaled_image') }}" download="upscaled.jpg">
        <button>Download Image</button>
      </a>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    global upscaled_image
    upscaled_image = None
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            img = Image.open(file.stream).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            output, _ = model.enhance(img_cv, outscale=4)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(output)
            img_io = io.BytesIO()
            result.save(img_io, "JPEG")
            img_io.seek(0)
            upscaled_image = img_io

    return render_template_string(HTML_TEMPLATE, upscaled=(upscaled_image is not None))

@app.route("/upscaled")
def get_upscaled_image():
    if upscaled_image:
        return send_file(upscaled_image, mimetype="image/jpeg")
    return "No image has been upscaled yet.", 404

if __name__ == "__main__":
    app.run(debug=True)
