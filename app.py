import os
import torch
import requests
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# === PATCHING BASICSR TO FIX IMPORT ERROR ===
def patch_basicsr():
    import sys
    import fileinput

    basicsr_path = next((p for p in sys.path if "basicsr" in p), None)
    if basicsr_path:
        degradations_file = os.path.join(basicsr_path, "data", "degradations.py")
        if os.path.exists(degradations_file):
            with fileinput.FileInput(degradations_file, inplace=True) as file:
                for line in file:
                    print(line.replace(
                        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                        "from torchvision.transforms.functional import rgb_to_grayscale"
                    ), end="")
            print("Patched basicsr successfully!")

patch_basicsr()

# Flask app
app = Flask(__name__)

# Model URL and Path
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = "RealESRGAN_x4plus.pth"

# Automatically download the model if not found
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully!")

# Load Model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4, 
    model_path=MODEL_PATH, 
    model=model, 
    tile=0, 
    tile_pad=10, 
    pre_pad=0, 
    half=True if torch.cuda.is_available() else False
)

# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Function to check allowed file type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upscale", methods=["POST"])
def upscale_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Load and process image
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)

        # Perform upscaling
        output, _ = upsampler.enhance(img_np, outscale=4)
        output_img = Image.fromarray(output)

        # Convert to bytes
        img_io = BytesIO()
        output_img.save(img_io, format="PNG")
        img_io.seek(0)

        return jsonify({"message": "Upscaling successful"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
