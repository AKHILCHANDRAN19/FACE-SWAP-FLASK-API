import os
import sys
from torchvision.transforms import functional
sys.modules["torchvision.transforms.functional_tensor"] = functional

from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import io

# Initialize Flask app
app = Flask(__name__)

# Download Required Models (if not already downloaded)
if not os.path.exists('realesr-general-x4v3.pth'):
    os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")
if not os.path.exists('GFPGANv1.2.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth -P .")
if not os.path.exists('GFPGANv1.3.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P .")
if not os.path.exists('GFPGANv1.4.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")
if not os.path.exists('RestoreFormer.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth -P .")

# Load RealESRGAN model
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

# GFPGAN Enhancement Function
def upscaler(img, version, scale):
    try:
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        face_enhancer = GFPGANer(
            model_path=f'{version}.pth',
            upscale=2,
            arch='RestoreFormer' if version == 'RestoreFormer' else 'clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

        if scale != 2:
            interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
            h, w = img.shape[0:2]
            output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output
    except Exception as error:
        print('Error:', error)
        return None

# Flask API Endpoint
@app.route('/enhance', methods=['POST'])
def enhance_image():
    try:
        # Check if an image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Get the image file
        file = request.files['image']
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)

        # Get parameters from the request
        version = request.form.get('version', 'GFPGANv1.3')
        scale = float(request.form.get('scale', 2))

        # Process the image
        output_image = upscaler(file_path, version, scale)

        if output_image is None:
            return jsonify({"error": "Failed to process the image"}), 500

        # Save the output image to a byte stream
        _, buffer = cv2.imencode('.jpg', output_image)
        byte_stream = io.BytesIO(buffer)

        # Return the processed image
        return send_file(byte_stream, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
