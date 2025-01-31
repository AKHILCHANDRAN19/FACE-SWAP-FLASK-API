import os
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from gfpgan import GFPGANer

# Initialize Flask app
app = Flask(__name__)

# Load GFPGAN model
model_path = 'experiments/pretrained_models/GFPGANv1.3.pth'
restorer = GFPGANer(
    model_path=model_path,
    upscale=1,  # Default upscale factor (will be overridden)
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None  # Optional: Use RealESRGAN for background upscaling
)

# Ensure the output directory exists
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

@app.route('/restore', methods=['POST'])
def restore_image():
    """
    Endpoint to restore a face in an uploaded image using GFPGAN.
    Query Parameters:
        - upscale: (Optional) Upscale factor (2, 3, or 4). Default is 2.
    """
    try:
        # Check if an image file is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Get the upscale factor from the query parameters
        upscale_factor = request.args.get('upscale', default=2, type=int)
        if upscale_factor not in [2, 3, 4]:
            return jsonify({"error": "Invalid upscale factor. Choose 2, 3, or 4."}), 400

        # Update the GFPGAN upscale factor
        restorer.upscale = upscale_factor

        # Read the uploaded image
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Restore the face using GFPGAN
        _, _, restored_img = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        # Save the restored image
        output_path = os.path.join(output_dir, 'restored_image.jpg')
        cv2.imwrite(output_path, restored_img)

        # Return the restored image as a response
        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
