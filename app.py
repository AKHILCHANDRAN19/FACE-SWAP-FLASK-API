# app.py
import os
import cv2
import numpy as np
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import insightface
from insightface.app import FaceAnalysis
import gdown
from io import BytesIO
import zipfile

# Configure for low memory usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Initialize models with CPU-only and reduced precision
print("🚀 Initializing face analysis model...")
face_app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider']
)
face_app.prepare(ctx_id=-1, det_size=(320, 320))  # Reduced input size

# Model setup
MODEL_PATH = 'inswapper_128.onnx'
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading face swap model...")
    gdown.download(
        'https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF',
        MODEL_PATH,
        quiet=False
    )

print("⚙️ Loading swapper model...")
swapper = insightface.model_zoo.get_model(
    MODEL_PATH,
    download=False,
    providers=['CPUExecutionProvider']
)

def validate_image(file_stream):
    """Validate image using magic numbers"""
    header = file_stream.read(4)
    file_stream.seek(0)
    if header.startswith(b'\xff\xd8\xff'):
        return 'jpg'
    elif header.startswith(b'\x89PNG'):
        return 'png'
    elif header[:2] == b'\xff\xd8':
        return 'jpeg'
    else:
        raise ValueError("Unsupported image format")

def swap_faces(img1_bytes, img2_bytes):
    """Perform face swap with memory optimization"""
    # Convert bytes to numpy arrays
    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Detect faces (using first face found)
    face1 = face_app.get(img1)
    face2 = face_app.get(img2)
    
    if not face1:
        raise ValueError("❌ No face detected in first image")
    if not face2:
        raise ValueError("❌ No face detected in second image")

    # Perform swap with garbage collection
    result1 = swapper.get(img1, face1[0], face2[0], paste_back=True)
    result2 = swapper.get(img2, face2[0], face1[0], paste_back=True)
    
    # Release memory
    del img1, img2, face1, face2
    
    return result1, result2

@app.route('/swap', methods=['POST'])
def handle_swap():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return {"error": "Missing image files"}, 400

        # Process files directly from memory
        file1 = request.files['image1']
        file2 = request.files['image2']

        # Validate images
        for f in [file1, file2]:
            f.stream.seek(0)
            try:
                validate_image(f.stream)
            except ValueError as e:
                return {"error": str(e)}, 400

        # Read bytes
        img1_bytes = file1.read()
        img2_bytes = file2.read()

        # Process
        result1, result2 = swap_faces(img1_bytes, img2_bytes)

        # Create in-memory ZIP file
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, 'w') as zf:
            # Convert images to bytes
            _, img1_bytes = cv2.imencode('.jpg', result1)
            _, img2_bytes = cv2.imencode('.jpg', result2)
            
            zf.writestr('swapped_1.jpg', img1_bytes.tobytes())
            zf.writestr('swapped_2.jpg', img2_bytes.tobytes())

        mem_zip.seek(0)
        
        # Clean up
        del result1, result2, img1_bytes, img2_bytes

        return send_file(
            mem_zip,
            mimetype='application/zip',
            as_attachment=True,
            download_name='swapped_results.zip'
        )

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
