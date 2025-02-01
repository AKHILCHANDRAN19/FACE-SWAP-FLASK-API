# Updated app.py
import cv2
import numpy as np
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import insightface
from insightface.app import FaceAnalysis
import gdown
from io import BytesIO

app = Flask(__name__)

# Initialize models
print("Initializing models...")
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Download model if missing
MODEL_PATH = 'inswapper_128.onnx'
if not os.path.exists(MODEL_PATH):
    gdown.download(
        'https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF',
        MODEL_PATH,
        quiet=False
    )
swapper = insightface.model_zoo.get_model(MODEL_PATH, download=False)

def validate_image(file_stream):
    """Validate image file using magic numbers"""
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
    """Swap faces between two in-memory images"""
    # Convert bytes to numpy arrays
    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Detect faces (using first face found)
    face1 = face_app.get(img1)
    face2 = face_app.get(img2)
    
    if not face1 or not face2:
        raise ValueError("Could not detect faces in both images")
    
    # Perform swap
    result1 = swapper.get(img1, face1[0], face2[0], paste_back=True)
    result2 = swapper.get(img2, face2[0], face1[0], paste_back=True)
    
    return result1, result2

@app.route('/swap', methods=['POST'])
def handle_swap():
    if 'image1' not in request.files or 'image2' not in request.files:
        return {"error": "Missing image files"}, 400
    
    try:
        # Process files directly from memory
        file1 = request.files['image1']
        file2 = request.files['image2']

        # Validate image formats
        for f in [file1, file2]:
            f.stream.seek(0)
            validate_image(f.stream)

        # Convert to bytes
        img1_bytes = file1.read()
        img2_bytes = file2.read()

        # Perform face swap
        result1, result2 = swap_faces(img1_bytes, img2_bytes)

        # Create in-memory files
        output = BytesIO()
        output.write(cv2.imencode('.jpg', result1)[1].tobytes())
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='swapped_result.jpg'
        )

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
