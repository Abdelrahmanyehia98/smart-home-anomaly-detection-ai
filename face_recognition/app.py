from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import pymongo
from bson.binary import Binary
import pickle
import time
import uuid
import logging
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FaceRecognitionAPI')


class FaceRecognitionAPI:
    def __init__(self, mongodb_uri, db_name, collection_name):
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = pymongo.MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.initialize_model()
        self.upload_folder = 'uploads'
        os.makedirs(self.upload_folder, exist_ok=True)

    def initialize_model(self):
        logger.info("Downloading and initializing AuraFace model...")
        try:
            snapshot_download("fal/AuraFace-v1", local_dir="models/auraface")
            self.face_app = FaceAnalysis(
                name="auraface",
                providers=["CPUExecutionProvider"],
                root=".",
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def process_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, "Failed to read image"
            faces = self.face_app.get(image)
            if not faces:
                return None, "No face detected in image"
            return faces, "Success"
        except Exception as e:
            return None, f"Error processing image: {str(e)}"

    def check_face_quality(self, face, image):
        try:
            bbox = face.bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            img_h, img_w = image.shape[:2]
            if x1 < 0 or y1 < 0 or x2 >= img_w or y2 >= img_h:
                return False, "Face is partially out of frame"
            face_width = x2 - x1
            face_height = y2 - y1
            face_area = face_width * face_height
            image_area = img_w * img_h
            face_ratio = face_area / image_area
            if face_ratio < 0.005:
                return False, "Face is too small, please move closer"
            if face_width < 20 or face_height < 20:
                return False, "Face is extremely small, please move closer"
            if hasattr(face, 'det_score') and face.det_score < 0.3:
                return False, "Face detection confidence is low"
            return True, "Face check passed"
        except Exception as e:
            return True, f"Face quality check had issues but proceeding: {str(e)}"

    def find_closest_match(self, embedding, threshold=0.5):
        try:
            all_faces = list(self.collection.find())
            if not all_faces:
                return None, float('inf')
            closest_match = None
            min_distance = float('inf')
            for face_doc in all_faces:
                if 'embedding' in face_doc:
                    stored_embedding = pickle.loads(face_doc['embedding'])
                    distance = 1 - np.dot(embedding, stored_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        closest_match = face_doc
            if min_distance <= threshold:
                return closest_match, min_distance
            else:
                return None, min_distance
        except Exception as e:
            return None, float('inf')


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file):
    """Validate and save uploaded file. Returns (file_path, error_response)."""
    if file.filename == '':
        return None, (jsonify({'status': 'error', 'message': 'No selected file'}), 400)
    if not allowed_file(file.filename):
        return None, (jsonify({'status': 'error', 'message': 'Invalid file format'}), 400)
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{time.time()}_{filename}")
    file.save(file_path)
    img = cv2.imread(file_path)
    if img is None:
        os.remove(file_path)
        return None, (jsonify({'status': 'error', 'message': 'Invalid image'}), 400)
    return file_path, None


@app.route('/face-register', methods=['POST'])
def face_register():
    """Register a known face once per person."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    name = request.form.get('name', 'unknown')
    file_path, err = save_upload(file)
    if err:
        return err

    try:
        faces, message = face_api.process_image(file_path)
        if faces is None:
            os.remove(file_path)
            return jsonify({'status': 'error', 'message': message}), 400

        if len(faces) > 1:
            os.remove(file_path)
            return jsonify({'status': 'error', 'message': 'Multiple faces detected, please provide a single face image'}), 400

        face = faces[0]
        image = cv2.imread(file_path)
        is_quality, quality_message = face_api.check_face_quality(face, image)
        if not is_quality:
            os.remove(file_path)
            return jsonify({'status': 'error', 'message': quality_message}), 400

        embedding = face.normed_embedding

        # Check if face is already registered
        existing, distance = face_api.find_closest_match(embedding, threshold=0.5)
        if existing:
            os.remove(file_path)
            return jsonify({
                'status': 'already_registered',
                'message': f'Face already registered as {existing.get("name", "unknown")}',
                'user_id': existing.get('user_id'),
                'name': existing.get('name')
            }), 200

        doc = {
            'user_id': str(uuid.uuid4()),
            'name': name,
            'embedding': Binary(pickle.dumps(embedding)),
            'timestamp': time.time()
        }
        face_api.collection.insert_one(doc)
        os.remove(file_path)
        return jsonify({
            'status': 'success',
            'message': f'Face registered successfully as {name}',
            'user_id': doc['user_id'],
            'name': name
        }), 201

    except Exception as e:
        try:
            os.remove(file_path)
        except:
            pass
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/face-auth', methods=['POST'])
def face_auth():
    """Verify all faces in an image. Returns result per face."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    file_path, err = save_upload(file)
    if err:
        return err

    threshold = request.form.get('threshold', 0.5)
    try:
        threshold = float(threshold)
    except:
        threshold = 0.5

    try:
        faces, message = face_api.process_image(file_path)
        if faces is None:
            os.remove(file_path)
            return jsonify({'status': 'error', 'message': message}), 400

        image = cv2.imread(file_path)
        results = []

        for face in faces:
            is_quality, quality_message = face_api.check_face_quality(face, image)
            if not is_quality:
                results.append({
                    'known': False,
                    'reason': quality_message
                })
                continue

            embedding = face.normed_embedding
            match, distance = face_api.find_closest_match(embedding, threshold=threshold)

            if match:
                results.append({
                    'known': True,
                    'name': match.get('name', 'unknown'),
                    'user_id': match.get('user_id'),
                    'confidence': round(float(1 - distance), 2)
                })
            else:
                results.append({
                    'known': False,
                    'confidence': round(float(1 - distance), 2)
                })

        os.remove(file_path)
        return jsonify({
            'status': 'success',
            'faces_count': len(results),
            'results': results
        }), 200

    except Exception as e:
        try:
            os.remove(file_path)
        except:
            pass
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.get("/")
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=7000, type=int)
    parser.add_argument('--mongodb-uri', default="mongodb://localhost:27017/")
    parser.add_argument('--db-name', default="Face-Id")
    parser.add_argument('--collection', default="face_embeddings")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    face_api = FaceRecognitionAPI(args.mongodb_uri, args.db_name, args.collection)
    app.run(host=args.host, port=args.port, debug=args.debug)