from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import os, uuid

from app.config import CONFIG
from app.cost_estimator import estimate_cost

app = Flask(__name__)
CORS(app)

# load config values
APP_CONFIG = CONFIG["app"]
PATHS = CONFIG["paths"]
INFERENCE = CONFIG["inference"]
CLASS_NAMES = CONFIG["classes"]

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", PATHS["upload_folder"])
MODEL_PATH = os.path.join(BASE_DIR, "..", PATHS["damage_detection_model_path"])
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# load the damage detection model
model = YOLO(MODEL_PATH)


###
## This method is used to check whether the backend runs healthy.
###
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "service": APP_CONFIG["name"],
        "version": APP_CONFIG["version"],
        "status": "OK",
        "model_loaded": True
    })


###
## This method is used to predict images.
###
@app.route("/predict", methods=["POST"])
def predict():

    make = request.form.get("make")
    model_name = request.form.get("model")

    if not make or not model_name:
        return jsonify({"error": "Vehicle make and model are required"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = get_image_from_request()

    if image.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # save the uploaded image temporarily
    image_path = save_uploaded_image(image)

    results = model(
        image_path,
        conf=INFERENCE["confidence_threshold"],
        iou=INFERENCE["iou_threshold"],
        imgsz=INFERENCE["image_size"]
    )

    detections = process_detections(results, make, model_name)

    grand_total = sum(
        d["cost_estimation"]["total_cost"] for d in detections
    )

    os.remove(image_path)

    return jsonify({
        "vehicle": {
            "make": make,
            "model": model_name
        },
        "detections": detections,
        "count": len(detections),
        "grand_total_estimated_cost": grand_total
    })


###
## This method is used decide the image format given, and read the image from the request.
###
def get_image_from_request():

    # Case 1: Standard file upload
    if 'image' in request.files:
        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image

    # Case 2: iPhone / Safari base64 upload
    if request.json and 'image_bytes' in request.json:
        image_data = request.json['image_bytes']
        decoded = base64.b64decode(image_data)
        file_bytes = np.frombuffer(decoded, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image

    return None


###
## This method is used to read damages in the image and process the total repair cost.
###
def process_detections(results, make, model_name):
    detections = []

    for r in results:
        image_height, image_width = r.orig_shape

        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            part = CLASS_NAMES.get(cls_id, "unknown")

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            severity, damage_ratio = calculate_severity(
                x1, y1, x2, y2, image_width, image_height
            )

            # predict and estimate the total repair cost
            cost_info = estimate_cost(part, severity, make, model_name)

            detections.append({
                "part": part,
                "confidence": round(confidence, 3),
                "severity": severity,
                "damage_ratio": round(damage_ratio, 4),
                "cost_estimation": cost_info,
                "bounding_box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "original_width": image_width,
                "original_height": image_height,
            })

    return detections


###
## This method is used to save an image in the temp folder.
###
def save_uploaded_image(file):
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)
    return image_path


###
## This method is used to calculate the damage level.
###
def calculate_severity(x1, y1, x2, y2, image_width, image_height):
    box_area = (x2 - x1) * (y2 - y1)
    image_area = image_width * image_height
    damage_ratio = box_area / image_area

    if damage_ratio < 0.05:
        severity = "low"
    elif damage_ratio < 0.15:
        severity = "medium"
    else:
        severity = "high"

    return severity, damage_ratio



if __name__ == "__main__":
    app.run(debug=APP_CONFIG["debug"])
