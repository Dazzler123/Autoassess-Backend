from flask import Flask, jsonify, request
from ultralytics import YOLO
import os, uuid
from app.config import CONFIG

app = Flask(__name__)

# load config values
APP_CONFIG = CONFIG["app"]
PATHS = CONFIG["paths"]
INFERENCE = CONFIG["inference"]
CLASS_NAMES = CONFIG["classes"]

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", PATHS["upload_folder"])
MODEL_PATH = os.path.join(BASE_DIR, "..", PATHS["model_path"])

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # save the image temporarily
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # run detection
    results = model(
        image_path,
        conf=INFERENCE["confidence_threshold"],
        iou=INFERENCE["iou_threshold"],
        imgsz=INFERENCE["image_size"]
    )

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append({
                "part": CLASS_NAMES.get(cls_id, "unknown"),
                "confidence": round(confidence, 3)
            })

    # remove uploaded file
    os.remove(image_path)

    return jsonify({
        "detections": detections,
        "count": len(detections)
    })


if __name__ == "__main__":
    app.run(debug=APP_CONFIG["debug"])
