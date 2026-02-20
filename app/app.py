from flask import Flask, jsonify, request
from ultralytics import YOLO
import os, uuid

from app.config import CONFIG
from app.cost_estimator import estimate_cost

app = Flask(__name__)

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

# load the model
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
            part = CLASS_NAMES.get(cls_id, "unknown")

            # get the bounding box coordinates (xyxy)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # estimate the repair cost
            cost_info = estimate_cost(part, confidence, make, model_name)

            detections.append({
                "part": part,
                "confidence": round(confidence, 3),
                "severity": cost_info["severity"],
                "cost_estimation": {
                    "labour_cost": cost_info["labour_cost"],
                    "part_cost": cost_info["part_cost"],
                    "total_cost": cost_info["total_cost"]
                },
                "bounding_box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })


    grand_total = sum(
        d["cost_estimation"]["total_cost"] for d in detections
    )

    # remove uploaded file
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


if __name__ == "__main__":
    app.run(debug=APP_CONFIG["debug"])
