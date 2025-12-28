from flask import Flask, jsonify, request
from ultralytics import YOLO
from PIL import Image
import os
import uuid

app = Flask(__name__)

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "autoassess_yolov8_final.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

# class mapping 
CLASS_NAMES = {
    0: "door",
    1: "bumper",
    2: "headlight",
    3: "taillight",
    4: "fender",
    5: "hood",
    6: "trunk",
    7: "mirror",
    8: "windscreen"
}

###
## This method is used to check whether the backend runs healthy.
###
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "service": "AutoAssess Backend",
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
        conf=0.25,
        iou=0.6,
        imgsz=640
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

    return jsonify({'/Users/dazzler/Downloads/test-images/images.jpeg'
        "detections": detections,
        "count": len(detections)
    })


if __name__ == "__main__":
    app.run(debug=True)
