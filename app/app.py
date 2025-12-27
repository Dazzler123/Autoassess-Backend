from flask import Flask, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "autoassess_yolov8_final.pt")
model = YOLO(MODEL_PATH)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "service": "AutoAssess Backend - Running Healthy...",
        "model_loaded": True
    })

if __name__ == "__main__":
    app.run(debug=True)
