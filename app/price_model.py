import joblib
import os
import numpy as np
import pandas as pd
from app.config import CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", CONFIG["paths"]["price_prediction_model_path"])

with open(MODEL_PATH, "rb") as f:
    price_model = joblib.load(MODEL_PATH)


def predict_part_price(part, make, model_name):
    try:
        # create dataframe
        input_df = pd.DataFrame([{
            "part": part,
            "make": make,
            "model": model_name
        }])

        # predict log price
        predicted_log = price_model.predict(input_df)[0]

        # convert back to actual LKR
        predicted_price = np.expm1(predicted_log)

        # prevent negative or nonsense values
        predicted_price = max(predicted_price, 0)

        return int(predicted_price)

    except Exception as e:
        print("Price prediction error:", e)
        return 0