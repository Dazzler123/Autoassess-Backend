from app.config import CONFIG
from app.price_model import predict_part_price

COST_CONFIG = CONFIG["cost_estimation"]

###
##  This method can be used to estimate repair cost based on detected part and confidence score.
##  Confidence is used as a proxy for damage severity.
###
def estimate_cost(part, severity, make, model_name):
    predicted_part_price = predict_part_price(part, make.lower(), model_name.lower())

    labour_cost = COST_CONFIG["labour_costs"].get(part, 0)
    multiplier = COST_CONFIG["severity_multipliers"][severity]

    estimated_labour = int(labour_cost)

    # part-specific override
    if part in ["windscreen", "headlight", "taillight", "mirror"]:
        estimated_part = int(predicted_part_price)
    else:
        estimated_part = int(predicted_part_price * multiplier)

    total_cost = estimated_labour + estimated_part

    return {
        "labour_cost": estimated_labour,
        "part_cost": estimated_part,
        "total_cost": total_cost
    }
