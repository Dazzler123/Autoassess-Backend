import yaml
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config.yaml")

with open(CONFIG_PATH, "r") as file:
    CONFIG = yaml.safe_load(file)