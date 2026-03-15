import tomli
import os

def load_config(path="config.toml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file {path} not found.")
    with open(path, "rb") as f:
        config = tomli.load(f)
    return config
