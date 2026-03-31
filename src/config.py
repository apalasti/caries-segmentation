import tomli
import os

def load_config(path=None):
    if path is None:
        path = os.getenv("CARIES_CONFIG_PATH", "config.toml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file {path} not found.")
    with open(path, "rb") as f:
        config = tomli.load(f)
    return config
