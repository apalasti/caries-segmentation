import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import gdown
import requests
from tqdm import tqdm
from roboflow import Roboflow
from dotenv import load_dotenv


load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
DC1000_URL = "https://drive.google.com/uc?id=1Xn1oGHvhGF9GbkcLEtCOV5QvWWqt1y62"
KAGGLE_DENTAL_URL = "https://www.kaggle.com/api/v1/datasets/download/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # DC1000 dataset (Google Drive)
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        gdown.download(
            DC1000_URL,
            tmp.name,
            log_messages={"start": "Downloading DC1000 dataset...\n"},
        )

        print("Extracting DC1000 archive...")
        with zipfile.ZipFile(tmp.name, "r") as zf:
            zf.extractall(OUTPUT_DIR)

    # Children's dental panoramic radiographs
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        print("Downloading children's dental panoramic radiographs dataset...")
        r = requests.get(KAGGLE_DENTAL_URL, stream=True, timeout=60)
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        chunk_size = 8192
        with tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                tmp.write(chunk)
                pbar.update(len(chunk))

        tmp.flush()
        print("Extracting dental archive...")
        with zipfile.ZipFile(tmp.name, "r") as zf:
            zf.extractall(OUTPUT_DIR)

    # Roboflow dataset
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("arshs-workspace-radio").project("vzrad2")
    version = project.version(6)
    dataset = version.download("yolov5")
    shutil.move(
        dataset.location, OUTPUT_DIR
    )  # For some reason version.download 'location' argument did not put data to OUTPUT_DIR
    print(f"Dataset extracted to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
