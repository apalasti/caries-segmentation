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

ROBOFLOW_API_KEY = "58rqeTg0l5nzy3JWH1Sl"#os.getenv("ROBOFLOW_API_KEY")

DC1000_URL = "https://drive.google.com/uc?id=1Xn1oGHvhGF9GbkcLEtCOV5QvWWqt1y62"
KAGGLE_DENTAL_URL = "https://www.kaggle.com/api/v1/datasets/download/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_file(url: str, dest: Path):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def move_equivalent(src: Path, dst: Path):
    """
    Windows-safe replacement for shutil.move
    -> copy + delete = same end result
    """

    shutil.copytree(src, dst)
    shutil.rmtree(src)  # fontos: ugyanaz mint move!


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # temp mappa (nem NamedTemporaryFile!)
    temp_dir = Path(tempfile.mkdtemp())

    # -------------------------
    # DC1000 (Google Drive)
    # -------------------------
    dc_zip = temp_dir / "dc1000.zip"

    print("Downloading DC1000 dataset...")
    gdown.download(DC1000_URL, str(dc_zip), quiet=False)

    print("Extracting DC1000 archive...")
    extract_zip(dc_zip, OUTPUT_DIR)

    # -------------------------
    # Dental dataset
    # -------------------------
    dental_zip = temp_dir / "dental.zip"

    print("Downloading dental dataset...")
    download_file(KAGGLE_DENTAL_URL, dental_zip)

    print("Extracting dental archive...")
    extract_zip(dental_zip, OUTPUT_DIR)

    # -------------------------
    # Roboflow
    # -------------------------
    print("Downloading Roboflow dataset...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("arshs-workspace-radio").project("vzrad2")
    version = project.version(6)

    dataset = version.download("yolov5")

    # 🔥 ugyanaz mint eredeti move, csak Windows safe
    move_equivalent(Path(dataset.location), OUTPUT_DIR)

    print(f"Dataset extracted to {OUTPUT_DIR}")

    # cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()