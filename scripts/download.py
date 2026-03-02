import tempfile
import zipfile
from pathlib import Path

import gdown


DC1000_URL = "https://drive.google.com/uc?id=1Xn1oGHvhGF9GbkcLEtCOV5QvWWqt1y62"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        zip_path = Path(tmp.name)

        gdown.download(DC1000_URL, str(zip_path), quiet=False)

        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(OUTPUT_DIR)


if __name__ == "__main__":
    main()
