import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from inference.model import model

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Dental Segmentation API Running"}


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):

    print("PREDICT CALLED")

    results = []

    for file in files:

        print("PROCESSING:", file.filename)
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        mask, overlay = model.predict(image_np)

        _, overlay_encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay_bytes = overlay_encoded.tobytes()

        import base64

        overlay_b64 = base64.b64encode(overlay_bytes).decode("utf-8")

        results.append(
            {
                "filename": file.filename,
                "model": model.name,
                "overlay": overlay_b64,
            }
        )

    return JSONResponse(content={"results": results})