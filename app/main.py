from fastapi import FastAPI, File, UploadFile, Query
import io
import cv2
import numpy as np
from PIL import Image
import base64

from inference.model import model

app = FastAPI()


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...),
    mode: str = Query("soft"),   # "soft" | "binary"
    threshold: float = Query(0.5)
):

    results = []

    for file in files:

        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("L")
        image_np = np.array(image)

        probs  = model.predict(image_np, return_prob=True)

        base = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        if mode == "binary":

            mask = (probs > threshold).astype(np.uint8)

            color_mask = np.zeros_like(base)
            color_mask[:, :, 2] = mask * 255  # red

            final = cv2.addWeighted(base, 1.0, color_mask, 0.4, 0)

        else:
            heatmap = (probs * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            final = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)

        _, buffer = cv2.imencode(".png", final)
        b64 = base64.b64encode(buffer).decode("utf-8")

        results.append({
            "filename": file.filename,
            "overlay": b64,
            "mode": mode,
            "model":model.name
        })

    return {"results": results}