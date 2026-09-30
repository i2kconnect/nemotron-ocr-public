"""Transport-only subset of NIM 2.x OCR. All document reconstruction is in Java."""
import asyncio
import base64
import binascii
import io
import math
import os
import numpy as np
from contextlib import asynccontextmanager

from huggingface_hub import snapshot_download
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel, ConfigDict
from typing import Literal
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

MAX_BODY = 32 * 1024 * 1024
MAX_PIXELS = 8_000_000
model = None
lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    global model
    snapshot = await asyncio.to_thread(snapshot_download,
        repo_id="nvidia/nemotron-ocr-v2",
        revision=os.environ["NEMOTRON_REVISION"], allow_patterns=["v2_english/*"])
    model = await asyncio.to_thread(NemotronOCRV2,
        model_dir=os.path.join(snapshot, "v2_english"),
        lang=os.getenv("NEMOTRON_LANG", "en"),
        infer_length=int(os.getenv("NEMOTRON_INFER_LENGTH", "1536")))
    yield
    model = None


app = FastAPI(title="Public NVIDIA Nemotron OCR — raw NIM API", lifespan=lifespan)


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image_url"] = "image_url"
    url: str


class OcrInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: list[ImageInput]
    merge_levels: list[Literal["word", "sentence", "paragraph"]] | None = None


@app.get("/v1/health/ready")
def ready():
    return {"object": "health.response", "ready": model is not None,
            "message": "ready" if model is not None else "loading"}


def recognize(image, level):
    header, separator, encoded = image.url.partition(",")
    if not separator or header not in ("data:image/png;base64", "data:image/jpeg;base64", "data:image/jpg;base64"):
        raise HTTPException(422, "Use a base64 PNG or JPEG data URL")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as raster:
            width, height = raster.size
            if width * height > MAX_PIXELS:
                raise HTTPException(413, "Raster exceeds 8 million pixels; tile in the client")
            rgb = raster.convert("RGB")
    except (ValueError, binascii.Error, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(422, "Invalid image") from error
    try:
        predictions = model(np.array(rgb, copy=True), merge_level=level)
    finally:
        rgb.close()
    detections = []
    for item in predictions:
        # This pinned upstream pipeline returns normalized axis bounds.
        # Its "upper" is max(y); emit conventional top-left polygon ordering.
        coords = [float(item[key]) for key in ("left", "upper", "right", "lower")]
        if not all(math.isfinite(value) for value in coords):
            continue
        left, upper, right, lower = [min(1.0, max(0.0, value)) for value in coords]
        x0, x1 = sorted((left, right))
        y0, y1 = sorted((upper, lower))
        confidence = float(item.get("confidence", 0.5))
        if not math.isfinite(confidence):
            confidence = 0.5
        points = [{"x": x0, "y": y0}, {"x": x1, "y": y0},
                  {"x": x1, "y": y1}, {"x": x0, "y": y1}]
        detections.append({"bounding_box": {"points": points},
            "text_prediction": {"text": str(item.get("text", "")),
                                "confidence": min(1.0, max(0.0, confidence))}})
    return detections, len(raw)


@app.post("/v1/ocr")
async def ocr(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BODY:
            raise HTTPException(413, "Request exceeds 32 MiB")
        body.extend(chunk)
    try:
        payload = OcrInput.model_validate_json(body)
    except ValueError as error:
        raise HTTPException(422, "Invalid OCR request") from error
    if not 1 <= len(payload.input) <= 16:
        raise HTTPException(422, "Provide 1 to 16 images")
    levels = payload.merge_levels or ["word"]
    if len(levels) != 1:
        raise HTTPException(422, "This local adapter supports one merge level per request")
    if model is None:
        raise HTTPException(503, "Model is loading")
    data, total = [], 0
    async with lock:
        for index, image in enumerate(payload.input):
            detections, size = await asyncio.to_thread(recognize, image, levels[0])
            total += size
            data.append({"index": index, "text_detections": detections})
    return {"model": "nvidia/nemotron-ocr-v2", "data": data,
            "usage": {"images_size_mb": total / 1_000_000}}
