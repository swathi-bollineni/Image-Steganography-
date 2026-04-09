from base64 import b64encode
from io import BytesIO
from typing import Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from models.loader import stego_models
from utils.image_utils import (
    array_to_pil,
    preprocess_pair,
    preprocess_single,
)
from utils.metrics import compute_psnr_ssim

app = FastAPI(title="GAN Image Steganography API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _file_to_pil(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    return Image.open(BytesIO(data))


def _to_base64(img: Image.Image, format: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return b64encode(buf.read()).decode("utf-8")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/hide-reveal")
async def hide_and_reveal(
    cover_image: UploadFile = File(...),
    secret_image: UploadFile = File(...),
):
    """
    Full pipeline:
    1. Hide: (cover, secret) -> stego
    2. Reveal: stego -> (recovered_secret, reconstructed_cover)
    3. Metrics: PSNR/SSIM for cover↔stego and secret↔recovered_secret
    """
    try:
        cover_pil = _file_to_pil(cover_image)
        secret_pil = _file_to_pil(secret_image)

        cover_arr_b, secret_arr_b = preprocess_pair(cover_pil, secret_pil)

        # Hide
        stego_tensor = stego_models.hide(
            cover=cover_arr_b.astype("float32"),
            secret=secret_arr_b.astype("float32"),
        )

        # Reveal
        secret_rec_tensor, cover_rec_tensor = stego_models.reveal(
            stego_tensor
        )

        # Convert tensors to numpy images in [0,1], drop batch dim
        cover_arr = cover_arr_b[0]
        secret_arr = secret_arr_b[0]
        stego_arr = np.array(stego_tensor[0], dtype="float32")
        secret_rec_arr = np.array(secret_rec_tensor[0], dtype="float32")
        cover_rec_arr = np.array(cover_rec_tensor[0], dtype="float32")

        # Metrics
        m_cover_stego = compute_psnr_ssim(cover_arr, stego_arr)
        m_secret_rec = compute_psnr_ssim(secret_arr, secret_rec_arr)

        # Convert to PIL for front-end display/download
        stego_pil = array_to_pil(stego_arr)
        secret_rec_pil = array_to_pil(secret_rec_arr)
        cover_rec_pil = array_to_pil(cover_rec_arr)

        response_payload = {
            "stego_image_base64": _to_base64(stego_pil),
            "recovered_secret_base64": _to_base64(secret_rec_pil),
            "reconstructed_cover_base64": _to_base64(cover_rec_pil),
            "metrics": {
                "psnr_cover_stego": m_cover_stego["psnr"],
                "ssim_cover_stego": m_cover_stego["ssim"],
                "psnr_secret_recovered": m_secret_rec["psnr"],
                "ssim_secret_recovered": m_secret_rec["ssim"],
            },
        }

        return JSONResponse(content=response_payload)

    except Exception as exc:  # pragma: no cover - defensive
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal error: {exc}"},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

