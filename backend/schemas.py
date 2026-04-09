from typing import Optional

from pydantic import BaseModel


class Metrics(BaseModel):
    psnr_cover_stego: float
    ssim_cover_stego: float
    psnr_secret_recovered: float
    ssim_secret_recovered: float


class HideRevealResponse(BaseModel):
    stego_image_base64: str
    recovered_secret_base64: str
    reconstructed_cover_base64: str
    metrics: Metrics


class ErrorResponse(BaseModel):
    detail: str

