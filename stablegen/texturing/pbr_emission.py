"""Emission detectors shared by PBR generation (pbr.py) and Masked Edit's PBR update.

Plain numpy/OpenCV/PIL, so both pipelines produce identical maps from the same view.
"""

import cv2
import numpy as np
from PIL import Image


def emission_from_residual(residual_img, metallic_img=None, threshold=0.2):
    """Emission from the IID-Lighting non-diffuse residual (an RGB image).

    *metallic_img* (an 'L' image, resized to the residual when needed)
    suppresses the specular reflections of metal, which are not emission.
    A soft cutoff then removes weak residual: luminance below half the
    threshold is dropped and full strength is reached at the threshold.
    Returns an RGB image.
    """
    residual = np.asarray(residual_img.convert('RGB'), np.float32) / 255.0
    if metallic_img is not None:
        res_h, res_w = residual.shape[:2]
        if metallic_img.size != (res_w, res_h):
            metallic_img = metallic_img.resize((res_w, res_h), Image.Resampling.BILINEAR)
        metallic = np.asarray(metallic_img.convert('L'), np.float32) / 255.0
        residual = residual * (1.0 - metallic)[:, :, np.newaxis]
    luminance = np.mean(residual, axis=2)
    fade = np.clip((luminance - threshold * 0.5) / max(threshold * 0.5, 1e-6),
                   0.0, 1.0)[:, :, np.newaxis]
    return Image.fromarray(np.clip(residual * fade * 255, 0, 255).astype(np.uint8), mode='RGB')


def emission_from_hsv(image, sat_min=0.5, val_min=0.85, bloom=5.0, threshold=0.2):
    """Emission from bright, saturated pixels of a generated image.

    Glowing parts keep their colour intensity while normally lit surfaces
    desaturate under bright light. The detection mask is blurred by *bloom*
    pixels, cut at *threshold* and multiplied by the image's own colour.
    Returns an RGB image.
    """
    rgb = np.asarray(image.convert('RGB'), np.float32) / 255.0
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
    mask = ((saturation >= sat_min) & (maxc >= val_min)).astype(np.float32)
    if bloom > 0:
        ksize = int(bloom * 4) | 1  # must be odd
        mask = cv2.GaussianBlur(mask, (ksize, ksize), bloom)
    mask = np.where(mask > threshold, mask, 0.0)
    return Image.fromarray(np.clip(rgb * mask[:, :, np.newaxis] * 255, 0, 255).astype(np.uint8), mode='RGB')
