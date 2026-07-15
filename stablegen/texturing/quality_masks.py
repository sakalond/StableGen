"""Binary context/write-mask helpers for sequential edit-model texturing."""

from __future__ import annotations

import cv2
import numpy as np


def regularize_repair_mask(mask, threshold=127, dilation=0):
    """Return an exact uint8 binary repair mask (255 means editable)."""
    if mask is None:
        raise ValueError("mask is required")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = np.where(mask > int(threshold), 255, 0).astype(np.uint8)
    dilation = max(0, int(dilation))
    if dilation:
        size = dilation * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def apply_binary_context_mask(context_bgr, repair_mask, fallback_rgb):
    """Replace every editable context pixel with the exact fallback colour."""
    if context_bgr is None:
        raise ValueError("context image is required")
    if context_bgr.shape[:2] != repair_mask.shape[:2]:
        repair_mask = cv2.resize(
            repair_mask,
            (context_bgr.shape[1], context_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    result = context_bgr.copy()
    fallback_bgr = np.array(
        [round(fallback_rgb[2] * 255), round(fallback_rgb[1] * 255), round(fallback_rgb[0] * 255)],
        dtype=np.uint8,
    )
    result[repair_mask > 0, :3] = fallback_bgr
    return result


def reject_fallback_from_write_mask(
    write_mask,
    generated_bgr,
    fallback_rgb,
    tolerance=8,
):
    """Disable write pixels where Qwen left the sentinel colour behind."""
    if generated_bgr.shape[:2] != write_mask.shape[:2]:
        generated_bgr = cv2.resize(
            generated_bgr,
            (write_mask.shape[1], write_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    fallback_bgr = np.array(
        [round(fallback_rgb[2] * 255), round(fallback_rgb[1] * 255), round(fallback_rgb[0] * 255)],
        dtype=np.int16,
    )
    delta = generated_bgr[:, :, :3].astype(np.int32) - fallback_bgr.astype(np.int32)[None, None, :]
    sentinel = np.sqrt(np.sum(delta * delta, axis=2)) <= max(0, int(tolerance))
    result = write_mask.copy()
    result[(write_mask > 0) & sentinel] = 0
    return result
