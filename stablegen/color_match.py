# stablegen/color_match.py

from __future__ import annotations
import numpy as np

def _rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB [0,1] to YUV (BT.601-ish).
    rgb: (..., 3)
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b

    return np.stack((y, u, v), axis=-1)


def _yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    """
    Convert YUV back to RGB [0,1] and clamp.
    yuv: (..., 3)
    """
    y = yuv[..., 0]
    u = yuv[..., 1]
    v = yuv[..., 2]

    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u

    rgb = np.stack((r, g, b), axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def _boost_chroma_yuv(
    tgt_rgb: np.ndarray,
    matched_rgb: np.ndarray,
    extra: float,
) -> np.ndarray:
    """
    Exaggerate the chroma shift from tgt_rgb -> matched_rgb, while
    preserving luminance from tgt_rgb.

    extra: in [0, 1], where 0 = no extra, 1 = max extra.
    """
    extra = float(max(0.0, min(extra, 1.0)))

    yuv_tgt = _rgb_to_yuv(tgt_rgb)      # Y from target
    yuv_mat = _rgb_to_yuv(matched_rgb)  # UV from matched

    y = yuv_tgt[..., 0]
    uv_t = yuv_tgt[..., 1:]
    uv_m = yuv_mat[..., 1:]

    # Extrapolate UV away from target toward matched:
    #   extra = 0 -> uv = uv_m   (no extra beyond matched)
    #   extra = 1 -> uv = uv_t + 2 * (uv_m - uv_t)
    scale = 1.0 + extra
    uv = uv_t + scale * (uv_m - uv_t)

    yuv = np.empty_like(yuv_tgt)
    yuv[..., 0] = y
    yuv[..., 1:] = uv

    return _yuv_to_rgb(yuv)

def _split_rgb_alpha(img: np.ndarray):
    """Return (rgb, alpha or None) from [H, W, 3/4] float32 [0, 1]."""
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected [H, W, 3 or 4], got {img.shape}")
    if img.shape[2] == 4:
        return img[..., :3], img[..., 3:4]
    return img, None


def _merge_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray | None, like: np.ndarray) -> np.ndarray:
    """Rebuild into 3 or 4 channels to match `like`."""
    if like.shape[2] == 4:
        if alpha is None:
            alpha = like[..., 3:4]
        return np.concatenate([rgb, alpha], axis=-1)
    return rgb


def _reinhard_preserve_luma(ref_rgb: np.ndarray, tgt_rgb: np.ndarray) -> np.ndarray:
    """
    Reinhard-style color transfer that matches only chroma (U/V),
    while preserving the target luminance (Y).

    This avoids the typical darkening/brightening side effects and
    is better suited for 'make it bluer' style corrections.
    """
    ref_yuv = _rgb_to_yuv(ref_rgb)
    tgt_yuv = _rgb_to_yuv(tgt_rgb)

    # Flatten U/V for stats
    ref_uv = ref_yuv[..., 1:].reshape(-1, 2)
    tgt_uv = tgt_yuv[..., 1:].reshape(-1, 2)

    ref_mu = ref_uv.mean(axis=0)
    ref_sigma = ref_uv.std(axis=0) + 1e-6

    tgt_mu = tgt_uv.mean(axis=0)
    tgt_sigma = tgt_uv.std(axis=0) + 1e-6

    out = np.copy(tgt_yuv)

    uv = out[..., 1:]
    uv = (uv - tgt_mu) / tgt_sigma
    uv = uv * ref_sigma + ref_mu
    out[..., 1:] = uv  # keep Y as-is, only adjust U/V

    return _yuv_to_rgb(out)



def _hist_match(ref_rgb: np.ndarray, tgt_rgb: np.ndarray) -> np.ndarray:
    """Simple per-channel histogram matching."""
    out = tgt_rgb.copy()
    ref = (np.clip(ref_rgb, 0, 1) * 255.0).astype(np.float32)
    tgt = (np.clip(tgt_rgb, 0, 1) * 255.0).astype(np.float32)

    for c in range(3):
        ref_c = ref[..., c].ravel()
        tgt_c = tgt[..., c].ravel()

        tgt_vals, tgt_idx, tgt_counts = np.unique(
            tgt_c, return_inverse=True, return_counts=True
        )
        ref_vals, ref_counts = np.unique(ref_c, return_counts=True)

        tgt_cdf = np.cumsum(tgt_counts).astype(np.float64)
        tgt_cdf /= tgt_cdf[-1]

        ref_cdf = np.cumsum(ref_counts).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        mapped = np.interp(tgt_cdf, ref_cdf, ref_vals)
        matched = mapped[tgt_idx].reshape(tgt_rgb.shape[:2])
        out[..., c] = matched / 255.0

    return np.clip(out, 0.0, 1.0)


def _mvgd(ref_rgb: np.ndarray, tgt_rgb: np.ndarray) -> np.ndarray:
    """Covariance-based color transfer (MVGD-ish)."""
    eps = 1e-6
    ref_flat = ref_rgb.reshape(-1, 3).astype(np.float64)
    tgt_flat = tgt_rgb.reshape(-1, 3).astype(np.float64)

    mu_ref = ref_flat.mean(axis=0, keepdims=True)
    mu_tgt = tgt_flat.mean(axis=0, keepdims=True)

    ref_c = ref_flat - mu_ref
    tgt_c = tgt_flat - mu_tgt

    cov_ref = np.cov(ref_c, rowvar=False) + np.eye(3) * eps
    cov_tgt = np.cov(tgt_c, rowvar=False) + np.eye(3) * eps

    eval_t, evec_t = np.linalg.eigh(cov_tgt)
    eval_r, evec_r = np.linalg.eigh(cov_ref)

    D_t_inv_sqrt = np.diag(1.0 / np.sqrt(eval_t))
    D_r_sqrt = np.diag(np.sqrt(eval_r))

    W_t = evec_t @ D_t_inv_sqrt @ evec_t.T  # whitening
    C_r = evec_r @ D_r_sqrt @ evec_r.T      # coloring

    tgt_whitened = tgt_c @ W_t.T
    tgt_colored = tgt_whitened @ C_r.T

    matched_flat = tgt_colored + mu_ref
    matched = matched_flat.reshape(tgt_rgb.shape)
    return np.clip(matched, 0.0, 1.0).astype(np.float32)


def _apply_core(ref_rgb: np.ndarray, tgt_rgb: np.ndarray, method: str) -> np.ndarray:
    m = (method or "reinhard").lower()
    if m == "reinhard":
        return _reinhard_preserve_luma(ref_rgb, tgt_rgb)
    elif m == "hm":
        return _hist_match(ref_rgb, tgt_rgb)
    elif m in ("mvgd", "mkl"):  # treat MKL ~ MVGD
        return _mvgd(ref_rgb, tgt_rgb)
    elif m == "hm-mvgd-hm":
        s1 = _hist_match(ref_rgb, tgt_rgb)
        s2 = _mvgd(ref_rgb, s1)
        s3 = _hist_match(ref_rgb, s2)
        return s3
    elif m == "hm-mkl-hm":
        s1 = _hist_match(ref_rgb, tgt_rgb)
        s2 = _mvgd(ref_rgb, s1)
        s3 = _hist_match(ref_rgb, s2)
        return s3
    else:
        return _reinhard(ref_rgb, tgt_rgb)


def color_match_single(
    ref: np.ndarray,
    target: np.ndarray,
    method: str = "reinhard",
    strength: float = 1.0,
) -> np.ndarray:
    """
    Color-match a single target to a reference.
    ref / target: [H, W, 3 or 4] in [0, 1].

    UI strength is interpreted as:
      0.0–1.0 : blend from original -> matched
      1.0–2.0 : full match + extra chroma push (no big luminance change)
    """
    if ref.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"Size mismatch: ref {ref.shape[:2]} vs target {target.shape[:2]}"
        )

    # Split out RGB / A
    ref_rgb, _ = _split_rgb_alpha(ref)
    tgt_rgb, tgt_a = _split_rgb_alpha(target)

    # Full-strength match result (what "1.0" really means)
    base_matched_rgb = _apply_core(ref_rgb, tgt_rgb, method)

    # Clamp UI strength to [0, 2]
    ui_s = float(max(0.0, min(strength, 2.0)))

    if ui_s <= 1.0:
        # 0–1: regular blend between target and matched
        s = ui_s
        matched_rgb = tgt_rgb * (1.0 - s) + base_matched_rgb * s
    else:
        # >1: start from fully matched, then push chroma harder
        matched_rgb = base_matched_rgb
        extra = ui_s - 1.0  # in (0, 1]
        matched_rgb = _boost_chroma_yuv(tgt_rgb, matched_rgb, extra)

    matched_rgb = np.clip(matched_rgb, 0.0, 1.0)
    out = _merge_rgb_alpha(matched_rgb, tgt_a, like=target)
    return out.astype(np.float32)
