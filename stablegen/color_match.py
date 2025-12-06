# stablegen/color_match.py

from __future__ import annotations
import numpy as np


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


def _reinhard(ref_rgb: np.ndarray, tgt_rgb: np.ndarray) -> np.ndarray:
    """Per-channel mean/std transfer (Reinhard-style)."""
    out = tgt_rgb.copy()
    eps = 1e-8
    for c in range(3):
        ref_c = ref_rgb[..., c]
        tgt_c = tgt_rgb[..., c]

        m_ref, s_ref = ref_c.mean(), ref_c.std() + eps
        m_tgt, s_tgt = tgt_c.mean(), tgt_c.std() + eps

        out[..., c] = (tgt_c - m_tgt) * (s_ref / s_tgt) + m_ref
    return np.clip(out, 0.0, 1.0)


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
        return _reinhard(ref_rgb, tgt_rgb)
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
    method: str = "hm-mvgd-hm",
    strength: float = 1.0,
) -> np.ndarray:
    """
    Color-match a single target to a reference.
    ref / target: [H, W, 3 or 4] in [0, 1].
    """
    if ref.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"Size mismatch: ref {ref.shape[:2]} vs target {target.shape[:2]}"
        )

    ref_rgb, _ = _split_rgb_alpha(ref)
    tgt_rgb, tgt_a = _split_rgb_alpha(target)

    matched_rgb = _apply_core(ref_rgb, tgt_rgb, method)

    s = float(max(0.0, min(strength, 10.0)))
    if s != 1.0:
        matched_rgb = tgt_rgb * (1.0 - s) + matched_rgb * s

    matched_rgb = np.clip(matched_rgb, 0.0, 1.0)
    out = _merge_rgb_alpha(matched_rgb, tgt_a, like=target)
    return out.astype(np.float32)
