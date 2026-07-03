# -*- coding: utf-8 -*-
"""2D midsagittal contour → kinematic symmetric 3D lift (3_kinematic_lift.py 축약)."""
import numpy as np

from .contour import mask2contour
from modules.utils import mask_label_2d


def _mask_2d(mask):
    return mask_label_2d(mask)


def width_profile(s, half_w=30.0, width_end=0.35):
    """Half-width along normalized arclength s in [0,1]."""
    bump = np.sin(np.pi * s) ** 0.6
    return half_w * (width_end + (1.0 - width_end) * bump)


def lift_frame(curve_mm, nz=15, half_w=30.0, edge_drop=9.0, width_end=0.35):
    """Midline (N,2) mm tip→root → symmetric dome surface (N, Nz, 3) mm.

    Coordinates: x=ant-post, y=up (image), z=lateral (symmetric dome).
    """
    curve_mm = np.asarray(curve_mm, dtype=float)
    if curve_mm.ndim != 2 or curve_mm.shape[1] < 2:
        raise ValueError("curve_mm must be (N, 2+)")
    xy = curve_mm[:, :2]
    N = len(xy)
    s = np.linspace(0, 1, N)
    W = width_profile(s, half_w=half_w, width_end=width_end)
    zt = np.linspace(-1, 1, int(nz))
    surf = np.zeros((N, int(nz), 3), dtype=float)
    for i in range(N):
        x, y = xy[i]
        z = W[i] * zt
        drop = edge_drop * (1.0 - np.sqrt(np.clip(1.0 - zt ** 2, 0.0, 1.0)))
        surf[i, :, 0] = x
        surf[i, :, 1] = y - drop
        surf[i, :, 2] = z
    return surf


def lift(contours, nz=15, half_w=30.0, edge_drop=9.0, width_end=0.35):
    """Contour 시퀀스 → 3D lifted surface (T, N, Nz, 3) mm.

    Parameters
    ----------
    contours : (T, N, 3) array or list of (N, 3)
        image-mm dorsal arcs (mask2contour 출력).

    Returns
    -------
    lifted : (T, N, Nz, 3) float mm
    """
    contours = np.asarray(contours, dtype=float)
    if contours.ndim == 2:
        contours = contours[np.newaxis, ...]
    if contours.ndim != 3 or contours.shape[2] < 2:
        raise ValueError("contours must be (T, N, 3) or (N, 3)")
    frames = [
        lift_frame(contours[k], nz=nz, half_w=half_w,
                   edge_drop=edge_drop, width_end=width_end)
        for k in range(contours.shape[0])
    ]
    return np.stack(frames, axis=0)


def lift_masks(masks, mm_per_px=None, nz=15, half_w=30.0, edge_drop=9.0,
               width_end=0.35):
    """Mask 시퀀스 (T,H,W,C) → lift().

    Parameters
    ----------
    masks : (T, H, W, C) array
        ``load_video()`` 출력.

    Returns
    -------
    lifted : (T, N, Nz, 3) float mm
    """
    masks = np.asarray(masks)
    if masks.ndim != 4:
        raise ValueError("masks must be (T, H, W, C)")
    m2c = {}
    if mm_per_px is not None:
        m2c["mm_per_px"] = mm_per_px
    contours = np.stack([
        mask2contour(_mask_2d(masks[k]), **m2c) for k in range(masks.shape[0])
    ], axis=0)
    return lift(contours, nz=nz, half_w=half_w, edge_drop=edge_drop,
                width_end=width_end)
