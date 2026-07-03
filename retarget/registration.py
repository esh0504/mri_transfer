# -*- coding: utf-8 -*-
"""rest mask + 3D reference → registration.csv (image↔model affine anchors)."""
import os
import sys

import numpy as np

from modules.utils import save_csv, read_csv_dicts

_WORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WORK_ROOT not in sys.path:
    sys.path.insert(0, _WORK_ROOT)

from tongue_contour import anatomical_landmarks  # noqa: E402


def _require_file(path, label="path"):
    p = os.path.abspath(str(path))
    if not os.path.isfile(p):
        raise FileNotFoundError("%s not found: %s" % (label, p))
    return p


def _mask_2d(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        return mask[..., 0]
    if mask.ndim == 2:
        return mask
    raise ValueError("mask must be (H,W) or (H,W,C)")


def _model_landmarks_m(ref_3d, y_tol=0.003):
    """3D ref mesh verts (metres) → tip/dorsum/root (x,z) metres on midsagittal."""
    if getattr(ref_3d, "verts", None) is None:
        raise ValueError("ref_3d must have verts (N,3)")
    V = np.asarray(ref_3d.verts, dtype=float)
    mid = V[np.abs(V[:, 1]) <= y_tol]
    if len(mid) < 10:
        mid = V
    tip_i = int(np.argmin(mid[:, 0]))
    root_i = int(np.argmax(mid[:, 0]))
    dorsum_i = int(np.argmax(mid[:, 2]))
    return {
        "tip": (float(mid[tip_i, 0]), float(mid[tip_i, 2])),
        "dorsum": (float(mid[dorsum_i, 0]), float(mid[dorsum_i, 2])),
        "root": (float(mid[root_i, 0]), float(mid[root_i, 2])),
    }


def _image_landmarks_mm(rest_mask, mm_per_px):
    """rest mask → tip/dorsum/root in image-mm (x,y)."""
    H = _mask_2d(rest_mask).shape[0]
    lm = anatomical_landmarks(_mask_2d(rest_mask))
    if lm is None:
        raise ValueError("rest mask에서 anatomical landmarks 추출 실패")
    out = {}
    for name in ("tip", "dorsum", "root"):
        if name not in lm:
            raise ValueError("landmark '%s' missing in rest mask" % name)
        r, c = lm[name]
        out[name] = (float(c) * mm_per_px, float((H - 1) - r) * mm_per_px)
    return out


def _load_landmark_map(path):
    """landmark_map.csv → {label: (imageX, imageY, modelX_m, modelZ_m)}."""
    path = _require_file(path, "landmark_map")
    out = {}
    for row in read_csv_dicts(path):
        try:
            out[row["label"].strip()] = (
                float(row["imageX"]), float(row["imageY"]),
                float(row["modelX_m"]), float(row["modelZ_m"]),
            )
        except (ValueError, KeyError, TypeError):
            continue
    if len(out) < 3:
        raise ValueError(
            "landmark_map needs >=3 usable rows: %s" % path)
    return out


def _fit_affine(img_xy, mod_xz):
    M = np.column_stack([img_xy, np.ones(len(img_xy))])
    A, *_ = np.linalg.lstsq(M, mod_xz, rcond=None)
    pred = M @ A
    res = np.linalg.norm(pred - mod_xz, axis=1)
    return A, float(np.sqrt((res ** 2).mean())), res


def _pairs_from_auto(rest_mask, ref_3d, mm_per_px):
    img_lm = _image_landmarks_mm(rest_mask, mm_per_px)
    mod_lm = _model_landmarks_m(ref_3d)
    names = ["tip", "dorsum", "root"]
    img = np.array([img_lm[k] for k in names], dtype=float)
    mod = np.array([mod_lm[k] for k in names], dtype=float)
    return names, img, mod


def _pairs_from_map(landmark_map):
    names = list(landmark_map.keys())
    img = np.array([[v[0], v[1]] for v in landmark_map.values()], dtype=float)
    mod = np.array([[v[2], v[3]] for v in landmark_map.values()], dtype=float)
    return names, img, mod


def register(rest_mask, ref_3d, out_csv, landmark_map=None, mm_per_px=1.164):
    """rest MRI mask + 3D reference mesh → registration.csv 저장.

    Parameters
    ----------
    rest_mask : (H, W, C) array
    ref_3d : TongueModel (verts/faces metres)
    out_csv : str
        저장할 registration.csv 경로.
    landmark_map : str, optional
        ``label,imageX,imageY,modelX_m,modelZ_m`` CSV (>=3 anchors).
        없으면 rest mask 자동 랜드마크 + ref_3d mesh 랜드마크 사용.

    Returns
    -------
    dict
        path, names, rms_mm, worst_mm, anchors (list of dicts)
    """
    if landmark_map is not None:
        names, img_xy, mod_m = _pairs_from_map(_load_landmark_map(landmark_map))
    else:
        names, img_xy, mod_m = _pairs_from_auto(rest_mask, ref_3d, mm_per_px)

    A, rms_m, res_m = _fit_affine(img_xy, mod_m)
    rms_mm = rms_m * 1000.0
    worst_mm = float(res_m.max()) * 1000.0

    anchors = []
    for name, ixy, mxz in zip(names, img_xy, mod_m):
        anchors.append({
            "label": name,
            "imageX": round(float(ixy[0]), 3), "imageY": round(float(ixy[1]), 3),
            "modelX": round(float(mxz[0]) * 1000.0, 3),
            "modelZ": round(float(mxz[1]) * 1000.0, 3),
        })
    out_csv = save_csv(
        out_csv, ["label", "imageX", "imageY", "modelX", "modelZ"], anchors)

    return {
        "path": out_csv,
        "names": names,
        "rms_mm": rms_mm,
        "worst_mm": worst_mm,
        "anchors": anchors,
        "affine_m": A,
    }
