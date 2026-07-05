# -*- coding: utf-8 -*-
"""3D reference mesh + 2D source/target mask → retargeted mesh obj."""
import os

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import uniform_filter1d

from retarget.contour import mask2contour
from modules.utils import mask_label_2d, read_csv_dicts


def _require_file(path, label="path"):
    p = os.path.abspath(str(path))
    if not os.path.isfile(p):
        raise FileNotFoundError("%s not found: %s" % (label, p))
    return p


def _require_ref_3d(ref_3d):
    if ref_3d is None:
        raise ValueError("ref_3d is required")
    if getattr(ref_3d, "verts", None) is None or getattr(ref_3d, "faces", None) is None:
        raise ValueError("ref_3d must have verts (N,3) and faces (F,3)")
    reg = getattr(ref_3d, "registration_csv", None)
    if not reg:
        raise ValueError(
            "ref_3d.registration_csv is required — "
            "use attach_registration(ref_3d, registration_csv)")
    return _require_file(reg, "registration_csv")


def attach_registration(ref_3d, registration_csv):
    """3D reference model에 image→model affine CSV 경로를 붙인다.

    Parameters
    ----------
    ref_3d : TongueModel or verts/faces 핸들
    registration_csv : str
        label,imageX,imageY,modelX,modelZ anchor CSV.

    Returns
    -------
    ref_3d
        ``registration_csv`` 속성이 설정된 동일 객체.
    """
    ref_3d.registration_csv = _require_file(registration_csv, "registration_csv")
    return ref_3d


def _mask_label_2d(mask):
    return mask_label_2d(mask)


def _ref_mesh(ref_3d):
    """ref_3d verts/faces (metres) → V_mm, F, V_rest_m."""
    V_rest_m = np.asarray(ref_3d.verts, dtype=float)
    F = np.asarray(ref_3d.faces, dtype=int)
    if V_rest_m.ndim != 2 or V_rest_m.shape[1] != 3:
        raise ValueError("ref_3d.verts must be (N, 3)")
    V_mm = V_rest_m * 1000.0
    return V_mm, F, V_rest_m


def _affine_image_to_model(reg_csv):
    """image (x,y) mm → model (x,z) mm affine."""
    img, mod = [], []
    for row in read_csv_dicts(reg_csv):
        img.append([float(row["imageX"]), float(row["imageY"])])
        mod.append([float(row["modelX"]), float(row["modelZ"])])
    if len(img) < 3:
        raise ValueError(
            "registration.csv needs >=3 anchors: %s" % reg_csv)
    img = np.asarray(img, dtype=float)
    mod = np.asarray(mod, dtype=float)
    A, *_ = np.linalg.lstsq(
        np.column_stack([img, np.ones(len(img))]), mod, rcond=None)
    return lambda xy: np.column_stack([xy, np.ones(len(xy))]) @ A


def _resample_curve(curve_xz, n):
    curve_xz = np.asarray(curve_xz, dtype=float)
    d = np.r_[0, np.cumsum(np.hypot(np.diff(curve_xz[:, 0]),
                                    np.diff(curve_xz[:, 1])))]
    if d[-1] == 0:
        return np.repeat(curve_xz[:1], n, axis=0)
    u = np.linspace(0, d[-1], n)
    return np.column_stack([
        np.interp(u, d, curve_xz[:, 0]),
        np.interp(u, d, curve_xz[:, 1]),
    ])


def _model_dorsal_curve(V_mm, nb):
    x = V_mm[:, 0]
    z = V_mm[:, 2]
    xq = np.linspace(x.min(), x.max(), nb)
    half = (x.max() - x.min()) / max(nb, 1)
    zc = np.full(nb, np.nan)
    for i, xi in enumerate(xq):
        sel = z[np.abs(x - xi) <= half]
        if len(sel):
            zc[i] = sel.max()
    ok = ~np.isnan(zc)
    zc = np.interp(xq, xq[ok], zc[ok])
    k = np.ones(3) / 3.0
    zs = np.convolve(zc, k, mode="same")
    zs[0], zs[-1] = zc[0], zc[-1]
    return np.column_stack([xq, zs])


def _displacement_colors(V_rest_m, V_def_m):
    disp = np.linalg.norm(V_def_m - V_rest_m, axis=1)
    dmax = float(disp.max()) if disp.size else 1.0
    if dmax <= 0:
        dmax = 1.0
    t = np.clip(disp / dmax, 0.0, 1.0)
    try:
        import matplotlib.cm as cm
        rgba = cm.viridis(t)
        return (rgba[:, :3] * 255.0).astype(np.uint8)
    except Exception:
        g = (t * 255.0).astype(np.uint8)
        return np.column_stack([g, g, g])


def retarget(ref_3d, source, target, nctrl=13, rbf_len=18.0, spatial_win=3,
             mm_per_px=None):
    """3D reference mesh + 2D rest/target mask → deformed mesh obj.

    Parameters
    ----------
    ref_3d : TongueModel
        3D rest tongue (``verts``/``faces`` metres).
        ``attach_registration(ref_3d, csv)`` 로 ``registration_csv`` 필요.
    source : (H, W, C) array
        2D **rest** MRI mask (``load_mask`` 출력).
    target : (H, W, C) array
        2D **target** MRI mask.

    Returns
    -------
    obj : dict
        ``points_cloud`` (metres), ``Mesh``, ``Color``.
    """
    reg_csv = _require_ref_3d(ref_3d)
    V_mm, F, V_rest_m = _ref_mesh(ref_3d)

    m2c = {}
    if mm_per_px is not None:
        m2c["mm_per_px"] = mm_per_px

    source_c = mask2contour(_mask_label_2d(source), **m2c)
    target_c = mask2contour(_mask_label_2d(target), **m2c)

    dorsal = _model_dorsal_curve(V_mm, nb=nctrl)
    to_model = _affine_image_to_model(reg_csv)

    source_xz = _resample_curve(to_model(source_c[:, :2]), nctrl)
    target_xz = _resample_curve(to_model(target_c[:, :2]), nctrl)
    delta = target_xz - source_xz

    if spatial_win > 1 and delta.shape[0] >= spatial_win:
        delta = uniform_filter1d(delta, spatial_win, axis=0, mode="nearest")

    Vxz = V_mm[:, [0, 2]]
    rbf = RBFInterpolator(
        dorsal, delta, kernel="gaussian",
        epsilon=1.0 / rbf_len, degree=-1, smoothing=1e-3)
    d_xz = rbf(Vxz)

    V_def_m = V_rest_m.copy()
    V_def_m[:, 0] += d_xz[:, 0] / 1000.0
    V_def_m[:, 2] += d_xz[:, 1] / 1000.0

    return {
        "points_cloud": V_def_m,
        "Mesh": F,
        "Color": _displacement_colors(V_rest_m, V_def_m),
    }
