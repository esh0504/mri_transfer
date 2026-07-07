# -*- coding: utf-8 -*-
"""
V2/retarget/retarget.py

retarget 패키지의 *공개 API*:

    mask2contour(mask)                  2D 혀 마스크 → dorsal arc contour (image-mm)
    register(rest_mask, ref_3d, csv)    image↔model affine anchor CSV 저장
    attach_registration(ref_3d, csv)    ref_3d에 registration_csv 경로 부착
    lift(contours) / lift_masks(masks)  2D contour → 대칭 3D dome surface
    retarget(ref_3d, source, target)    3D rest mesh + 2D rest/target mask → 변형 mesh

내부 헬퍼(contour primitive / affine / landmark 루틴)와 설정(configure)은
retarget/utils.py 에 있다. 파일 IO(mask/CSV)는 modules.utils 사용.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import uniform_filter1d

from modules.utils import mask_label_2d, save_csv

from . import utils
from .utils import (
    precise_contour, require_file, require_ref_3d, ref_mesh,
    affine_image_to_model, resample_curve, model_dorsal_curve, displacement_colors,
    load_landmark_map, fit_affine, pairs_from_auto, pairs_from_map,
)


# --------------------------------------------------------------------------- #
# contour
# --------------------------------------------------------------------------- #
def mask2contour(mask, n_markers=None, mm_per_px=None, clip_root=None,
                 clip_drop_frac=None):
    """2D 혀 마스크 → dorsal arc contour (image-mm).

    Parameters
    ----------
    mask : (H, W) array
        RT-MRI segmentation (label 4=tongue, 5=airway).

    Returns
    -------
    contour : (N, 3) float
        image-mm 좌표: x=col*mm/px (ant→post), y=(H-1-row)*mm/px (up), z=0.
    """
    if n_markers is None:
        n_markers = utils.N_MARKERS
    if mm_per_px is None:
        mm_per_px = utils.MM_PER_PX
    if clip_root is None:
        clip_root = utils.CLIP_ROOT
    if clip_drop_frac is None:
        clip_drop_frac = utils.CLIP_DROP_FRAC

    mask = np.asarray(mask)
    rc = precise_contour(
        mask, n=n_markers, clip_root=clip_root, clip_drop_frac=clip_drop_frac)
    if rc is None:
        raise ValueError("mask에서 dorsal contour를 추출하지 못했습니다.")

    H = mask.shape[0]
    x = rc[:, 1] * mm_per_px
    y = (H - 1 - rc[:, 0]) * mm_per_px
    return np.column_stack([x, y, np.zeros_like(x)])


def dorsal_contours_video(masks, mm_per_px=None, n=None, med=5, ma=3):
    """마스크 시퀀스 → 시간 일관 dorsal contour (image-mm).

    프레임마다 독립적으로 tip을 뽑지 않고, envelope apex tip을 프레임축으로 트래킹
    (median+이동평균 평활 후 표면 재투영)하여 tip↔root 대응이 튀지 않게 한다.
    비디오 전체를 retarget할 때 mask2contour 대신 사용.

    Parameters
    ----------
    masks : (T,H,W,C) / (T,H,W) array 또는 프레임 리스트.

    Returns
    -------
    contours : list[(n,3) float or None]   image-mm dorsal arcs (프레임별)
    tips : (T,2) float                      트래킹된 tip (row,col)
    """
    if n is None:
        n = utils.N_MARKERS
    if mm_per_px is None:
        mm_per_px = utils.MM_PER_PX
    masks2d = [mask_label_2d(m) for m in masks]
    tips, rcs = utils.track_dorsal_tips(masks2d, n=n, med=med, ma=ma)
    out = []
    for m2d, rc in zip(masks2d, rcs):
        if rc is None:
            out.append(None)
            continue
        H = m2d.shape[0]
        x = rc[:, 1] * mm_per_px
        y = (H - 1 - rc[:, 0]) * mm_per_px
        out.append(np.column_stack([x, y, np.zeros_like(x)]))
    return out, tips


# --------------------------------------------------------------------------- #
# Hybrid landmarks (anatomical anchors → 등쪽 리샘플) — DLC-swappable
# --------------------------------------------------------------------------- #
def hybrid_landmarks(mask, anchors=None, n=20):
    """tip→root N개 landmark를 '해부학적 anchor + 구간별 리샘플'로 추출 (row,col).

    anchors : {'tip','dorsum','root': (row,col)} dict. None이면 기하로 자동 계산.
        **DeepLabCut 예측 anchor를 그대로 넣으면** landmark identity가 그 anchor에
        고정되어 rest↔target 대응이 안정된다(이게 하이브리드의 핵심 이점).
    Returns (N,2) row,col 또는 None."""
    m2d = mask_label_2d(mask)
    if anchors is None:
        anchors, arc = utils.dorsal_anchors(m2d)
        if anchors is None:
            return None
    else:
        arc = utils.dorsal_arc(m2d)
        if arc is None:
            return None
    order = [anchors["tip"], anchors["dorsum"], anchors["root"]]
    c0 = n // 2 + 1
    c1 = n - (c0 - 1)
    return utils.landmarks_from_anchors(arc, order, [c0, c1])


def hybrid_landmarks_video(masks, n=20, med=5, ma=3):
    """마스크 시퀀스 → (T,n,2) landmark. anchor(tip/dorsum/root)를 프레임축으로
    median+이동평균 평활 후 각 프레임 표면에 재투영하여 identity를 안정화한다.
    (DLC 도입 시 프레임별 anchor를 예측값으로 바꾸면 그대로 대체됨.)"""
    masks2d = [mask_label_2d(m) for m in masks]
    arcs = [utils.dorsal_arc(m) for m in masks2d]
    names = ["tip", "dorsum", "root"]
    raw = {k: [] for k in names}
    for e in arcs:
        if e is None:
            for k in names:
                raw[k].append((np.nan, np.nan))
        else:
            raw["tip"].append(tuple(e[0]))
            raw["root"].append(tuple(e[-1]))
            raw["dorsum"].append(tuple(e[utils.dorsum_index(e)]))
    sm = {}
    for k in names:
        a = np.array(raw[k], float)
        valid = ~np.isnan(a[:, 0]); idx = np.where(valid)[0]
        for t in range(len(a)):
            if not valid[t] and len(idx):
                a[t] = a[idx[np.argmin(np.abs(idx - t))]]
        ref = utils.median_filter_2d(a, med)
        if ma > 1:
            kk = np.ones(ma) / ma
            rs = np.convolve(ref[:, 0], kk, "same"); cs = np.convolve(ref[:, 1], kk, "same")
            rs[0], cs[0] = ref[0]; rs[-1], cs[-1] = ref[-1]
            ref = np.column_stack([rs, cs])
        sm[k] = ref
    c0 = n // 2 + 1; c1 = n - (c0 - 1)
    out = []
    for t, e in enumerate(arcs):
        if e is None:
            out.append(None); continue
        order = []
        for k in names:
            r = sm[k][t]
            j = int(np.argmin(np.hypot(e[:, 0] - r[0], e[:, 1] - r[1])))
            order.append(tuple(e[j]))
        out.append(utils.landmarks_from_anchors(e, order, [c0, c1]))
    return out


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
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
        names, img_xy, mod_m = pairs_from_map(load_landmark_map(landmark_map))
    else:
        names, img_xy, mod_m = pairs_from_auto(rest_mask, ref_3d, mm_per_px)

    A, rms_m, res_m = fit_affine(img_xy, mod_m)
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
    ref_3d.registration_csv = require_file(registration_csv, "registration_csv")
    return ref_3d


# --------------------------------------------------------------------------- #
# kinematic lift (2D contour → symmetric 3D dome)
# --------------------------------------------------------------------------- #
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
        mask2contour(mask_label_2d(masks[k]), **m2c) for k in range(masks.shape[0])
    ], axis=0)
    return lift(contours, nz=nz, half_w=half_w, edge_drop=edge_drop,
                width_end=width_end)


# --------------------------------------------------------------------------- #
# retarget (3D reference mesh + 2D rest/target mask → deformed mesh)
# --------------------------------------------------------------------------- #
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
    reg_csv = require_ref_3d(ref_3d)
    V_mm, F, V_rest_m = ref_mesh(ref_3d)

    m2c = {}
    if mm_per_px is not None:
        m2c["mm_per_px"] = mm_per_px

    source_c = mask2contour(mask_label_2d(source), **m2c)
    target_c = mask2contour(mask_label_2d(target), **m2c)

    dorsal = model_dorsal_curve(V_mm, nb=nctrl)
    to_model = affine_image_to_model(reg_csv)

    source_xz = resample_curve(to_model(source_c[:, :2]), nctrl)
    target_xz = resample_curve(to_model(target_c[:, :2]), nctrl)
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
        "Color": displacement_colors(V_rest_m, V_def_m),
    }
