#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2/retarget/utils.py

retarget 패키지의 *헬퍼 계층* — 설정 상태 + contour/registration/retarget 내부 루틴.
공개 API(mask2contour / register / lift / retarget / attach_registration)는
retarget/retarget.py 에 있다. 파일 IO(mask/CSV)는 modules.utils 사용.

contour 추출 primitive:
    precise_contour(mask, n)      airway를 향한 dorsal 표면 (row,col), tip→root
    full_boundary_contour(mask)   전체 닫힌 혀 윤곽
    anatomical_landmarks(mask)    tip/dorsum/root/floor 랜드마크

환경 변수 (기본값):
    MM_PER_PX (1.164)  픽셀→mm 배율
    N_MARKERS (25)     dorsal contour 샘플 수
    CLIP_ROOT (1)      후방 spur 제거 여부
    CLIP_DROP_FRAC (1.0)
"""
import os

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours, label as sklabel
from skimage.morphology import remove_small_objects, binary_closing, disk

from modules.utils import read_csv_dicts

# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #
MM_PER_PX = float(os.environ.get("MM_PER_PX", "1.164"))
N_MARKERS = int(os.environ.get("N_MARKERS", "25"))

LBL_TONGUE = 4
LBL_AIRWAY = 5

CLIP_ROOT = os.environ.get("CLIP_ROOT", "1").lower() not in ("0", "false", "no")
CLIP_DROP_FRAC = float(os.environ.get("CLIP_DROP_FRAC", "1.0"))

# 사용자가 수작업으로 지정한 midsag dorsal 정점 인덱스(tip→root 순서). None이면 자동 추정.
# configure(midsag_indices=[...]) 또는 config의 contour.midsag_indices 로 설정.
MIDSAG_ORDER = None

# contour 알고리즘 버전: v1(legacy) / v2(envelope+temporal) / v3(normal+dorsum-run).
# 버전별 설명은 versionmanagement.md 참조.
CONTOUR_MODE = os.environ.get("CONTOUR_MODE", "v2")


def configure(cfg=None, **overrides):
    """contour 기본 파라미터를 설정으로 덮어쓴다 (Hydra 등에서 호출).

    사용: configure(cfg.retarget.contour, mm_per_px=cfg.retarget.mm_per_px).
    인식 키: n_markers, clip_root, clip_drop_frac, mm_per_px, mode(=contour 버전).
    """
    global MM_PER_PX, N_MARKERS, CLIP_ROOT, CLIP_DROP_FRAC, CONTOUR_MODE, MIDSAG_ORDER
    opts = dict(cfg) if cfg else {}
    opts.update(overrides)
    if opts.get("midsag_indices") is not None:   # 수작업 지정 midsag dorsal 정점(tip→root)
        MIDSAG_ORDER = list(opts["midsag_indices"])
    if opts.get("n_markers") is not None:
        N_MARKERS = int(opts["n_markers"])
    if opts.get("clip_root") is not None:
        CLIP_ROOT = bool(opts["clip_root"])
    if opts.get("clip_drop_frac") is not None:
        CLIP_DROP_FRAC = float(opts["clip_drop_frac"])
    if opts.get("mm_per_px") is not None:
        MM_PER_PX = float(opts["mm_per_px"])
    if opts.get("mode") is not None:
        CONTOUR_MODE = str(opts["mode"]).lower()


# --------------------------------------------------------------------------- #
# contour 추출 primitive
# --------------------------------------------------------------------------- #
def longest_true_run_cyclic(mask_bool):
    n = len(mask_bool)
    if mask_bool.all():
        return 0, n
    f2 = np.concatenate([mask_bool, mask_bool])
    best_len, best = 0, (0, 0)
    cur = start = 0
    for i in range(2 * n):
        if f2[i]:
            if cur == 0:
                start = i
            cur += 1
            if cur > best_len:
                best_len, best = cur, (start, i + 1)
        else:
            cur = 0
    return best[0], best[1]


def clip_posterior_spur(c, drop_frac=1.0, x_reversal=True, rev_tol=0.5):
    """Trim the posterior 'spur' from a tip->root arc (row,col) so the 2D tongue
    extent matches the 3D model (no deep pharyngeal/curl-back limb).

    Walk from the dorsum peak (max z = min row) toward the root and cut at the
    first of:
      (a) x (col) reversal  -> the contour doubles back (the notch/spur)
      (b) z descended more than drop_frac * (peak_z - tip_z) below the peak
    drop_frac<=0 disables (b). Returns the clipped arc (tip..cut)."""
    if len(c) < 4:
        return c
    z = -c[:, 0]
    col = c[:, 1]
    pk = int(np.argmax(z))
    rise = max(1e-6, z[pk] - z[0])
    cut = len(c)
    for i in range(pk + 1, len(c)):
        if x_reversal and col[i] < col[i - 1] - rev_tol:
            cut = i
            break
        if drop_frac > 0 and z[i] < z[pk] - drop_frac * rise:
            cut = i
            break
    return c[:max(pk + 2, cut)]


def smooth_closed(c, win=3):
    """Cyclic moving-average smoothing of a closed (row,col) boundary."""
    if win <= 1 or len(c) < win:
        return c
    pad = win
    cc = np.vstack([c[-pad:], c, c[:pad]])
    k = np.ones(win) / win
    r = np.convolve(cc[:, 0], k, "same")[pad:-pad]
    col = np.convolve(cc[:, 1], k, "same")[pad:-pad]
    return np.column_stack([r, col])


def facing_arc(mask, facing_thresh=2.5, smooth_win=3):
    """airway를 향한 dorsal 경계의 가장 긴 연속 구간, tip(최소 col)→root 순서로.

    Returns (M,2) row,col sub-pixel points, or None."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return None
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return None
    c = smooth_closed(max(cs, key=len), smooth_win)
    airway = (mask == LBL_AIRWAY)
    if airway.sum() == 0:
        return c
    dt = distance_transform_edt(~airway)       # distance to nearest airway pixel
    rr = np.clip(c[:, 0].round().astype(int), 0, mask.shape[0] - 1)
    cc = np.clip(c[:, 1].round().astype(int), 0, mask.shape[1] - 1)
    facing = dt[rr, cc] <= facing_thresh
    if facing.sum() < 5:
        return None
    s, e = longest_true_run_cyclic(facing)
    arc = c[np.arange(s, e) % len(c)]
    if arc[0, 1] > arc[-1, 1]:                  # orient tip(min col) first
        arc = arc[::-1]
    return arc


def clip_anterior_drop(arc, drop_frac=0.5):
    """앞쪽 급강하(앞-바닥 노치)를 잘라 contour가 dorsal apex에서 시작하게 한다.

    dorsum 정점에서 tip 방향으로 걸으며 tip→peak 상승폭의 drop_frac 이상 내려간
    지점을 잘라, 혀 앞끝 돌기의 바닥이 아니라 등쪽 apex를 tip으로 잡도록 한다."""
    if len(arc) < 4:
        return arc
    z = -arc[:, 0]
    pk = int(np.argmax(z))
    if pk == 0:
        return arc
    rise = max(1e-6, z[pk] - z[:pk + 1].min())
    cut = 0
    for i in range(pk - 1, -1, -1):
        if z[i] < z[pk] - drop_frac * rise:
            cut = i + 1
            break
    return arc[cut:]


def dorsal_envelope(arc):
    """facing arc의 상단 포락선: 열(column)마다 가장 위쪽(min row) 점만 취함.

    단일값(top surface)이라 혀 앞끝 돌기를 감는 루프가 생기지 않는다. tip→root(col)."""
    env = {}
    for r, c in arc:
        cb = int(round(c))
        if cb not in env or r < env[cb][0]:
            env[cb] = (r, c)
    return np.array([env[k] for k in sorted(env)], float)


def resample_rowcol(e, n, smooth_win=3):
    """(row,col) 곡선을 평활 후 호길이 등간격으로 n점 리샘플."""
    if len(e) < 3:
        return None
    r_, c_ = e[:, 0].astype(float), e[:, 1].astype(float)
    if smooth_win > 1 and len(e) >= smooth_win:
        k = np.ones(smooth_win) / smooth_win
        rs = np.convolve(r_, k, "same"); cs = np.convolve(c_, k, "same")
        rs[0], cs[0] = r_[0], c_[0]; rs[-1], cs[-1] = r_[-1], c_[-1]; r_, c_ = rs, cs
    d = np.r_[0, np.cumsum(np.hypot(np.diff(c_), np.diff(r_)))]
    if d[-1] == 0:
        return None
    u = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(u, d, r_), np.interp(u, d, c_)])


def precise_contour(mask, n=60, facing_thresh=2.5, smooth_win=3,
                    clip_root=False, clip_drop_frac=1.0, ant_drop=0.5, mode=None):
    """Dorsal contour 디스패처 — CONTOUR_MODE(또는 mode 인자)로 버전 선택.

    v1 = legacy(min-col + longest-run), v2 = envelope + anterior-clip(기본),
    v3 = normal + dorsum-run + upper-anterior tip(ContourExtract.md),
    v4 = V3 tip + V2 body/root, v5 = V1 root + tip wrap 제거,
    v6 = V5 root + tip을 normal(underside 꺾임)로 연장,
    v7 = V6와 유사하되 tip을 normal 각도 '급변(corner)'+좌측각 cap,
    v8 = V7 tip + root도 normal(완전 우측 90° 직전)로.
    버전별 알고리즘 설명은 versionmanagement.md 참조."""
    mode = (mode or CONTOUR_MODE).lower()
    if mode == "v1":
        return precise_contour_v1(mask, n, facing_thresh, smooth_win,
                                  clip_root, clip_drop_frac)
    if mode == "v3":
        return precise_contour_v3(mask, n, facing_thresh, clip_root=clip_root,
                                  clip_drop_frac=clip_drop_frac, smooth_win=smooth_win)
    if mode == "v4":
        return precise_contour_v4(mask, n, facing_thresh, clip_root=clip_root,
                                  clip_drop_frac=clip_drop_frac, smooth_win=smooth_win)
    if mode == "v5":
        return precise_contour_v5(mask, n, facing_thresh, smooth_win,
                                  clip_root, clip_drop_frac)
    if mode == "v6":
        return precise_contour_v6(mask, n, facing_thresh, smooth_win,
                                  clip_root, clip_drop_frac)
    if mode == "v7":
        return precise_contour_v7(mask, n, facing_thresh, smooth_win,
                                  clip_root, clip_drop_frac)
    if mode == "v8":
        return precise_contour_v8(mask, n, facing_thresh, smooth_win,
                                  clip_root, clip_drop_frac)
    return precise_contour_v2(mask, n, facing_thresh, smooth_win,
                              clip_root, clip_drop_frac, ant_drop)


def precise_contour_v1(mask, n=60, facing_thresh=2.5, smooth_win=3,
                       clip_root=False, clip_drop_frac=1.0):
    """V1 (legacy): airway-facing 가장 긴 run + min-col tip. tip이 앞-바닥으로
    감기는 실패가 있었던 원래 방식(비교/fallback용). (n,2) row,col 또는 None."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return None
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    airway = (mask == LBL_AIRWAY)
    if airway.sum() > 0:
        dt = distance_transform_edt(~airway)
        rr = np.clip(c[:, 0].round().astype(int), 0, mask.shape[0] - 1)
        cc = np.clip(c[:, 1].round().astype(int), 0, mask.shape[1] - 1)
        facing = dt[rr, cc] <= facing_thresh
        if facing.sum() >= 5:
            s, e = longest_true_run_cyclic(facing)
            c = c[np.arange(s, e) % len(c)]
    if c[0, 1] > c[-1, 1]:
        c = c[::-1]
    if clip_root:
        c = clip_posterior_spur(c, drop_frac=clip_drop_frac)
    return resample_rowcol(c, n, smooth_win)


def precise_contour_v2(mask, n=60, facing_thresh=2.5, smooth_win=3,
                       clip_root=False, clip_drop_frac=1.0, ant_drop=0.5):
    """V2: 상단 포락선(upper-envelope) + anterior-clip. 열마다 airway를 향한 가장
    위쪽 경계점만 취해(단일값 top surface → tip 루프 없음) apex를 tip으로 잡는다.
    (video는 track_dorsal_tips로 시간 평활.) (n,2) row,col 또는 None."""
    arc = facing_arc(mask, facing_thresh, smooth_win=1)
    if arc is None:
        return None
    e = dorsal_envelope(arc)
    if ant_drop:
        e = clip_anterior_drop(e, ant_drop)
    if clip_root:
        e = clip_posterior_spur(e, drop_frac=clip_drop_frac)
    return resample_rowcol(e, n, smooth_win)


def full_boundary_contour(mask, n=80, smooth_win=3):
    """Full CLOSED tongue (label 4) outline, sub-pixel, arc-length resampled to
    n points with consistent cross-frame correspondence (start at tip = min col,
    dorsum first, evenly spaced around the loop). Returns (n,2) row,col or None."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return None
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    if len(c) < 8:
        return None

    t0 = int(np.argmin(c[:, 1]))               # start at tip = anterior-most
    c = np.roll(c, -t0, axis=0)

    k = max(3, len(c) // 10)                    # orient dorsum (superior) first
    fwd_sup = -c[1:1 + k, 0].mean()
    bwd_sup = -c[-k:, 0].mean()
    if bwd_sup > fwd_sup:
        c = np.vstack([c[0], c[1:][::-1]])

    if smooth_win and smooth_win > 1 and len(c) >= smooth_win:
        pad = smooth_win
        cc = np.vstack([c[-pad:], c, c[:pad]])
        ker = np.ones(smooth_win) / smooth_win
        rs = np.convolve(cc[:, 0], ker, "same")[pad:-pad]
        cm = np.convolve(cc[:, 1], ker, "same")[pad:-pad]
        c = np.column_stack([rs, cm])

    cl = np.vstack([c, c[0]])
    d = np.r_[0, np.cumsum(np.hypot(np.diff(cl[:, 1]), np.diff(cl[:, 0])))]
    if d[-1] == 0:
        return None
    u = np.linspace(0, d[-1], n, endpoint=False)
    return np.column_stack([np.interp(u, d, cl[:, 0]), np.interp(u, d, cl[:, 1])])


def anatomical_landmarks(mask):
    """Stable midsagittal tongue landmarks as {name:(row,col)} on the sub-pixel
    boundary: tip=anterior-most, dorsum=superior-most, root=posterior-most,
    floor=inferior-most. Returns None if the tongue is missing."""
    c = full_boundary_contour(mask, n=400, smooth_win=3)
    if c is None:
        return None
    return {
        "tip":    (float(c[np.argmin(c[:, 1]), 0]), float(c[np.argmin(c[:, 1]), 1])),
        "dorsum": (float(c[np.argmin(c[:, 0]), 0]), float(c[np.argmin(c[:, 0]), 1])),
        "root":   (float(c[np.argmax(c[:, 1]), 0]), float(c[np.argmax(c[:, 1]), 1])),
        "floor":  (float(c[np.argmax(c[:, 0]), 0]), float(c[np.argmax(c[:, 0]), 1])),
    }


# --------------------------------------------------------------------------- #
# 공통 입력 검증 / 좌표 헬퍼
# --------------------------------------------------------------------------- #
def require_file(path, label="path"):
    p = os.path.abspath(str(path))
    if not os.path.isfile(p):
        raise FileNotFoundError("%s not found: %s" % (label, p))
    return p


def mask_2d(mask):
    """(H,W,C) or (H,W) → (H,W). 첫 채널만 사용."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        return mask[..., 0]
    if mask.ndim == 2:
        return mask
    raise ValueError("mask must be (H,W) or (H,W,C)")


# --------------------------------------------------------------------------- #
# retarget 내부 루틴 (mesh / affine / curve)
# --------------------------------------------------------------------------- #
def require_ref_3d(ref_3d):
    if ref_3d is None:
        raise ValueError("ref_3d is required")
    if getattr(ref_3d, "verts", None) is None or getattr(ref_3d, "faces", None) is None:
        raise ValueError("ref_3d must have verts (N,3) and faces (F,3)")
    reg = getattr(ref_3d, "registration_csv", None)
    if not reg:
        raise ValueError(
            "ref_3d.registration_csv is required — "
            "use attach_registration(ref_3d, registration_csv)")
    return require_file(reg, "registration_csv")


def ref_mesh(ref_3d):
    """ref_3d verts/faces (metres) → V_mm, F, V_rest_m."""
    V_rest_m = np.asarray(ref_3d.verts, dtype=float)
    F = np.asarray(ref_3d.faces, dtype=int)
    if V_rest_m.ndim != 2 or V_rest_m.shape[1] != 3:
        raise ValueError("ref_3d.verts must be (N, 3)")
    V_mm = V_rest_m * 1000.0
    return V_mm, F, V_rest_m


def affine_image_to_model(reg_csv):
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


def resample_curve(curve_xz, n):
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


def midsag_dorsal_order(V_mm, y_tol=6.0):
    """rest 모델의 midsag 정점(|y|≤y_tol)을 무게중심 각도로 loop 정렬한 뒤, dorsal(위쪽)
    아크만 tip(최소 x)→root(최대 x) 순서로. **정점 집합이 고정**이라 프레임 간 일관된
    dorsal contour를 준다(매 프레임 max-z 재계산으로 생기는 스파이크 없음). 반환: 정점 인덱스.

    사용자가 MIDSAG_ORDER(수작업 지정 인덱스)를 설정하면 자동 추정 대신 그걸 그대로 쓴다."""
    if MIDSAG_ORDER is not None:
        return np.asarray(MIDSAG_ORDER, dtype=int)
    y = V_mm[:, 1]
    idx = np.where(np.abs(y) <= y_tol)[0]
    if len(idx) < 6:
        idx = np.arange(len(V_mm))
    P = V_mm[idx][:, [0, 2]]
    ctr = P.mean(axis=0)
    order = np.argsort(np.arctan2(P[:, 1] - ctr[1], P[:, 0] - ctr[0]))   # loop 정렬
    Lidx = idx[order]; Lp = P[order]; n = len(Lp)
    ti = int(np.argmin(Lp[:, 0])); ri = int(np.argmax(Lp[:, 0]))
    arc1 = [(ti + k) % n for k in range((ri - ti) % n + 1)]
    arc2 = [(ri + k) % n for k in range((ti - ri) % n + 1)]
    sel = np.array(arc1 if Lp[arc1, 1].mean() >= Lp[arc2, 1].mean() else arc2[::-1])
    if Lp[sel[0], 0] > Lp[sel[-1], 0]:                 # tip(min x)이 먼저
        sel = sel[::-1]
    return Lidx[sel]


def model_dorsal_curve(V_mm, nb, y_tol=6.0, order=None):
    """혀 중앙(midsag) dorsal 곡선 [x, z] (mm), nb점 (호길이 등간격).

    rest에서 고정한 midsag dorsal 정점(midsag_dorsal_order)을 리샘플한다. order를 주면
    그 고정 인덱스를 그대로 사용(변형 메쉬에 rest 순서 적용). 없으면 V_mm에서 계산."""
    idx = midsag_dorsal_order(V_mm, y_tol) if order is None else np.asarray(order)
    D = V_mm[idx][:, [0, 2]].astype(float)
    keep = np.r_[True, (np.abs(np.diff(D[:, 0])) + np.abs(np.diff(D[:, 1]))) > 1e-6]
    D = D[keep]
    if len(D) < 2:
        return np.repeat(D[:1], nb, axis=0) if len(D) else np.zeros((nb, 2))
    s = np.r_[0, np.cumsum(np.hypot(np.diff(D[:, 0]), np.diff(D[:, 1])))]
    if s[-1] == 0:
        return np.repeat(D[:1], nb, axis=0)
    u = np.linspace(0, s[-1], nb)
    return np.column_stack([np.interp(u, s, D[:, 0]), np.interp(u, s, D[:, 1])])


def displacement_colors(V_rest_m, V_def_m):
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


# --------------------------------------------------------------------------- #
# registration 내부 루틴 (landmark / affine fit)
# --------------------------------------------------------------------------- #
def model_landmarks_m(ref_3d, y_tol=0.003):
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


def image_landmarks_mm(rest_mask, mm_per_px):
    """rest mask → tip/dorsum/root in image-mm (x,y)."""
    H = mask_2d(rest_mask).shape[0]
    lm = anatomical_landmarks(mask_2d(rest_mask))
    if lm is None:
        raise ValueError("rest mask에서 anatomical landmarks 추출 실패")
    out = {}
    for name in ("tip", "dorsum", "root"):
        if name not in lm:
            raise ValueError("landmark '%s' missing in rest mask" % name)
        r, c = lm[name]
        out[name] = (float(c) * mm_per_px, float((H - 1) - r) * mm_per_px)
    return out


def load_landmark_map(path):
    """landmark_map.csv → {label: (imageX, imageY, modelX_m, modelZ_m)}."""
    path = require_file(path, "landmark_map")
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


def fit_affine(img_xy, mod_xz):
    M = np.column_stack([img_xy, np.ones(len(img_xy))])
    A, *_ = np.linalg.lstsq(M, mod_xz, rcond=None)
    pred = M @ A
    res = np.linalg.norm(pred - mod_xz, axis=1)
    return A, float(np.sqrt((res ** 2).mean())), res


def pairs_from_auto(rest_mask, ref_3d, mm_per_px):
    img_lm = image_landmarks_mm(rest_mask, mm_per_px)
    mod_lm = model_landmarks_m(ref_3d)
    names = ["tip", "dorsum", "root"]
    img = np.array([img_lm[k] for k in names], dtype=float)
    mod = np.array([mod_lm[k] for k in names], dtype=float)
    return names, img, mod


def pairs_from_map(landmark_map):
    names = list(landmark_map.keys())
    img = np.array([[v[0], v[1]] for v in landmark_map.values()], dtype=float)
    mod = np.array([[v[2], v[3]] for v in landmark_map.values()], dtype=float)
    return names, img, mod


# --------------------------------------------------------------------------- #
# 시간축 tip 트래킹 (video) — 프레임 간 tip 일관성 (Normal/envelope + smallest-change)
# --------------------------------------------------------------------------- #
def dorsal_contour_from_tip(mask, tip, n=25, facing_thresh=2.5, smooth_win=3,
                            clip_drop_frac=1.0):
    """정해진 tip(row,col)에서 시작하도록 앵커링한 dorsal contour.

    트래킹된 tip에서 root까지 상단 포락선을 따라 추출 → 프레임 간 tip↔root 대응 일관."""
    arc = facing_arc(mask, facing_thresh, smooth_win=1)
    if arc is None:
        return None
    e = dorsal_envelope(arc)
    e = e[e[:, 1] >= tip[1] - 1.0]              # tip 앞쪽(더 anterior) 열 제거
    if len(e) < 3:
        return None
    e[0] = np.asarray(tip, float)              # 정확히 tip에서 시작
    e = clip_posterior_spur(e, drop_frac=clip_drop_frac)
    return resample_rowcol(e, n, smooth_win)


def median_filter_2d(a, size=5):
    """프레임축(axis 0) 슬라이딩 median — tip 궤적의 outlier 제거."""
    T = len(a); h = size // 2; out = a.copy()
    for t in range(T):
        lo, hi = max(0, t - h), min(T, t + h + 1)
        out[t] = np.median(a[lo:hi], axis=0)
    return out


def track_dorsal_tips(masks, n=25, med=5, ma=3, dev_project=True):
    """마스크 시퀀스 → 시간 일관 dorsal tip + contour.

    프레임별로 envelope apex tip을 구한 뒤, 시간축 'smallest-change' 평활을 적용:
    median filter(outlier 제거) + moving average, 그리고 각 프레임 tip을 자기 표면
    (facing arc)에 다시 투영한다. Returns (tips (T,2) row,col, contours list)."""
    arcs = [facing_arc(m, smooth_win=1) for m in masks]
    tips = []
    for a in arcs:
        c = resample_rowcol(clip_anterior_drop(dorsal_envelope(a), 0.5), n) \
            if a is not None else None
        tips.append(tuple(c[0]) if c is not None else (np.nan, np.nan))
    tips = np.array(tips, float)
    valid = ~np.isnan(tips[:, 0])
    if valid.sum() == 0:
        return tips, [None] * len(masks)
    idx = np.where(valid)[0]
    for t in range(len(tips)):                 # 빈 프레임은 가장 가까운 유효 프레임으로
        if not valid[t]:
            tips[t] = tips[idx[np.argmin(np.abs(idx - t))]]
    ref = median_filter_2d(tips, med)          # 1) outlier 제거
    if ma > 1:                                 # 2) 이동평균 평활
        k = np.ones(ma) / ma
        rs = np.convolve(ref[:, 0], k, "same"); cs = np.convolve(ref[:, 1], k, "same")
        rs[0], cs[0] = ref[0]; rs[-1], cs[-1] = ref[-1]
        ref = np.column_stack([rs, cs])
    final = tips.copy()
    if dev_project:                            # 3) 평활 tip을 각 프레임 표면에 재투영
        for t, a in enumerate(arcs):
            if a is not None:
                j = int(np.argmin(np.hypot(a[:, 0] - ref[t, 0], a[:, 1] - ref[t, 1])))
                final[t] = a[j]
    contours = [dorsal_contour_from_tip(m, final[t], n=n) if arcs[t] is not None else None
                for t, m in enumerate(masks)]
    return final, contours


# --------------------------------------------------------------------------- #
# Hybrid landmarks: anatomical anchors → segment별 리샘플 (DLC-swappable)
# --------------------------------------------------------------------------- #
def dorsal_arc(mask, facing_thresh=2.5, ant_drop=0.5, clip_drop_frac=1.0):
    """정리된 dorsal 표면 (row,col) tip(apex)→root, 가변 길이(리샘플 전). None 가능."""
    a = facing_arc(mask, facing_thresh, smooth_win=1)
    if a is None:
        return None
    e = dorsal_envelope(a)
    if ant_drop:
        e = clip_anterior_drop(e, ant_drop)
    e = clip_posterior_spur(e, drop_frac=clip_drop_frac)
    return e if len(e) >= 3 else None


def dorsum_index(e, min_frac=0.30):
    """dorsum = tip 영역(호길이 min_frac 이내) 제외 후 최상단(min row) 점 인덱스.

    혀끝이 들린 자세에서 raised tip을 dorsum으로 오인하지 않도록 앞부분을 배제한다."""
    s = np.r_[0, np.cumsum(np.hypot(np.diff(e[:, 0]), np.diff(e[:, 1])))]
    frac = s / (s[-1] + 1e-9)
    idx = np.where(frac >= min_frac)[0]
    if len(idx) == 0:
        idx = np.arange(len(e))
    return int(idx[np.argmin(e[idx, 0])])


def dorsal_anchors(mask):
    """기하 기반 anchor(추후 DLC 예측으로 교체) — tip/dorsum/root (row,col).

    Returns (anchors dict, dorsal_arc (M,2)) 또는 (None, None)."""
    e = dorsal_arc(mask)
    if e is None:
        return None, None
    anchors = {"tip": tuple(e[0]), "dorsum": tuple(e[dorsum_index(e)]), "root": tuple(e[-1])}
    return anchors, e


def landmarks_from_anchors(arc, anchors_ordered, seg_counts):
    """dorsal arc를 순서 anchor '사이 구간'별로 호길이 등간격 리샘플.

    arc : (M,2) row,col tip→root.
    anchors_ordered : [(row,col)] tip→root 순서(양 끝은 arc[0], arc[-1] 권장).
    seg_counts : 구간별 점 수(길이 = anchor 수 − 1); 인접 구간은 경계 anchor를 공유.
    반환 (N,2): landmark index가 프레임 간 같은 해부학적 위치(anchor로 고정)."""
    e = np.asarray(arc, float)
    s = np.r_[0, np.cumsum(np.hypot(np.diff(e[:, 0]), np.diff(e[:, 1])))]
    pos = []
    for a in anchors_ordered:
        j = int(np.argmin(np.hypot(e[:, 0] - a[0], e[:, 1] - a[1])))
        pos.append(s[j])
    pos = np.sort(np.array(pos, float))
    pos[0], pos[-1] = s[0], s[-1]
    out = []
    for i in range(len(pos) - 1):
        u = np.linspace(pos[i], pos[i + 1], seg_counts[i])
        seg = np.column_stack([np.interp(u, s, e[:, 0]), np.interp(u, s, e[:, 1])])
        out.append(seg if i == 0 else seg[1:])   # 공유 경계 중복 제거
    return np.vstack(out)


# --------------------------------------------------------------------------- #
# V3 — ContourExtract.md precise_contour_v2
#   outward-normal filter + airway cleanup + dorsum-guided run 선택 +
#   upper-anterior tip + dorsal-side trim + (옵션) temporal prev_tip.
# --------------------------------------------------------------------------- #
def clean_airway(airway, min_size=20, closing_radius=1, keep_largest=False):
    """작은 airway 파편 제거 + closing. keep_largest는 oral airway가 여러 조각으로
    갈라지는 프레임에서 문제될 수 있어 기본 off (ContourExtract.md §8)."""
    a = remove_small_objects(np.asarray(airway).astype(bool), min_size=min_size)
    if closing_radius > 0:
        a = binary_closing(a, disk(closing_radius))
    if keep_largest:
        lab = sklabel(a)
        if lab.max() > 0:
            counts = np.bincount(lab.ravel()); counts[0] = 0
            a = lab == counts.argmax()
    return a


def outward_normals_from_contour(cont, tongue_mask, eps=1.5):
    """(row,col) contour의 각 점에서 tongue 밖을 향한 단위 outward normal (N,2)."""
    prev_p = np.roll(cont, 1, axis=0); next_p = np.roll(cont, -1, axis=0)
    tangent = next_p - prev_p
    tangent = tangent / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8)
    n1 = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    H, W = tongue_mask.shape

    def inside(p):
        rr = np.clip(np.rint(p[:, 0]).astype(int), 0, H - 1)
        cc = np.clip(np.rint(p[:, 1]).astype(int), 0, W - 1)
        return tongue_mask[rr, cc]
    inside1 = inside(cont + eps * n1)
    return np.where((~inside1)[:, None], n1, -n1)


def close_small_false_gaps_cyclic(keep, max_gap=4):
    """cyclic bool에서 짧은 False 구간(≤max_gap)을 메워 run이 잘게 쪼개지는 것 방지."""
    n = len(keep); k = np.asarray(keep).copy()
    if k.all() or not k.any():
        return k
    start = int(np.where(k)[0][0]); rolled = np.roll(k, -start); i = 0
    while i < n:
        if not rolled[i]:
            j = i
            while j < n and not rolled[j]:
                j += 1
            if (j - i) <= max_gap:
                rolled[i:j] = True
            i = j
        else:
            i += 1
    return np.roll(rolled, start)


def true_runs_cyclic_indices(keep):
    """cyclic bool → True 연속 구간들의 인덱스 배열 리스트(랩어라운드 처리)."""
    n = len(keep); keep = np.asarray(keep)
    if not keep.any():
        return []
    if keep.all():
        return [np.arange(n)]
    start = int(np.where(~keep)[0][0]); rolled = np.roll(keep, -start); runs = []; i = 0
    while i < n:
        if rolled[i]:
            j = i
            while j < n and rolled[j]:
                j += 1
            runs.append(np.array([(start + t) % n for t in range(i, j)]))
            i = j
        else:
            i += 1
    return runs


def cyclic_index_distance_to_run(idx, run_indices, n):
    d = np.abs(np.asarray(run_indices) - idx)
    return float(np.minimum(d, n - d).min())


def score_run(run_indices, cont, normal, dorsum_idx, prev_tip=None,
              w_dorsum=2.5, w_length=1.0, w_ventral=1.5, w_temporal=1.0):
    """run 점수(클수록 좋음): dorsum 포함 + 길이 + 비-ventral + (옵션) prev-tip 근접."""
    n = len(cont); pts = cont[run_indices]; normals = normal[run_indices]
    length_score = len(run_indices) / n
    dorsum_dist = cyclic_index_distance_to_run(dorsum_idx, run_indices, n)
    dorsum_score = 1.0 - min(dorsum_dist / max(n * 0.15, 1), 1.0)
    ventral_score = 1.0 - np.mean(normals[:, 0] > 0.55)
    cmin = pts[:, 1].min(); cand = np.where(pts[:, 1] <= cmin + 8)[0]
    tip = pts[cand[np.argmin(pts[cand, 0])]] if len(cand) else pts[np.argmin(pts[:, 1])]
    temporal_score = 0.0
    if prev_tip is not None:
        temporal_score = -min(np.linalg.norm(tip - prev_tip) / 20.0, 2.0)
    return (w_dorsum * dorsum_score + w_length * length_score
            + w_ventral * ventral_score + w_temporal * temporal_score)


def choose_upper_anterior_tip(seg, tip_band_px=8):
    """anterior band(최소 col + tip_band_px) 안에서 가장 위쪽(min row) 점 = true tip."""
    cols = seg[:, 1]; rows = seg[:, 0]; cmin = cols.min()
    cand = np.where(cols <= cmin + tip_band_px)[0]
    if len(cand) == 0:
        return int(np.argmin(cols))
    return int(cand[np.argmin(rows[cand])])


def keep_dorsal_side_after_tip(seg, tip_idx):
    """tip 주변 lower/floor curl 제거 — 더 rootward·dorsal한 쪽만 유지."""
    side_a = seg[tip_idx:]; side_b = seg[:tip_idx + 1][::-1]

    def side_score(side):
        if len(side) < 3:
            return -1e9
        return 2.0 * (side[-1, 1] - side[0, 1]) + 0.2 * len(side) + 0.05 * (-np.mean(side[:, 0]))
    return side_a if side_score(side_a) >= side_score(side_b) else side_b


def precise_contour_v3(mask, n=25, facing_thresh=2.5, normal_row_max=0.35,
                       tip_band_px=8, gap_close_pts=4, prev_tip=None, clip_root=True,
                       clip_drop_frac=1.0, airway_min_size=20, airway_closing_radius=1,
                       airway_keep_largest=False, smooth_win=3, return_debug=False):
    """V3 (ContourExtract.md): normal 필터 + dorsum-guided run + upper-anterior tip.

    후보 run이 없으면 V2로 fallback. return_debug면 (contour, dbg dict) 반환."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return (None, {}) if return_debug else None
    airway = clean_airway(mask == LBL_AIRWAY, airway_min_size, airway_closing_radius,
                          airway_keep_largest)
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return (None, {}) if return_debug else None
    cont = max(cs, key=len)

    dt = distance_transform_edt(~airway) if airway.sum() else np.full(mask.shape, 1e9)
    rr = np.clip(np.rint(cont[:, 0]).astype(int), 0, mask.shape[0] - 1)
    cc = np.clip(np.rint(cont[:, 1]).astype(int), 0, mask.shape[1] - 1)
    airway_ok = dt[rr, cc] <= facing_thresh
    normal = outward_normals_from_contour(cont, tongue)
    normal_ok = normal[:, 0] < normal_row_max
    keep = close_small_false_gaps_cyclic(airway_ok & normal_ok, gap_close_pts)

    runs = true_runs_cyclic_indices(keep)
    dbg = {"num_runs": len(runs), "fallback_used": False}
    if len(runs) == 0:                               # fallback → V2
        dbg["fallback_used"] = True
        out = precise_contour_v2(mask, n=n, facing_thresh=facing_thresh,
                                 clip_root=clip_root, clip_drop_frac=clip_drop_frac)
        return (out, dbg) if return_debug else out

    dorsum_idx = int(np.argmin(cont[:, 0]))
    best_score, best_run = max(
        ((score_run(r, cont, normal, dorsum_idx, prev_tip), r) for r in runs),
        key=lambda x: x[0])
    seg = cont[best_run]
    seg = keep_dorsal_side_after_tip(seg, choose_upper_anterior_tip(seg, tip_band_px))
    if seg[-1, 1] < seg[0, 1]:
        seg = seg[::-1]
    if clip_root:
        seg = clip_posterior_spur(seg, drop_frac=clip_drop_frac)
    out = resample_rowcol(seg, n, smooth_win)
    if return_debug and out is not None:
        ddist = cyclic_index_distance_to_run(dorsum_idx, best_run, len(cont))
        vfrac = float(np.mean(normal[best_run][:, 0] > 0.55))
        dbg.update(selected_run_length=len(best_run), selected_run_score=float(best_score),
                   dorsum_distance=ddist, ventral_frac=vfrac, TT=tuple(np.round(out[0], 1)),
                   confidence=float(0.35 * (1 - min(ddist / max(len(cont) * 0.15, 1), 1))
                                    + 0.30 * (1 - vfrac) + 0.35 * (len(best_run) / len(cont))))
    return (out, dbg) if return_debug else out


def precise_contour_v4(mask, n=25, facing_thresh=2.5, clip_root=True,
                       clip_drop_frac=1.0, smooth_win=3, ant_drop=0.5):
    """V4 = V3 tip + V2 root/body (두 버전의 강점 결합).

    tip anchor는 V3(normal 필터 + upper-anterior)로 정밀하게, root anchor는 V2
    (envelope + posterior-clip)로 깔끔하게 잡고, 그 사이를 V2 envelope 표면을 따라
    리샘플한다. V3/V2 실패 시 V2로 fallback. (n,2) row,col tip→root 또는 None."""
    arc = facing_arc(mask, facing_thresh, smooth_win=1)
    if arc is None:
        return precise_contour_v2(mask, n=n, facing_thresh=facing_thresh,
                                  clip_root=clip_root, clip_drop_frac=clip_drop_frac)
    env = dorsal_envelope(arc)
    # V2 root anchor
    e2 = clip_anterior_drop(env, ant_drop) if ant_drop else env
    if clip_root:
        e2 = clip_posterior_spur(e2, drop_frac=clip_drop_frac)
    if len(e2) < 3:
        return precise_contour_v2(mask, n=n, facing_thresh=facing_thresh,
                                  clip_root=clip_root, clip_drop_frac=clip_drop_frac)
    root_v2 = e2[-1]
    # V3 tip anchor
    v3 = precise_contour_v3(mask, n=n, facing_thresh=facing_thresh, clip_root=clip_root,
                            clip_drop_frac=clip_drop_frac, smooth_win=smooth_win)
    tt = v3[0] if v3 is not None else e2[0]
    # V2 envelope 표면을 tip→root 사이로 리샘플
    i_tt = int(np.argmin(np.hypot(env[:, 0] - tt[0], env[:, 1] - tt[1])))
    i_root = int(np.argmin(np.hypot(env[:, 0] - root_v2[0], env[:, 1] - root_v2[1])))
    lo, hi = sorted([i_tt, i_root])
    seg = env[lo:hi + 1].copy()
    if np.hypot(*(seg[0] - tt)) > np.hypot(*(seg[-1] - tt)):
        seg = seg[::-1]
    seg[0] = tt
    seg[-1] = root_v2
    return resample_rowcol(seg, n, smooth_win)


def precise_contour_v5(mask, n=60, facing_thresh=2.5, smooth_win=3,
                       clip_root=True, clip_drop_frac=1.0, tip_band_px=8):
    """V5 = V1 root + tip fix.

    V1(경계 추종)의 root가 좋으므로 그대로 두고, tip이 혀끝을 U자로 감는 문제만 고친다.
    V1을 끝까지(posterior clip으로 root 확정) 돌린 뒤, 앞쪽 band에서 upper-anterior apex를
    찾아 그 앞의 wrap을 잘라낸다(root=끝점은 그대로 보존). (n,2) row,col 또는 None."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return None
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    airway = (mask == LBL_AIRWAY)
    if airway.sum() > 0:
        dt = distance_transform_edt(~airway)
        rr = np.clip(c[:, 0].round().astype(int), 0, mask.shape[0] - 1)
        cc = np.clip(c[:, 1].round().astype(int), 0, mask.shape[1] - 1)
        facing = dt[rr, cc] <= facing_thresh
        if facing.sum() >= 5:
            s, e = longest_true_run_cyclic(facing)
            c = c[np.arange(s, e) % len(c)]
    if c[0, 1] > c[-1, 1]:
        c = c[::-1]
    if clip_root:                                   # V1 root 확정 (그대로 보존)
        c = clip_posterior_spur(c, drop_frac=clip_drop_frac)
    tip_idx = choose_upper_anterior_tip(c, tip_band_px)   # 앞쪽 apex
    if tip_idx < len(c) - 3:                         # 앞쪽 wrap만 제거, root(끝) 유지
        c = c[tip_idx:]
    return resample_rowcol(c, n, smooth_win)


def outward_normals(rc, tongue, eps=1.5):
    """open curve (row,col)의 각 점에서 tongue 밖(airway 쪽)을 향한 단위 outward normal.

    tangent를 유한차분으로 구해 90° 회전 후, tongue 밖으로 나가는 방향을 선택."""
    t = np.gradient(rc, axis=0)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    H, W = tongue.shape

    def inside(p):
        rr = np.clip(np.rint(p[:, 0]).astype(int), 0, H - 1)
        cc = np.clip(np.rint(p[:, 1]).astype(int), 0, W - 1)
        return tongue[rr, cc]
    return np.where((~inside(rc + eps * nrm))[:, None], nrm, -nrm)


def precise_contour_v6(mask, n=60, facing_thresh=2.5, smooth_win=3,
                       clip_root=True, clip_drop_frac=1.0, down_thresh=0.35):
    """V6 = V5 root + tip을 normal 기반으로 진짜 혀끝까지 연장.

    V5는 apex에서 멈춰 혀끝 앞면을 놓친다. V6는 root를 V1/V5와 동일하게 확정한 뒤,
    dorsum peak에서 앞쪽으로 걸으며 outward normal이 **아래(underside)로 꺾이는 지점 직전**
    까지 tip을 연장한다 → 좌측(anterior)을 향하는 혀끝 앞면(front face)을 포함하고
    바닥 wrap은 제외. (n,2) row,col 또는 None."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return None
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    airway = (mask == LBL_AIRWAY)
    if airway.sum() > 0:
        dt = distance_transform_edt(~airway)
        rr = np.clip(c[:, 0].round().astype(int), 0, mask.shape[0] - 1)
        cc = np.clip(c[:, 1].round().astype(int), 0, mask.shape[1] - 1)
        facing = dt[rr, cc] <= facing_thresh
        if facing.sum() >= 5:
            s, e = longest_true_run_cyclic(facing)
            c = c[np.arange(s, e) % len(c)]
    if c[0, 1] > c[-1, 1]:
        c = c[::-1]
    if clip_root:                                   # root = V1/V5와 동일하게 확정
        c = clip_posterior_spur(c, drop_frac=clip_drop_frac)
    nv = outward_normals(c, tongue)                 # 각 점 outward normal
    nrow = nv[:, 0]
    if smooth_win > 1 and len(nrow) >= smooth_win:  # row성분 평활(노이즈 방지)
        nrow = np.convolve(nrow, np.ones(smooth_win) / smooth_win, "same")
    pk = int(np.argmin(c[:, 0]))                    # dorsum peak
    tip_idx = 0
    for i in range(pk, -1, -1):                     # dorsum → anterior
        if nrow[i] > down_thresh:                   # normal이 아래(underside wrap)를 향함
            tip_idx = i + 1                          # 그 직전까지 tip 연장
            break
    tip_idx = min(tip_idx, pk)
    c = c[tip_idx:]
    return resample_rowcol(c, n, smooth_win)


def precise_contour_v7(mask, n=60, facing_thresh=2.5, smooth_win=3, clip_root=True,
                       clip_drop_frac=1.0, dthresh_deg=45.0, angle_win=3,
                       presmooth=7, guard_deg=40.0, max_left_deg=40.0):
    """V7 = V6와 유사하되 tip을 normal 각도의 '급변(corner)' 지점으로.

    dorsum에서 앞쪽으로 걸으며, normal이 dorsum 대비 guard_deg 이상 꺾인 뒤 각도 변화가
    급격(window Δangle > dthresh_deg)한 지점까지 tip을 연장한다. root는 V1/V5와 동일.
    주의: 순수 급변만 쓰면 마스크 계단 노이즈가 dorsum에서 가짜 corner를 만들어 tip이
    뒤로 튄다 → guard_deg + presmooth로 앞쪽(혀끝) 영역으로 한정. (결과는 V6에 가깝다.)

    max_left_deg: tip normal이 수직(up)에서 좌상단으로 이 각도(30~45° 권장)를 넘지 않도록
    상한. corner가 더 좌측까지 가면 이 각도 지점까지 tip을 당긴다(좌측 과연장 방지)."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return None
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    airway = (mask == LBL_AIRWAY)
    if airway.sum() > 0:
        dt = distance_transform_edt(~airway)
        rr = np.clip(c[:, 0].round().astype(int), 0, mask.shape[0] - 1)
        cc = np.clip(c[:, 1].round().astype(int), 0, mask.shape[1] - 1)
        facing = dt[rr, cc] <= facing_thresh
        if facing.sum() >= 5:
            s, e = longest_true_run_cyclic(facing)
            c = c[np.arange(s, e) % len(c)]
    if c[0, 1] > c[-1, 1]:
        c = c[::-1]
    if clip_root:
        c = clip_posterior_spur(c, drop_frac=clip_drop_frac)
    csm = c
    if presmooth > 1 and len(c) >= presmooth:       # 계단 노이즈 완화
        k = np.ones(presmooth) / presmooth
        csm = np.column_stack([np.convolve(c[:, 0], k, "same"),
                               np.convolve(c[:, 1], k, "same")])
    nv = outward_normals(csm, tongue)
    ang = np.unwrap(np.arctan2(nv[:, 0], nv[:, 1]))
    N = len(c)
    dang = np.array([np.degrees(abs(ang[min(N - 1, i + angle_win)] - ang[max(0, i - angle_win)]))
                     for i in range(N)])
    # normal의 좌측각(수직 up 기준, +=좌상단): up=0, upper-left45=45, left=90
    lang = np.degrees(np.arctan2(-nv[:, 1], -nv[:, 0]))
    pk = int(np.argmin(c[:, 0]))
    ad = ang[pk]
    tip_idx = 0
    for i in range(pk, -1, -1):                     # dorsum → anterior
        turned = np.degrees(abs(ang[i] - ad))
        if turned > guard_deg and dang[i] > dthresh_deg:   # 충분히 꺾인 뒤 급변
            tip_idx = i
            break
    # cap: tip normal이 좌상단 max_left_deg를 넘지 않도록 (넘으면 그 직전까지 당김)
    cap_idx = 0
    for i in range(pk, -1, -1):
        if lang[i] > max_left_deg:
            cap_idx = i + 1
            break
    tip_idx = max(tip_idx, cap_idx)                 # 좌측 과연장 방지
    c = c[tip_idx:]
    return resample_rowcol(c, n, smooth_win)


def precise_contour_v8(mask, n=60, facing_thresh=2.5, smooth_win=3, clip_root=True,
                       clip_drop_frac=1.0, dthresh_deg=45.0, angle_win=3, presmooth=7,
                       guard_deg=40.0, max_left_deg=40.0, right_thresh=80.0):
    """V8 = V7 tip(corner + 좌측각 cap) + root도 normal 기반.

    root는 clip_posterior_spur 대신, dorsum에서 뒤쪽으로 걸으며 normal이 **완전 우측(90°)을
    보기 직전**(우측각 > right_thresh 되는 지점의 직전)을 root로 잡는다. tip은 V7과 동일."""
    tongue = (mask == LBL_TONGUE)
    if tongue.sum() < 10:
        return None
    cs = find_contours(tongue.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    airway = (mask == LBL_AIRWAY)
    if airway.sum() > 0:
        dt = distance_transform_edt(~airway)
        rr = np.clip(c[:, 0].round().astype(int), 0, mask.shape[0] - 1)
        cc = np.clip(c[:, 1].round().astype(int), 0, mask.shape[1] - 1)
        facing = dt[rr, cc] <= facing_thresh
        if facing.sum() >= 5:
            s, e = longest_true_run_cyclic(facing)
            c = c[np.arange(s, e) % len(c)]
    if c[0, 1] > c[-1, 1]:
        c = c[::-1]
    csm = c
    if presmooth > 1 and len(c) >= presmooth:
        k = np.ones(presmooth) / presmooth
        csm = np.column_stack([np.convolve(c[:, 0], k, "same"),
                               np.convolve(c[:, 1], k, "same")])
    nv = outward_normals(csm, tongue)
    ang = np.unwrap(np.arctan2(nv[:, 0], nv[:, 1]))
    N = len(c)
    dang = np.array([np.degrees(abs(ang[min(N - 1, i + angle_win)] - ang[max(0, i - angle_win)]))
                     for i in range(N)])
    lang = np.degrees(np.arctan2(-nv[:, 1], -nv[:, 0]))   # 좌측각 (up=0, left=90)
    rang = np.degrees(np.arctan2(nv[:, 1], -nv[:, 0]))    # 우측각 (up=0, right=90)
    pk = int(np.argmin(c[:, 0]))
    ad = ang[pk]
    # TIP: V7 (corner + 좌측각 cap)
    tip_idx = 0
    for i in range(pk, -1, -1):
        if np.degrees(abs(ang[i] - ad)) > guard_deg and dang[i] > dthresh_deg:
            tip_idx = i
            break
    cap_idx = 0
    for i in range(pk, -1, -1):
        if lang[i] > max_left_deg:
            cap_idx = i + 1
            break
    tip_idx = max(tip_idx, cap_idx)
    # ROOT: normal이 완전 우측(90°) 보기 직전
    root_idx = N - 1
    if clip_root:
        for i in range(pk, N):
            if rang[i] > right_thresh:
                root_idx = i - 1
                break
    root_idx = max(root_idx, pk + 1)
    c = c[tip_idx:root_idx + 1]
    return resample_rowcol(c, n, smooth_win)
