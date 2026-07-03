# -*- coding: utf-8 -*-
"""2D midsagittal MRI mask → dorsal contour (1_extract_contours.py 축약)."""
import os
import sys

import numpy as np

# tongue_contour.py 는 프로젝트 루트(/work)에 있음
_WORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _WORK_ROOT not in sys.path:
    sys.path.insert(0, _WORK_ROOT)

from tongue_contour import precise_contour  # noqa: E402

MM_PER_PX = float(os.environ.get("MM_PER_PX", "1.164"))
N_MARKERS = int(os.environ.get("N_MARKERS", "25"))
CLIP_ROOT = os.environ.get("CLIP_ROOT", "1").lower() not in ("0", "false", "no")
CLIP_DROP_FRAC = float(os.environ.get("CLIP_DROP_FRAC", "1.0"))


def configure(cfg=None, **overrides):
    """contour 기본 파라미터를 설정으로 덮어쓴다 (Hydra 등에서 호출).

    사용: configure(cfg.retarget.contour, mm_per_px=cfg.retarget.mm_per_px).
    인식 키: n_markers, clip_root, clip_drop_frac, mm_per_px.
    """
    global MM_PER_PX, N_MARKERS, CLIP_ROOT, CLIP_DROP_FRAC
    opts = dict(cfg) if cfg else {}
    opts.update(overrides)
    if opts.get("n_markers") is not None:
        N_MARKERS = int(opts["n_markers"])
    if opts.get("clip_root") is not None:
        CLIP_ROOT = bool(opts["clip_root"])
    if opts.get("clip_drop_frac") is not None:
        CLIP_DROP_FRAC = float(opts["clip_drop_frac"])
    if opts.get("mm_per_px") is not None:
        MM_PER_PX = float(opts["mm_per_px"])


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
        1_extract_contours.py 의 tongue_targets 한 프레임과 동일 convention.
    """
    if n_markers is None:
        n_markers = N_MARKERS
    if mm_per_px is None:
        mm_per_px = MM_PER_PX
    if clip_root is None:
        clip_root = CLIP_ROOT
    if clip_drop_frac is None:
        clip_drop_frac = CLIP_DROP_FRAC

    mask = np.asarray(mask)
    rc = precise_contour(
        mask, n=n_markers, clip_root=clip_root, clip_drop_frac=clip_drop_frac)
    if rc is None:
        raise ValueError("mask에서 dorsal contour를 추출하지 못했습니다.")

    H = mask.shape[0]
    x = rc[:, 1] * mm_per_px
    y = (H - 1 - rc[:, 0]) * mm_per_px
    return np.column_stack([x, y, np.zeros_like(x)])
