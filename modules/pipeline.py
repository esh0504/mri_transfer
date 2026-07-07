# -*- coding: utf-8 -*-
"""파이프라인 공개 API — main.py 및 외부 스크립트 진입점."""
import os

import numpy as np

from artisynth import (
    TongueModel, MUSCLE_NAMES, load_model as _load_model, fem as _fem,
    configure as _art_configure,
)
from modules.utils import (
    OUT_DIR, ensure_dir, load_mask, load_video, visualization as _visualization,
)
from retarget import (
    register, attach_registration, retarget as _retarget,
    configure as _retarget_configure,
)

_CFG = None
_RETARGET = {}
_REG_CSV = None


def configure(cfg):
    """Hydra cfg → artisynth / retarget / render 전역 설정."""
    global _CFG, _RETARGET, _REG_CSV
    _CFG = cfg
    _art_configure(cfg.artisynth)
    rc = cfg.retarget
    _retarget_configure(rc.contour, mm_per_px=rc.mm_per_px)
    _RETARGET = dict(
        mm_per_px=rc.mm_per_px,
        nctrl=rc.retarget.nctrl,
        rbf_len=rc.retarget.rbf_len,
        spatial_win=rc.retarget.spatial_win,
    )
    import retarget.utils as _ru
    print("retarget: contour mode=%s, n_markers=%d, mm_per_px=%.3f, nctrl=%d"
          % (_ru.CONTOUR_MODE, _ru.N_MARKERS, _RETARGET["mm_per_px"], _RETARGET["nctrl"]))
    outdir = ensure_dir(cfg.paths.out_dir or OUT_DIR)
    _REG_CSV = os.path.join(outdir, "registration.csv")


def load_model(file_path):
    """OBJ rest mesh → TongueModel."""
    return _load_model(file_path)


def fem(model, activations):
    """11D 근육 활성값 forward → TongueModel."""
    return _fem(model, activations)


def visualization(model, output_path):
    """TongueModel → PNG 저장. 반환: 저장 경로(str)."""
    vis_cfg = {"out_path": output_path}
    if _CFG is not None:
        try:
            from omegaconf import OmegaConf
            vis_cfg = dict(OmegaConf.to_container(_CFG.render, resolve=True))
        except Exception:
            pass
        vis_cfg["out_path"] = output_path
    return _visualization(model, vis_cfg)


def _as_mask(mask_or_path):
    if isinstance(mask_or_path, str):
        return load_mask(mask_or_path)
    return mask_or_path


def _to_tongue_model(src_model, result):
    out = TongueModel()
    out.verts = result["points_cloud"]
    out.faces = result["Mesh"]
    out.names = getattr(src_model, "names", list(MUSCLE_NAMES))
    act = getattr(src_model, "activation", None)
    out.activation = (np.asarray(act, dtype=float).copy() if act is not None
                      else np.zeros(len(MUSCLE_NAMES)))
    if getattr(src_model, "registration_csv", None):
        out.registration_csv = src_model.registration_csv
    return out


def _retarget_one(src_model, ref, target):
    ref = _as_mask(ref)
    target = _as_mask(target)
    if not getattr(src_model, "registration_csv", None):
        register(ref, src_model, _REG_CSV, mm_per_px=_RETARGET["mm_per_px"])
        attach_registration(src_model, _REG_CSV)
    else:
        attach_registration(src_model, src_model.registration_csv)
    result = _retarget(src_model, ref, target, **_RETARGET)
    return _to_tongue_model(src_model, result)


def retargeting(src_model, ref, target):
    """3D src_model + 2D rest/target mask → retargeted TongueModel.

    ref, target: mask array 또는 ``.mat`` 파일 경로.
    target이 폴더면 ``mask_*.mat`` 전체를 retarget → ``list[TongueModel]`` 반환.
    """
    if isinstance(target, str) and os.path.isdir(target):
        return [_retarget_one(src_model, ref, frame)
                for frame in load_video(target)]
    return _retarget_one(src_model, ref, target)
