#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2/main.py — 2-stage 실험 드라이버 (Hydra config).

설정은 configs/ 에 있다:
  configs/configs.yaml            # stage · paths · render · activations (+ defaults)
  configs/artisynth/default.yaml  # fem 파라미터
  configs/retarget/default.yaml   # register/lift/retarget 파라미터

실행 (Hydra CLI 오버라이드):
  python main.py                                  # stage=all
  python main.py stage=fem
  python main.py stage=retarget
  python main.py stage=fem artisynth.activations.GGP=0.3 artisynth.activations.HG=0.2
  python main.py artisynth.nramp=40 retarget.mm_per_px=1.2
  python main.py render.size=[512,512] paths.out_dir=/tmp/run1
  # target_mask 를 폴더로 주면 전체 시퀀스를 retarget:
  python main.py stage=retarget paths.target_mask=datasets/GT_Segmentations/Subject3

구조: 알고리즘 artisynth/·retarget/, IO·시각화 modules/utils.py, 설정 configs/.
출력: _test_out/ (rest.png·fem.png·retargeted.obj·registration.csv·…)
headless 렌더: apt install xvfb libxrender1 && xvfb-run -a python main.py
"""
import os
import sys

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# main.py를 V2 밖에서 실행해도 패키지(artisynth/retarget/modules)를 찾도록 경로 보강
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import artisynth.utils as art                     # noqa: E402  (configure 용)
import retarget.contour as rcontour               # noqa: E402  (configure 용)
from modules.utils import (                       # noqa: E402
    OUT_DIR, ensure_dir, repo_path, save_png, save_npy, save_obj, extract_obj,
    load_mask, load_video, vis_with_activations,
)
from artisynth.utils import fem, model_from_obj, MUSCLE_NAMES, TongueModel  # noqa: E402
from retarget import register, lift_masks, attach_registration, retarget    # noqa: E402


# --------------------------------------------------------------------------- #
# 헬퍼
# --------------------------------------------------------------------------- #
def _resolve(path):
    """상대경로면 리포 루트(V2의 부모) 기준으로 해석."""
    return path if os.path.isabs(path) else repo_path(path)


def _outdir(cfg):
    return ensure_dir(cfg.paths.out_dir or OUT_DIR)


def _render_settings(cfg):
    return dict(upper_degree=cfg.render.upper_degree,
                right_degree=cfg.render.right_degree,
                size=tuple(cfg.render.size))


def _synthetic_tongue():
    """JVM/OBJ 없이 렌더 파이프라인을 돌리기 위한 더미 혀 메쉬."""
    outline = np.array([
        [0.00, 0.0, 0.00], [0.02, 0.0, 0.015], [0.04, 0.0, 0.022],
        [0.06, 0.0, 0.020], [0.08, 0.0, 0.012], [0.09, 0.0, 0.000],
        [0.07, 0.0, -0.008], [0.04, 0.0, -0.010], [0.01, 0.0, -0.006],
    ], dtype=float)
    verts = np.vstack([outline.mean(0, keepdims=True), outline])
    n = len(outline)
    faces = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)], dtype=int)
    m = TongueModel()
    m.verts, m.faces = verts, faces
    m.names = list(MUSCLE_NAMES)
    m.activation = np.zeros(len(MUSCLE_NAMES))
    return m


def _rest_model(tongue_obj):
    """rest 메쉬(OBJ) 로드. 없으면 합성 더미로 폴백."""
    if os.path.isfile(tongue_obj):
        m = model_from_obj(tongue_obj)
        print("   rest mesh: %s (V=%d, F=%d)" % (tongue_obj, len(m.verts), len(m.faces)))
        return m
    print("   (rest OBJ 없음 → 합성 더미: %s)" % tongue_obj)
    return _synthetic_tongue()


# --------------------------------------------------------------------------- #
# Stage 1: retarget  (MRI mask → register → lift → retarget)
# --------------------------------------------------------------------------- #
def stage_retarget(cfg):
    rc = cfg.retarget
    mm = rc.mm_per_px
    rcontour.configure(rc.contour, mm_per_px=mm)     # contour 전역 주입
    outdir = _outdir(cfg)

    def op(name):
        return os.path.join(outdir, name)

    tongue_obj = _resolve(cfg.paths.tongue_obj)
    reg_csv = op("registration.csv")
    ref_3d = model_from_obj(tongue_obj)
    rest = load_mask(_resolve(cfg.paths.rest_mask))

    print("[retarget/1] register")
    info = register(rest, ref_3d, reg_csv, mm_per_px=mm)
    print("  ", info["path"], "| anchors:", info["names"], "| RMS %.2f mm" % info["rms_mm"])

    print("[retarget/2] lift")
    video = load_video(_resolve(cfg.paths.mask_dir))
    lifted = lift_masks(video, mm_per_px=mm, nz=rc.lift.nz, half_w=rc.lift.half_w,
                        edge_drop=rc.lift.edge_drop, width_end=rc.lift.width_end)
    save_npy(op("tongue_lift_3d.npy"), lifted)
    disp = np.linalg.norm(lifted - lifted[0:1], axis=3)
    print("  ", lifted.shape, "| max disp %.1f mm" % disp.max())

    print("[retarget/3] retarget")
    ref_3d = attach_registration(ref_3d, reg_csv)
    target_path = _resolve(cfg.paths.target_mask)
    rkw = dict(nctrl=rc.retarget.nctrl, rbf_len=rc.retarget.rbf_len,
               spatial_win=rc.retarget.spatial_win, mm_per_px=mm)

    if os.path.isdir(target_path):
        # 폴더 → 내부 mask_*.mat 전체를 프레임별 retarget
        targets = load_video(target_path)
        seq_dir = ensure_dir(op("retargeted_objs"))
        for i in range(targets.shape[0]):
            obj = retarget(ref_3d, rest, targets[i], **rkw)
            save_obj(obj, os.path.join(seq_dir, "frame_%03d.obj" % i))
        print("  folder target → %d frames: %s" % (targets.shape[0], seq_dir))
        return None

    # 단일 파일 → 1프레임 retarget
    target = load_mask(target_path)
    result = retarget(ref_3d, rest, target, **rkw)
    save_obj(result, op("retargeted.obj"))
    print("  ", op("retargeted.obj"),
          "| verts:", result["points_cloud"].shape[0], "faces:", result["Mesh"].shape[0])
    return result


# --------------------------------------------------------------------------- #
# Stage 2: fem  (11D 활성값 → ArtiSynth forward)
# --------------------------------------------------------------------------- #
def stage_fem(cfg):
    art.configure(cfg.artisynth)                     # fem 전역 주입
    render = _render_settings(cfg)
    activations = OmegaConf.to_container(cfg.artisynth.activations, resolve=True)
    outdir = _outdir(cfg)

    def op(name):
        return os.path.join(outdir, name)

    print("[fem] activations:", activations)

    # rest 렌더 + OBJ (참고용)
    rest_model = _rest_model(_resolve(cfg.paths.tongue_obj))
    save_png(vis_with_activations(rest_model, render), op("rest.png"))
    save_obj(extract_obj(rest_model), op("rest.obj"))

    # forward
    try:
        model = fem(None, activations)
    except Exception as e:
        print("fem 실패:", e)
        print("ARTISYNTH_HOME / Java / 컴파일된 ArtiSynth 모델 확인")
        return None

    print("  solver_ok:", getattr(model, "ok", None),
          "| verts:", model.verts.shape, "faces:", model.faces.shape)
    for name, val in zip(MUSCLE_NAMES, model.activation):
        if val > 0:
            print("    %s=%.3f" % (name, val))

    path = save_png(vis_with_activations(model, render), op("fem.png"))
    save_obj(extract_obj(model), op("fem.obj"))
    if path:
        print("  저장:", path)
    return model


# --------------------------------------------------------------------------- #
# Hydra 엔트리
# --------------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(cfg: DictConfig):
    stage = cfg.stage
    if stage not in ("retarget", "fem", "all"):
        raise SystemExit("stage 는 retarget|fem|all 중 하나 (받은 값: %s)" % stage)

    print("muscles(11D):", ", ".join(MUSCLE_NAMES))
    print("stage:", stage, "| out_dir:", cfg.paths.out_dir or OUT_DIR, "\n")

    if stage in ("retarget", "all"):
        stage_retarget(cfg)
    if stage in ("fem", "all"):
        stage_fem(cfg)


if __name__ == "__main__":
    main()
