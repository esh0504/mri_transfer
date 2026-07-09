#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2-stage 파이프라인 드라이버 (Hydra config).

실행:
  python main.py
  python main.py stage=fem
  python main.py stage=retarget
  python main.py stage=fem artisynth.activations.HG=0.3

Retarget contour 버전 선택 (v1~v8, 알고리즘 설명은 versionmanagement.md):
  python main.py stage=retarget retarget.contour.mode=v6
  python main.py stage=retarget retarget.contour.mode=v8
  (또는 configs/retarget/default.yaml 의 contour.mode 값을 바꿔 기본값 지정)
"""
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from modules.pipeline import configure, load_model, fem, visualization, retargeting  # noqa: E402
from modules.utils import ensure_dir, OUT_DIR, save_gif  # noqa: E402


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(cfg: DictConfig):
    stage = cfg.stage
    if stage not in ("retarget", "fem", "all"):
        raise SystemExit("stage 는 retarget|fem|all 중 하나 (받은 값: %s)" % stage)

    configure(cfg)
    outdir = ensure_dir(cfg.paths.out_dir or OUT_DIR)
    root = cfg.paths.data_root
    tongue = os.path.join(root, cfg.paths.tongue_obj)
    activations = OmegaConf.to_container(cfg.artisynth.activations, resolve=True)

    if stage in ("retarget", "all"):
        model = load_model(tongue)
        result = retargeting(
            model,
            os.path.join(root, cfg.paths.rest_mask),
            os.path.join(root, cfg.paths.target_mask),
        )
        frames = result if isinstance(result, list) else [result]
        png_dir = ensure_dir(os.path.join(outdir, "retargeted"))   # PNG는 retargeted/ 폴더에
        png_paths = []
        for i, m in enumerate(frames):
            name = "retargeted.png" if len(frames) == 1 else "retargeted_%03d.png" % i
            try:                                    # 렌더 실패(headless 등)해도 GIF는 계속
                png_paths.append(visualization(m, os.path.join(png_dir, name)))
            except Exception as e:
                print("   (retargeted 렌더 실패 frame %d: %s)" % (i, e))
                png_paths.append(None)
        # 폴더(비디오) 전체를 retarget하면 PNG들을 순서대로 묶어 GIF 생성
        if len(frames) > 1:
            from modules.utils import load_video, mask_label_2d
            from modules.compare import (build_compare_gif, build_overlay_gif,
                                         build_points3d_gif)
            fps = cfg.retarget.get("gif_fps", 10)
            slow = max(1.0, fps / 2.0)          # 비교 GIF는 2배 느리게(천천히 보기)
            g = save_gif(png_paths, os.path.join(outdir, "retargeted.gif"), fps=fps)
            if g:
                print("saved: %s (%d frames, %s fps)" % (g, len(png_paths), fps))
            # 비교용: MRI 마스크(2D)
            tmasks = [mask_label_2d(mk)
                      for mk in load_video(os.path.join(root, cfg.paths.target_mask))]
            reg_csv = getattr(frames[0], "registration_csv", None)
            gc = build_compare_gif(tmasks, frames,
                                   os.path.join(outdir, "compare.gif"),
                                   reg_csv=reg_csv, rest_verts=model.verts,
                                   png_paths=png_paths,
                                   mm_per_px=cfg.retarget.mm_per_px, fps=slow)
            go = build_overlay_gif(tmasks, frames, reg_csv,
                                   os.path.join(outdir, "overlay.gif"),
                                   mm_per_px=cfg.retarget.mm_per_px, fps=slow,
                                   rest_verts=model.verts)
            # 3D 변형 점(포인트 클라우드) 시퀀스
            gp = build_points3d_gif(frames, os.path.join(outdir, "points3d.gif"),
                                    rest_verts=model.verts, fps=slow)
            print("saved: %s, %s, %s (fps=%.1f, 느리게)" % (gc, go, gp, slow))
            print(f"Retargeted {name} saved to {os.path.join(outdir, name)}")

    if stage in ("fem", "all"):
        model = load_model(tongue)
        visualization(model, os.path.join(outdir, "rest.png"))
        model = fem(model, activations)
        visualization(model, os.path.join(outdir, "fem.png"))


if __name__ == "__main__":
    main()
