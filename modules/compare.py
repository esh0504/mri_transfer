# -*- coding: utf-8 -*-
"""Retargeting 비교 GIF 빌더.

compare.gif : [MRI 마스크 | 변형 모델 실루엣] 나란히
overlay.gif : MRI에서 뽑은 dorsal contour + 변형 모델의 dorsal contour(이미지로 역매핑)를 겹침

모두 matplotlib(Agg) 기반이라 headless(디스플레이 없음)에서도 생성된다.
main.py의 retarget 스테이지에서 target이 폴더(비디오)일 때 호출한다.
"""
import os

import numpy as np


def _mask_rgb(mask2d):
    """2D 라벨 마스크 → 컬러 이미지 (tongue=빨강, airway=파랑, 기타=회색)."""
    m = np.asarray(mask2d)
    rgb = np.full(m.shape + (3,), 55, np.uint8)
    rgb[m == 4] = (220, 70, 60)      # tongue
    rgb[m == 5] = (70, 140, 220)     # airway
    return rgb


def _affine_model_to_image(reg_csv):
    """registration.csv anchor로 model-mm(x,z) → image-mm(x,y) affine (3x2)."""
    from modules.utils import read_csv_dicts
    img, mod = [], []
    for r in read_csv_dicts(reg_csv):
        img.append([float(r["imageX"]), float(r["imageY"])])
        mod.append([float(r["modelX"]), float(r["modelZ"])])
    img = np.asarray(img, float); mod = np.asarray(mod, float)
    A, *_ = np.linalg.lstsq(np.column_stack([mod, np.ones(len(mod))]), img, rcond=None)
    return A


def build_compare_gif(target_masks2d, deformed_models, out_path, fps=5, size=(300, 300)):
    """프레임별 [MRI 마스크 | 변형 모델 실루엣] → GIF. 반환: 저장 경로 또는 None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from modules.utils import vis_mask, save_gif

    allv = np.vstack([np.asarray(dm.verts, float) for dm in deformed_models])
    bounds = (allv[:, 0].min(), allv[:, 0].max(), allv[:, 2].min(), allv[:, 2].max())
    outdir = os.path.dirname(os.path.abspath(out_path))
    tmp = []
    for i, (mk, dm) in enumerate(zip(target_masks2d, deformed_models)):
        sil = vis_mask(np.asarray(dm.verts, float), np.asarray(dm.faces, int),
                       size=size, bounds=bounds, plane="midsag")
        fig, ax = plt.subplots(1, 2, figsize=(6, 3.1))
        ax[0].imshow(_mask_rgb(mk)); ax[0].set_title("MRI (mask)", fontsize=9); ax[0].axis("off")
        ax[1].imshow(sil); ax[1].set_title("Retargeting", fontsize=9); ax[1].axis("off")
        fig.suptitle("frame %d" % i, fontsize=9)
        fig.tight_layout()
        p = os.path.join(outdir, "_cmp_%03d.png" % i)
        fig.savefig(p, dpi=90); plt.close(fig); tmp.append(p)
    gif = save_gif(tmp, out_path, fps=fps)
    for p in tmp:
        try:
            os.remove(p)
        except OSError:
            pass
    return gif


def build_points3d_gif(deformed_models, out_path, rest_verts=None, fps=5,
                       elev=18, azim=-70):
    """변형된 3D 점(모델 정점)을 프레임 순서대로 3D scatter로 → GIF.

    rest_verts를 주면 rest 대비 변위(mm)로 색을 입힌다. 시점은 고정(변형이 보이게)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from modules.utils import save_gif

    allv = np.vstack([np.asarray(dm.verts, float) for dm in deformed_models])
    ctr = allv.mean(axis=0)
    half = float(np.abs(allv - ctr).max())        # 큐빅 bounds(등축 비율)
    rest = np.asarray(rest_verts, float) if rest_verts is not None else None
    dmax = 1.0
    if rest is not None:
        dmax = max(1e-6, max(np.linalg.norm(np.asarray(dm.verts, float) - rest, axis=1).max()
                             for dm in deformed_models) * 1000.0)
    outdir = os.path.dirname(os.path.abspath(out_path))
    tmp = []
    for i, dm in enumerate(deformed_models):
        V = np.asarray(dm.verts, float)
        col = (np.linalg.norm(V - rest, axis=1) * 1000.0) if rest is not None else V[:, 2]
        fig = plt.figure(figsize=(4.2, 4.2))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=col, cmap="viridis",
                        s=7, vmin=0, vmax=(dmax if rest is not None else None))
        ax.set_xlim(ctr[0]-half, ctr[0]+half); ax.set_ylim(ctr[1]-half, ctr[1]+half)
        ax.set_zlim(ctr[2]-half, ctr[2]+half)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("frame %d" % i, fontsize=9)
        if rest is not None and i == 0:
            fig.colorbar(sc, ax=ax, shrink=0.6, label="disp (mm)")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        p = os.path.join(outdir, "_p3d_%03d.png" % i)
        fig.savefig(p, dpi=90); plt.close(fig); tmp.append(p)
    gif = save_gif(tmp, out_path, fps=fps)
    for p in tmp:
        try:
            os.remove(p)
        except OSError:
            pass
    return gif


def build_overlay_gif(target_masks2d, deformed_models, reg_csv, out_path,
                      mm_per_px=1.164, fps=5, nctrl=25, rest_verts=None):
    """프레임별 MRI 위에 (관측 dorsal contour + 변형 모델 dorsal contour) 겹침 → GIF.

    rest_verts를 주면 rest에서 고정한 midsag dorsal 정점 순서를 변형 메쉬에 그대로 적용한다
    (매 프레임 재계산 없이 일관된 중앙 dorsal 라인)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from modules.utils import save_gif
    from retarget import mask2contour               # 설정된 CONTOUR_MODE/CLIP_ROOT 사용(=retargeting과 동일)
    from retarget.utils import model_dorsal_curve, midsag_dorsal_order

    A = _affine_model_to_image(reg_csv) if reg_csv else None
    order = (midsag_dorsal_order(np.asarray(rest_verts, float) * 1000.0)
             if rest_verts is not None else None)     # rest에서 고정
    outdir = os.path.dirname(os.path.abspath(out_path))
    tmp = []
    for i, (mk, dm) in enumerate(zip(target_masks2d, deformed_models)):
        H = mk.shape[0]
        fig, a = plt.subplots(figsize=(4.2, 4.2))
        a.imshow(_mask_rgb(mk)); a.axis("off"); a.set_title("frame %d" % i, fontsize=9)
        try:                                           # 관측 dorsal contour (retargeting과 동일한 mask2contour)
            cimg = mask2contour(mk)                     # (N,3) image-mm (x,y)
            a.plot(cimg[:, 0] / mm_per_px, (H - 1) - cimg[:, 1] / mm_per_px,
                   "-", c="yellow", lw=2.4, label="MRI contour")
        except Exception:
            pass
        if A is not None:
            V_mm = np.asarray(dm.verts, float) * 1000.0
            dor = model_dorsal_curve(V_mm, nctrl, order=order)   # 고정 midsag dorsal 정점
            im = np.column_stack([dor, np.ones(len(dor))]) @ A   # → image-mm (x,y)
            col = im[:, 0] / mm_per_px
            row = (H - 1) - im[:, 1] / mm_per_px
            a.plot(col, row, "-", c="cyan", lw=2.2, label="deformed model contour")
        a.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        p = os.path.join(outdir, "_ov_%03d.png" % i)
        fig.savefig(p, dpi=90); plt.close(fig); tmp.append(p)
    gif = save_gif(tmp, out_path, fps=fps)
    for p in tmp:
        try:
            os.remove(p)
        except OSError:
            pass
    return gif
