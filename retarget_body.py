#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""retarget_body.py — Tongue Body full-contour retargeting + 4-panel 시각화.

기존 dorsal-arc retargeting 을 **닫힌 full contour** 버전으로 확장:
  - 3D 모델의 midsag full 외곽선 = tongue mesh 의 y=0 평면 cross-section (닫힌 폴리곤).
  - MRI Tongue Body contour(extract_body_contour.py 산출, image space)를 registration
    affine 으로 model 공간으로 옮김.
  - 두 닫힌 contour 를 tip(min x) 시작·같은 방향으로 정렬·N점 리샘플 → 점 대응.
  - delta[k] = body[k] - body[rest] (model 공간), 곡선/시간축 스무딩 후 Gaussian-RBF
    skinning(centers=model full contour) 으로 mesh 변형.
  - 프레임별 4패널 GIF: MRI | model midsag full | 3D mesh(shaded) | 3D mesh(disp).

사용: python retarget_body.py --subject Subject3
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


# ---- 지오메트리 ---------------------------------------------------------------
def load_obj(path):
    V, F = [], []
    for L in open(path):
        t = L.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            F.append([int(s.split("/")[0]) - 1 for s in t[1:4]])
    return np.array(V), np.array(F)


def model_midsag_contour(V_mm, F, n):
    """tongue mesh 를 y=0 로 잘라 닫힌 midsag 외곽선 (x,z) mm, n점."""
    segs = []
    for tri in F:
        p = V_mm[tri]; s = p[:, 1]
        pts = []
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            if (s[a] <= 0 < s[b]) or (s[b] <= 0 < s[a]):
                t = s[a] / (s[a] - s[b]); pts.append((p[a] + t * (p[b] - p[a]))[[0, 2]])
        if len(pts) == 2:
            segs.append((pts[0], pts[1]))
    used = [False] * len(segs)
    poly = [np.array(segs[0][0]), np.array(segs[0][1])]; used[0] = True
    for _ in range(len(segs)):
        cur = poly[-1]; nxt = None
        for i, sg in enumerate(segs):
            if used[i]:
                continue
            a, b = np.array(sg[0]), np.array(sg[1])
            if np.hypot(*(a - cur)) < 0.5:
                nxt = (i, b); break
            if np.hypot(*(b - cur)) < 0.5:
                nxt = (i, a); break
        if nxt is None:
            break
        used[nxt[0]] = True; poly.append(nxt[1])
    return align_resample(np.array(poly), n)


# ---- 닫힌 contour 정렬/리샘플 (tip=min x 시작, dorsal-up 방향) -----------------
def align_resample(c, n):
    c = np.asarray(c, float)
    i0 = int(np.argmin(c[:, 0]))                 # tip = 최소 x
    c = np.roll(c, -i0, axis=0)
    if c[1, 1] < c[-1, 1]:                       # 다음 점이 위(z↑, dorsal)로 가게
        c = np.roll(c[::-1], 1, axis=0)
    cc = np.vstack([c, c[:1]])
    d = np.r_[0, np.cumsum(np.hypot(np.diff(cc[:, 0]), np.diff(cc[:, 1])))]
    if d[-1] == 0:
        return None
    u = np.linspace(0, d[-1], n, endpoint=False)
    return np.column_stack([np.interp(u, d, cc[:, 0]), np.interp(u, d, cc[:, 1])])


def affine_image_to_model(reg_csv):
    img, mod = [], []
    for r in csv.DictReader(open(reg_csv)):
        img.append([float(r["imageX"]), float(r["imageY"])])
        mod.append([float(r["modelX"]), float(r["modelZ"])])
    img = np.array(img); mod = np.array(mod)
    A, *_ = np.linalg.lstsq(np.column_stack([img, np.ones(len(img))]), mod, rcond=None)
    return lambda xy: np.column_stack([xy, np.ones(len(xy))]) @ A


# ---- retarget ----------------------------------------------------------------
def build(subject, data, n=60, rbf_len=18.0, spatial_win=3, temporal_win=9, temporal_poly=2,
          mm_per_px=1.164, H=256):
    sub = os.path.join(data, "MRI_SSFP_10fps", subject)
    obj = os.path.join(data, "tongue_model", "tongue_rest_m.obj")
    reg = os.path.join("test", "v8", subject, "registration.csv")
    V, F = load_obj(obj); V = V * 1000.0; V[:, 0] += 2.0        # mm
    mc = model_midsag_contour(V, F, n)                          # (n,2) 모델 full contour
    to_model = affine_image_to_model(reg)
    d = np.load(os.path.join(sub, "body_contour.npz"))
    fids = [int(x) for x in d["frame_ids"]]
    all_fr = list(range(min(fids), max(fids) + 1))

    def body_model(mi):
        key = "resamp_f%d" % mi
        if key not in d.files:
            return None
        rc = d[key]                                            # (,2) row,col
        xy = np.column_stack([rc[:, 1] * mm_per_px, (H - 1 - rc[:, 0]) * mm_per_px])
        return align_resample(to_model(xy), n)                # model (x,z) mm, n점

    stack = np.full((len(all_fr), n, 2), np.nan)
    for t, mi in enumerate(all_fr):
        b = body_model(mi)
        if b is not None:
            stack[t] = b
    for c in range(2):                                        # 결측 프레임 시간보간(예: f35)
        for j in range(n):
            col = stack[:, j, c]; ok = ~np.isnan(col)
            stack[:, j, c] = np.interp(np.arange(len(all_fr)), np.where(ok)[0], col[ok])

    rest = 0
    delta = stack - stack[rest]                               # (T,n,2)
    if spatial_win > 1:                                       # 곡선 따라(닫힘) 스무딩
        delta = uniform_filter1d(delta, spatial_win, axis=1, mode="wrap")
    if temporal_win > 1 and len(all_fr) >= temporal_win:
        delta = savgol_filter(delta, temporal_win, temporal_poly, axis=0, mode="interp")
    delta = delta - delta[rest]

    Vxz = V[:, [0, 2]]
    deformed = []
    for k in range(len(all_fr)):
        rbf = RBFInterpolator(mc, delta[k], kernel="gaussian",
                              epsilon=1.0 / rbf_len, degree=-1, smoothing=1e-3)
        dxz = rbf(Vxz)
        Vd = V.copy(); Vd[:, 0] += dxz[:, 0]; Vd[:, 2] += dxz[:, 1]
        deformed.append(Vd)
    return dict(all_fr=all_fr, V=V, F=F, mc=mc, deformed=deformed, stack=stack,
                delta=delta, sub=sub, mm_per_px=mm_per_px, H=H, npz=d)


# ---- 4-panel gif -------------------------------------------------------------
def render(res, out_gif, fps=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    import imageio.v2 as imageio
    F = res["F"]; mc = res["mc"]; mm = res["mm_per_px"]; H = res["H"]
    allv = np.vstack(res["deformed"])
    xl = (allv[:, 0].min(), allv[:, 0].max()); yl = (res["V"][:, 1].min(), res["V"][:, 1].max())
    zl = (allv[:, 2].min(), allv[:, 2].max())
    gmax = max(np.linalg.norm(res["deformed"][k] - res["V"], axis=1).max()
               for k in range(len(res["all_fr"]))) or 1.0     # 전 프레임 공통 색 스케일
    rest_mid = res["V"][np.abs(res["V"][:, 1]) < 6.0]
    tmp = []; outdir = os.path.dirname(os.path.abspath(out_gif)) or "."
    for k, mi in enumerate(res["all_fr"]):
        Vd = res["deformed"][k]; disp = np.linalg.norm(Vd - res["V"], axis=1)
        fig = plt.figure(figsize=(15, 4.4))
        # 1) MRI + body contour (혀 영역으로 crop)
        a0 = fig.add_subplot(1, 4, 1)
        g = imageio.imread(os.path.join(res["sub"], "png", "image_%d.png" % mi))
        a0.imshow(g, cmap="gray"); a0.axis("off"); a0.set_title("MRI f%d" % mi, fontsize=9)
        key = "contour_f%d" % mi
        if key in res["npz"].files:
            c = res["npz"][key]; cc = np.vstack([c, c[:1]]); a0.plot(cc[:, 1], cc[:, 0], "-", c="lime", lw=1.5)
            a0.set_xlim(c[:, 1].min() - 22, c[:, 1].max() + 22)
            a0.set_ylim(c[:, 0].max() + 22, c[:, 0].min() - 22)
        # 2) model midsag full contour
        a1 = fig.add_subplot(1, 4, 2)
        a1.scatter(rest_mid[:, 0], rest_mid[:, 2], s=5, c="0.8")
        mcd = mc + res["delta"][k]                          # 변형된 모델 contour
        a1.plot(np.r_[mcd[:, 0], mcd[:1, 0]], np.r_[mcd[:, 1], mcd[:1, 1]], "b-", lw=1.4, label="model")
        bm = res["stack"][k]
        a1.plot(np.r_[bm[:, 0], bm[:1, 0]], np.r_[bm[:, 1], bm[:1, 1]], "r--", lw=1.2, label="MRI body")
        a1.set_aspect("equal", adjustable="box"); a1.set_xlim(xl); a1.set_ylim(zl)
        a1.set_xticks([]); a1.set_yticks([]); a1.set_title("model midsag full", fontsize=9)
        if k == 0:
            a1.legend(fontsize=6, loc="lower left")
        # 3) 3D mesh shaded
        a2 = fig.add_subplot(1, 4, 3, projection="3d")
        a2.plot_trisurf(Vd[:, 0], Vd[:, 1], Vd[:, 2], triangles=F, color="lightblue",
                        alpha=0.9, linewidth=0.1, edgecolor="0.5")
        a2.set_xlim(xl); a2.set_ylim(yl); a2.set_zlim(zl); a2.view_init(20, -70)
        a2.set_xticklabels([]); a2.set_yticklabels([]); a2.set_zticklabels([])
        a2.set_title("3D mesh", fontsize=9)
        # 4) 3D mesh colored by displacement
        a3 = fig.add_subplot(1, 4, 4, projection="3d")
        tv = disp[F].mean(1)
        tp = a3.plot_trisurf(Vd[:, 0], Vd[:, 1], Vd[:, 2], triangles=F, cmap="viridis",
                             linewidth=0.1, edgecolor="0.4")
        tp.set_array(tv); tp.set_clim(0, gmax)
        a3.set_xlim(xl); a3.set_ylim(yl); a3.set_zlim(zl); a3.view_init(20, -70)
        a3.set_xticklabels([]); a3.set_yticklabels([]); a3.set_zticklabels([])
        a3.set_title("3D disp (max %.1fmm)" % disp.max(), fontsize=9)
        fig.suptitle("Tongue-Body full-contour retargeting  f%d" % mi, fontsize=10)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.03, wspace=0.06)
        p = os.path.join(outdir, "_bp_%03d.png" % k); fig.savefig(p, dpi=95); plt.close(fig); tmp.append(p)
    imageio.mimsave(out_gif, [imageio.imread(p) for p in tmp], duration=1.0 / fps, loop=0)
    for p in tmp:
        try:
            os.remove(p)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets")
    ap.add_argument("--subject", default="Subject3")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rbf-len", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = build(args.subject, args.data, n=args.n, rbf_len=args.rbf_len)
    out = args.out or os.path.join(res["sub"], "body_retarget_4panel.gif")
    render(res, out, fps=args.fps)
    print("frames=%d  max disp=%.1fmm  saved: %s"
          % (len(res["all_fr"]), max(np.linalg.norm(res["deformed"][k]-res["V"],axis=1).max()
                                     for k in range(len(res["all_fr"]))), out))


if __name__ == "__main__":
    main()
