#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_body_contour.py — Tongue Body class 의 전체(닫힌) contour 추출.

정의: **Tongue Body = fill(수작업 Labels.tif outline) ∩ mask(label 4)** 의 최대 연결성분.
Labels.tif 는 프레임마다 손으로 그린 폐곡선 outline(채워지지 않은 링)이므로,
closing→fill 로 영역을 만든 뒤 실제 혀 마스크(label 4)와 교집합하고 최대 성분만 남긴다.
그 영역의 닫힌 전체 외곽선을 프레임별로 뽑는다.

출력(기본 <subject> 폴더):
    body_contour.npz   프레임별 full contour(row,col, subpixel) + resample(고정 N) + body 마스크
    body_contour.gif   grayscale 위 body contour 오버레이(확인용)

사용:
    python extract_body_contour.py --subject Subject3
    python extract_body_contour.py --subject Subject3 --resample 120 --out datasets/.../body_contour.npz
"""
import argparse
import glob
import os
import re

import numpy as np
import scipy.io as sio
import tifffile
from scipy.ndimage import binary_closing, binary_fill_holes, label as cc_label
from skimage.measure import find_contours


def _fidx(p):
    return int(re.search(r"(\d+)", os.path.basename(p)).group(1))


def load_mask4(path):
    d = sio.loadmat(path)
    arr = np.asarray(next(v for k, v in d.items() if not k.startswith("__")))
    arr = arr[..., 0] if arr.ndim == 3 else arr
    return arr == 4


def body_region(outline, mask4, closing_iter=2):
    """손그림 outline → 채운 영역 ∩ mask4 → 최대 연결성분(잔조각 제거)."""
    filled = binary_fill_holes(binary_closing(outline > 0, iterations=closing_iter))
    reg = filled & mask4
    lab, n = cc_label(reg)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        reg = lab == sizes.argmax()
    return reg


def full_contour(region):
    """영역의 닫힌 전체 외곽선(row,col, subpixel). 가장 긴 것 선택. 없으면 None."""
    cs = find_contours(region.astype(float), 0.5)
    return max(cs, key=len) if cs else None


def resample_closed(c, n):
    """닫힌 곡선을 호길이 등간격 n점으로."""
    cc = np.vstack([c, c[:1]])                      # 닫기
    d = np.r_[0, np.cumsum(np.hypot(np.diff(cc[:, 0]), np.diff(cc[:, 1])))]
    if d[-1] == 0:
        return None
    u = np.linspace(0, d[-1], n, endpoint=False)
    return np.column_stack([np.interp(u, d, cc[:, 0]), np.interp(u, d, cc[:, 1])])


def extract_subject(data, subject, resample=120, closing_iter=2):
    sub_dir = os.path.join(data, "MRI_SSFP_10fps", subject)
    seg_dir = os.path.join(data, "GT_Segmentations", subject)
    lab = tifffile.imread(os.path.join(sub_dir, "Labels.tif"))     # (T,H,W) 0/1
    mpaths = sorted(glob.glob(os.path.join(seg_dir, "mask_*.mat")), key=_fidx)
    midx = [_fidx(p) for p in mpaths]
    if lab.shape[0] != len(mpaths):
        print("경고: Labels.tif 프레임(%d) != mask(%d) — 앞에서부터 짝지음"
              % (lab.shape[0], len(mpaths)))
    T = min(lab.shape[0], len(mpaths))

    contours, resampled, regions, frame_ids = {}, {}, {}, []
    for k in range(T):
        mi = midx[k]
        reg = body_region(lab[k], load_mask4(mpaths[k]), closing_iter)
        c = full_contour(reg)
        frame_ids.append(mi)
        regions["f%d" % mi] = reg.astype(np.uint8)
        if c is None:
            continue
        contours["f%d" % mi] = c.astype(np.float32)               # (N,2) row,col
        if resample:
            r = resample_closed(c, resample)
            if r is not None:
                resampled["f%d" % mi] = r.astype(np.float32)
    return dict(frame_ids=np.array(frame_ids), contours=contours,
                resampled=resampled, regions=regions, midx=midx, sub_dir=sub_dir)


def save_npz(res, out_path, resample):
    payload = {"frame_ids": res["frame_ids"]}
    for k, v in res["contours"].items():
        payload["contour_" + k] = v
    for k, v in res["resampled"].items():
        payload["resamp_" + k] = v
    if resample and res["resampled"]:
        stack = np.stack([res["resampled"][k] for k in
                          ["f%d" % i for i in res["frame_ids"]
                           if "f%d" % i in res["resampled"]]])
        payload["resampled_stack"] = stack           # (T, resample, 2) 파이프라인용
    np.savez_compressed(out_path, **payload)


def save_gif(res, out_path, fps=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    tmp = []
    outdir = os.path.dirname(os.path.abspath(out_path)) or "."
    for mi in res["frame_ids"]:
        key = "f%d" % mi
        g = imageio.imread(os.path.join(res["sub_dir"], "png", "image_%d.png" % mi))
        fig, a = plt.subplots(figsize=(4, 4))
        a.imshow(g, cmap="gray")
        c = res["contours"].get(key)
        if c is not None:
            cc = np.vstack([c, c[:1]])
            a.plot(cc[:, 1], cc[:, 0], "-", c="lime", lw=1.6)
        reg = res["regions"][key]
        ys, xs = np.where(reg)
        if len(xs):
            a.set_xlim(xs.min() - 12, xs.max() + 12); a.set_ylim(ys.max() + 12, ys.min() - 12)
        a.set_title("Tongue Body  frame %d" % mi, fontsize=10); a.axis("off")
        p = os.path.join(outdir, "_body_%03d.png" % mi)
        fig.tight_layout(); fig.savefig(p, dpi=100); plt.close(fig); tmp.append(p)
    imageio.mimsave(out_path, [imageio.imread(p) for p in tmp], duration=1.0 / fps, loop=0)
    for p in tmp:
        try:
            os.remove(p)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets")
    ap.add_argument("--subject", default="Subject3")
    ap.add_argument("--resample", type=int, default=120, help="닫힌 contour 리샘플 점수(0=off)")
    ap.add_argument("--closing", type=int, default=2, help="outline 닫기 반복(끊긴 링 보정)")
    ap.add_argument("--out", default="", help="npz 저장 경로(기본 <subject>/body_contour.npz)")
    ap.add_argument("--gif", default="", help="확인용 gif 경로(기본 <subject>/body_contour.gif)")
    args = ap.parse_args()

    res = extract_subject(args.data, args.subject, args.resample, args.closing)
    out = args.out or os.path.join(res["sub_dir"], "body_contour.npz")
    gif = args.gif or os.path.join(res["sub_dir"], "body_contour.gif")
    save_npz(res, out, args.resample)
    save_gif(res, gif)
    n = len(res["contours"])
    print("frames=%d  contour 추출=%d  저장: %s , %s"
          % (len(res["frame_ids"]), n, out, gif))


if __name__ == "__main__":
    main()
