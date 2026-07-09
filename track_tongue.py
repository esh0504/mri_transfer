#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""track_tongue.py — optical-flow 로 기준(clean) 프레임의 혀 라벨을 앞뒤로 전파.

아이디어: 혀가 주변과 **떨어져 또렷한 프레임**(air gap 존재)에서 라벨을 만들어 두면,
붙어서 경계가 사라진 프레임은 grayscale MRI 의 optical flow 로 그 라벨을 흘려보내
채운다. **다중 라벨** 지원 → 기준 프레임에 body/floor/base(보라/빨강/하늘)를 그려두면
세 부위가 함께 전파된다.

핵심 함수
    propagate_labels(grays, order, refs) -> {idx: label_img}
        grays : {frame_idx: uint8 grayscale}
        order : 정렬된 frame_idx 리스트
        refs  : {frame_idx: int label 이미지}  (0=배경, 1,2,3..=부위)
        각 프레임은 좌/우 가장 가까운 기준에서 양방향 전파 후 거리가중 블렌드,
        라벨별 soft map 을 argmax 로 합쳐 겹침 없이 반환.

CLI (검증/데모: GT label 4 를 기준으로 사용)
    python track_tongue.py --subject Subject3 --refs 1,20,40,60
    python track_tongue.py --subject Subject3 --label-dir myrefs --refs 1,36,71 --out out_track
"""
import argparse
import glob
import os
import re

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv 필요:  pip install opencv-python-headless")


# ---- optical flow 전파 코어 ---------------------------------------------------
def farneback(a, b):
    """motion a→b (dense). 반환 flow[y,x] = (dx, dy)."""
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 5, 7, 1.5, 0)


def warp(soft, flow, grid):
    """soft 를 flow 로 backward-warp (grid = (gx, gy))."""
    gx, gy = grid
    mx = (gx + flow[..., 0]).astype(np.float32)
    my = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(soft.astype(np.float32), mx, my, cv2.INTER_LINEAR)


def _propagate_soft(soft_ref, ref_idx, order, grays, grid, direction):
    """ref 한 방향으로 soft mask 흘리기 → {idx: soft}."""
    pos = {i: k for k, i in enumerate(order)}
    out = {ref_idx: soft_ref}
    k0 = pos[ref_idx]
    rng = range(k0 + 1, len(order)) if direction > 0 else range(k0 - 1, -1, -1)
    prev = ref_idx
    for kk in rng:
        f = order[kk]
        fl = farneback(grays[f], grays[prev])      # f→prev 로 backward warp
        out[f] = warp(out[prev], fl, grid)
        prev = f
    return out


def propagate_labels(grays, order, refs):
    """다중 라벨을 기준 프레임들에서 전 프레임으로 전파. {idx: int label img} 반환."""
    H, W = grays[order[0]].shape
    grid = np.meshgrid(np.arange(W), np.arange(H))
    ref_idxs = sorted(refs)
    label_vals = sorted({int(v) for r in ref_idxs for v in np.unique(refs[r]) if v != 0})
    if not label_vals:
        raise ValueError("기준 라벨에 전경(>0)이 없습니다.")

    # 라벨별로 각 ref 양방향 전파 soft map 사전계산
    fwd = {r: {} for r in ref_idxs}
    bwd = {r: {} for r in ref_idxs}
    for L in label_vals:
        for r in ref_idxs:
            s = (refs[r] == L).astype(np.float32)
            fwd[r][L] = _propagate_soft(s, r, order, grays, grid, +1)
            bwd[r][L] = _propagate_soft(s, r, order, grays, grid, -1)

    out = {}
    for i in order:
        left = [r for r in ref_idxs if r <= i]
        right = [r for r in ref_idxs if r >= i]
        rl = max(left) if left else min(right)
        rr = min(right) if right else max(left)
        w = 0.0 if rl == rr else (i - rl) / (rr - rl)   # 우측 기준 가중치
        soft = np.zeros((H, W, len(label_vals)), np.float32)
        for c, L in enumerate(label_vals):
            ml = fwd[rl][L][i] if i >= rl else bwd[rl][L][i]
            mr = bwd[rr][L][i] if i <= rr else fwd[rr][L][i]
            soft[..., c] = (1 - w) * ml + w * mr
        lab = np.zeros((H, W), np.int32)
        amax = soft.argmax(2)
        fg = soft.max(2) > 0.5                          # 배경 임계
        for c, L in enumerate(label_vals):
            lab[fg & (amax == c)] = L
        out[i] = lab
    return out


# ---- IO 헬퍼 -----------------------------------------------------------------
def _fidx(p):
    return int(re.search(r"(\d+)", os.path.basename(p)).group(1))


def load_gray_sequence(subject_dir):
    """<subject>/png/image_*.png → {idx: uint8}."""
    paths = sorted(glob.glob(os.path.join(subject_dir, "png", "image_*.png")), key=_fidx)
    return {_fidx(p): cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths}


def load_gt_tongue(seg_dir):
    """<seg>/mask_*.mat → {idx: binary tongue(label4)}. (검증/데모용)"""
    import scipy.io as sio
    out = {}
    for p in sorted(glob.glob(os.path.join(seg_dir, "mask_*.mat")), key=_fidx):
        d = sio.loadmat(p)
        arr = next(v for k, v in d.items() if not k.startswith("__"))
        arr = np.asarray(arr)
        arr = arr[..., 0] if arr.ndim == 3 else arr
        out[_fidx(p)] = (arr == 4).astype(np.int32)
    return out


def dice(a, b):
    a = a > 0
    b = b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


# ---- CLI ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets")
    ap.add_argument("--subject", default="Subject3")
    ap.add_argument("--refs", default="", help="기준 프레임 번호 콤마구분(예: 1,20,40,60). 비우면 stride 사용")
    ap.add_argument("--stride", type=int, default=20, help="--refs 없을 때 기준 간격")
    ap.add_argument("--label-dir", default="", help="기준 라벨 PNG 폴더(image_{n}.png, 0/1/2/3..). 없으면 GT label4")
    ap.add_argument("--out", default="", help="전파 결과 라벨 PNG 저장 폴더")
    args = ap.parse_args()

    sub_dir = os.path.join(args.data, "MRI_SSFP_10fps", args.subject)
    seg_dir = os.path.join(args.data, "GT_Segmentations", args.subject)
    grays = load_gray_sequence(sub_dir)
    order = sorted(grays)
    if not order:
        raise SystemExit("png 없음. 먼저 datasets/prepare.py 실행: %s" % sub_dir)

    ref_list = ([int(x) for x in args.refs.split(",") if x.strip()]
                or order[::args.stride] + ([order[-1]] if order[-1] not in order[::args.stride] else []))

    gt = load_gt_tongue(seg_dir) if os.path.isdir(seg_dir) else {}
    if args.label_dir:                                   # 사용자 기준 라벨(부위 포함 가능)
        refs = {}
        for r in ref_list:
            p = os.path.join(args.label_dir, "image_%d.png" % r)
            refs[r] = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.int32)
    else:                                                # 데모: GT 혀(binary)
        refs = {r: gt[r] for r in ref_list if r in gt}

    result = propagate_labels(grays, order, refs)

    if gt:                                               # 정확도 리포트(비-기준 프레임)
        nonref = [i for i in order if i not in refs]
        d_non = np.mean([dice(result[i] > 0, gt[i]) for i in nonref]) if nonref else 1.0
        d_min = np.min([dice(result[i] > 0, gt[i]) for i in nonref]) if nonref else 1.0
        print("refs=%d  non-ref Dice mean=%.3f min=%.3f" % (len(refs), d_non, d_min))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for i in order:
            cv2.imwrite(os.path.join(args.out, "label_%d.png" % i), result[i].astype(np.uint8))
        print("saved %d label PNGs -> %s" % (len(order), args.out))


if __name__ == "__main__":
    main()
