#!/usr/bin/env python3
"""Ordered, block-tagged variant of the activation pool.

기존 build_pool.py 와 '값은 동일'(같은 seed)하되:
  · 블록을 BLOCK_ORDER 순서로 '연속 배치'(블록 간 셔플 없음) → 섹션 분리가 자명
  · 각 행에 어떤 샘플링 블록인지 'block' 열로 마킹

주의: 원본 build_pool 은 head -n 으로도 primitive 가 보존되도록 랜덤 블록을 섞는
'prefix property' 를 씁니다. 이 태그드 버전은 그 성질을 포기하는 대신,
블록별 슬라이싱을 쉽게 만듭니다 (index 범위가 곧 블록).

사용:
    python sampling/build_pool_tagged.py --n 200000 --out datasets/pool_v2_200000_tagged.txt
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muscles import MUSCLE_NAMES          # noqa: E402
import build_pool as bp                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--out", default="datasets/pool_v2_200000_tagged.txt")
    args = ap.parse_args()

    # 결정론적 생성 (원본 pool 과 동일한 숫자)
    A, tags = bp.build(args.n, args.cap, args.seed)

    # 블록을 BLOCK_ORDER 순서로 연속 배치 (블록 내부 순서는 안정 유지)
    rank = {b: i for i, b in enumerate(bp.BLOCK_ORDER)}
    perm = np.argsort([rank[t] for t in tags], kind="stable")
    A, tags = A[perm], tags[perm]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Stratified muscle activation pool -- ORDERED by block, block-tagged\n")
        f.write("# blocks (in order): " + ",".join(bp.BLOCK_ORDER) + "\n")
        f.write("# columns: index,block," + ",".join(MUSCLE_NAMES) + "\n")
        f.write("index,block," + ",".join(MUSCLE_NAMES) + "\n")
        for i in range(len(A)):
            f.write("%d,%s,%s\n" % (i, tags[i], ",".join("%.6f" % v for v in A[i])))

    print("wrote %s  (%d rows)" % (args.out, len(A)))
    print("block ranges  [start:end)  (index 범위가 곧 블록):")
    s = 0
    for b in bp.BLOCK_ORDER:
        k = int((tags == b).sum())
        if k:
            print("  %-7s %7d   [%d:%d]" % (b, k, s, s + k)); s += k


if __name__ == "__main__":
    main()
