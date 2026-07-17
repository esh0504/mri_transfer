#!/usr/bin/env python3
"""
Stratified 11D muscle-activation pool generator for the TongueX static PoseBank.

WHY NOT PLAIN LHS?
------------------
Uniform LHS over [0,1]^11 suffers from concentration of measure. Empirically, on
the previous pool_50000.txt:

    sum of 11 activations                    : 5.50 +/- 0.96 (never below 1.61)
    muscles > 0.5 per sample                 : 5.5 on average
    sparse samples (<=3 active, >=8 near-off): 0 / 50000
    L2 distance to rest pose (all zeros)     : 0.655 (never reached)
    L2 distance to any single-muscle posture : 0.56 - 0.79 (never reached)

i.e. every single sample was a whole-tongue co-contraction. The rest pose, the
single-muscle primitives (which are exactly what the counterfactual XAI claim
needs), and every sparse speech-like synergy were entirely absent.

DESIGN
------
Total activation "effort" and activation "direction" are decoupled:

    effort E    ~ stratified over [E_MIN, E_MAX]
    direction d ~ Dirichlet(alpha), alpha < 1 => sparse simplex
    a = clip(E * d, 0, cap)

plus deterministic structural blocks that guarantee the primitives exist.

Blocks:
    REST     rest pose (all zeros)                              1
    SINGLE   single-muscle sweeps                    11 x levels
    PAIR     pairwise grids                    55 x levels^2
    TRIPLE   sparse 3-muscle random draws
    ANCHOR   literature-informed synergies + perturbations
    EFFORT   effort/Dirichlet stratified core (the bulk)
    LHS      legacy uniform LHS, kept as an OOD / generalization split

Output is format-compatible with the previous pool file, plus a sidecar metadata
CSV (index, block, effort, n_active) for the audit and for train/OOD splits.

Usage:
    python sampling/build_pool.py --n 200000 --out datasets/pool_v2_200000.txt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muscles import (  # noqa: E402
    MUSCLE_NAMES,
    N_MUSCLES,
    ARTICULATORY_ANCHORS,
    anchor_to_vec,
)

BLOCK_ORDER = ["REST", "SINGLE", "PAIR", "TRIPLE", "ANCHOR", "EFFORT", "LHS"]


def block_rest():
    return np.zeros((1, N_MUSCLES)), "REST"


def block_single(cap, n_levels=12):
    """Single-muscle sweeps. These are the basis of dM/da_i and are REQUIRED for
    the counterfactual / muscle-importance claim to have any data support."""
    levels = np.linspace(cap / n_levels, cap, n_levels)
    rows = []
    for i in range(N_MUSCLES):
        for lv in levels:
            v = np.zeros(N_MUSCLES)
            v[i] = lv
            rows.append(v)
    return np.asarray(rows), "SINGLE"


def block_pair(cap, n_levels=4):
    """Pairwise grids -> 2nd-order muscle interaction. Co-contraction is
    nonlinear; this cannot be recovered from single-muscle sweeps alone."""
    levels = np.linspace(cap / n_levels, cap, n_levels)
    rows = []
    for i in range(N_MUSCLES):
        for j in range(i + 1, N_MUSCLES):
            for a in levels:
                for b in levels:
                    v = np.zeros(N_MUSCLES)
                    v[i], v[j] = a, b
                    rows.append(v)
    return np.asarray(rows), "PAIR"


def block_triple(rng, k, cap):
    """Random sparse 3-muscle combinations."""
    rows = np.zeros((k, N_MUSCLES))
    for r in range(k):
        idx = rng.choice(N_MUSCLES, size=3, replace=False)
        rows[r, idx] = rng.uniform(0.1, cap, size=3)
    return rows, "TRIPLE"


def block_anchor(rng, k, cap, sigma=0.05, p_dropout=0.0, p_extra=0.0,
                 elo=0.7, ehi=1.3):
    """Literature-informed articulatory synergies plus local perturbations.

    교란 폭이 클러스터 구별도를 좌우한다. 기본값은 'tight'(각 중심 주변에 조밀).
    - sigma      : active 근육 gaussian jitter (작을수록 조밀)
    - elo, ehi   : effort 재스케일 범위 (좁을수록 방향 유지)
    - p_dropout  : 근육 하나 제거 확률 (정체성 흐림 → 0 권장)
    - p_extra    : 엉뚱한 근육 추가 확률 (다른 클러스터로 새어감 → 0 권장)
    넓은 커버리지(예전 동작)를 원하면 build(..., anchor_wide=True) 사용.
    """
    names = list(ARTICULATORY_ANCHORS)
    base = np.stack([anchor_to_vec(ARTICULATORY_ANCHORS[n], cap) for n in names])
    rows = np.zeros((k, N_MUSCLES))
    for r in range(k):
        b = base[rng.integers(len(names))].copy()
        b *= rng.uniform(elo, ehi)                                   # weak -> strong
        b += rng.normal(0.0, sigma, size=N_MUSCLES) * (b > 0)        # jitter
        act = np.flatnonzero(b > 1e-6)
        if len(act) > 1 and rng.random() < p_dropout:                # drop a muscle
            b[rng.choice(act)] = 0.0
        if rng.random() < p_extra:                                   # recruit an unexpected one
            off = np.flatnonzero(b <= 1e-6)
            if len(off):
                b[rng.choice(off)] = rng.uniform(0.05, 0.45)
        rows[r] = np.clip(b, 0.0, cap)
    return rows, "ANCHOR"


def block_effort(rng, k, cap, e_min=0.05, e_max=5.5,
                 alphas=(0.15, 0.25, 0.4, 0.7, 1.2, 2.5)):
    """Effort/Dirichlet stratified core -- the workhorse block.

    effort E:    stratified over [e_min, e_max], so near-rest configurations are
                 as densely sampled as strong ones (LHS gave 5.5 +/- 0.96 only).
    direction d: Dirichlet(alpha). alpha << 1 -> sparse, single-muscle-dominant;
                 alpha > 1 -> dense co-contraction. Cycling alpha spans the whole
                 sparse->dense axis, which plain LHS collapses to a point.

    Clipping at `cap` reduces realized effort; the realized value (not nominal E)
    is what the metadata sidecar records.
    """
    edges = np.linspace(e_min, e_max, k + 1)
    E = edges[:-1] + rng.random(k) * np.diff(edges)
    rng.shuffle(E)

    a_idx = rng.integers(0, len(alphas), size=k)
    rows = np.zeros((k, N_MUSCLES))
    for r in range(k):
        d = rng.dirichlet(np.full(N_MUSCLES, alphas[a_idx[r]]))
        rows[r] = np.clip(E[r] * d, 0.0, cap)
    return rows, "EFFORT"


def block_lhs(rng, k, cap):
    """Legacy uniform LHS. Retained as a small OOD / generalization split ONLY.
    It is a poor training distribution, but it is a fair stress test of whether
    the surrogate extrapolates to dense co-contraction."""
    A = np.zeros((k, N_MUSCLES))
    for j in range(N_MUSCLES):
        perm = rng.permutation(k)
        A[:, j] = (perm + rng.random(k)) / k
    return A * cap, "LHS"


def build(n_total, cap, seed, frac_effort=0.62, frac_anchor=0.14,
          frac_triple=0.06, frac_lhs=0.18, anchor_wide=False):
    rng = np.random.default_rng(seed)

    fixed = [block_rest(), block_single(cap), block_pair(cap)]
    n_fixed = sum(b[0].shape[0] for b in fixed)
    n_rand = max(0, n_total - n_fixed)

    s = frac_effort + frac_anchor + frac_triple + frac_lhs
    k_anc = int(round(n_rand * frac_anchor / s))
    k_tri = int(round(n_rand * frac_triple / s))
    k_lhs = int(round(n_rand * frac_lhs / s))
    k_eff = n_rand - k_anc - k_tri - k_lhs

    # anchor_wide=True → 예전 넓은 교란(커버리지↑, 구별도↓), 기본은 tight(구별도↑)
    anc_kw = dict(sigma=0.12, p_dropout=0.15, p_extra=0.25, elo=0.4, ehi=1.8) if anchor_wide else {}

    rand = [
        block_effort(rng, k_eff, cap),
        block_anchor(rng, k_anc, cap, **anc_kw),
        block_triple(rng, k_tri, cap),
        block_lhs(rng, k_lhs, cap),
    ]

    # PREFIX PROPERTY.
    # The deterministic blocks (REST, SINGLE, PAIR) go FIRST, unshuffled; only the
    # random blocks are shuffled. So:
    #   - any prefix >= n_fixed contains ALL 1013 primitives, and
    #   - any prefix is otherwise a balanced subsample of the random blocks.
    # If everything were shuffled together, `head -50000` of a 200k pool would keep
    # only ~1/4 of the single-muscle sweeps -- and those are the entire data basis
    # for the counterfactual / muscle-importance claim (0714.md Sec 13).
    # This is what makes it safe to stop the ArtiSynth run early, and makes the
    # 10k/25k/50k dataset-utility ablation (Sec 16.4) a plain `head -n`.
    Ar = np.concatenate([b[0] for b in rand], axis=0)
    Tr = np.concatenate([np.full(b[0].shape[0], b[1]) for b in rand])
    order = rng.permutation(Ar.shape[0])

    A = np.concatenate([b[0] for b in fixed] + [Ar[order]], axis=0)
    tags = np.concatenate([np.full(b[0].shape[0], b[1]) for b in fixed] + [Tr[order]])
    return A, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000,
                    help="target pool size (deterministic blocks may push it slightly over)")
    ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--out", default="datasets/pool_v2.txt")
    ap.add_argument("--meta", default=None,
                    help="metadata sidecar csv (default: <out>.meta.csv)")
    args = ap.parse_args()

    A, tags = build(args.n, args.cap, args.seed)
    n = A.shape[0]
    meta_path = args.meta or (os.path.splitext(args.out)[0] + ".meta.csv")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "w") as f:
        f.write("# Stratified muscle activation pool (MUSCLE_NAMES order, 0..cap)\n")
        f.write("# blocks: REST/SINGLE/PAIR/TRIPLE/ANCHOR/EFFORT/LHS (see meta csv)\n")
        f.write("# columns: " + ",".join(MUSCLE_NAMES) + "\n")
        f.write("index," + ",".join(MUSCLE_NAMES) + "\n")
        for i in range(n):
            f.write("%d,%s\n" % (i, ",".join("%.6f" % v for v in A[i])))

    effort = A.sum(1)
    n_act = (A > 0.05).sum(1)
    with open(meta_path, "w") as f:
        f.write("index,block,effort,n_active\n")
        for i in range(n):
            f.write("%d,%s,%.6f,%d\n" % (i, tags[i], effort[i], n_act[i]))

    print("wrote %s  (%d samples)" % (args.out, n))
    print("wrote %s" % meta_path)
    print("")
    print("block composition:")
    for t in BLOCK_ORDER:
        k = int((tags == t).sum())
        print("  %-7s %7d  (%.1f%%)" % (t, k, 100.0 * k / n))


if __name__ == "__main__":
    main()
    main()
