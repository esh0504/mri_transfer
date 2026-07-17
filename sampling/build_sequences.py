#!/usr/bin/env python3
"""
Keyframe sequence builder for the dynamic MotionBank (0714.md Sec 6.2).

REPLACES the previous sampler, which drew 4 pool indices uniformly at random.
That produced (a) transitions between two arbitrary co-contraction blobs, i.e.
physiologically meaningless control directions, and (b) 2796 degenerate A-B-A-B
oscillations. 0714.md Sec 6.2 already specifies the right thing --
"Valid static PoseBank -> pose-space nearest-neighbor graph -> smooth activation
transition" -- this implements it.

Speech is a sequence of *targets* reached from and returned toward a neutral
posture, with bounded control velocity. So:

  1. Restrict to physically VALID poses (needs the watchdog labels).
  2. Build a kNN graph. Prefer SHAPE space (mesh descriptors) over activation
     space: two distant activations can give the same tongue, and the transition
     that matters is the one that is smooth *in shape*.
  3. Each sequence = rest-ish start -> graph walk through K-2 targets -> rest-ish
     end, with every consecutive hop constrained to be a graph edge (i.e. an
     achievable, bounded control step).
  4. Enforce distinct keyframes and a minimum path diversity so you do not
     regenerate 400k near-duplicates.

Run this AFTER the static ArtiSynth pass, once you have validity labels.

Usage:
    python sampling/build_sequences.py \
        --pool datasets/pool_v2_100000.txt \
        --validity datasets/validity.csv \        # index,label  (VALID/MARGINAL/...)
        --descriptors datasets/shape_desc.npy \   # optional (N,D) mesh descriptors
        --n-seq 200000 --seq-len 4 \
        --out datasets/sequences_v2.txt
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muscles import N_MUSCLES  # noqa: E402
from audit_pool import load_pool  # noqa: E402


def load_validity(path, n):
    """index,label csv. Anything not VALID/MARGINAL is dropped."""
    keep = np.zeros(n, dtype=bool)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("index"):
                continue
            p = line.split(",")
            i, lab = int(p[0]), p[1].strip().upper()
            if i < n and lab in ("VALID", "MARGINAL"):
                keep[i] = True
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--validity", default=None,
                    help="index,label csv from the physical-validity watchdog. "
                         "If omitted, ALL poses are treated as valid (not recommended).")
    ap.add_argument("--descriptors", default=None,
                    help=".npy (N,D) per-pose shape descriptors. Strongly preferred "
                         "over activation space for graph construction.")
    ap.add_argument("--n-seq", type=int, default=200000)
    ap.add_argument("--seq-len", type=int, default=4)
    ap.add_argument("--knn", type=int, default=24,
                    help="graph degree; each hop must be within this neighbourhood")
    ap.add_argument("--rest-frac", type=float, default=0.25,
                    help="fraction of lowest-effort valid poses treated as 'neutral'")
    ap.add_argument("--p-rest-start", type=float, default=0.7)
    ap.add_argument("--p-rest-end", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--out", default="datasets/sequences_v2.txt")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    A = load_pool(args.pool)
    n = A.shape[0]

    valid = load_validity(args.validity, n) if args.validity else np.ones(n, bool)
    vidx = np.flatnonzero(valid)
    if len(vidx) < args.seq_len * 2:
        sys.exit("too few valid poses (%d)" % len(vidx))
    print("valid poses: %d / %d" % (len(vidx), n))

    # --- graph feature space -------------------------------------------------
    if args.descriptors:
        F = np.load(args.descriptors)[vidx]
        F = (F - F.mean(0)) / (F.std(0) + 1e-8)
        space = "shape descriptors"
    else:
        F = A[vidx]
        space = "activation (fallback -- provide --descriptors for the real thing)"
    print("graph space: %s  dim=%d" % (space, F.shape[1]))

    tree = cKDTree(F)
    _, nbr = tree.query(F, k=min(args.knn + 1, len(vidx)))
    nbr = nbr[:, 1:]  # drop self

    # --- neutral / rest set --------------------------------------------------
    effort = A[vidx].sum(1)
    thr = np.quantile(effort, args.rest_frac)
    rest_pool = np.flatnonzero(effort <= thr)
    print("neutral pool: %d poses (effort <= %.2f)" % (len(rest_pool), thr))

    # --- generate ------------------------------------------------------------
    K = args.seq_len
    seqs = np.zeros((args.n_seq, K), dtype=int)
    seen = set()
    w = 0
    guard = 0
    while w < args.n_seq and guard < args.n_seq * 20:
        guard += 1
        start = (rng.choice(rest_pool) if rng.random() < args.p_rest_start
                 else rng.integers(len(vidx)))
        path = [int(start)]
        for step in range(K - 1):
            last = path[-1]
            cand = nbr[last]
            cand = cand[~np.isin(cand, path)]          # no immediate repeats
            if len(cand) == 0:
                break
            if step == K - 2 and rng.random() < args.p_rest_end:
                # bias the final hop back toward neutral (speech returns to rest)
                ce = effort[cand]
                p = np.exp(-2.0 * (ce - ce.min()) / (ce.ptp() + 1e-8))
                nxt = rng.choice(cand, p=p / p.sum())
            else:
                nxt = rng.choice(cand)
            path.append(int(nxt))
        if len(path) != K:
            continue
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        seqs[w] = vidx[np.asarray(path)]  # map back to pool indices
        w += 1

    seqs = seqs[:w]
    print("generated %d unique sequences (requested %d)" % (w, args.n_seq))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# keyframe sequences (pool index paths), pose-graph walks\n")
        f.write("# graph space: %s ; knn=%d\n" % (space, args.knn))
        f.write("# total_sequences=%d  seq_len=%d\n" % (w, K))
        f.write("seq_id," + ",".join("k%d" % i for i in range(K)) + "\n")
        for i in range(w):
            f.write("%d,%s\n" % (i, ",".join(str(int(x)) for x in seqs[i])))
    print("wrote %s" % args.out)

    # sanity
    dups = sum(1 for r in seqs if len(set(r)) < K)
    print("sequences with a repeated keyframe: %d (%.2f%%)" % (dups, 100.0 * dups / max(w, 1)))


if __name__ == "__main__":
    main()
