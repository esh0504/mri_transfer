#!/usr/bin/env python3
"""
Activation-space audit (0714.md Sec 7.1). Compares one or more pools on the
diagnostics that actually matter for this project.

Deliberately does NOT report 11D fill distance as a coverage claim: with N=1e5,
N^(1/11) = 2.8 grid levels per muscle, so no pool of any realistic size "covers"
[0,1]^11. Fill distance is reported only as a descriptive statistic. The real
coverage claim must be made in SHAPE space against real MRI contours -- see
sampling/README.md.

Usage:
    python sampling/audit_pool.py old=datasets/pool_50000.txt new=datasets/pool_v2_100000.txt
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from muscles import MUSCLE_NAMES, N_MUSCLES, ARTICULATORY_ANCHORS, anchor_to_vec  # noqa: E402


def load_pool(path, max_rows=None):
    """Reads either pool format; stops at the first blank/comment after the header."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                break
            if line.startswith("#") or line.startswith("index,"):
                continue
            parts = line.split(",")
            if len(parts) != N_MUSCLES + 1:
                break
            rows.append([float(x) for x in parts[1:]])
            if max_rows and len(rows) >= max_rows:
                break
    return np.asarray(rows)


def audit(name, A):
    n = A.shape[0]
    eff = A.sum(1)
    n_on = (A > 0.5).sum(1)
    n_off = (A < 0.1).sum(1)
    sparse = ((A > 0.3).sum(1) <= 3) & ((A < 0.1).sum(1) >= 8)
    near_rest = eff < 0.3

    tree = cKDTree(A)
    nn = tree.query(A, k=2)[0][:, 1]

    print("=" * 68)
    print("%s   N=%d" % (name, n))
    print("=" * 68)
    print("  effort (sum of 11)      : %.2f +/- %.2f   [min %.2f, max %.2f]"
          % (eff.mean(), eff.std(), eff.min(), eff.max()))
    print("  effort quantiles 5/50/95: %.2f / %.2f / %.2f"
          % tuple(np.quantile(eff, [.05, .5, .95])))
    print("  muscles >0.5 per sample : %.2f" % n_on.mean())
    print("  muscles <0.1 per sample : %.2f" % n_off.mean())
    print("  SPARSE (<=3 on, >=8 off): %6d  (%.2f%%)" % (sparse.sum(), 100 * sparse.mean()))
    print("  NEAR-REST (effort<0.3)  : %6d  (%.2f%%)" % (near_rest.sum(), 100 * near_rest.mean()))
    print("  11D NN dist (median)    : %.3f   [descriptive only, not a coverage claim]"
          % np.median(nn))

    print("  -- distance to structural primitives (0 = present in pool) --")
    probes = {"REST (all zero)": np.zeros(N_MUSCLES)}
    for i, m in enumerate(MUSCLE_NAMES):
        e = np.zeros(N_MUSCLES)
        e[i] = 1.0
        probes["only %s" % m] = e
    worst = 0.0
    for k, v in probes.items():
        d = tree.query(v)[0]
        worst = max(worst, d)
        print("     %-16s : %.3f" % (k, d))
    print("     >> worst primitive gap: %.3f" % worst)

    print("  -- distance to literature articulatory anchors --")
    ad = [tree.query(anchor_to_vec(v))[0] for v in ARTICULATORY_ANCHORS.values()]
    print("     median %.3f   worst %.3f  (%s)"
          % (np.median(ad), np.max(ad),
             list(ARTICULATORY_ANCHORS)[int(np.argmax(ad))]))
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    for a in args:
        name, path = a.split("=", 1) if "=" in a else (a, a)
        audit(name, load_pool(path))
