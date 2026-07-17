#!/usr/bin/env python3
"""
Loader / validator for the static PoseBank produced by
artisynth_scripts/export_static.py.

Converts the big-endian float32 shards into a single memmapped .npy per field,
joins the per-shard validity metadata, and prints the yield table (which is
figure F2a in sampling/EVIDENCE_PLAN.md).

Usage:
    python sampling/load_static.py --root D:/tonguex/static_v2 --pool datasets/pool_v2_200000.txt
"""
import argparse
import glob
import os

import numpy as np


def read_topology(root):
    info = {}
    with open(os.path.join(root, "topology_info.txt")) as f:
        for line in f:
            for kv in line.strip().split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    info[k] = v
    return int(info["n_surf_verts"]), int(info["n_fem_nodes"])


def load_meta(root):
    """index,label,reason,max_vel,peak_ramp_vel,vol_ratio,n_inverted,min_elem_vol,settle_t,secs"""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "meta", "shard_*.csv"))):
        with open(p) as f:
            next(f)
            for line in f:
                c = line.rstrip("\n").split(",")
                if len(c) < 10:
                    continue
                rows.append((int(c[0]), c[1], c[2], float(c[3]), float(c[4]),
                             np.nan if c[5] == "NA" else float(c[5]),
                             -1 if c[6] == "NA" else int(c[6]),
                             np.nan if c[7] == "NA" else float(c[7]),
                             float(c[8]), float(c[9])))
    rows.sort(key=lambda r: r[0])
    return rows


def stack(root, sub, n_pts, out_npy):
    """Concatenate big-endian float32 shards into one (N, n_pts, 3) float32 array."""
    shards = sorted(glob.glob(os.path.join(root, sub, "shard_*.bin")))
    if not shards:
        raise SystemExit("no shards in %s/%s" % (root, sub))
    rec = n_pts * 3 * 4
    total = sum(os.path.getsize(s) // rec for s in shards)
    arr = np.lib.format.open_memmap(out_npy, mode="w+",
                                    dtype=np.float32, shape=(total, n_pts, 3))
    w = 0
    for s in shards:
        raw = np.fromfile(s, dtype=">f4")
        k = raw.size // (n_pts * 3)
        if k * n_pts * 3 != raw.size:
            raise SystemExit("shard %s is truncated (partial write?)" % s)
        arr[w:w + k] = raw.reshape(k, n_pts, 3).astype(np.float32)
        w += k
    arr.flush()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default=None, help="default: <root>/npy")
    args = ap.parse_args()

    out = args.out or os.path.join(args.root, "npy")
    os.makedirs(out, exist_ok=True)

    n_surf, n_node = read_topology(args.root)
    print("topology: %d surface verts, %d fem nodes" % (n_surf, n_node))

    meta = load_meta(args.root)
    idx = np.array([m[0] for m in meta])
    label = np.array([m[1] for m in meta])
    max_vel = np.array([m[3] for m in meta])
    peak = np.array([m[4] for m in meta])
    vol = np.array([m[5] for m in meta])
    ninv = np.array([m[6] for m in meta])
    emin = np.array([m[7] for m in meta])
    settle_t = np.array([m[8] for m in meta])
    secs = np.array([m[9] for m in meta])
    n = len(meta)

    nv = stack(args.root, "verts", n_surf, os.path.join(out, "surf_verts.npy"))
    nn = stack(args.root, "nodes", n_node, os.path.join(out, "fem_nodes.npy"))
    if not (nv == nn == n):
        raise SystemExit("row count mismatch: meta=%d verts=%d nodes=%d" % (n, nv, nn))

    # realized activations, aligned to the same row order
    pool = np.loadtxt(args.pool, delimiter=",", skiprows=3, comments="#")
    lut = {int(r[0]): r[1:] for r in pool}
    act = np.stack([lut[i] for i in idx]).astype(np.float32)

    np.save(os.path.join(out, "pool_index.npy"), idx)
    np.save(os.path.join(out, "activation.npy"), act)
    np.save(os.path.join(out, "label.npy"), label)
    np.save(os.path.join(out, "max_vel.npy"), max_vel)
    np.save(os.path.join(out, "vol_ratio.npy"), vol)
    np.save(os.path.join(out, "min_elem_vol.npy"), emin)
    np.save(os.path.join(out, "n_inverted.npy"), ninv)

    # ---- F2a: yield table -------------------------------------------------
    print("\nvalidity yield  (N=%d)" % n)
    for k in ["VALID", "MARGINAL", "INVALID_PHYSICAL", "FAILED_NUMERICAL"]:
        c = int((label == k).sum())
        print("  %-18s %7d  (%.1f%%)" % (k, c, 100.0 * c / max(n, 1)))
    print("\nthroughput: %.2f s/sample  ->  %.1f h for 200k on ONE worker"
          % (secs.mean(), secs.mean() * 200000 / 3600))

    # ---- watchdog sanity ---------------------------------------------------
    if (ninv < 0).all():
        print("\n!! n_inverted is NA on every row -> the watchdog is BLIND.")
    if np.isnan(emin).all():
        print("!! min_elem_vol is NA on every row -> the watchdog is BLIND.")

    # ---- F2b: element quality (the signal that actually matters) ----------
    # Global vol_ratio stays ~1.0 even when an element folds through itself,
    # because incompressibility is enforced globally. Inversion is LOCAL.
    ok = ~np.isnan(emin)
    if ok.any():
        print("\nmin per-element volume ratio:")
        print("  P01 %.3f | P05 %.3f | median %.3f | min %.3f"
              % (np.percentile(emin[ok], 1), np.percentile(emin[ok], 5),
                 np.median(emin[ok]), emin[ok].min()))
        print("  elements inverted in %d samples" % int((ninv > 0).sum()))

    # ---- settling sanity ---------------------------------------------------
    print("\nadaptive settle: median %.2fs, P95 %.2fs, hit cap in %d samples"
          % (np.median(settle_t), np.percentile(settle_t, 95),
             int((settle_t >= settle_t.max() - 1e-6).sum())))
    print("peak velocity during ramp: median %.3g, P95 %.3g, max %.3g"
          % (np.median(peak), np.percentile(peak, 95), peak.max()))
    print("max node speed at end of settle: median %.3g, P95 %.3g"
          % (np.median(max_vel), np.percentile(max_vel, 95)))


if __name__ == "__main__":
    main()
