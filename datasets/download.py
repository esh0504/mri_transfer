#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the gated Hugging Face dataset into V2/datasets/.

Requires Hugging Face access approval for:
  https://huggingface.co/datasets/SeunghoEum/mri-tongue-dataset

Authenticate first:
  hf auth login
  # or: export HF_TOKEN=...

Layout after download (V2/datasets/):
  GT_Segmentations/Subject{1-5}/mask_*.mat
  MRI_SSFP_10fps/Subject{1-5}/image_*.dcm   (optional)
  tongue_model/tongue_rest_m.obj

Example pipeline run (absolute paths):
  python main.py stage=retarget \\
    paths.tongue_obj=$(pwd)/datasets/tongue_model/tongue_rest_m.obj \\
    paths.mask_dir=$(pwd)/datasets/GT_Segmentations/Subject3 \\
    paths.rest_mask=$(pwd)/datasets/GT_Segmentations/Subject3/mask_1.mat \\
    paths.target_mask=$(pwd)/datasets/GT_Segmentations/Subject3/mask_51.mat
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

DEFAULT_REPO = "SeunghoEum/mri-tongue-dataset"
DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(DATASETS_DIR)

# HF repo path -> local folder under V2/datasets/
LAYOUT = (
    ("datasets/GT_Segmentations", "GT_Segmentations"),
    ("datasets/MRI_SSFP_10fps", "MRI_SSFP_10fps"),
    ("tongue_model", "tongue_model"),
)


def _require_hub():
    try:
        from huggingface_hub import HfApi, snapshot_download
        return HfApi, snapshot_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from e


def _check_access(repo_id: str, token: str | None) -> None:
    HfApi, _ = _require_hub()
    api = HfApi(token=token)
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except Exception as e:
        raise SystemExit(
            "Cannot access %s.\n"
            "  1) Request access: https://huggingface.co/datasets/%s\n"
            "  2) After approval: hf auth login\n"
            "Error: %s" % (repo_id, repo_id, e)
        ) from e
    gated = getattr(info, "gated", False)
    if gated:
        print("Dataset is gated (%s). Ensure your account is approved." % gated)


def _install_layout(staging: str, out_dir: str, *, skip_dicom: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for src_rel, dst_name in LAYOUT:
        if skip_dicom and dst_name == "MRI_SSFP_10fps":
            print("Skipping DICOM: %s" % src_rel)
            continue
        src = os.path.join(staging, src_rel)
        dst = os.path.join(out_dir, dst_name)
        if not os.path.isdir(src):
            raise SystemExit("Missing in Hub snapshot: %s" % src_rel)
        if os.path.lexists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        n = sum(1 for _root, _dirs, files in os.walk(dst) for _ in files)
        print("  %s  (%d files)" % (dst_name, n))


def download(
    repo_id: str = DEFAULT_REPO,
    out_dir: str | None = None,
    *,
    token: str | None = None,
    skip_dicom: bool = False,
    revision: str | None = None,
) -> str:
    """Download and flatten the HF dataset. Returns output directory."""
    _, snapshot_download = _require_hub()
    out_dir = out_dir or DATASETS_DIR
    out_dir = os.path.abspath(out_dir)
    token = token or os.environ.get("HF_TOKEN")

    print("Repo:   %s" % repo_id)
    print("Output: %s" % out_dir)
    _check_access(repo_id, token)

    with tempfile.TemporaryDirectory(prefix="hf_mri_tongue_") as staging:
        print("Downloading snapshot …")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=staging,
            token=token,
            revision=revision,
        )
        print("Installing layout …")
        _install_layout(staging, out_dir, skip_dicom=skip_dicom)

    tongue = os.path.join(out_dir, "tongue_model", "tongue_rest_m.obj")
    masks = os.path.join(out_dir, "GT_Segmentations", "Subject3", "mask_1.mat")
    if not os.path.isfile(tongue):
        raise SystemExit("Expected mesh missing: %s" % tongue)
    if not os.path.isfile(masks):
        raise SystemExit("Expected mask missing: %s" % masks)

    print("\nDone.")
    print("\nSuggested run from V2/:")
    print("  python main.py stage=retarget \\")
    print("    paths.tongue_obj=%s \\" % tongue)
    print("    paths.mask_dir=%s \\" % os.path.join(out_dir, "GT_Segmentations", "Subject3"))
    print("    paths.rest_mask=%s \\" % masks)
    print("    paths.target_mask=%s" % os.path.join(
        out_dir, "GT_Segmentations", "Subject3", "mask_51.mat"))
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument("--out", default=DATASETS_DIR, help="Output directory (default: V2/datasets)")
    p.add_argument("--revision", default=None, help="Git revision / tag on the Hub")
    p.add_argument("--skip-dicom", action="store_true", help="Skip MRI_SSFP_10fps DICOM cine")
    p.add_argument("--token", default=None, help="HF token (default: HF_TOKEN env or cached login)")
    args = p.parse_args(argv)
    download(args.repo, args.out, token=args.token, skip_dicom=args.skip_dicom, revision=args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
