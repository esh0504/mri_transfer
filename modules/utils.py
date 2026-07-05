#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2/modules/utils.py

프로젝트 공통 유틸 — 파일 IO와 시각화(렌더링) 전부.
artisynth·retarget 알고리즘 어디에도 종속되지 않는다(모델은 duck typing:
model.verts / model.faces / model.names 만 사용). 알고리즘 모듈은 여기의
프리미티브(load/save/vis)를 호출해서 쓰고, 자기 자신은 계산에만 집중한다.

구성:
  · 경로        V2_DIR / REPO_DIR / OUT_DIR / ensure_dir / out_path / repo_path / data_path
  · 이미지 IO   save_png
  · OBJ  IO     load_obj / extract_obj / save_obj
  · mask IO     load_mask / load_video / mask_label_2d
  · CSV / npy   save_csv / read_csv_dicts / save_npy
  · 시각화      vis / vis3d / vis_mask / activation_caption / vis_with_activations

알고리즘(artisynth/·retarget/)은 여기의 프리미티브를 호출해서 IO/렌더를 처리한다.
"""
import csv
import glob
import os
import re

import numpy as np

# --------------------------------------------------------------------------- #
# 경로
# --------------------------------------------------------------------------- #
V2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(V2_DIR)
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(V2_DIR, "_test_out"))


def ensure_dir(directory):
    """디렉터리 생성(존재하면 무시). 인자를 그대로 반환."""
    if directory:
        os.makedirs(directory, exist_ok=True)
    return directory


def out_path(name):
    """OUT_DIR 아래 경로를 만들고 OUT_DIR을 보장한다. name이 절대경로면 그대로."""
    ensure_dir(OUT_DIR)
    return name if os.path.isabs(name) else os.path.join(OUT_DIR, name)


def repo_path(*parts):
    """리포 루트(=V2의 부모) 기준 경로.

    예: repo_path("tongue_model", "tongue_rest_m.obj")
        repo_path("datasets", "GT_Segmentations", "Subject3")
    """
    return os.path.join(REPO_DIR, *parts)


def data_path(data_root, *parts):
    """data_root 기준 경로. data_root가 상대면 V2_DIR(프로젝트 루트) 기준."""
    root = data_root if os.path.isabs(data_root) else os.path.join(V2_DIR, data_root)
    return os.path.normpath(os.path.join(root, *parts))


# --------------------------------------------------------------------------- #
# 이미지 IO
# --------------------------------------------------------------------------- #
def save_png(img, name_or_path):
    """(H, W, 3) uint8 → PNG 저장. 반환: 저장 경로(str) 또는 None.

    bare name(디렉터리 구분자 없음)이면 OUT_DIR 아래에 저장한다.
    imageio가 없으면 조용히 건너뛰고 None을 반환한다.
    """
    if os.path.isabs(name_or_path) or os.path.dirname(name_or_path):
        path = name_or_path
    else:
        path = out_path(name_or_path)
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, img)
        return path
    except Exception as e:
        print("   (PNG 저장 건너뜀: %s)" % e)
        return None


def save_npy(path, arr):
    """np.save 래퍼(상위 디렉터리 보장). 저장 경로 반환."""
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    np.save(path, np.asarray(arr))
    return path


# --------------------------------------------------------------------------- #
# CSV IO (프리미티브)
# --------------------------------------------------------------------------- #
def save_csv(path, fieldnames, rows):
    """rows(list[dict])를 fieldnames 순서로 CSV 저장. 저장 경로 반환."""
    out = os.path.abspath(str(path))
    ensure_dir(os.path.dirname(out) or ".")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def read_csv_dicts(path):
    """CSV → list[dict] (csv.DictReader)."""
    p = os.path.abspath(str(path))
    if not os.path.isfile(p):
        raise FileNotFoundError("csv not found: %s" % p)
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# OBJ IO
# --------------------------------------------------------------------------- #
DEFAULT_MESH_COLOR = (230, 90, 75)


def _require_mesh(model):
    """model.verts / model.faces 존재 검증(duck typing)."""
    if (model is None or getattr(model, "verts", None) is None
            or getattr(model, "faces", None) is None):
        raise ValueError("mesh가 비었습니다 (verts/faces 필요).")


def load_obj(path):
    """Wavefront OBJ → (verts (N,3) float, faces (F,3) int).

    다각형 face는 fan 삼각분할. 단위는 파일 그대로(혀 OBJ는 metres)."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in t[1:]]
                for k in range(1, len(idx) - 1):       # fan triangulation
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def extract_obj(model, color=None):
    """TongueModel(또는 verts/faces 핸들) → OBJ 데이터 dict.

    반환 키:
      points_cloud : (N, 3) float — 정점 좌표 (metres)
      Mesh         : (F, 3) int   — 삼각형 face 인덱스 (0-based)
      Color        : (N, 3) uint8 — 정점 RGB (0..255)
    """
    _require_mesh(model)
    verts = np.asarray(model.verts, dtype=float)
    faces = np.asarray(model.faces, dtype=int)
    n = len(verts)
    if color is None:
        rgb = np.tile(np.asarray(DEFAULT_MESH_COLOR, dtype=np.uint8), (n, 1))
    else:
        c = np.asarray(color, dtype=np.uint8)
        rgb = np.tile(c.reshape(1, 3), (n, 1)) if c.shape == (3,) else c.reshape(n, 3)
    return {"points_cloud": verts, "Mesh": faces, "Color": rgb}


def _obj_path(path):
    p = str(path)
    return p if p.lower().endswith(".obj") else p + ".obj"


def save_obj(obj, path):
    """extract_obj() 결과(동일 키 dict)를 Wavefront OBJ로 저장. 저장 경로 반환.

    Color가 있으면 ``v x y z r g b`` (RGB 0..1) 확장 형식으로 쓴다.
    """
    verts = np.asarray(obj["points_cloud"], dtype=float)
    faces = obj.get("Mesh")
    colors = obj.get("Color")
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8).reshape(len(verts), 3)

    out = _obj_path(path)
    ensure_dir(os.path.dirname(os.path.abspath(out)) or ".")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# modules.utils save_obj\n")
        for i, v in enumerate(verts):
            if colors is not None:
                r, g, b = colors[i] / 255.0
                f.write("v %.6f %.6f %.6f %.6f %.6f %.6f\n"
                        % (v[0], v[1], v[2], r, g, b))
            else:
                f.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        if faces is not None:
            for t in np.asarray(faces, dtype=int):
                f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))
    return out


# --------------------------------------------------------------------------- #
# MRI mask IO
# --------------------------------------------------------------------------- #
MAT_VAR_NAME = "mask_frame"


def _natkey(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else 0


def _require_dir(folder_path, label="folder_path"):
    folder = os.path.abspath(str(folder_path))
    if not os.path.isdir(folder):
        raise NotADirectoryError("%s is not a directory: %s" % (label, folder))
    return folder


def mask_label_2d(mask):
    """(H,W,C) 또는 (H,W) → 2D label slice."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        return mask[..., 0]
    if mask.ndim == 2:
        return mask
    raise ValueError("mask must be (H,W) or (H,W,C), got %s" % (mask.shape,))


def _as_hwc(arr):
    """array → (H, W, C). 2D label/image → C=1."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a[..., np.newaxis]
    if a.ndim == 3:
        return a
    raise ValueError("mask must be 2D (H,W) or 3D (H,W,C), got %s" % (a.shape,))


def _load_mat(path, mat_var=MAT_VAR_NAME):
    import scipy.io as sio
    data = sio.loadmat(path)
    if mat_var in data:
        return data[mat_var]
    for key, val in data.items():
        if not key.startswith("__"):
            return val
    raise ValueError("no array found in %s" % path)


def load_mask(mask_path, mat_var=MAT_VAR_NAME):
    """단일 마스크 파일(.mat/.npy/.npz/이미지) → (H, W, C)."""
    path = os.path.abspath(str(mask_path))
    if not os.path.isfile(path):
        raise FileNotFoundError("mask_path not found: %s" % path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        arr = _load_mat(path, mat_var=mat_var)
    elif ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        if mat_var in data:
            arr = data[mat_var]
        else:
            keys = [k for k in data.files if not k.startswith("_")]
            if not keys:
                raise ValueError("empty npz: %s" % path)
            arr = data[keys[0]]
    elif ext in (".png", ".tif", ".tiff", ".bmp"):
        try:
            import imageio.v2 as imageio
            arr = imageio.imread(path)
        except Exception:
            from PIL import Image
            arr = np.asarray(Image.open(path))
    else:
        raise ValueError("unsupported mask format: %s" % ext)
    return _as_hwc(arr)


def load_video(folder_path):
    """폴더 내 ``mask_*.mat`` → (T, H, W, C)."""
    folder = _require_dir(folder_path)
    files = sorted(glob.glob(os.path.join(folder, "mask_*.mat")), key=_natkey)
    if not files:
        raise FileNotFoundError("no mask_*.mat files under %s" % folder)
    frames = [load_mask(p) for p in files]
    return np.stack(frames, axis=0)


# --------------------------------------------------------------------------- #
# 시각화 — 2D 정중시상 실루엣 마스크 (matplotlib)
# --------------------------------------------------------------------------- #
def _silhouette_mask(verts, faces, size=(256, 256), bounds=None, axes=(0, 2),
                     margin=0.05, fill=(255, 255, 255), bg=(0, 0, 0)):
    """표면 삼각형을 2D 평면(axes)에 투영해 채운 실루엣 마스크 (H,W,3) uint8.

    matplotlib(Agg)만 사용 → JVM 불필요. axes=(0,2)는 정중시상(x-z) 평면.
    bounds=(amin,amax,bmin,bmax)를 주면 좌표계 고정(시계열 비교용).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    H, W = int(size[0]), int(size[1])
    ai, bi = axes
    P = np.asarray(verts, dtype=float)[:, [ai, bi]]
    F = np.asarray(faces, dtype=int)

    if bounds is None:
        amin, amax = P[:, 0].min(), P[:, 0].max()
        bmin, bmax = P[:, 1].min(), P[:, 1].max()
        da = (amax - amin) or 1.0
        db = (bmax - bmin) or 1.0
        amin -= da * margin; amax += da * margin
        bmin -= db * margin; bmax += db * margin
    else:
        amin, amax, bmin, bmax = bounds

    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(amin, amax)
    ax.set_ylim(bmin, bmax)
    ax.axis("off")
    ax.set_facecolor(tuple(c / 255.0 for c in bg))
    fig.patch.set_facecolor(tuple(c / 255.0 for c in bg))
    coll = PolyCollection(P[F], closed=True,
                          facecolors=[tuple(c / 255.0 for c in fill)],
                          edgecolors="none", antialiaseds=False)
    ax.add_collection(coll)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
    plt.close(fig)
    return buf


def vis_mask(model_or_verts, faces=None, size=(256, 256), bounds=None,
             plane="midsag", fill=(255, 255, 255), bg=(0, 0, 0)):
    """2D midsagittal 실루엣 마스크 (matplotlib). 최적화 loss / MRI 비교용."""
    axes = {"midsag": (0, 2), "axial": (0, 1), "coronal": (1, 2)}[plane]
    if faces is not None:
        verts = np.asarray(model_or_verts, dtype=float)
        f = np.asarray(faces, dtype=int)
    else:
        _require_mesh(model_or_verts)
        verts, f = model_or_verts.verts, model_or_verts.faces
    return _silhouette_mask(verts, f, size=size, bounds=bounds, axes=axes,
                            fill=fill, bg=bg)


# --------------------------------------------------------------------------- #
# 시각화 — 3D 표면 렌더 (trimesh + pyrender, off-screen)
# --------------------------------------------------------------------------- #
# ArtiSynth 혀 mesh 해부학 좌표 (world, metres): +Z=위, -Y=우, -X=앞
AX_UP = np.array([0.0, 0.0, 1.0])
AX_RIGHT = np.array([0.0, -1.0, 0.0])
AX_FRONT = np.array([-1.0, 0.0, 0.0])


def _look_at_pose(eye, target, up=None):
    """World-from-camera pose (pyrender convention). up 기본 +Z(superior)."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = AX_UP if up is None else np.asarray(up, dtype=float)
    z = eye - target
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    if n < 1e-8:
        up = AX_FRONT if abs(np.dot(up, AX_UP)) > 0.9 else AX_UP
        x = np.cross(up, z)
        n = np.linalg.norm(x)
    x /= n + 1e-12
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def _camera_pose_from_angles(center, distance, upper_degree, right_degree):
    """해부학 구면좌표로 eye를 잡고 centroid를 바라본다.

    기준축 +Z=위, -Y=우, -X=앞.
    upper_degree : +Z 쪽으로 기울임(0=수평, 90=위에서).
    right_degree : 앞(0)→우(90)→뒤(180)→좌(270) CCW.
    right_degree=90, upper_degree=0 → 정중시상(sagittal).
    """
    elev = np.deg2rad(float(upper_degree))
    azim = np.deg2rad(float(right_degree))
    c = np.asarray(center, dtype=float)
    ce, se = np.cos(elev), np.sin(elev)
    ca, sa = np.cos(azim), np.sin(azim)
    offset = float(distance) * (ce * ca * AX_FRONT + ce * sa * AX_RIGHT + se * AX_UP)
    return _look_at_pose(c + offset, c, up=AX_UP)


_virtual_display = None


def _ensure_headless_display():
    """DISPLAY 없는 headless 환경에서 pyrender용 가상 X11(xvfb)을 띄운다."""
    global _virtual_display
    if os.environ.get("DISPLAY") or _virtual_display is not None:
        return
    try:
        from pyvirtualdisplay import Display
    except ImportError as e:
        raise RuntimeError(
            "vis: DISPLAY가 없습니다 (headless 환경).\n"
            "  apt install xvfb libxrender1 libx11-6\n"
            "  pip install PyVirtualDisplay\n"
            "또는: xvfb-run -a python main.py"
        ) from e
    _virtual_display = Display(visible=0, size=(1024, 768))
    _virtual_display.start()


def _import_pyrender():
    """pyrender 서브모듈 lazy import (Viewer/pyglet 창은 사용하지 않음)."""
    from pyrender.offscreen import OffscreenRenderer
    from pyrender.scene import Scene
    from pyrender.mesh import Mesh
    from pyrender.material import MetallicRoughnessMaterial
    from pyrender.light import DirectionalLight
    from pyrender.camera import PerspectiveCamera
    return {
        "OffscreenRenderer": OffscreenRenderer, "Scene": Scene, "Mesh": Mesh,
        "MetallicRoughnessMaterial": MetallicRoughnessMaterial,
        "DirectionalLight": DirectionalLight, "PerspectiveCamera": PerspectiveCamera,
    }


def _render_mesh3d_pyrender(verts, faces, size=(768, 768), bg=(28, 28, 36),
                            color=(230, 90, 75), upper_degree=45.0,
                            right_degree=90.0):
    """trimesh + pyrender off-screen 3D → (H, W, 3) uint8 RGB."""
    _ensure_headless_display()
    try:
        import trimesh
        pr = _import_pyrender()
    except ImportError as e:
        raise ImportError(
            "vis: pip install trimesh pyrender (+ headless: apt install xvfb libxrender1)"
        ) from e

    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vis: verts must be (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("vis: faces must be (F, 3)")

    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    rgba = [c / 255.0 for c in color] + [1.0]
    material = pr["MetallicRoughnessMaterial"](
        baseColorFactor=rgba, metallicFactor=0.15, roughnessFactor=0.55)
    pr_mesh = pr["Mesh"].from_trimesh(tm, material=material, smooth=True)

    scene = pr["Scene"](
        bg_color=[bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0],
        ambient_light=[0.25, 0.25, 0.28])
    scene.add(pr_mesh)

    center = tm.centroid
    dist = float(max(tm.extents.max(), 1e-4) * 2.4)
    cam_pose = _camera_pose_from_angles(center, dist, upper_degree, right_degree)
    camera = pr["PerspectiveCamera"](yfov=np.pi / 4.0, znear=dist * 0.01,
                                     zfar=dist * 20.0)
    scene.add(camera, pose=cam_pose)
    light = pr["DirectionalLight"](color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)
    fill = pr["DirectionalLight"](color=[0.9, 0.9, 1.0], intensity=1.2)
    scene.add(fill, pose=_camera_pose_from_angles(
        center, dist, upper_degree + 15, right_degree + 90))

    H, W = int(size[0]), int(size[1])
    renderer = pr["OffscreenRenderer"](viewport_width=W, viewport_height=H)
    try:
        img, _ = renderer.render(scene)
    finally:
        renderer.delete()
    img = np.asarray(img)[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _vis_png_path(out_path_):
    p = os.path.abspath(str(out_path_))
    return p if p.lower().endswith(".png") else p + ".png"


def _write_vis_png(img, out_path_):
    path = _vis_png_path(out_path_)
    ensure_dir(os.path.dirname(path))
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, img)
    except ImportError as e:
        raise ImportError("vis save: pip install imageio") from e
    return path


def _apply_vis_settings(settings, defaults):
    """rendering_settings dict → vis() 옵션 (elev/azim 구 이름 호환)."""
    if not settings:
        return dict(defaults)
    out = dict(defaults)
    for key in ("size", "bg", "color", "out_path", "upper_degree", "right_degree"):
        if key in settings:
            out[key] = settings[key]
    if "elev" in settings:
        out["upper_degree"] = settings["elev"]
    if "azim" in settings:
        out["right_degree"] = settings["azim"]
    return out


def vis(model_or_verts, faces=None, out_path=None, size=(768, 768),
        bg=(28, 28, 36), color=(230, 90, 75), upper_degree=45.0,
        right_degree=90.0, **kwargs):
    """3D mesh를 trimesh+pyrender로 렌더한다.

    1) vis(model)                        → (H, W, 3) uint8
    2) vis(model, rendering_settings)    → settings dict
       (예: dict(upper_degree=30, right_degree=90, size=(640, 640)))
    3) vis(model, upper_degree=30, ...)  → 키워드 인자
    4) vis(verts, faces, out_path, ...)  → PNG 저장, 경로(str) 반환
    """
    defaults = dict(size=size, bg=bg, color=color, upper_degree=upper_degree,
                    right_degree=right_degree, out_path=out_path)

    if isinstance(faces, dict):                 # vis(model, settings)
        opts = _apply_vis_settings(faces, defaults)
        opts.update({k: v for k, v in kwargs.items()
                     if k in defaults or k in ("elev", "azim")})
        if "elev" in kwargs:
            opts["upper_degree"] = kwargs["elev"]
        if "azim" in kwargs:
            opts["right_degree"] = kwargs["azim"]
        faces = None
    else:
        opts = _apply_vis_settings(kwargs, defaults)

    size = opts["size"]; bg = opts["bg"]; color = opts["color"]
    upper_degree = opts["upper_degree"]; right_degree = opts["right_degree"]
    out_path = opts["out_path"]

    if out_path is not None:                    # vis(verts, faces, out_path)
        if faces is None:
            raise ValueError("vis(verts, faces, out_path): faces가 필요합니다.")
        img = _render_mesh3d_pyrender(
            model_or_verts, faces, size=size, bg=bg, color=color,
            upper_degree=upper_degree, right_degree=right_degree)
        return _write_vis_png(img, out_path)

    model = model_or_verts
    _require_mesh(model)
    return _render_mesh3d_pyrender(
        model.verts, model.faces, size=size, bg=bg, color=color,
        upper_degree=upper_degree, right_degree=right_degree)


def vis3d(model, size=(768, 768), bg=(28, 28, 36), color=(230, 90, 75),
          upper_degree=45.0, right_degree=90.0, show_edges=False):
    """vis()와 동일한 trimesh+pyrender 3D 렌더 (show_edges 현재 미사용)."""
    _require_mesh(model)
    return _render_mesh3d_pyrender(
        model.verts, model.faces, size=size, bg=bg, color=color,
        upper_degree=upper_degree, right_degree=right_degree)


# --------------------------------------------------------------------------- #
# 시각화 — 11D 활성값 오버레이
# --------------------------------------------------------------------------- #
def activation_caption(names, activation, cols=6, pad=8, line_h=18,
                       bg=(28, 28, 36), fg=(220, 220, 220), hi=(255, 180, 90)):
    """11D 활성값 텍스트 패널 (H, W, 3) uint8. PIL 없으면 빈 1행 패널."""
    pairs = [(n, float(v)) for n, v in zip(names, activation)]
    lines = [pairs[i:i + cols] for i in range(0, len(pairs), cols)]
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        text = "  ".join("%s=%.2f" % (n, v) for n, v in pairs)
        w = max(len(text) * 7, 320)
        return np.full((line_h + 2 * pad, w, 3), bg, dtype=np.uint8)

    font = ImageFont.load_default()
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    col_w = max(draw_probe.textlength("%s=0.00" % n, font=font) for n in names) + 12
    w = int(cols * col_w + 2 * pad)
    h = len(lines) * line_h + 2 * pad
    panel = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(panel)
    for row, chunk in enumerate(lines):
        y = pad + row * line_h
        for col, (name, val) in enumerate(chunk):
            x = pad + col * col_w
            draw.text((x, y), "%s=%.2f" % (name, val),
                      fill=(hi if val > 0 else fg), font=font)
    return np.asarray(panel)


def vis_with_activations(model, settings=None, names=None):
    """vis() 렌더 + 하단 활성값 패널을 세로로 이어 붙인 (H, W, 3) uint8.

    names 미지정 시 model.names 사용(둘 다 없으면 활성값 패널 생략)."""
    settings = settings or {}
    img = vis(model, settings) if settings else vis(model)
    act = getattr(model, "activation", None)
    if act is None:
        return img
    names = names or getattr(model, "names", None)
    if names is None:
        return img
    act = np.asarray(act, dtype=float).reshape(-1)
    cap = activation_caption(list(names)[: len(act)], act)
    bg = settings.get("bg", (28, 28, 36)) if isinstance(settings, dict) else (28, 28, 36)
    w = max(img.shape[1], cap.shape[1])
    if img.shape[1] < w:
        img = np.hstack([img, np.full((img.shape[0], w - img.shape[1], 3), bg, dtype=np.uint8)])
    if cap.shape[1] < w:
        cap = np.hstack([cap, np.full((cap.shape[0], w - cap.shape[1], 3), (28, 28, 36), dtype=np.uint8)])
    return np.vstack([img, cap])
