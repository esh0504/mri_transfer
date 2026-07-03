#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2/artisynth/utils.py

ArtiSynth FEM 혀 모델 알고리즘 — 로드 + 11D 근육 활성값 forward 구동.
JPype로 ArtiSynth를 *인프로세스* 로드하고, 활성값을 램프로 가해 평형까지 시뮬한다.
파일 IO·시각화는 이 모듈에 없다(전부 modules.utils). 여기는 계산만 담당.

핵심 API:

    model = fem(model, muscle_values)   # 로드 + 활성값 적용 → TongueModel
    model = model_from_obj(path)        # OBJ rest 메쉬 → TongueModel (JVM 불필요)

사용 예:

    from artisynth.utils import fem, MUSCLE_NAMES
    from modules.utils import vis, save_obj, extract_obj

    model = fem(None, [0.3] + [0.0] * 10)      # 11D 활성값 (MUSCLE_NAMES 순서)
    model = fem(model, {"GGP": 0.5, "HG": 0.2})  # 핸들 재사용(로드 1회만)

환경 변수 (기본값):
    ARTISYNTH_HOME   ArtiSynth 트리 경로 (classes + lib/*.jar)
    TONGUE_MODEL     기본 artisynth.models.tongue3d.HexTongueDemo
    SETTLE_T  (0.4)  hold 구간 기본 시간(초); ramp step당 시간은 RAMP_STEP_T
    MAXSTEP   (0.0005) FEM 적분 스텝(초)
    RAMP_STEP_T (0.01) ramp 각 단계당 시뮬 시간(초)
    NRAMP     (20)   활성도 램프 단계 수
    INCOMP    (AUTO) 비압축성 방법(OFF/AUTO/ELEMENT/NODAL)
    ADAPTIVE_STEPPING (1) GUI Play와 동일하게 adaptive stepping 사용
    JVM_XMX   (4g)   JVM 힙

요구 사항: Java(JDK) + 컴파일된 ArtiSynth 트리, pip install JPype1
"""
import glob
import os

import numpy as np

from modules.utils import load_obj

# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #
ARTISYNTH_HOME = os.environ.get(
    "ARTISYNTH_HOME", r"C:\Users\d11\artisynth\artisynth_core")
TONGUE_MODEL = os.environ.get(
    "TONGUE_MODEL", "artisynth.models.tongue3d.HexTongueDemo")
SETTLE_T = float(os.environ.get("SETTLE_T", "0.4"))
MAXSTEP = float(os.environ.get("MAXSTEP", "0.0005"))
RAMP_STEP_T = float(os.environ.get("RAMP_STEP_T", "0.01"))
NRAMP = int(os.environ.get("NRAMP", "20"))
NRAMP_RETRY_START = int(os.environ.get("NRAMP_RETRY_START", "50"))
NRAMP_RETRY_MAX = int(os.environ.get("NRAMP_RETRY_MAX", "500"))
INCOMP = os.environ.get("INCOMP", "AUTO").upper()  # GUI 기본값과 일치(부피 보존)
# adaptive stepping: GUI Play와 동일하게 켬(True). 불안정 스텝을 자동 세분해 재시도 →
# inverted-elements 예외 방지. 0/false로 두면 기존(headless) 동작처럼 바로 예외.
ADAPTIVE_STEPPING = os.environ.get("ADAPTIVE_STEPPING", "1").lower() not in (
    "0", "false", "no", "off")
JVM_XMX = os.environ.get("JVM_XMX", "4g")

# 11D 제어 공간(SKILL.md). inverse(6번)/activations CSV 헤더 순서와 동일.
MUSCLE_NAMES = ["GGP", "GGM", "GGA", "STY", "GH", "MH",
                "HG", "VERT", "TRANS", "IL", "SL"]

# configure()가 덮어쓸 수 있는 전역 설정 키 → 형변환 함수.
_CONFIG_KEYS = {
    "artisynth_home": str, "tongue_model": str, "settle_t": float,
    "maxstep": float, "ramp_step_t": float, "nramp": int,
    "nramp_retry_start": int, "nramp_retry_max": int,
    "incomp": lambda v: str(v).upper(), "adaptive_stepping": bool, "jvm_xmx": str,
}


def configure(cfg=None, **overrides):
    """모듈 전역 fem 파라미터를 설정으로 덮어쓴다 (Hydra 등에서 호출).

    사용: configure(cfg.artisynth)  또는  configure(nramp=40, incomp="NODAL").
    cfg는 dict/DictConfig(매핑) 무엇이든 가능. 인식하는 키는 _CONFIG_KEYS 참조.
    """
    opts = dict(cfg) if cfg else {}
    opts.update(overrides)
    g = globals()
    for key, cast in _CONFIG_KEYS.items():
        if key in opts and opts[key] is not None:
            g[key.upper()] = cast(opts[key])


# --------------------------------------------------------------------------- #
# Model 핸들
# --------------------------------------------------------------------------- #
class TongueModel:
    """ArtiSynth FEM 혀 모델 상태 핸들.

    JPype 핸들(main/tongue/exciters)과 토폴로지(faces), 그리고 마지막으로 적용한
    활성값에 대한 변형 표면 정점(verts)을 담는다. fem()이 채워서 반환한다.
    """

    def __init__(self):
        self.main = None          # artisynth.core.driver.Main
        self.tongue = None        # FemMuscleModel
        self.exciters = None      # list[MuscleExciter] (model 순서)
        self.names = None         # list[str] 모델이 보고한 exciter 이름(순서)
        self.mesh = None          # 표면 메쉬 핸들
        self.faces = None         # (F,3) int  표면 삼각형 인덱스(불변)
        self.verts = None         # (N,3) float  현재 표면 정점(metres)
        self.activation = None    # (11,) float  마지막 적용 활성값

    @property
    def loaded(self):
        return self.main is not None


def model_from_obj(path):
    """OBJ 표면 메쉬를 TongueModel로 래핑(verts/faces만 채움).

    JVM/ArtiSynth 없이 실제 혀 rest 메쉬로 렌더/테스트할 때 사용한다.
    근육 활성값에 의한 *변형*은 여전히 fem()(ArtiSynth)이 필요하다.
    (OBJ 파싱은 modules.utils.load_obj — IO는 그쪽에 모여 있다.)"""
    v, f = load_obj(path)
    m = TongueModel()
    m.verts = v
    m.faces = f
    m.names = list(MUSCLE_NAMES)
    m.activation = np.zeros(len(MUSCLE_NAMES))
    return m


# --------------------------------------------------------------------------- #
# JPype / JVM
# --------------------------------------------------------------------------- #
def _start_jvm():
    import jpype
    if jpype.isJVMStarted():
        return
    cp = ([os.path.join(ARTISYNTH_HOME, "classes")]
          + glob.glob(os.path.join(ARTISYNTH_HOME, "lib", "*.jar")))
    libdir = os.path.join(ARTISYNTH_HOME, "lib")
    jpype.startJVM(
        "-Xmx%s" % JVM_XMX,
        "-Djava.awt.headless=true",
        "-Dartisynth.home=%s" % ARTISYNTH_HOME,
        "-Djava.library.path=%s" % libdir,
        classpath=cp,
    )


def _jclass(name):
    import jpype
    return jpype.JClass(name)


def _find_tongue(root):
    """muscle exciter를 가진 FemMuscleModel을 재귀 탐색."""
    def rec(m):
        try:
            if hasattr(m, "getMuscleExciters") and m.getMuscleExciters().size() > 0:
                return m
        except Exception:
            pass
        try:
            subs = m.models()
            for i in range(subs.size()):
                r = rec(subs.get(i))
                if r is not None:
                    return r
        except Exception:
            pass
        return None
    tops = root.models()
    for i in range(tops.size()):
        r = rec(tops.get(i))
        if r is not None:
            return r
    return None


def _extract_faces(mesh):
    """표면 메쉬 face 인덱스 (F,3). 토폴로지는 불변이라 rest에서 1회만 추출."""
    faces = mesh.getFaces()
    nf = faces.size()
    F = np.empty((nf, 3), dtype=int)
    for i in range(nf):
        vi = faces.get(i).getVertexIndices()
        F[i, 0] = vi[0]
        F[i, 1] = vi[1]
        F[i, 2] = vi[2]
    return F


def _deactivate_probes(root):
    try:
        ips = root.getInputProbes()
        for i in range(ips.size()):
            try:
                ips.get(i).setActive(False)
            except Exception:
                pass
    except Exception:
        pass


def _read_surface(model):
    """현재 표면 정점 (N,3) metres."""
    mesh = model.mesh
    verts = mesh.getVertices()
    nv = verts.size()
    out = np.empty((nv, 3))
    for i in range(nv):
        p = verts.get(i).getPosition()
        out[i, 0] = p.x
        out[i, 1] = p.y
        out[i, 2] = p.z
    return out


def _configure_fem_stability(root, tongue):
    """FEM 적분 안정화 — GUI Play와 동일 조건으로 맞춤.

    핵심: adaptive stepping을 켜둔다. GUI는 이 안전장치로 element가 뒤집힐 스텝을
    자동으로 잘게 쪼개 재시도하므로 inverted-elements 예외가 나지 않는다. 이를 끄면
    (headless의 기존 동작) 뒤집힐 뻔한 한 스텝에서 바로 NumericalException이 터진다.
    incompressibility도 모델 기본값(AUTO)로 두어 부피 보존을 유지한다."""
    try:
        root.setMaxStepSize(MAXSTEP)
        root.setAdaptiveStepping(ADAPTIVE_STEPPING)   # GUI와 동일하게 기본 True
    except Exception:
        pass
    try:
        FemModel = _jclass("artisynth.core.femmodels.FemModel")
        tongue.setIncompressible(getattr(FemModel.IncompMethod, INCOMP))
        tongue.setMaxStepSize(MAXSTEP)
    except Exception as e:
        print("note: setIncompressible failed:", e)


def _load(model_name=None):
    """JVM 시작 + 모델 빌드 → 채워진 TongueModel 반환."""
    import jpype

    _start_jvm()
    JString = _jclass("java.lang.String")
    JArray = jpype.JArray
    Main = _jclass("artisynth.core.driver.Main")
    ArrayList = _jclass("java.util.ArrayList")

    model_name = model_name or TONGUE_MODEL
    m = Main.getMain()
    if m is None:
        try:
            Main.main(JArray(JString)(["-noGui"]))
        except Exception as e:
            print("note: Main.main(-noGui) raised:", e)
        m = Main.getMain()
    if m is None:
        m = Main("forward", False)
        m.start(ArrayList())

    if not m.loadModel(model_name, model_name.split(".")[-1], JArray(JString)([])):
        raise RuntimeError("loadModel failed: " + str(m.getErrorMessage()))
    root = m.getRootModel()
    tongue = _find_tongue(root)
    if tongue is None:
        raise RuntimeError(
            "no FemMuscleModel with exciters found in " + model_name)

    # 중력 제거 (활성값에 의한 변형만 보기 위함)
    try:
        root.models().get(0).setGravity(0, 0, 0)
        tongue.setGravity(0, 0, 0)
    except Exception:
        pass
    _configure_fem_stability(root, tongue)
    _deactivate_probes(root)

    exlist = tongue.getMuscleExciters()
    exciters = [exlist.get(i) for i in range(exlist.size())]
    names = [str(e.getName()) for e in exciters]
    mesh = tongue.getSurfaceMesh()

    model = TongueModel()
    model.main = m
    model.tongue = tongue
    model.exciters = exciters
    model.names = names
    model.mesh = mesh
    model.faces = _extract_faces(mesh)
    model.verts = _read_surface(model)
    print("ArtiSynth ready: %d exciters, %d surface verts, %d FEM nodes. order: %s"
          % (len(exciters), mesh.numVertices(), tongue.numNodes(), ",".join(names)))
    print("  fem settings: maxStep=%.4f rampStep=%.3fs incompressible=%s adaptive=%s"
          % (MAXSTEP, RAMP_STEP_T, INCOMP, ADAPTIVE_STEPPING))
    return model


# --------------------------------------------------------------------------- #
# 활성값 적용
# --------------------------------------------------------------------------- #
def _coerce_activation(muscle_values, names):
    """muscle_values(list/np/dict) → exciter 순서(names)에 맞춘 (M,) float 벡터."""
    if muscle_values is None:
        return np.zeros(len(names), dtype=float)
    if isinstance(muscle_values, dict):
        return np.array([float(muscle_values.get(n, 0.0)) for n in names],
                        dtype=float)
    a = np.asarray(muscle_values, dtype=float).ravel()
    # 입력 벡터는 MUSCLE_NAMES(SKILL 11D) 순서로 간주한다. 모델이 보고한 exciter
    # 순서(names)가 다르면 이름 기준으로 재배열해 위치를 맞춘다.
    if a.shape[0] == len(MUSCLE_NAMES) and set(names) == set(MUSCLE_NAMES):
        idx = [MUSCLE_NAMES.index(n) for n in names]
        return a[idx]
    if a.shape[0] == len(names):
        # 표준 11D가 아니거나 이름이 다른 모델 → 이미 모델 순서라고 가정.
        return a
    raise ValueError(
        "muscle_values 길이 %d != exciter 수 %d (순서: %s)"
        % (a.shape[0], len(names), ",".join(names)))


def _nramp_retry_schedule(start=None, max_nramp=None):
    """fem() 실패 시 재시도할 NRAMP 목록: start, 2×, …, max_nramp."""
    if start is None:
        start = NRAMP_RETRY_START
    if max_nramp is None:
        max_nramp = NRAMP_RETRY_MAX
    n = int(start)
    max_nramp = int(max_nramp)
    while True:
        yield n
        if n >= max_nramp:
            break
        n = min(n * 2, max_nramp)


def _apply_activation(model, a, settle=None, nramp=None):
    """활성값을 open-loop로 가해 평형까지 forward 시뮬. 성공 여부(bool) 반환.

    매 호출: rest로 reset → nramp 단계로 서서히 올림 → 최종값으로 hold.
    ramp 각 단계는 RAMP_STEP_T(기본 0.01s) 고정; 총 ramp 시간 = RAMP_STEP_T × nramp.
    실제 ArtiSynth forward 솔버를 사용한다."""
    if settle is None:
        settle = SETTLE_T
    if nramp is None:
        nramp = NRAMP
    nramp = int(nramp)
    m = model.main
    exciters = model.exciters
    root = m.getRootModel()
    m.reset()
    _deactivate_probes(root)
    _configure_fem_stability(root, model.tongue)
    seg = float(RAMP_STEP_T)
    ok = True
    fail_step = None
    fail_ex = None
    for k in range(1, nramp + 1):
        frac = float(k) / nramp
        for i, e in enumerate(exciters):
            e.setExcitation((float(a[i]) if i < len(a) else 0.0) * frac)
        m.playAndWait(seg)
        ex = m.getSimulationException()
        if ex is not None:
            fail_step = k
            fail_ex = ex
            ok = False
            break
    if ok:
        m.playAndWait(float(settle) * 2.0)   # 최종값 hold하여 평형
        ex = m.getSimulationException()
        if ex is not None:
            fail_ex = ex
            ok = False
    if not ok:
        names = model.names or MUSCLE_NAMES
        if fail_step is not None:
            frac = float(fail_step) / nramp
            active = [(n, float(v) * frac) for n, v in zip(names, a) if v > 0]
            print("WARNING: solver exception during ramp step %d/%d (NRAMP=%d, excitation=%s): %s"
                  % (fail_step, nramp, nramp, active, fail_ex))
        else:
            print("WARNING: solver exception during hold (NRAMP=%d): %s"
                  % (nramp, fail_ex))
    return ok


# --------------------------------------------------------------------------- #
# 공개 API: fem
# --------------------------------------------------------------------------- #
def fem(model=None, muscle_values=None, settle=None, model_name=None):
    """ArtiSynth FEM 혀 모델을 로드(필요 시)하고 11D 근육 활성값을 적용한다.

    Parameters
    ----------
    model : TongueModel or None
        이전에 로드한 핸들. None이면 JVM 시작 + 모델 빌드(첫 호출). 이후 프레임에선
        반환된 핸들을 다시 넘겨 재사용(로드 1회만).
    muscle_values : array-like(11,) or dict or None
        근육 활성값 0..1. 길이 11 벡터(MUSCLE_NAMES 순서) 또는 {이름: 값} dict.
        None이면 rest(전부 0).
    settle : float, optional
        평형까지 hold 시뮬 시간(초). 기본 SETTLE_T.
    model_name : str, optional
        ArtiSynth 모델 클래스명. 기본 TONGUE_MODEL.

    Raises
    ------
    RuntimeError
        NRAMP 재시도(50→100→200→400→500) 후에도 forward가 실패하면 발생.

    Returns
    -------
    TongueModel
        변형된 표면 정점(model.verts, (N,3) metres), faces, 적용 활성값이 채워진 핸들.
    """
    if model is None or not getattr(model, "loaded", False):
        model = _load(model_name)

    a = _coerce_activation(muscle_values, model.names)
    ok = False
    used_nramp = None
    for nramp in _nramp_retry_schedule():
        if nramp > NRAMP_RETRY_START:
            print("fem: NRAMP=%d 재시도 …" % nramp)
        ok = _apply_activation(model, a, settle=settle, nramp=nramp)
        if ok:
            used_nramp = nramp
            if nramp > NRAMP_RETRY_START:
                print("fem: NRAMP=%d 에서 성공" % nramp)
            break
        if nramp < NRAMP_RETRY_MAX:
            print("fem: NRAMP=%d 실패 — ramp 단계 수 증가" % nramp)

    if not ok:
        active = [(n, float(v)) for n, v in zip(model.names, a) if v > 0]
        raise RuntimeError(
            "fem forward failed after NRAMP retries (%d..%d): "
            "inverted elements / solver error. active=%s "
            "(lower activation or MAXSTEP may help)"
            % (NRAMP_RETRY_START, NRAMP_RETRY_MAX, active))

    model.activation = a
    model.ok = ok
    model.nramp = used_nramp
    model.verts = _read_surface(model)
    return model


def shutdown():
    """JVM 종료(프로세스 당 1회만 시작/종료 가능)."""
    import jpype
    if jpype.isJVMStarted():
        jpype.shutdownJVM()
