# -*- coding: utf-8 -*-
"""
V2/artisynth/artisynth.py

ArtiSynth FEM 혀 모델의 *공개 API* — 모델 로드 + 11D 근육 활성값 forward 구동.
내부 헬퍼(설정/JVM/솔버 루틴)는 artisynth/utils.py 에 있다.

핵심 API:

    model = fem(model, muscle_values)   # 로드 + 활성값 적용 → TongueModel
    model = model_from_obj(path)        # OBJ rest 메쉬 → TongueModel (JVM 불필요)

사용 예:

    from artisynth import fem, MUSCLE_NAMES
    from modules.utils import vis, save_obj, extract_obj

    model = fem(None, [0.3] + [0.0] * 10)      # 11D 활성값 (MUSCLE_NAMES 순서)
    model = fem(model, {"GGP": 0.5, "HG": 0.2})  # 핸들 재사용(로드 1회만)
"""
import numpy as np

from modules.utils import load_obj

from . import utils
from .utils import (
    MUSCLE_NAMES, TongueModel,
    load, coerce_activation, nramp_retry_schedule, apply_activation, read_surface,
)


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


def load_model(file_path):
    """OBJ rest mesh → TongueModel."""
    return model_from_obj(file_path)


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
        model = load(model_name)

    a = coerce_activation(muscle_values, model.names)
    ok = False
    used_nramp = None
    for nramp in nramp_retry_schedule():
        if nramp > utils.NRAMP_RETRY_START:
            print("fem: NRAMP=%d 재시도 …" % nramp)
        ok = apply_activation(model, a, settle=settle, nramp=nramp)
        if ok:
            used_nramp = nramp
            if nramp > utils.NRAMP_RETRY_START:
                print("fem: NRAMP=%d 에서 성공" % nramp)
            break
        if nramp < utils.NRAMP_RETRY_MAX:
            print("fem: NRAMP=%d 실패 — ramp 단계 수 증가" % nramp)

    if not ok:
        active = [(n, float(v)) for n, v in zip(model.names, a) if v > 0]
        raise RuntimeError(
            "fem forward failed after NRAMP retries (%d..%d): "
            "inverted elements / solver error. active=%s "
            "(lower activation or MAXSTEP may help)"
            % (utils.NRAMP_RETRY_START, utils.NRAMP_RETRY_MAX, active))

    model.activation = a
    model.ok = ok
    model.nramp = used_nramp
    model.verts = read_surface(model)
    return model


def shutdown():
    """JVM 종료(프로세스 당 1회만 시작/종료 가능)."""
    import jpype
    if jpype.isJVMStarted():
        jpype.shutdownJVM()
