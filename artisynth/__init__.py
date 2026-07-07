# -*- coding: utf-8 -*-
"""artisynth 패키지: ArtiSynth FEM 혀 모델 알고리즘 (fem / 모델 로드).

구조:
    artisynth/artisynth.py  공개 API (fem, load_model, model_from_obj, shutdown)
    artisynth/utils.py      헬퍼 계층 (설정 + configure + TongueModel + JVM/솔버 루틴)

시각화·파일 IO는 여기 없다 → modules.utils 사용.
"""
from .artisynth import fem, shutdown, model_from_obj, load_model
from .utils import MUSCLE_NAMES, TongueModel, configure

__all__ = [
    "fem", "shutdown", "model_from_obj", "load_model",
    "MUSCLE_NAMES", "TongueModel", "configure",
]
