# -*- coding: utf-8 -*-
"""artisynth 패키지: ArtiSynth FEM 혀 모델 알고리즘 (fem / 모델 로드).

시각화·파일 IO는 여기 없다 → modules.utils 사용.
"""
from .utils import fem, shutdown, MUSCLE_NAMES, TongueModel, model_from_obj

__all__ = ["fem", "shutdown", "MUSCLE_NAMES", "TongueModel", "model_from_obj"]
