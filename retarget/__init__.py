# -*- coding: utf-8 -*-
"""retarget 패키지: MRI mask → contour / registration / lift / retarget 알고리즘.

구조:
    retarget/retarget.py  공개 API (mask2contour, register, attach_registration,
                          lift, lift_frame, lift_masks, width_profile, retarget)
    retarget/utils.py     헬퍼 계층 (설정 + configure + contour primitive +
                          affine/landmark 내부 루틴)

파일 IO(mask/CSV 로딩·저장)와 시각화는 여기 없다 → modules.utils 사용.
"""
from .retarget import (
    mask2contour, dorsal_contours_video, register, attach_registration,
    lift, lift_frame, lift_masks, width_profile, retarget,
    hybrid_landmarks, hybrid_landmarks_video,
)
from .utils import configure

__all__ = [
    "mask2contour", "dorsal_contours_video",
    "hybrid_landmarks", "hybrid_landmarks_video",
    "register", "attach_registration",
    "lift", "lift_frame", "lift_masks", "width_profile",
    "retarget",
    "configure",
]
