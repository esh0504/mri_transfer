# versionmanagement.md — Dorsal Contour 추출 알고리즘 버전 관리

혀 dorsal contour(Stage 2) 추출을 개선하며 시도한 알고리즘들을 **V1/V2/V3** 로 보존하고,
각 버전의 알고리즘·근거·장단점·선택 방법을 정리한다. 모든 버전은 `retarget/utils.py` 에
공존하며, `configs/retarget/default.yaml` 의 `contour.mode` 또는 `configure(mode=...)` 로 전환한다.

관련 문서: 문제 진단·논문 근거는 `ContourExtract.md`, 파이프라인 전체는 `Retargeting_skill.md`.

---

## 요약 표

| 버전 | 한 줄 요약 | tip 오검출 | frame jitter(px)\* | 복잡도 | 상태 |
|---|---|---|---|---|---|
| **V1** | legacy: airway-facing 최장 run + min-col tip | 있음(앞-바닥 감김) | 2.6 (틀린 위치에 안정) | 낮음 | 보존(비교/fallback) |
| **V2** | upper-envelope + anterior-clip (+temporal) | 해소 | 4.2 → **2.1**(temporal) | 낮음 | **기본(default)** |
| **V3** | normal 필터 + dorsum-run + upper-anterior tip | 해소 | 3.5 | 높음 | 선택형(ContourExtract.md) |
| **V4** | V3 tip + V2 root/body 결합 | 해소 | tip 3.5·root 5.6 | 중간 | 선택형(권장 조합) |
| **V5** | V1 root(경계 추종) + tip wrap 제거 | 해소 | root=V1(더 posterior) | 낮음 | 선택형(root 우선) |
| **V6** | V5 + normal로 tip을 진짜 혀끝까지 연장 | 해소 | root=V5, tip 더 anterior | 낮음 | 선택형(tip 정밀) |
| **V7** | V6와 유사, tip을 normal 각도 급변(corner)+좌측각 cap | 해소 | ≈V6 | 낮음 | 실험형(corner 기준) |
| **V8** | V7 tip + root도 normal(완전 우측 90° 직전) | 해소 | tip·root 모두 normal | 낮음 | 실험형(양끝 normal) |

\* Subject3 71프레임 tip의 프레임 간 평균 이동량(작을수록 안정, 단 V1은 *틀린* 위치에 안정적임에 주의).
V2의 2.1은 `track_dorsal_tips`(시간 평활) 적용값. V4는 tip jitter=V3(3.5), root jitter=V2(5.6).

---

## V1 — legacy (원래 방식)

함수: `precise_contour_v1()`

```
1. tongue(label 4) 최장 closed boundary 추출 (find_contours)
2. airway distance ≤ facing_thresh 인 boundary point 선택
3. cyclic boundary에서 가장 긴 True run 선택
4. col이 작은 쪽을 tip으로 정렬 (min-col = tip)
5. clip_posterior_spur → smoothing → arc-length resample
```

**실패 원인.** 혀끝이 올라간 자세(예: /t/·/s/·/l/)에서는 앞끝에 얇은 돌기+노치가 생기고
airway가 그 주위를 감싼다. "airway에 가깝다"가 돌기 위·아래·앞 모두 참이라 contour가 돌기를
**한 바퀴 감아 앞-바닥으로 내려갔다** 올라오고, `min-col`이 등쪽 apex가 아니라 **바닥 고정부**를
tip으로 잡는다. jitter는 낮게 보이지만 *틀린 위치*에 안정적으로 붙는 착시다.

**보존 이유.** 비교 baseline, 그리고 V3의 fallback 후보.

---

## V2 — upper-envelope + anterior-clip (현재 기본)

함수: `precise_contour_v2()` (+ 시간축 `track_dorsal_tips()`, `dorsal_contours_video()`)

```
1. facing_arc(): airway-facing 최장 run (V1의 2~3단계)
2. dorsal_envelope(): 열(column)마다 '가장 위쪽(min row)' 점만 취함
     → x에 대해 단일값(single-valued) → 돌기를 감는 루프 원천 제거
3. clip_anterior_drop(): dorsum 정점에서 tip쪽으로 급강하 지점을 잘라
     tip을 앞-바닥 노치가 아니라 '등쪽 apex'에 고정
4. (clip_root) clip_posterior_spur → smoothing → arc-length resample
```

**핵심 아이디어.** "열마다 최상단"이 곧 outward-normal이 위를 향하는 dorsal 표면 선택과 같아,
V3의 normal 필터가 하려는 일을 **threshold sweep 없이 구조적으로** 달성한다.

**시간축(video).** `track_dorsal_tips()`: 프레임별 envelope apex tip → median(outlier 제거) +
이동평균 평활 → 각 프레임 표면에 재투영. `dorsal_contours_video(masks)`가 이를 감싸 프레임 간
tip↔root 대응이 일관된 image-mm contour를 반환한다. (jitter 2.6→2.1px, 돌기 프레임 해소.)

**장점.** 단순, knob 적음(`ant_drop`, `clip_drop_frac`), 검증됨.
**한계.** 등쪽이 x에 대해 단일값이라 가정 → retroflex/말린 혀끝 등 **접힌(folded) 형상**에서 취약.
이 경우 V3(경계 추종)이 더 일반적.

---

## V3 — normal + dorsum-run + upper-anterior tip (ContourExtract.md 안)

함수: `precise_contour_v3()` (+ 헬퍼: `clean_airway`, `outward_normals_from_contour`,
`close_small_false_gaps_cyclic`, `true_runs_cyclic_indices`, `score_run`,
`choose_upper_anterior_tip`, `keep_dorsal_side_after_tip`)

```
A. airway cleanup: remove_small_objects + closing (keep_largest 기본 off)
B. 후보 = (airway distance ≤ thresh) AND (outward normal_row < normal_row_max=0.35)
     → dorsal은 normal이 위(row<0), floor/ventral(아래 향함)은 제거
C. 짧은 False gap 메움 → cyclic True run들로 분할
D. run 점수 = w_dorsum·dorsum포함 + w_length·길이 + w_ventral·비-ventral
             + w_temporal·prev_tip근접  → 최고 점수 run 선택
E. upper-anterior tip: anterior band 안에서 가장 위쪽 점을 tip으로,
     keep_dorsal_side_after_tip로 tip 주변 floor curl 제거
F. tip→root 정렬 → clip_posterior_spur → smoothing → resample
G. 후보 run이 없으면 V2로 fallback. confidence/ventral_frac/TT 등 debug 반환.
```

**근거 논문(ContourExtract.md).** Su 2018(geometric snake), Peng 2010·Labrunie 2018(shape/ASM
prior, correspondence), Eslami 2020(TT heatmap), Wrench 2022·Sun 2025(DeepLabCut contour),
Somandepalli 2017(semantic edge).

**장점.** 실제 경계를 추종하므로 folded 형상까지 일반적, confidence+fallback 내장.
**한계.** knob이 많음(`normal_row_max`, `tip_band_px`, `gap_close_pts`, run score 가중치 등) →
sweep 필요. `dorsum_idx = argmin(row)`는 raised-tip에서 흔들릴 수 있어(V2 실험에서 확인) 보정 여지.
너무 strict하면 true tip이 잘려 fallback율↑.

**주의(문서 자체 경고).** `normal_row_max`가 너무 작으면 tip 잘림 → `facing_thresh`와 함께 sweep.
`airway_keep_largest=True`는 oral airway가 여러 조각으로 갈라지는 프레임에서 문제.

---

## V4 — V3 tip + V2 root/body (권장 조합)

함수: `precise_contour_v4()`

관찰: **V2는 root/body가 깔끔**(envelope + posterior-clip)하고 **V3는 tip이 정밀**(normal +
upper-anterior)하다. V4는 둘을 앵커로 결합한다.

```
1. facing_arc → dorsal_envelope (= V2의 dorsal 표면)
2. root anchor = V2 방식: clip_anterior_drop + clip_posterior_spur 의 끝점
3. tip anchor  = V3 방식: precise_contour_v3()[0] (normal 필터 + upper-anterior)
4. envelope 표면을 tip anchor ~ root anchor 구간으로 arc-length 리샘플
   (양 끝을 각각 V3 tip / V2 root 로 고정)
5. V3 또는 표면 추출 실패 시 V2로 fallback
```

**결과.** tip은 V3와 동일(jitter 3.5), root는 V2와 동일(jitter 5.6). 즉 각 부위에서 더 정확한
버전을 그대로 가져온다. body는 V2 envelope 표면을 공유하므로 tip↔root 사이가 매끄럽다.

**한계.** V3를 내부에서 호출하므로 V3의 knob(normal_row_max 등)을 그대로 물려받고 비용도
V2보다 큼. tip/root anchor가 envelope 상에서 너무 가까우면(짧은 혀) 구간이 좁아질 수 있음.

---

## V5 — V1 root + tip fix (root 우선 조합)

함수: `precise_contour_v5()`

관찰: **V1은 root를 매우 잘 잡는다**(경계를 그대로 추종해 후방 dorsum 끝까지, col ~108–118).
문제는 tip에서 혀끝을 U자로 감는 것뿐. V5는 **V1의 root를 그대로 두고 tip만 고친다.**

```
1. V1 그대로: 최장 airway-facing run → min-col 정렬 → clip_posterior_spur
   (이 단계까지 root가 V1과 100% 동일하게 확정됨)
2. tip fix: choose_upper_anterior_tip 로 앞쪽 band의 apex를 찾고,
   그 앞의 wrap 구간(arc[:tip_idx])만 잘라냄 → root(끝점)는 건드리지 않음
3. arc-length 리샘플
```

**결과.** root는 V1과 **완전히 동일**(실측 f1/f30/f45/f51/f60 = 108.5/118.5/109.5/107.5/113.0),
tip은 apex로 교정(U자 감김 제거). V2/V4보다 root가 더 posterior까지 감.

**V4 vs V5 (둘 다 "좋은 tip + 좋은 root" 조합):**
- **V4** = V3 tip + **V2 root**(envelope, posterior-clip이 짧게 자를 때 있음).
- **V5** = 교정 tip + **V1 root**(경계 추종, 더 멀리 감).
→ root를 더 뒤까지 원하면 V5, envelope 기반 부드러운 표면을 원하면 V4. (정량 비교는 golden-set 권장.)

---

## V6 — V5 + normal 기반 tip 연장 (tip 정밀)

함수: `precise_contour_v6()`

관찰: V5는 tip이 apex에서 멈춰 **혀끝 앞면(front face)을 놓친다**(실제 혀끝까지 안 감).
outward normal로 보면 dorsal 표면은 위(↑), 혀끝 앞면은 좌측(←, anterior), 바닥 wrap은 아래(↓)를
가리킨다. V6는 이를 이용해 tip을 진짜 혀끝까지 연장한다.

```
1. V5와 동일: 최장 airway-facing run → min-col 정렬 → clip_posterior_spur (root 확정)
2. 각 점 outward normal 계산 (outward_normals: 유한차분 tangent 90° 회전 + 혀 밖 방향)
3. dorsum peak에서 앞쪽으로 걸으며, normal의 row성분이 down_thresh(0.35) 초과 = **아래(underside
   wrap)로 꺾이는 지점 직전**까지 tip을 연장 → 좌측을 향하는 혀끝 앞면 포함, 바닥 wrap 제외
4. arc-length 리샘플
```

**결과.** root는 V5와 **동일**, tip은 V5보다 더 anterior(실측 col 55~57 vs V5 59~64)로 실제 혀끝에
도달. **root=V1의 깊이 + tip=진짜 혀끝** → 현재 가장 완전한 조합.

**튜닝.** `down_thresh` ↑(예: 0.5)면 tip을 더 앞까지(바닥 가까이), ↓면 더 보수적으로 연장.

---

## V7 — normal 각도 급변(corner) 기반 tip (실험형)

함수: `precise_contour_v7()`

아이디어: 혀끝은 곡률이 높은 코너 → normal 각도가 급변하는 지점이 tip. V6가 "normal이 아래로
꺾이는 지점"을 쓰는 대신, V7은 "각도 변화가 급격한 지점"을 tip으로 쓴다.

```
1. V5/V6와 동일: 최장 airway-facing run → 정렬 → clip_posterior_spur (root)
2. contour presmooth(계단 노이즈 완화) → outward normal → 각도(unwrap)
3. dorsum에서 앞으로 걸으며, normal이 dorsum 대비 guard_deg(40°) 이상 꺾인 뒤
   window 각도변화 > dthresh_deg(45°) 인 첫 지점(=corner)까지 tip 연장
```

**중요(정직한 결과).** **순수 급변 기준만 쓰면 실패한다** — 마스크 계단 노이즈가 dorsum 위에서
가짜 corner를 만들어 tip이 오히려 뒤로 튄다. 그래서 `guard_deg`(앞쪽으로 충분히 꺾인 뒤에만
corner 인정) + `presmooth`가 필수이고, 이를 넣으면 **결과가 V6에 매우 가깝다**(실측 tip col ≈ V6).
즉 V7은 "corner 기준"이라는 대안을 구현한 실험형이며, 실사용 견고성은 V6가 더 단순·안정적이다.

**좌측각 상한(`max_left_deg`, 기본 40°, 권장 30~45°).** tip의 normal이 수직(up)에서 좌상단으로
이 각도를 넘지 않도록 cap을 건다. corner가 더 좌측(예: 완전히 좌측 90°)까지 가면 그 직전(각도 도달
지점)까지 tip을 당겨, tip normal이 항상 "위~좌상단 30~45° 이내"가 되게 한다. (실측: cap 없으면
일부 프레임 tip 좌측각 90°까지 → cap 40°면 전부 ≤40°.)

**튜닝.** `dthresh_deg`↓/`guard_deg`↓ → 더 앞까지, ↑ → 보수적. `presmooth`↑ → 노이즈에 강함.
`max_left_deg`↓ → tip을 덜 좌측(더 위쪽)에서 멈춤.

---

## V8 — tip·root 둘 다 normal 기반 (양끝 normal)

함수: `precise_contour_v8()`

V7이 tip을 normal로 잡았듯, V8은 **root도 normal로** 잡는다. `clip_posterior_spur` 대신,
dorsum에서 뒤쪽으로 걸으며 normal이 **완전 우측(90°)을 보기 직전**(우측각 > `right_thresh` 되는
지점의 직전)을 root로 한다.

```
tip  = V7과 동일 (corner + 좌측각 cap ≤ max_left_deg)
root = dorsum → 뒤쪽으로 걸으며 normal 우측각(up=0, right=90)이 right_thresh(기본 80°)를
       넘기 직전 지점. (넘으면 그 직전을 root)
```

**결과.** tip은 앞쪽 혀끝(좌상단 ≤40°), root는 후방 dorsum 끝(우상단, 완전 우측 직전).
실측 root col ≈ 110~118 (V5 clip root 108~111보다 조금 더 posterior).

**튜닝.** `right_thresh`↑(85~88) → root를 더 뒤(인두쪽)까지, ↓ → 보수적. 너무 높이면(≈90)
인두벽까지 넘어갈 수 있음(f45에서 확인). 기본 80이 안정적.

---

## 관련 구성요소 (버전과 직교)

- **temporal tip tracking** (`track_dorsal_tips`, `dorsal_contours_video`): V2 위에서 동작하는
  시간 평활. 어떤 버전이든 per-frame contour를 시퀀스로 안정화하는 후처리로 확장 가능.
- **hybrid landmarks** (`hybrid_landmarks`, `hybrid_landmarks_video`): tip/dorsum/root **anchor +
  구간별 리샘플**로 20점 landmark를 뽑아 index-identity 대응을 고정. anchor를 **DeepLabCut 예측값**
  으로 교체하면(`hybrid_landmarks(mask, anchors=dlc_pred)`) 그대로 상위 버전이 됨. 현재는 기하
  anchor라 단순 arc-length 대비 이득이 작음(dorsum anchor가 약점 → DLC가 채울 영역).

---

## 버전 선택 방법

```yaml
# configs/retarget/default.yaml
contour:
  mode: v2      # v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8
```

```python
import retarget
retarget.configure(mode="v3")          # 전역 전환 (mask2contour/retarget 반영)

from retarget.utils import precise_contour
precise_contour(mask, n=25, mode="v3") # 호출 단위 지정
precise_contour(mask, n=25)            # mode 생략 시 CONTOUR_MODE(기본 v2)
```

환경변수 `CONTOUR_MODE=v3` 로도 기본값 변경 가능.

---

## 실험 관찰 요약 (Subject3, frames 1–71)

- **tip 정확도(실패 프레임 f45/f51 등):** V1 실패(앞-바닥) → V2·V3 모두 등쪽 apex로 해소.
- **frame jitter:** V1 2.6(틀린 위치), V2 4.2(per-frame)→2.1(temporal), V3 3.5(per-frame).
- **V3 debug 예(f51):** ventral_frac 0.06, confidence 0.79, fallback 없음 → 정상 선택.
- **결론:** 실패 케이스는 V2·V3 둘 다 해결. V2가 단순·검증됨(기본), V3는 folded 형상·
  confidence/fallback이 필요할 때. 정량 비교는 ContourExtract.md §12의 golden-set 평가
  하베스트(수동 TT/TD/TR + TT error/MSD/false-tip/direction-fail 지표)로 확정하는 것을 권장.

---

## 다음 단계 (권장)

1. **golden-set 평가 하베스트**(ContourExtract.md §12·§16) 구축 → V1/V2/V3/hybrid 정량 ablation.
2. 평가에서 V2가 folded 프레임에서 지면 V3를 기본으로, 아니면 V2 유지 + V3 fallback.
3. 남는 false-tip이 mask artifact성이면 **TT/TD/TR heatmap 또는 DeepLabCut anchor** 학습 →
   `hybrid_landmarks(anchors=dlc_pred)` 로 연결(Phase 3/4).
