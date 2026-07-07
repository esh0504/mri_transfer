# Retargeting_skill.md: 2D MRI 마스크 → 3D 혀 메쉬 Retargeting 파이프라인

## 📌 개요

Retargeting은 **3D rest 혀 메쉬**(ArtiSynth `tongue_rest_m.obj`)를, 2D 정중시상(midsagittal) MRI
분할 마스크가 나타내는 자세로 **변형(deform)** 시키는 과정입니다. 근육 활성값(FEM)을 쓰지 않고,
2D 마스크에서 뽑은 혀 등쪽(dorsal) 윤곽의 **rest→target 변위**를 3D 메쉬에 RBF로 전파해 기하학적으로
변형합니다. 즉 "MRI가 보여주는 표면 움직임"을 "3D 모델 표면 움직임"으로 옮기는 것이 목표입니다.

전체 흐름은 다음과 같습니다.

```
rest mask ─┐
           ├─▶ [1] mask→2D label ─▶ [2] dorsal contour ─┐
target mask┘                                            │
                                                        ▼
rest mask + 3D mesh ─▶ [3] registration (image↔model affine)
                                                        │
                                                        ▼
      [4] retarget core: contour→model 좌표 → Δ(rest→target) → RBF → mesh 변형
                                                        │
                                                        ▼
                          [5] 출력: points_cloud / Mesh / Color
```

관련 코드: `retarget/retarget.py`(공개 API), `retarget/utils.py`(헬퍼/알고리즘),
`modules/pipeline.py`(오케스트레이션), `modules/utils.py`(IO/시각화).

---

## 🧭 좌표계 규약 (모든 단계의 기준)

Retargeting에서 가장 헷갈리고 버그가 잘 생기는 부분이 좌표계입니다. 세 가지 공간을 오갑니다.

| 공간 | 단위 | 축 정의 | 비고 |
|---|---|---|---|
| **image (px)** | 픽셀 | `row`=아래로 증가, `col`=오른쪽으로 증가 | 마스크 원본 (H=W=256) |
| **image-mm** | mm | `x = col·mm_per_px`, `y = (H-1-row)·mm_per_px` | `y`축을 뒤집어 **위쪽이 +** |
| **model** | m (내부 mm) | `x`=전-후(ant-post), `y`=좌우(lateral≈0), `z`=위(up) | 렌더 규약: +Z=위, −X=앞 |

핵심 매핑: registration이 만드는 affine은 **image-mm `(x,y)` → model `(x,z)`** 를 잇습니다.
즉 이미지의 수평(col)은 모델의 전-후(x)로, 이미지의 수직(위쪽 y)은 모델의 수직(z)으로 대응됩니다.
모델 메쉬는 metre 단위지만, 내부 계산은 대부분 **mm(×1000)** 로 수행한 뒤 마지막에 다시 metre로 되돌립니다.

---

## 🔧 Stage 0 — 입력과 설정

입력은 (1) 3D rest 메쉬(`ref_3d`: `verts (N,3)` metres, `faces (F,3)`), (2) rest 2D 마스크,
(3) target 2D 마스크입니다. 마스크는 `.mat`(`mask_frame` 변수, 256×256, 라벨 0–6)이며
**tongue=4, airway=5** 를 사용합니다. 설정은 `configs/retarget/default.yaml`에서 오고
`retarget.utils.configure()`가 모듈 전역값을 덮어씁니다.

| 파라미터 | 기본값 | 의미 | 쓰이는 단계 |
|---|---|---|---|
| `mm_per_px` | 1.164 | 픽셀→mm 배율 | 2·3·4 |
| `contour.n_markers` | 25 | dorsal contour 샘플 점 수 | 2 |
| `contour.clip_root` | true | 후방 spur(인두쪽 꼬리) 제거 | 2 |
| `contour.clip_drop_frac` | 1.0 | spur 절단 임계 | 2 |
| `retarget.nctrl` | 13 | RBF 제어점(=곡선 리샘플) 수 | 4 |
| `retarget.rbf_len` | 18.0 | RBF 가우시안 반경(mm), `epsilon=1/rbf_len` | 4 |
| `retarget.spatial_win` | 3 | Δ 공간 스무딩 창 | 4 |

---

## 🩻 Stage 1 — 마스크 → 2D 라벨

함수: `modules.utils.mask_label_2d(mask)`

마스크가 `(H,W,C)`면 첫 채널만, `(H,W)`면 그대로 2D 라벨 맵으로 만듭니다. 이후 모든 contour
연산은 이 2D 라벨(값 4=tongue, 5=airway)을 입력으로 받습니다.

---

## 🪝 Stage 2 — Dorsal contour 추출 (upper-envelope 방식)

함수: `retarget.utils.precise_contour` → `retarget.mask2contour`

혀 표면 중 **airway(기도)를 향한 등쪽 곡선**만 tip(앞끝 apex)→root(뿌리) 순서로 뽑습니다.
초기 구현은 "airway-facing 경계의 가장 긴 연속 구간"을 그대로 따라갔는데, 혀 앞끝이 올라간
자세(예: `/t/`, `/s/`, `/l/`)에서는 앞끝에 **얇은 돌기 + 노치**가 생기고 airway가 그 주위를
감싸므로, 곡선이 돌기를 **한 바퀴 감아 앞-바닥으로 내려갔다가** 올라오는 루프가 생겼습니다.
그러면 tip(최소 col)이 등쪽 apex가 아니라 **바닥 고정부**에 찍혔습니다. 이를 해결하기 위해
현재는 **upper-envelope(상단 포락선)** 방식을 씁니다.

1. **airway-facing arc** `facing_arc()`: `find_contours`로 서브픽셀 경계를 얻고,
   `distance_transform_edt(~airway) ≤ facing_thresh(=2.5)` 인 점들 중 `longest_true_run_cyclic()`
   로 가장 긴 연속 구간을 골라 tip(최소 col)→root로 정렬합니다.
2. **상단 포락선** `dorsal_envelope()`: **열(column)마다 가장 위쪽(min row) 점만** 취합니다.
   x에 대해 단일값(single-valued)이 되어 돌기를 감는 루프가 원천적으로 사라집니다.
3. **앞쪽 급강하 컷** `clip_anterior_drop(drop_frac=0.5)`: dorsum 정점에서 tip 방향으로 걸으며
   높이가 급락하는 지점을 잘라, tip을 **앞-바닥 노치가 아니라 등쪽 apex**에 맞춥니다.
4. **후방 spur 제거**(옵션 `clip_root`): `clip_posterior_spur()`로 인두쪽 꼬리 제거.
5. **평활·리샘플** `resample_rowcol()`: 3-tap 이동평균 후 호길이 등간격 `n`점 리샘플.

`mask2contour`는 이 `(row,col)` 결과를 **image-mm**로 변환합니다:
`x = col·mm_per_px`, `y = (H−1−row)·mm_per_px`, `z = 0`, 결과 `(N,3)`.

### 시간축 tip 트래킹 (video) — `dorsal_contours_video()` / `track_dorsal_tips()`

프레임마다 독립적으로 tip을 뽑으면 애매한 프레임에서 여전히 튈 수 있습니다. 10fps 비디오라
tip은 프레임 간 조금씩만 움직이므로, **시간 일관성(smallest-change)** 으로 이를 안정화합니다.
`track_dorsal_tips(masks)` 는 (1) 프레임별 envelope apex tip을 구하고, (2) 프레임축으로
**median 필터(outlier 제거) + 이동평균 평활**을 적용한 뒤, (3) 평활된 tip을 각 프레임의 표면
(`facing_arc`)에 **다시 투영**합니다. 그리고 `dorsal_contour_from_tip()` 으로 그 tip에서 root까지
앵커링해 추출하므로 **tip↔root 대응이 프레임 간 일관**됩니다. 폴더(비디오) 리타게팅 시
`mask2contour` 대신 `retarget.dorsal_contours_video(masks)` 를 쓰면 됩니다.
(실측 Subject3 71프레임: tip jitter 2.6px → 2.1px, 돌기 프레임 tip 오검출 해소.)

> ⚠️ 디버깅 포인트: tip이 앞끝 apex에 있는지, 등쪽을 따라가는지, 시퀀스에서 tip 궤적이 매끄러운지
> 확인하세요(노트북 Stage 2 / 2b). 방향이 뒤집히거나 tip이 바닥으로 내려가면 rest/target 대응이 어긋납니다.

---

## 📐 Stage 3 — Registration (image ↔ model affine)

함수: `retarget.register` (보조: `anatomical_landmarks`, `model_landmarks_m`, `fit_affine`)

image-mm 좌표를 model 좌표로 옮기는 **affine 변환**을 추정해 `registration.csv`로 저장합니다.
anchor(대응점)를 얻는 방법은 두 가지입니다.

- **자동(landmark_map 미지정)**:
  - 이미지 랜드마크 `image_landmarks_mm()`: `anatomical_landmarks()`가 `full_boundary_contour`(닫힌
    전체 윤곽, n=400)에서 **tip=최소 col, dorsum=최소 row, root=최대 col, floor=최대 row**를 뽑고,
    tip/dorsum/root를 image-mm로 변환.
  - 모델 랜드마크 `model_landmarks_m()`: 정중시상(`|y|≤0.003 m`) 정점에서 **tip=최소 x, root=최대 x,
    dorsum=최대 z** 를 `(x,z)` metre로 추출.
- **수동(landmark_map.csv 지정)**: `label,imageX,imageY,modelX_m,modelZ_m` 형식의 ≥3개 대응점을 그대로 사용.

**Affine fit** `fit_affine(img_xy, mod_xz)`: `M = [img_xy | 1]` 로 두고
`A = lstsq(M, mod_xz)` (최소자승, `A`는 3×2). 예측 오차의 RMS/최대값을 함께 반환합니다.
`register`는 anchor를 `imageX,imageY`(image-mm)와 `modelX,modelZ`(model **mm**, 즉 `×1000`)로
CSV에 기록합니다. 반환 dict: `path, names, rms_mm, worst_mm, anchors, affine_m`.

`attach_registration(ref_3d, csv)`로 메쉬 핸들에 `registration_csv` 경로를 붙이면 Stage 4가 이를 씁니다.

> ⚠️ 디버깅 포인트: 자동 anchor가 3점이면 affine이 3점을 정확히 맞춰 RMS≈0이 나옵니다(과적합에 주의 —
> 낮은 RMS가 곧 좋은 정합을 뜻하진 않음). 두 랜드마크가 해부학적으로 같은 곳을 가리키는지 눈으로 확인하세요.

---

## 🎯 Stage 4 — Retarget core (변위 전파)

함수: `retarget.retarget(ref_3d, source, target, nctrl, rbf_len, spatial_win, mm_per_px)`

이 단계가 실제 변형을 만듭니다. 내부 순서:

1. **메쉬 준비** `ref_mesh()`: `V_mm = V_rest_m·1000`, `F`, `V_rest_m` 반환.
2. **contour 재추출**: `source_c = mask2contour(mask_label_2d(source))`, target도 동일(image-mm).
3. **모델 제어점 곡선** `model_dorsal_curve(V_mm, nctrl)`: 모델 x를 `nctrl`개로 균등분할(`linspace`)하고
   각 x-창에서 **최대 z**(등쪽 상단)를 취해 `(nctrl,2)` `[x, z]`(mm) 곡선을 만듭니다. 이것이 **RBF 제어점 위치**.
4. **image→model 매핑** `affine_image_to_model(reg_csv)`: CSV의 anchor로 다시 lstsq affine을 만들어
   image-mm `(x,y)` → model-mm `(x,z)` 함수를 반환.
5. **곡선 리샘플**: `source_xz = resample_curve(to_model(source_c), nctrl)`, target도 동일
   → 둘 다 model-mm `(nctrl,2)`.
6. **Δ 계산**: `delta = target_xz − source_xz` (제어점별 변위). `spatial_win>1`이면
   `uniform_filter1d`로 인접 제어점 간 공간 스무딩.
7. **RBF 보간**: `RBFInterpolator(dorsal, delta, kernel="gaussian", epsilon=1/rbf_len,
   degree=-1, smoothing=1e-3)`. 즉 **위치 `dorsal[k]` 에 변위 `delta[k]` 를 부여**하고, 모든 정점의
   `Vxz = V_mm[:,[0,2]]`에서 변위 `d_xz`를 보간.
8. **정점 변형**: `V_def = V_rest_m.copy()`; `V_def[:,0] += d_xz[:,0]/1000`;
   `V_def[:,2] += d_xz[:,1]/1000` (mm→m). **x,z만 변형, y(좌우)는 불변** → 정중시상 평면 내 변형.
9. **컬러** `displacement_colors(V_rest_m, V_def_m)`: 정점별 변위 크기를 최대값으로 정규화해 viridis
   컬러맵(0–255)으로 매핑.

반환: `{"points_cloud": V_def_m (m), "Mesh": F, "Color": (N,3) uint8}`.

> ⚠️ **핵심 디버깅 포인트 — 제어점 정렬 불일치**: RBF는 변위를 `dorsal[k]`(모델 x 균등분할) 위치에
> 붙이지만, `delta[k]`는 `source_xz[k]`(contour **호길이** 리샘플) 위치에서 잰 값입니다. 두 곡선의
> 매개변수화가 달라 k번째 점의 실제 위치가 어긋날 수 있고(실측 예: 평균 3.7mm, 최대 10mm), 그러면 변위가
> 살짝 엉뚱한 곳에 적용되어 변형이 왜곡됩니다. 노트북 Stage 4·5에서 `dorsal[k]↔source_xz[k]`를 잇는
> 주황선 길이로 이 오차를 직접 확인할 수 있습니다. 수정 방향은 변위를 `dorsal` 대신 `source_xz` 위치에
> 붙이는 것입니다.

---

## 📦 Stage 5 — 출력과 후처리

`retarget()`의 dict를 `modules.pipeline._to_tongue_model()`이 `TongueModel`(`verts`,`faces`,`names`,
`activation`, `registration_csv`)로 감싸고, `modules.utils.visualization()`으로 PNG를 렌더합니다.
`Color`는 정점별 변위 크기를 담고 있어, 어디가 많이 움직였는지 색으로 볼 수 있습니다.

---

## 🔁 오케스트레이션 & 배치(시계열) 모드

함수: `modules.pipeline.retargeting(src_model, ref, target)`

- `ref`/`target`은 마스크 배열 또는 `.mat` 경로를 받습니다.
- `registration_csv`가 아직 없으면 **자동으로 `register()`+`attach_registration()`** 을 먼저 실행합니다.
- `target`이 **폴더**면 `load_video()`로 `mask_*.mat` 전체를 읽어 프레임마다 retarget → `list[TongueModel]`
  반환(시계열). 단일 파일/배열이면 1프레임.

`main.py`는 Hydra 설정으로 `stage=retarget|fem|all`을 구동하며, retarget 스테이지는
`load_model(tongue_obj) → retargeting(model, rest_mask, target_mask) → visualization`을 수행합니다.

---

## 🧩 (보조 알고리즘) Kinematic symmetric 3D lift

함수: `retarget.lift_frame / lift / lift_masks` (보조: `width_profile`)

메인 retarget 경로와 **별개**로, 2D dorsal contour를 좌우대칭 3D 돔(dome) 표면으로 "부풀리는" 대안
알고리즘입니다. `width_profile(s)`가 정규화 호길이 `s∈[0,1]`에 따라 반폭을 주고(`sin(πs)^0.6` 형태),
`lift_frame`이 각 midline 점에서 측방(z) 아치를 만들며 가장자리를 `edge_drop`만큼 낮춥니다. 결과는
`(N, Nz, 3)` mm 표면이며, `lift`(시퀀스), `lift_masks`(마스크 시퀀스 → contour → lift)로 확장됩니다.
파라미터: `nz=15, half_w=30, edge_drop=9, width_end=0.35`. 3D GT 없이 형태를 근사할 때 사용합니다.

---

## 🐞 단계별 실패 지점 요약 (디버깅 체크리스트)

| 단계 | 흔한 증상 | 확인 방법 |
|---|---|---|
| 2 contour | 등쪽이 아니라 아래/옆을 땀, tip↔root 뒤집힘, clip 과다 | 노트북 Stage 2: 마스크 위 점 순서(보라→노랑)·tip 별표 |
| 2 tip 오검출 | 앞끝 돌기·노치에서 tip이 앞-바닥으로 감김 | envelope+`clip_anterior_drop`, video는 `dorsal_contours_video` (노트북 Stage 2b) |
| 3 registration | 랜드마크 오대응, 스케일/축 뒤집힘 | 노트북 Stage 3: 두 그림 랜드마크 위치, RMS |
| 4 매핑 | 매핑된 rest contour가 모델 등쪽에서 떨어짐 | 노트북 Stage 4: 회색 메쉬 위 빨강 곡선 |
| 4 제어점 정렬 | `dorsal[k]`↔`source_xz[k]` 어긋남 → 변위 오적용 | 노트북 Stage 5: 주황선 길이(평균/최대) |
| 4 RBF | 변위장 폭발/진동, 과평활 | 노트북 Stage 6: quiver, `rbf_len`/`smoothing` 조정 |

시각적 단계별 검증은 저장소 루트의 **`retarget_debug.ipynb`** 를 사용하세요(맨 위 셀에서 피험자·프레임만
바꾸면 됩니다).

---

## 🗂️ 함수 맵 (빠른 참조)

- **공개 API** (`retarget/retarget.py`): `mask2contour`, `dorsal_contours_video`, `register`,
  `attach_registration`, `lift`/`lift_frame`/`lift_masks`/`width_profile`, `retarget`
- **헬퍼/알고리즘** (`retarget/utils.py`): `precise_contour`, `facing_arc`, `dorsal_envelope`,
  `clip_anterior_drop`, `clip_posterior_spur`, `smooth_closed`, `resample_rowcol`,
  `dorsal_contour_from_tip`, `track_dorsal_tips`, `median_filter_2d`, `full_boundary_contour`,
  `anatomical_landmarks`, `longest_true_run_cyclic`, `ref_mesh`, `model_dorsal_curve`,
  `affine_image_to_model`, `resample_curve`, `displacement_colors`, `model_landmarks_m`,
  `image_landmarks_mm`, `fit_affine`, `pairs_from_auto/pairs_from_map`, `require_file`,
  `mask_2d`, `configure`
- **오케스트레이션** (`modules/pipeline.py`): `retargeting`, `configure`, `load_model`, `visualization`
- **IO/시각화** (`modules/utils.py`): `load_mask`, `load_video`, `load_obj`, `mask_label_2d`, `visualization`
