# ContourExtract.md

**목표:** 2D midsagittal MRI segmentation mask에서 혀의 **dorsal contour**를 안정적으로 추출하고, 특히 **tongue tip(TT)** 을 floor/ventral boundary와 혼동하지 않도록 개선한다.  
**작성 목적:** 현재 `precise_contour()`의 실패 케이스를 줄이기 위한 구현 우선순위, 논문 기반 근거, 디버깅/평가 계획을 정리한다.

---

## 0. 요약

현재 실패의 핵심은 **tip을 단순히 `min col` 또는 airway와 가까운 boundary의 시작점으로 잡는 것**이다. Target 예시에서는 혀끝 주변의 아래쪽/floor 쪽 경계가 airway와 가깝고 앞쪽으로 튀어나와 있어서, 알고리즘이 그 부분을 dorsal contour의 tip으로 착각한다.

내 추천은 다음 순서다.

1. **No-training hotfix:** 기존 contour extraction을 유지하되, 후보 boundary에 `outward normal`, `dorsum 포함 여부`, `upper-anterior tip 선택`을 추가한다.
2. **Shape/landmark prior:** rest contour 또는 소량의 corrected contour로 tip/blade/dorsum/root 순서를 보존하는 shape prior를 넣는다.
3. **Small supervised landmark:** 30~100 frame만 수동 라벨링해서 TT heatmap 또는 TT/TD/TR landmark 모델을 학습하고, 이 예측값을 contour 선택의 anchor로 사용한다.
4. **Full supervised contour:** annotation을 100~500 frame까지 할 수 있으면 DeepLabCut 또는 heatmap keypoint 방식으로 dorsal contour points 자체를 예측한다.

가장 먼저 넣어야 할 수정은 이것이다.

```text
현재:
  airway distance <= threshold
  -> longest cyclic run
  -> min-col endpoint를 tip으로 사용

수정:
  airway distance <= threshold
  + outward normal이 ventral/floor 방향이 아님
  + dorsum 근처를 지나는 run 우선
  + anterior band 안에서 가장 upper point를 true tip으로 선택
  + 필요 시 previous-frame tip prior 적용
```

---

## 1. 현재 알고리즘과 실패 원인

현재 Stage 2의 dorsal contour 추출은 대략 다음 구조다.

```text
1. tongue mask에서 가장 긴 closed boundary 추출
2. airway distance transform으로 airway-facing boundary point 선택
3. cyclic boundary에서 가장 긴 연속 run 선택
4. col이 작은 쪽을 tip으로 정렬
5. root spur clip
6. smoothing + arc-length resampling
```

이 방식은 단순하고 잘 동작하는 frame도 많지만, 다음 상황에서 취약하다.

- 혀 tip 주변 airway label이 혀의 위쪽뿐 아니라 아래쪽/floor 쪽 boundary도 감싼다.
- ventral/floor boundary가 실제 dorsal tip보다 더 anterior, 즉 더 작은 `col`을 갖는다.
- airway-facing run 중 가장 긴 구간이 반드시 dorsal surface가 아니다.
- `min col`은 “앞쪽” 기준일 뿐이고 “혀 등쪽 tip” 기준이 아니다.

따라서 현재 Target 실패는 registration/RBF 문제가 아니라, **Stage 2 contour extraction에서 tip anchor가 잘못 잡히는 문제**로 보는 것이 맞다.

---

## 2. 좌표와 용어

이미지 좌표 기준:

```text
row: 아래로 증가
col: 오른쪽으로 증가
```

현재 예시에서는 혀 tip이 이미지 왼쪽에 있으므로, anterior 방향은 보통 `col`이 작은 방향이다. 하지만 `min col`만으로 tip을 정의하면 아래쪽/floor boundary도 tip 후보가 된다.

권장 용어:

```text
TT: tongue tip, true dorsal contour의 시작 anchor
TB: tongue blade
TD: tongue dorsum, row가 가장 작은 dorsal 상단부 근처
TR: tongue root, dorsal contour의 posterior endpoint
ventral/floor false-tip: 혀끝 아래쪽 또는 구강 바닥 쪽 경계를 tip으로 잘못 잡은 점
```

---

## 3. 논문에서 가져올 핵심 아이디어

### 3.1 Su et al. 2018 — Geometrically constrained snake

**논문:** Zhihua Su et al., *Tongue Segmentation with Geometrically Constrained Snake Model*, Interspeech 2018.  
**링크:** https://www.isca-archive.org/interspeech_2018/su18c_interspeech.html

이 논문은 midsagittal MRI에서 혀가 주변 구조와 접촉하거나 boundary가 불명확할 때 traditional snake가 잘못된 edge로 수렴하는 문제를 다룬다. 네 문제와 가장 직접적으로 연결된다.

적용 아이디어:

```text
airway-facing edge만 보고 contour를 고르지 말고,
해부학적/geometric constraint를 energy 또는 score로 넣는다.
```

구현에서는 full snake를 다시 만들 필요 없이, 현재 후보 run 선택 단계에 다음 penalty를 넣으면 된다.

```text
run_score = airway_cost
          + normal_penalty
          + dorsum_miss_penalty
          + tip_prior_penalty
          + shape_prior_penalty
```

---

### 3.2 Peng et al. 2010 — PCA shape prior

**논문:** Ting Peng et al., *A Shape-Based Framework to Segmentation of Tongue Contours from MRI Data*, ICASSP 2010.  
**링크:** https://magrit.loria.fr/Papiers/icasspPeng2010.pdf

이 논문은 여러 tongue contour로 PCA shape model을 만들고, shape prior를 segmentation에 넣는다. 중요한 포인트는 **contour의 양 끝점이 physical correspondence를 가져야 한다**는 것이다. 즉, point index가 매 frame에서 같은 해부학적 의미를 가져야 한다.

적용 아이디어:

```text
rest dorsal contour 또는 manually corrected contour 몇 개로 평균 shape / PCA shape를 만든다.
후보 contour가 shape prior에서 너무 벗어나면 reject 또는 penalty를 준다.
```

네 retargeting에서는 rest→target displacement를 point index로 대응시키므로, shape prior는 단순 정확도보다 **tip→root ordering 보존**에 특히 중요하다.

---

### 3.3 Labrunie et al. 2018 — modified Active Shape Model

**논문:** Mathieu Labrunie et al., *Automatic segmentation of speech articulators from real-time midsagittal MRI based on supervised learning*, Speech Communication 2018.  
**링크:** https://www.sciencedirect.com/science/article/abs/pii/S0167639317301784

이 논문은 작은 수의 수동 contour로 modified Active Shape Model을 학습해서 rtMRI articulator contour를 자동 segmentation한다.

적용 아이디어:

```text
전체 segmentation을 CNN으로 갈아엎기 전에,
현재 contour 후보를 Active Shape Model처럼 해부학적 point set으로 다룬다.
```

즉, `contour[0] = tip`, `contour[중간] = dorsum`, `contour[-1] = root`가 유지되도록 score와 smoothing을 설계한다.

---

### 3.4 Eslami et al. 2020 — TT landmark heatmap

**논문:** Mohammad Eslami et al., *Automatic vocal tract landmark localization from midsagittal MRI data*, Scientific Reports 2020.  
**링크:** https://www.nature.com/articles/s41598-020-58103-6

이 논문은 midsagittal MRI에서 21개 vocal tract anatomical landmarks를 자동 localization한다. 이 landmarks에는 **tip of the tongue**가 포함된다.

적용 아이디어:

```text
TT만이라도 heatmap으로 예측한다.
그 다음 현재 contour 후보 중 TT heatmap peak와 가장 잘 맞는 endpoint를 가진 run을 선택한다.
```

이 방식은 annotation이 비교적 적어도 시작할 수 있고, 현재 문제처럼 `min col`이 실패하는 상황에 직접적으로 효과가 있다.

---

### 3.5 DeepLabCut / keypoint contour

**논문 A:** Wrench & Balch-Tomes, *Beyond the Edge: Markerless Pose Estimation of Speech Articulators from Ultrasound and Camera Images Using DeepLabCut*, Sensors 2022.  
**링크:** https://www.mdpi.com/1424-8220/22/3/1133

**논문 B:** Sun, Kitamura & Hayashi, *Extraction of Speech Organ Contours from Ultrasound and real-time MRI Data using DeepLabCut*, Acoustical Science and Technology 2025.  
**링크:** https://www.jstage.jst.go.jp/article/ast/46/4/46_e24.128/_pdf

핵심은 edge detection에 의존하지 않고, 사람이 정의한 keypoint 또는 contour point를 직접 예측하는 것이다. Sun et al. 2025는 rtMRI에서 tongue contour를 포함한 articulatory contours를 DeepLabCut으로 추출하는 방식을 보여준다.

적용 아이디어:

```text
입력: MRI frame 또는 현재 segmentation mask rendered image
출력: TT, TB, TD, TR 또는 dorsal contour 11~25 points
```

retargeting 목적이면 full mask segmentation보다, **dorsal contour points만 정확히 예측**하는 것이 더 효율적일 수 있다.

---

### 3.6 Somandepalli et al. 2017 — semantic edge detection

**논문:** Krishna Somandepalli et al., *Semantic Edge Detection for Tracking Vocal Tract Air-Tissue Boundaries in Real-Time MRI Images*, Interspeech 2017.  
**링크:** https://www.isca-archive.org/interspeech_2017/somandepalli17_interspeech.html

이 논문은 air-tissue boundary를 단순 edge가 아니라 articulator label이 붙은 semantic edge로 예측한다.

적용 아이디어:

```text
현재 mask 품질 자체가 자주 흔들린다면,
mask -> contour 후처리보다 semantic edge probability map을 직접 학습하는 방향도 가능하다.
```

다만 지금 당장은 기존 mask가 있으므로 우선순위는 낮다.

---

## 4. 추천 구현안: `precise_contour_v2`

### 4.1 핵심 변경점

현재 `precise_contour()`에서 바꿔야 할 부분은 크게 네 개다.

```text
A. airway-facing 후보에 outward normal 조건 추가
B. longest run 대신 dorsum 근처 run 선택
C. tip을 min-col endpoint가 아니라 upper-anterior point로 선택
D. 시계열이면 previous-frame tip prior 추가
```

---

## 5. Step A — airway-facing + outward normal filter

### 5.1 왜 필요한가

airway와 가깝다는 조건만 쓰면 dorsal surface와 ventral/floor surface를 구분하지 못한다. 둘 다 airway와 가까울 수 있기 때문이다.

혀의 dorsal surface는 outward normal이 대체로 위쪽을 향한다. 이미지 좌표에서는 row가 아래로 증가하므로, dorsal normal은 보통 `normal_row < 0` 또는 최소한 `normal_row`가 크게 양수가 아니어야 한다.

true tip은 normal이 anterior 방향을 향할 수 있어서 `normal_row ≈ 0`이다. 따라서 조건을 너무 빡세게 `normal_row < 0`으로 두면 tip이 잘릴 수 있다. 처음에는 아래처럼 완화된 threshold를 권장한다.

```python
normal_ok = outward_normal[:, 0] < 0.35
```

`0.35`는 시작값이다. target 실패 frame들을 보면서 `0.25 ~ 0.50` 범위에서 조정한다.

### 5.2 구현 스케치

```python
def outward_normals_from_contour(cont, tongue_mask, eps=1.5):
    """
    cont: (N, 2), row-col contour points
    tongue_mask: bool, True=tongue
    return: (N, 2), outward unit normal in row-col coordinate
    """
    import numpy as np

    prev_p = np.roll(cont, 1, axis=0)
    next_p = np.roll(cont, -1, axis=0)

    tangent = next_p - prev_p
    tangent = tangent / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8)

    # 90-degree rotation candidates in row-col plane
    n1 = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    n2 = -n1

    H, W = tongue_mask.shape

    def inside_tongue(p):
        rr = np.clip(np.rint(p[:, 0]).astype(int), 0, H - 1)
        cc = np.clip(np.rint(p[:, 1]).astype(int), 0, W - 1)
        return tongue_mask[rr, cc]

    p1 = cont + eps * n1
    p2 = cont + eps * n2

    inside1 = inside_tongue(p1)
    inside2 = inside_tongue(p2)

    # Prefer direction that moves outside the tongue mask.
    normals = np.where((~inside1)[:, None], n1, n2)

    # Rare ambiguous case: if both are outside/inside due to thin or jagged mask,
    # keep n1 as fallback. Later smoothing can absorb local noise.
    return normals
```

사용 예:

```python
tongue = mask == 4
airway = mask == 5

cont = longest_closed_tongue_boundary(tongue)  # existing find_contours result

dt = distance_transform_edt(~airway)
rr = np.clip(np.rint(cont[:, 0]).astype(int), 0, mask.shape[0] - 1)
cc = np.clip(np.rint(cont[:, 1]).astype(int), 0, mask.shape[1] - 1)

airway_ok = dt[rr, cc] <= facing_thresh
normal = outward_normals_from_contour(cont, tongue)
normal_ok = normal[:, 0] < normal_row_max  # e.g., 0.35

keep = airway_ok & normal_ok
```

---

## 6. Step B — longest run 대신 dorsum-containing run 선택

### 6.1 왜 필요한가

현재는 cyclic boundary에서 가장 긴 `True` run을 dorsal contour로 선택한다. 하지만 airway label이 tip 주변이나 floor 쪽에 붙으면, 가장 긴 run이 dorsal contour가 아닐 수 있다.

혀 등쪽 contour는 적어도 **dorsum**, 즉 tongue boundary 중 위쪽 상단부 근처를 지나야 한다. 따라서 run 선택 기준을 다음처럼 바꾼다.

```text
1순위: dorsum index를 포함하거나 가장 가까운 run
2순위: 충분히 긴 run
3순위: ventral normal 비율이 낮은 run
4순위: endpoint가 plausible tip/root를 형성하는 run
```

### 6.2 run scoring

```python
def cyclic_index_distance_to_run(idx, run_indices, n):
    import numpy as np
    run_indices = np.asarray(run_indices)
    d = np.abs(run_indices - idx)
    d = np.minimum(d, n - d)
    return float(d.min())


def score_run(run_indices, cont, normal, dorsum_idx, prev_tip=None):
    """
    Higher is better.
    """
    import numpy as np

    n = len(cont)
    pts = cont[run_indices]
    normals = normal[run_indices]

    length_score = len(run_indices) / n

    dorsum_dist = cyclic_index_distance_to_run(dorsum_idx, run_indices, n)
    dorsum_score = 1.0 - min(dorsum_dist / max(n * 0.15, 1), 1.0)

    # Strongly downweight runs dominated by downward normals.
    ventral_frac = np.mean(normals[:, 0] > 0.55)
    ventral_score = 1.0 - ventral_frac

    # Candidate tip = upper-anterior point in this run.
    cmin = pts[:, 1].min()
    cand = np.where(pts[:, 1] <= cmin + 8)[0]
    if len(cand) > 0:
        tip = pts[cand[np.argmin(pts[cand, 0])]]
    else:
        tip = pts[np.argmin(pts[:, 1])]

    temporal_score = 0.0
    if prev_tip is not None:
        dist = np.linalg.norm(tip - prev_tip)
        temporal_score = -min(dist / 20.0, 2.0)

    return (
        2.5 * dorsum_score
        + 1.0 * length_score
        + 1.5 * ventral_score
        + 1.0 * temporal_score
    )
```

처음에는 `prev_tip` 없이 시작하고, 시계열 target에서 tip jumping이 보이면 `prev_tip` penalty를 추가한다.

---

## 7. Step C — upper-anterior tip 선택

### 7.1 왜 필요한가

selected run을 잘 골라도, tip 주변에서 contour가 U자 형태로 말려 있으면 lower/floor point가 더 anterior일 수 있다. 따라서 tip은 `min col`이 아니라 다음처럼 잡는다.

```text
selected segment 안에서
col이 가장 작은 anterior band를 찾고,
그 band 안에서 row가 가장 작은 point를 TT로 선택한다.
```

즉, “앞쪽 중 가장 위쪽”을 true dorsal tip으로 본다.

### 7.2 구현 스케치

```python
def choose_upper_anterior_tip(seg, tip_band_px=8):
    """
    seg: (M, 2), selected contour segment in row-col.
    return: local index of true dorsal tip.
    """
    import numpy as np

    cols = seg[:, 1]
    rows = seg[:, 0]

    cmin = cols.min()
    cand = np.where(cols <= cmin + tip_band_px)[0]

    if len(cand) == 0:
        return int(np.argmin(cols))

    return int(cand[np.argmin(rows[cand])])
```

만약 tip이 segment의 중간에 있다면, tip 기준으로 양쪽을 나누고 dorsal/root 방향 side만 유지한다.

```python
def keep_dorsal_side_after_tip(seg, tip_idx):
    """
    Tip 주변에 lower/floor curl이 포함된 경우 제거한다.
    두 side 중 더 rootward, 더 dorsal-like인 쪽을 선택한다.
    """
    import numpy as np

    side_a = seg[tip_idx:]                 # tip -> one endpoint
    side_b = seg[:tip_idx + 1][::-1]       # tip -> other endpoint

    def side_score(side):
        if len(side) < 3:
            return -1e9
        rootward = side[-1, 1] - side[0, 1]  # col increases toward posterior in current orientation
        length = len(side)
        upperness = -np.mean(side[:, 0])
        return 2.0 * rootward + 0.2 * length + 0.05 * upperness

    return side_a if side_score(side_a) >= side_score(side_b) else side_b
```

이후 항상 `seg[0] = TT`, `seg[-1] = root`가 되도록 한다.

---

## 8. Step D — airway mask cleanup

airway label이 tip 주변에 작게 튀거나 여러 component로 나뉘면 contour 후보가 오염될 수 있다.

추천 전처리:

```python
from skimage.measure import label
from skimage.morphology import remove_small_objects, binary_closing, disk


def clean_airway(airway, min_size=20, closing_radius=1, keep_largest=False):
    airway = remove_small_objects(airway.astype(bool), min_size=min_size)
    airway = binary_closing(airway, disk(closing_radius))

    if keep_largest:
        lab = label(airway)
        if lab.max() > 0:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            airway = lab == counts.argmax()

    return airway
```

주의:

- `keep_largest=True`는 oral airway가 실제로 여러 component로 분리되는 frame에서 문제를 만들 수 있다.
- 처음에는 `remove_small_objects + closing`만 적용하고, debug overlay로 확인한 뒤 largest component를 켠다.

---

## 9. `precise_contour_v2` 전체 의사코드

```python
def precise_contour_v2(
    mask,
    n_markers=25,
    facing_thresh=2.5,
    normal_row_max=0.35,
    tip_band_px=8,
    gap_close_pts=4,
    prev_tip=None,
    clip_root=True,
):
    """
    Returns dorsal contour points in row-col order: tip -> root.
    """
    tongue = mask == 4
    airway = clean_airway(mask == 5)

    # 1. Closed tongue boundary
    cont = longest_closed_tongue_boundary(tongue)  # existing find_contours wrapper

    # 2. Airway-facing candidate
    dt = distance_transform_edt(~airway)
    rr, cc = sample_indices(cont, mask.shape)
    airway_ok = dt[rr, cc] <= facing_thresh

    # 3. Normal filter
    normal = outward_normals_from_contour(cont, tongue)
    normal_ok = normal[:, 0] < normal_row_max

    keep = airway_ok & normal_ok
    keep = close_small_false_gaps_cyclic(keep, max_gap=gap_close_pts)

    # 4. Runs
    runs = true_runs_cyclic_indices(keep)
    if len(runs) == 0:
        # fallback to previous method or relaxed threshold
        return precise_contour_legacy(mask, n=n_markers)

    # 5. Dorsum-guided run selection
    dorsum_idx = int(np.argmin(cont[:, 0]))
    best_run = max(
        runs,
        key=lambda r: score_run(r, cont, normal, dorsum_idx, prev_tip=prev_tip),
    )

    seg = unwrap_contour_indices(cont, best_run)

    # 6. Upper-anterior tip selection and trimming
    tip_idx = choose_upper_anterior_tip(seg, tip_band_px=tip_band_px)
    seg = keep_dorsal_side_after_tip(seg, tip_idx)

    # 7. Ensure tip -> root direction
    if seg[-1, 1] < seg[0, 1]:
        seg = seg[::-1]

    # 8. Optional posterior spur clip
    if clip_root:
        seg = clip_posterior_spur(seg)

    # 9. Smooth + arc-length resample
    seg = smooth_3tap(seg)
    seg = resample_arc_length(seg, n_markers)

    return seg
```

---

## 10. Config 제안

`configs/retarget/default.yaml`에 contour v2 옵션을 추가한다.

```yaml
contour:
  mode: v2
  n_markers: 25
  facing_thresh: 2.5
  clip_root: true
  clip_drop_frac: 1.0

  v2:
    normal_row_max: 0.35
    tip_band_px: 8
    gap_close_pts: 4
    airway_min_size: 20
    airway_closing_radius: 1
    airway_keep_largest: false

    score:
      w_dorsum: 2.5
      w_length: 1.0
      w_ventral: 1.5
      w_temporal: 1.0

    temporal:
      enabled: false
      max_tip_jump_px: 20
```

---

## 11. Debug overlay 추가 항목

현재 overlay에 다음을 추가하면 실패 원인을 바로 볼 수 있다.

```text
cyan: 기존 extracted contour
yellow/red gradient: v2 selected contour tip -> root
red star: selected TT
blue dots: rejected airway-facing candidates
orange dots: rejected by normal filter
green dot: dorsum_idx
white arrows: outward normal vectors every k points
```

각 frame마다 콘솔에 다음 값도 출력한다.

```text
num_runs
selected_run_length
selected_run_score
dorsum_distance_to_run
ventral_frac
TT(row,col)
TT_prev_distance, if sequence mode
fallback_used: true/false
```

실패 frame에서는 `airway_ok`, `normal_ok`, `keep`을 각각 따로 overlay해야 한다. 특히 Target 예시의 false-tip 위치가 `airway_ok=True`이지만 `normal_ok=False`로 떨어지는지 확인한다.

---

## 12. 평가 계획

### 12.1 최소 golden set

먼저 20~50개 frame만 골라서 수동으로 다음을 찍는다.

```text
TT: true dorsal tongue tip
TD: dorsum point
TR: root endpoint
optional: dorsal contour 11 or 25 points
```

반드시 포함할 frame:

- 현재 보여준 Target 실패 frame
- rest frame
- tip이 palate/teeth 근처에 붙은 frame
- tongue body가 낮은 frame
- root spur가 긴 frame
- airway label이 끊긴 frame

### 12.2 지표

```text
Tip error px/mm:
  ||pred_TT - manual_TT||

Contour MSD:
  predicted contour point에서 manual contour polyline까지 평균 최단거리

False-tip count:
  TT가 manual TT에서 특정 threshold 이상 벗어난 frame 수

Direction failure count:
  contour[0]이 root 쪽이고 contour[-1]이 tip 쪽인 frame 수

Temporal jitter:
  ||TT_t - TT_{t-1}||의 median / 95 percentile

Fallback rate:
  v2 조건이 너무 strict해서 legacy fallback이 발생한 비율
```

### 12.3 목표 기준

초기 목표는 다음 정도로 잡는다.

```text
TT error median < 3 px
TT error 95 percentile < 8 px
False-tip count = 0 on golden failure set
Direction failure count = 0
Fallback rate < 5%
```

mm 기준은 `mm_per_px`를 곱해서 함께 보고한다.

---

## 13. Supervised landmark 버전

No-training v2로도 false-tip이 남으면 TT만 학습하는 것이 가장 비용 대비 효과가 좋다.

### 13.1 Annotation

처음에는 contour 전체가 아니라 landmark 3개만 찍는다.

```text
TT: tongue tip
TD: tongue dorsum
TR: tongue root
```

권장 수량:

```text
30 frames: prototype 가능
50~100 frames: subject-specific model 시작점
200+ frames: robust한 keypoint/contour model
```

### 13.2 입력 선택

두 가지 중 하나를 선택한다.

```text
A. raw MRI image 입력
B. segmentation mask를 color/one-hot image로 변환한 입력
```

현재 문제는 mask 기반 pipeline 내부에서 발생하므로, 빠르게 적용하려면 B가 좋다.

예:

```text
channel 0: tongue mask
channel 1: airway mask
channel 2: other tissue/background boundary or distance transform
```

### 13.3 출력

간단한 heatmap U-Net 또는 Flat-net 스타일 모델:

```text
input: 256x256xC
output: 3 heatmaps, TT/TD/TR
loss: MSE or focal heatmap loss
```

inference:

```python
TT_pred = argmax(heatmap_TT)
TD_pred = argmax(heatmap_TD)
TR_pred = argmax(heatmap_TR)
```

이후 contour 후보 run score에 landmark penalty를 넣는다.

```python
score += -w_tt * distance(candidate_tip, TT_pred)
score += -w_td * distance(run, TD_pred)
score += -w_tr * distance(candidate_root, TR_pred)
```

핵심은 CNN이 contour를 완전히 대체하지 않아도 된다는 점이다. **TT만 reliable하게 예측해도 현재 false-tip 문제는 크게 줄어든다.**

---

## 14. Full contour keypoint 버전

annotation을 더 할 수 있으면 dorsal contour 자체를 11~25 points로 라벨링한다.

추천 point 정의:

```text
P00 = TT
P01-P05 = blade
P06-P15 = dorsum/body
P16-P24 = root side
```

또는 11 points로 시작:

```text
P00 = TT
P05 = TD 근처
P10 = TR
```

이 방식의 장점:

- point correspondence가 학습 단계에서 강제된다.
- `min col`, `longest run`, `airway distance`에 덜 의존한다.
- retargeting의 rest→target displacement 계산과 잘 맞는다.

단점:

- annotation 비용이 증가한다.
- subject/domain이 바뀌면 fine-tuning이 필요할 수 있다.

추천 적용 방식:

```text
초기: v2 contour를 자동 생성
수동: 실패 frame만 corrected contour 라벨링
학습: DLC 또는 heatmap keypoint model
운영: model prediction을 primary, v2 contour를 fallback/regularizer로 사용
```

---

## 15. 구현 우선순위

### Phase 1 — 하루 안에 가능한 hotfix

```text
1. outward normal 계산 함수 추가
2. airway_ok & normal_ok로 후보 filtering
3. longest run 대신 dorsum-guided run scoring
4. upper-anterior tip 선택
5. Target 실패 frame overlay 확인
```

예상 효과:

```text
floor/ventral false-tip이 크게 줄어듦.
annotation 없이 바로 적용 가능.
```

주의:

```text
normal_row_max가 너무 작으면 true tip이 잘릴 수 있음.
facing_thresh와 normal threshold를 같이 sweep해야 함.
```

---

### Phase 2 — 2~3일 안정화

```text
1. airway cleanup 옵션 추가
2. gap closing along contour 추가
3. previous-frame tip prior 추가
4. debug metrics 로그 추가
5. golden set 20~50 frame 수동 평가
```

예상 효과:

```text
시계열에서 tip jumping 감소.
frame-by-frame 튐 현상 감소.
```

---

### Phase 3 — 1주 supervised TT anchor

```text
1. TT/TD/TR landmark 라벨링 50~100 frames
2. heatmap model 또는 DLC project 생성
3. TT_pred를 contour run score에 anchor로 추가
4. no-training v2와 ablation 비교
```

예상 효과:

```text
mask artifact가 심한 frame에서도 TT 안정성 증가.
```

---

### Phase 4 — 2주 이상 full contour keypoints

```text
1. dorsal contour 11~25 points 라벨링
2. DLC 또는 keypoint heatmap model 학습
3. predicted contour를 retargeting에 직접 사용
4. v2 contour는 fallback 또는 confidence check로 유지
```

예상 효과:

```text
retargeting point correspondence 안정화.
Stage 2 contour extraction의 rule-based heuristic 의존도 감소.
```

---

## 16. 추천 ablation 실험

아래 순서로 한 번에 하나씩 켠다.

```text
baseline: legacy precise_contour
v2-A: + normal filter
v2-B: + dorsum-guided run selection
v2-C: + upper-anterior tip selection
v2-D: + airway cleanup
v2-E: + temporal prior
v2-F: + TT heatmap anchor
```

표는 이렇게 만든다.

| Method | TT median err px | TT p95 err px | False-tip count | Direction fail | Contour MSD px | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|
| legacy |  |  |  |  |  |  |
| v2-A |  |  |  |  |  |  |
| v2-B |  |  |  |  |  |  |
| v2-C |  |  |  |  |  |  |
| v2-D |  |  |  |  |  |  |
| v2-E |  |  |  |  |  |  |
| v2-F |  |  |  |  |  |  |

---

## 17. 내가 권장하는 최종 구조

최종적으로는 다음 구조가 가장 안전하다.

```text
mask
 ├─ tongue boundary extraction
 ├─ airway distance candidate
 ├─ normal/dorsum/shape-prior filtering
 ├─ optional TT/TD/TR heatmap prior
 └─ dorsal contour tip→root output
```

운영 모드:

```text
1. supervised TT available:
     TT heatmap anchor + v2 contour selection

2. supervised TT unavailable:
     v2 deterministic selection

3. v2 confidence low:
     previous frame contour propagation or legacy fallback
```

confidence score 예:

```python
confidence = (
    0.35 * dorsum_score
    + 0.25 * normal_score
    + 0.20 * length_score
    + 0.20 * tip_prior_score
)
```

confidence가 낮은 frame은 자동 retargeting 결과를 바로 믿지 말고 debug queue로 보낸다.

---

## 18. 체크리스트

구현 후 다음이 모두 만족되어야 한다.

```text
[ ] Target 실패 frame에서 TT가 floor가 아니라 true dorsal tip에 찍힌다.
[ ] contour 색상 gradient가 tip -> root 순서로 일관된다.
[ ] dorsum_idx가 selected run 위 또는 매우 가까운 곳에 있다.
[ ] normal filter 후 ventral/floor candidate가 제거된다.
[ ] clip_root가 true root를 과하게 자르지 않는다.
[ ] rest와 target의 contour point correspondence가 시각적으로 유지된다.
[ ] retargeting Stage 4에서 source_xz[k]와 target_xz[k]가 같은 해부학적 위치를 의미한다.
```

---

## 19. 결론

이 문제는 “혀 tip detection”만의 문제가 아니라 **dorsal contour의 해부학적 correspondence 문제**다. Retargeting에서는 contour point index가 그대로 displacement correspondence가 되기 때문에, tip이 한 번만 floor 쪽으로 잘못 잡혀도 이후 RBF deformation이 전부 흔들릴 수 있다.

따라서 가장 좋은 방향은 다음이다.

```text
단기: precise_contour_v2 = normal + dorsum + upper-anterior tip
중기: TT/TD/TR landmark prior
장기: dorsal contour keypoint model
```

우선은 annotation 없이 `precise_contour_v2`를 구현하고, 실패 frame이 남으면 TT heatmap anchor를 붙이는 것을 추천한다.

---

## References

1. Su, Z., Wei, J., Fang, Q., Wang, J., Honda, K. (2018). *Tongue Segmentation with Geometrically Constrained Snake Model*. Interspeech 2018. https://www.isca-archive.org/interspeech_2018/su18c_interspeech.html
2. Peng, T., Kerrien, E., Berger, M.-O. (2010). *A Shape-Based Framework to Segmentation of Tongue Contours from MRI Data*. ICASSP 2010. https://magrit.loria.fr/Papiers/icasspPeng2010.pdf
3. Labrunie, M., Badin, P., Voit, D., Joseph, A. A., Frahm, J., et al. (2018). *Automatic segmentation of speech articulators from real-time midsagittal MRI based on supervised learning*. Speech Communication. https://www.sciencedirect.com/science/article/abs/pii/S0167639317301784
4. Eslami, M., Neuschaefer-Rube, C., Serrurier, A. (2020). *Automatic vocal tract landmark localization from midsagittal MRI data*. Scientific Reports. https://www.nature.com/articles/s41598-020-58103-6
5. Wrench, A., Balch-Tomes, J. (2022). *Beyond the Edge: Markerless Pose Estimation of Speech Articulators from Ultrasound and Camera Images Using DeepLabCut*. Sensors. https://www.mdpi.com/1424-8220/22/3/1133
6. Sun, J., Kitamura, T., Hayashi, R. (2025). *Extraction of Speech Organ Contours from Ultrasound and real-time MRI Data using DeepLabCut*. Acoustical Science and Technology. https://www.jstage.jst.go.jp/article/ast/46/4/46_e24.128/_pdf
7. Somandepalli, K., Toutios, A., Narayanan, S. S. (2017). *Semantic Edge Detection for Tracking Vocal Tract Air-Tissue Boundaries in Real-Time Magnetic Resonance Images*. Interspeech 2017. https://www.isca-archive.org/interspeech_2017/somandepalli17_interspeech.html
