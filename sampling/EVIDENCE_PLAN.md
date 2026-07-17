# Dataset Evidence Plan

"이 데이터셋이 실제 혀의 모양을 커버한다"를 **증명**하기 위해 만들어야 할
그림·표·분포 목록. 각 항목마다 *어떤 반론을 막는가*를 명시했다.

---

## 0. 증거 사다리 (무엇이 증거이고 무엇이 아닌가)

| 단계 | 무엇을 보이나 | 순환성 |
|---|---|---|
| L0 activation 분포 통계 | 설계 의도대로 뽑혔다 | **순환. 증거 아님** — 설계 설명일 뿐 |
| L1 valid yield | 시뮬이 성공한다 | 없음 |
| L2 shape-space 다양성 | S(a)의 상이 넓다 | 없음 (단, 시뮬레이터 내부) |
| L3 비대칭 포함 | v2 ⊃ old | 없음 (시뮬레이터 내부) |
| L4 **실제 MRI coverage/precision** | **진짜 혀를 덮는다** | **없음 — 결정적** |
| L5 downstream A/B | 재구성 오차가 낮다 | 없음 — 논문의 최종 근거 |

L0은 **Supplementary로 보내라.** 메인 논문에서 L0을 커버리지 근거로 제시하면
"당신이 넣기로 정한 걸 넣었다는 뜻일 뿐"이라는 반박을 그대로 받는다.

**메인 피겨는 F5다.** 나머지는 F5를 해석 가능하게 만드는 보조 장치다.

---

## 1. 반드시 통제해야 할 3가지 (안 하면 모든 그림이 무의미)

### (a) Matched-N
최근접이웃 커버리지는 N에 대해 **단조증가**한다. old 50k vs v2 200k 비교는
그 자체로 무효다. 반드시 v2를 50k로 서브샘플해서 비교하고, 그 위에 saturation
곡선(F5b)을 얹어 N 효과를 분리한다.

### (b) Anatomy floor
ArtiSynth는 단일 canonical anatomy, 실제 subject는 해부학이 다르다. 정렬 후에도
남는 잔차가 있고, 그게 자세 차이 0일 때의 **바닥값**이다.

```
floor_s = d( 실제 subject s의 rest frame contour , 합성 rest mesh의 midsag contour )
          (경구개 + 하악 symphysis + 인두 후벽 landmark로 rigid+scale 정렬 후)
```

이 수평선 없이 C(ε)만 그리면 숫자를 해석할 수 없다. F5a에 반드시 그린다.

### (c) 노이즈 바닥 ε* 는 측정하는 것
주장하지 말고 측정한다. 우선순위:
1. 동일 프레임 재분할(re-segmentation) 변동성
2. 불가능하면: steady-state hold 구간 내 연속 프레임 간 contour 거리
   ("같은 자세"의 경험적 거리 스케일)

`C(ε*)` 가 헤드라인 숫자이므로 ε* 의 출처를 논문에 명시한다.

---

## 2. 핵심 지표: Contour-space Precision–Recall

실제 contour 집합 `R = {c_k}` (GT_Segmentations, Subject 1–5 전 프레임),
합성 contour 집합 `S = {R_π(M_i)}` (watchdog-valid mesh만).

```
Coverage / Recall   C(ε) = (1/|R|)·#{ c∈R : min_{s∈S} d(c,s) ≤ ε }
                    "실제 혀 모양 중 데이터셋이 재현 가능한 비율"

Precision           P(ε) = (1/|S|)·#{ s∈S : min_{c∈R} d(s,c) ≤ ε }
                    "데이터셋 샘플 중 실제로 있을 법한 모양의 비율"
```

`d` = 대칭 point-to-curve 거리 (mean 및 P95, mm). raw pixel chamfer 금지.

**Recall만 보면 안 되는 이유:** 데이터셋을 키우기만 해도 recall은 오른다.
쓰레기를 넣어도 오른다. Precision이 그걸 잡는다. old pool은 생리학적으로
불가능한 co-contraction 덩어리이므로 precision이 낮아야 한다. 둘 중 하나만
개선되면 재설계가 틀린 것이다.

프레이밍 출처: PRD (Sajjadi et al. 2018), Improved P&R (Kynkäänniemi et al. 2019).

---

## 3. Figures

### F1 — Sampling design 진단 [Supplementary]
분포 그래프 3패널, old vs v2 겹쳐 그리기.

- (a) **effort 히스토그램** (11개 activation 합). old는 5.50±0.96에 뭉쳐 있고
  1.61 아래가 없음. v2는 0 → 8.9에 걸침.
- (b) **활성 근육 수 히스토그램** (>0.5 기준). old 평균 5.5개, v2 2.2개.
- (c) **근육별 marginal**, raw vs valid 두 겹 (0714.md §7.1의 `p_raw(a) ≠ p(a|valid)`).

캡션에 "이것은 설계 설명이지 커버리지 증거가 아니다"를 명시할 것.

### F2 — Valid yield & failure structure
- (a) validity label 비율 stacked bar (VALID/MARGINAL/INVALID_PHYSICAL/FAILED_NUMERICAL), old vs v2
- (b) **effort별 acceptance rate 곡선** — effort가 높을수록 element inversion.
  old가 왜 계산을 낭비하는지 여기서 보인다.
- (c) failure reason별 parallel coordinates (0714.md §7.1)

→ **계산 효율 논거.** "v2는 같은 컴퓨트로 더 많은 valid 샘플을 만든다."

### F3 — Shape-space 구조 (시뮬레이터 내부)
- (a) **PCA cumulative variance** (surface vertex 벡터), old vs v2, matched N.
  95% 분산에 필요한 PC 수 = effective dimension. 예측: v2 > old.
- (b) **N_effective(ε) 곡선** (0714.md §7.3) — ε에 따른 unique mesh 수.
- (c) **nearest-shape gap 분포** (독립 test activation → 최근접 거리 히스토그램)

### F4 — 비대칭 포함 테스트 (matched N)
양방향 커버리지 곡선 하나의 축에:
```
C_{v2 ← old}(ε)   : old의 mesh들이 v2 데이터셋 안에서 ε 이내 이웃을 갖는 비율
C_{old ← v2}(ε)   : 그 역
```
**v2가 old를 진부분집합으로 포함하면** 위 곡선은 빨리 1에 붙고 아래 곡선은
낮게 남는다. 통과하면 "v2는 old가 하던 걸 다 하면서 더 한다"가 성립하고,
실패하면 v2가 뭔가를 잃었다는 뜻이라 즉시 알 수 있다. 우아하고 반증 가능.

### F5 — ★ 실제 MRI 커버리지 [MAIN FIGURE]
4패널.

- **(a) C(ε) / P(ε) 곡선.** x축 ε (0–6 mm). 곡선: old-50k / v2-50k(matched) /
  v2-full. 부가선: **anatomy floor**(수평), **ε\***(수직),
  **real-LOO 상한**(실제 프레임끼리의 leave-one-out coverage = 그 크기의
  완벽한 데이터셋이 도달 가능한 천장).
  → **헤드라인 숫자: `C(ε*)` 와 `P(ε*)` 한 쌍.**

- **(b) Saturation.** `C(ε*)` vs N (log x축). Pool이 셔플되어 있으므로
  시뮬 체크포인트(5k/10k/25k/50k/100k/200k)마다 재계산하면 **공짜로 나온다.**
  예측: old는 낮은 값에서 조기 포화 → *"LHS는 아무리 늘려도 안 된다"*의 증거.
  실무적으로도 "언제 멈춰도 되는가"에 답한다.

- **(c) Phone class별 coverage bar.** macro-average와 **worst-class**를 함께.
  전체 평균은 소수 클래스 실패를 숨긴다. 예측: old는 혀끝 거상(/t/,/s/,/l/)에서
  붕괴 — SL 단독 활성이 데이터에 없으므로.

- **(d) 2D articulatory occupancy overlay.** 리뷰어가 실제로 보는 그림.
  실제 contour의 첫 2 PC(또는 tip x–z, dorsum height 같은 해석 가능 descriptor)
  평면에 실제 프레임을 점으로, 합성 occupancy를 밀도/hull로 겹침.
  빈 영역 = gap. old는 실제 데이터 구름을 빗나간 곳에 뭉쳐 있어야 한다.

### F6 — Identifiability (contour space)
0714.md §7.5를 mesh space가 아니라 **contour space**에서. 그게 실제로 푸는 문제다.

- (a) hexbin: x = activation distance, y = **contour** distance.
  좌상단(activation 멀지만 contour 같음) = 관측으로 구분 불가능한 등가집합.
- (b) 등가집합 크기 히스토그램: 각 샘플에 대해 contour가 ε* 이내인 다른
  activation 샘플의 수.

→ **"왜 deterministic regression이면 안 되는가"의 결정적 그림.**
posterior/uncertainty가 선택이 아니라 필연임을 보인다. 동시에 posterior가
원리적으로 좁혀질 수 있는 상한을 정량화한다.

---

## 4. Tables

### T1 — Pool composition
block별 개수 / 비율 / **목적**. `SINGLE`·`PAIR`가 counterfactual claim의 유일한
데이터 근거임을 표에 못박는다.

### T2 — Coverage 결과 [MAIN TABLE]

| pool | N | valid% | C(ε*)↑ | P(ε*)↑ | median gap | P90 | max | macro-phone | **worst-phone** |
|---|---|---|---|---|---|---|---|---|---|
| old LHS | 50k | | | | | | | | |
| v2 (matched) | 50k | | | | | | | | |
| v2 (full) | 200k | | | | | | | | |
| *real LOO (상한)* | — | | | | | | | | |
| *anatomy floor* | — | | | | | | | | |

subject별로도 쪼개서 보고(1행 → 5행) — 특정 해부학에만 맞은 게 아님을 보인다.

### T3 — Downstream utility (0714.md §16.4)

| 학습 데이터 | real-MRI contour→3D 재구성 오차 | few-shot 효율 | **∂M/∂aᵢ 오차** |
|---|---|---|---|
| no synthetic | | | |
| old LHS 50k | | | |
| v2 50k (matched) | | | |
| v2 full | | | |

마지막 열이 XAI claim의 직접 검증이다: 양쪽 pool에서 `SINGLE` 블록을 **학습에서
제외**하고 surrogate를 학습시킨 뒤, ArtiSynth ground truth 대비 단일근육
개입 반응 `∂M/∂aᵢ` 예측 오차를 비교한다. old로 학습한 모델은 무너져야 한다.

---

## 5. 사전 등록할 예측 (결과를 보기 전에 적어둘 것)

진짜 실험이 되려면 반증 가능해야 한다.

1. **Valid yield: v2 > old.** 전 근육 동시 최대수축은 element inversion을 부른다.
2. **비대칭 포함: v2 ⊃ old**, 역은 성립 안 함 (F4).
3. **Shape effective dim: v2 > old** (F3a).
4. **C(ε\*): v2 > old**, 특히 혀끝 거상 계열에서 격차가 큼 (F5c).
5. **P(ε\*): v2 > old.** old에는 생리학적으로 불가능한 모양이 대량.
6. **old 학습 surrogate는 단일근육 counterfactual에서 실패** (T3).

**하나라도 틀리면 재설계가 틀린 것이다.** 그걸 5천 개로 알아내는 게 40만 개
돌린 뒤에 아는 것보다 낫다.

---

## 6. 실행 순서

```
Stage -1  export_all 수정        settle hold 추가 / npy memmap / validity watchdog
Stage  0  throughput 측정        ~50 샘플 → 샘플당 벽시계 시간
Stage  1  matched-N A/B pilot    각 5,000 (old vs v2, 동일 설정)
            → F2, F3, F4, F5(예비), T2(예비)
            → 여기서 승부가 난다. v2가 지면 중단하고 재설계.
Stage  2  v2 full 확장           ~200,000, 체크포인트마다 C(ε*) → F5b saturation
            → F1, F5, F6, T1, T2 확정
Stage  3  dynamic MotionBank     pose-graph 시퀀스. 커버리지 검증 이후에만.
            → T3
```

**정직하게 밝힐 한계 (리뷰어가 먼저 지적하게 두지 말 것):**
실제 데이터는 midsagittal contour뿐이므로 F5의 모든 커버리지는 **projection
공간**에서의 것이다. Contour coverage는 3D shape coverage의 **필요조건이지
충분조건이 아니다.** 좌우 폭과 lateral groove는 이것으로 검증되지 않는다.
F3(synthetic 3D coverage)를 보조로 붙이고, 이 구분을 본문에 명시한다.
