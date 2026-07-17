# PhonoTwin / ActiTongue

> **Sparse 2D MRI video로부터 subject-specific 3D tongue motion을 복원하고, 생체역학적으로 가능한 muscle activation hypotheses를 추론하는 probabilistic inverse-biomechanics framework**

---

## 1. Project Overview

본 프로젝트의 목표는 2D real-time MRI(rtMRI) 또는 cine MRI 영상에서 관측되는 혀의 움직임을 이용해 다음을 추정하는 것이다.

1. 시간에 따라 변형되는 **3D tongue mesh motion**
2. 해당 3D motion을 생성할 수 있는 **muscle activation 또는 muscle synergy 후보들**
3. 근육 activation을 변화시켰을 때 혀의 3D 형상과 MRI contour가 어떻게 달라지는지 보여주는 **physics-grounded counterfactual explanation**

핵심 아이디어는 한 장 또는 소수 view의 MRI만으로 3D motion을 직접 결정하려 하지 않고, 미리 학습한 **3D tongue-motion prior**와 **biomechanical simulation library**를 활용하는 것이다.

```text
3D tongue-motion data
        ↓
3D motion latent prior 학습
        ↓
3D mesh를 MRI plane으로 slicing
        ↓
Synthetic 2D contour-motion ↔ 3D motion paired learning
        ↓
Real 2D MRI video에서 3D motion posterior 추론
        ↓
Predicted mesh를 다시 MRI plane으로 slicing
        ↓
Observed contour와 consistency loss로 adaptation
        ↓
3D motion에 대응하는 muscle activation Top-K 검색
        ↓
Biomechanical forward simulation 및 reranking
        ↓
Muscle activation posterior + counterfactual visualization
```

---

## 2. Core Research Question

관측된 2D MRI contour sequence를 다음과 같이 두자.

\[
C_{0:T}^{obs}
\]

환자의 reference tongue mesh는

\[
M_0=(V_0,F)
\]

이며, 시간별 3D mesh는

\[
M_t=(V_t,F)
\]

로 나타낸다. 모든 frame에서 topology와 vertex correspondence는 유지한다.

프로젝트가 추정하려는 것은 다음 조건부 분포이다.

\[
p(M_{0:T}, A_{0:T}\mid I_{0:T}, M_0, \pi)
\]

- \(I_{0:T}\): MRI video
- \(M_0\): subject-specific reference mesh
- \(\pi\): MRI slice 위치, 방향, pixel spacing, thickness
- \(M_{0:T}\): 3D tongue-motion sequence
- \(A_{0:T}\): muscle activation 또는 muscle-synergy sequence

단일 2D view에서는 서로 다른 3D motion이 동일하거나 유사한 contour를 만들 수 있다.

\[
\mathcal R_\pi(M^{(1)}_{0:T})
\approx
\mathcal R_\pi(M^{(2)}_{0:T})
\]

따라서 이 문제는 deterministic inverse mapping보다 **probabilistic reconstruction**으로 다루는 것이 적절하다.

---

## 3. Main Contributions to Target

최종 논문에서 목표로 하는 기술적 contribution은 다음과 같다.

### 3.1 Probabilistic 2D-to-3D Tongue Motion Reconstruction

Sparse 2D MRI video로부터 하나의 고정된 3D mesh가 아니라 가능한 3D tongue-motion hypotheses의 posterior를 추정한다.

### 3.2 Simulation-Supervised Cross-Modal Motion Prior

3D tongue-motion sequence에서 MRI acquisition geometry에 맞는 contour sequence를 자동 생성하고, 2D contour encoder와 3D motion encoder가 동일한 latent motion manifold를 공유하도록 학습한다.

### 3.3 Contour-Consistent Real-MRI Adaptation

예측된 3D mesh를 MRI plane으로 다시 slicing하고, 관측된 MRI contour와의 차이를 이용해 real data에 adaptation한다.

### 3.4 Retrieval-Augmented Inverse Biomechanics

예측된 3D motion과 유사한 biomechanical simulations를 검색하고, 대응하는 muscle activation 후보를 forward simulation으로 검증 및 refinement한다.

### 3.5 Physics-Grounded Explainability

근육 activation을 중간 concept로 사용하고, 특정 activation을 조절했을 때 biomechanical model이 생성하는 mesh와 MRI contour의 변화를 보여주는 counterfactual explanation을 제공한다.

---

## 4. Terminology and Claim Boundaries

### 권장 표현

- estimated muscle activation
- effective muscle-control coefficients
- muscle-synergy posterior
- biomechanically plausible activation hypothesis
- probabilistic 3D tongue-motion reconstruction
- contour-consistent 3D motion estimate
- model-based counterfactual explanation

### 피해야 할 표현

- true muscle activation from MRI
- ground-truth activation recovered from a single 2D view
- exact 3D motion reconstructed from one contour
- MRI directly measures muscle activation

MRI가 직접 관측하는 것은 anatomy, motion, deformation, strain에 가깝다. 실제 muscle activation은 biomechanical model과 prior를 통해 **추론되는 latent cause**이다.

### Ground Truth 사용 기준

| 데이터 출처 | 권장 명칭 |
|---|---|
| FEM에서 activation을 입력해 생성한 mesh/contour | Synthetic ground truth |
| 원본 3D mesh를 slicing해 만든 contour | Synthetic paired supervision |
| 독립적인 3D/4D MRI 또는 tagged MRI에서 얻은 motion | Reference standard |
| 실제 2D MRI에서 모델이 출력한 mesh | Prediction / reconstruction estimate |
| 검색 또는 최적화로 얻은 activation | Inferred activation hypothesis |

---

## 5. Recommended Model Architecture

전체 모델은 modality-specific encoder와 shared motion decoder로 구성한다.

```text
3D Motion Sequence ── Mesh Motion Encoder ──┐
                                            │
Synthetic Contour ── Contour Encoder ───────┼── Shared Motion Latent z
                                            │
Real MRI Video ── MRI Video Encoder ────────┘
                                                    ↓
                                      Subject-Conditioned Motion Decoder
                                                    ↓
                                       3D Tongue-Motion Sequence
```

### 5.1 Inputs

- MRI video \(I_{0:T}\)
- MRI contour sequence \(C_{0:T}\)
- reference mesh \(M_0\)
- DICOM-derived slice geometry \(\pi\)
- optional jaw, hyoid, palate geometry
- optional subject metadata

### 5.2 Outputs

- 3D vertex displacement sequence

\[
U_t=V_t-V_0
\]

- reconstructed mesh sequence

\[
\hat V_t=V_0+\hat U_t
\]

- latent posterior

\[
q(z\mid I_{0:T},M_0,\pi)
\]

- activation posterior or Top-K hypotheses

\[
p(A_{0:T}\mid I_{0:T})
\]

---

## 6. Why Predict Displacement Instead of Generating a New Mesh

전체 mesh를 frame마다 새로 생성하는 방식보다 reference mesh의 displacement를 예측하는 것이 적합하다.

```text
Reference mesh M0
        +
Predicted displacement ΔV0:T
        ↓
Subject-specific 3D motion M0:T
```

장점:

- 환자의 기본 anatomy 유지
- 모든 frame에서 vertex ID 유지
- topology 고정
- material correspondence 유지
- temporal tracking과 strain 계산 가능
- activation-to-motion 비교가 쉬움
- mesh generation artifact 감소

---

## 7. Training Pipeline

## Phase A. Learn the 3D Tongue-Motion Manifold

먼저 contour나 MRI 없이 3D mesh-motion sequence만으로 VAE 또는 autoencoder를 학습한다.

\[
q_M(z\mid U_{0:T})
\]

\[
\hat U_{0:T}=D(z,M_0)
\]

```text
3D mesh-motion sequence
        ↓
Mesh Motion Encoder
        ↓
Motion latent z
        ↓
Subject-Conditioned Motion Decoder
        ↓
Reconstructed 3D mesh-motion sequence
```

### Phase A Loss

\[
\mathcal L_A =
\lambda_v\mathcal L_{vertex}
+\lambda_n\mathcal L_{normal}
+\lambda_e\mathcal L_{edge}
+\lambda_l\mathcal L_{laplacian}
+\lambda_t\mathcal L_{temporal}
+\beta\mathcal L_{KL}
\]

권장 temporal regularization:

\[
\mathcal L_{velocity}
=
\sum_t \|V_t-V_{t-1}\|^2
\]

\[
\mathcal L_{acceleration}
=
\sum_t \|V_t-2V_{t-1}+V_{t-2}\|^2
\]

주기적 발화 또는 반복 동작이라면 cycle closure를 추가할 수 있다.

\[
\mathcal L_{cycle}=\|V_T-V_0\|^2
\]

---

## Phase B. Synthetic Contour-to-3D Motion Pretraining

3D motion sequence를 실제 MRI acquisition plane과 같은 방식으로 slicing한다.

\[
C_{0:T}^{syn}=\mathcal R_\pi(M_{0:T})
\]

이렇게 자동으로 생성된 contour와 원본 3D motion을 paired data로 사용한다.

```text
3D mesh-motion sequence
        ↓ Randomized MRI slicing
Synthetic contour-motion sequence
        ↓ Contour Temporal Encoder
Motion posterior z
        ↓ Shared Motion Decoder
Predicted 3D tongue motion
```

이 단계는 엄밀히는 pure self-supervised learning이라기보다 다음 표현이 적절하다.

> **Simulation-supervised cross-modal pretraining with automatically generated contour–mesh pairs**

### Phase B Loss

\[
\mathcal L_B=
\lambda_{3D}\mathcal L_{mesh}
+\lambda_c\mathcal L_{contour}
+\lambda_z\mathcal L_{posterior-align}
+\lambda_t\mathcal L_{temporal}
+\beta\mathcal L_{KL}
\]

#### Mesh Reconstruction

\[
\mathcal L_{mesh}
=
\sum_t\|\hat V_t-V_t^{GT}\|_1
\]

#### Contour Consistency

예측 mesh를 동일한 MRI plane으로 다시 slicing한다.

\[
\hat C_t=\mathcal R_\pi(\hat M_t)
\]

\[
\mathcal L_{contour}
=
d_{SDF}(\hat C_t,C_t^{syn})
\]

명시적인 contour point index correspondence보다 다음을 권장한다.

- signed-distance-field loss
- symmetric Chamfer distance
- soft occupancy loss
- Dice loss for slice mask

#### Cross-Modal Posterior Alignment

\[
\mathcal L_{posterior-align}
=
D_{SKL}
\left(
q_C(z\mid C),
q_M(z\mid U)
\right)
\]

동일한 motion에서 나온 2D contour와 3D mesh가 유사한 latent posterior를 갖게 한다.

---

## Phase C. Real MRI Adaptation

실제 MRI video를 입력으로 받아 3D motion을 예측한다.

\[
q_I(z\mid I_{0:T},M_0,\pi)
\]

\[
\hat M_{0:T}=M_0+D(z,M_0)
\]

예측 mesh를 다시 실제 MRI plane으로 slicing한다.

\[
\hat C_{0:T}=\mathcal R_\pi(\hat M_{0:T})
\]

그리고 MRI에서 추출한 contour와 비교한다.

```text
Real MRI Video
       ↓
MRI Temporal Encoder
       ↓
Motion posterior z
       ↓
Pretrained Motion Decoder
       ↓
Predicted 3D tongue motion
       ↓ Differentiable MRI slicing
Predicted 2D contour
       ↓
Observed MRI contour와 consistency loss
```

### Decoder Freeze Strategy

실제 MRI에서 contour loss만 사용해 decoder 전체를 업데이트하면, 관측 plane 밖의 3D shape가 비현실적으로 변할 수 있다.

권장 순서:

1. Motion decoder 전체 freeze
2. MRI encoder만 학습
3. 필요할 경우 small adapter 또는 decoder 마지막 layer만 낮은 learning rate로 조정
4. 독립적인 real 3D supervision이 있을 때만 전체 decoder fine-tuning

### Phase C Loss

\[
\begin{aligned}
\mathcal L_C={}&
\lambda_c d_{SDF}(\hat C,C^{obs})\\
&+\lambda_{teacher}D(q_I,q_C)\\
&+\lambda_{prior}D_{KL}(q_I\|p(z))\\
&+\lambda_t\mathcal L_{temporal}\\
&+\lambda_r\mathcal L_{mesh-reg}\\
&+\lambda_{aux}\mathcal L_{auxiliary}
\end{aligned}
\]

Contour encoder를 teacher로, MRI encoder를 student로 사용할 수 있다.

```text
Observed contour → Frozen Contour Encoder → teacher posterior
Real MRI video  → MRI Encoder            → student posterior
```

MRI encoder가 영상의 intensity와 내부 texture를 활용하도록 다음 auxiliary task를 함께 사용할 수 있다.

- tongue segmentation
- boundary heatmap prediction
- in-plane optical flow
- masked video reconstruction
- temporal correspondence prediction
- future-frame prediction

---

## 8. MRI Slice Operator

MRI에서 필요한 것은 일반적인 camera projection이 아니라 3D mesh와 acquisition plane의 교차 또는 finite-thickness slab rendering이다.

### 8.1 Geometry Inputs

- Image Position Patient
- Image Orientation Patient
- Pixel Spacing
- Slice Thickness
- patient-to-mesh transform

2D pixel \((c,r)\)은 slice plane의 3D point로 변환할 수 있다.

\[
q(c,r)=S+c\Delta_cX+r\Delta_rY
\]

### 8.2 Recommended Differentiable Implementations

#### Soft Slab Rasterization

Plane으로부터의 거리를 이용해 soft occupancy를 계산한다.

\[
w(x)=\exp\left(-\frac{d_\pi(x)^2}{2\sigma^2}\right)
\]

#### Implicit SDF Sampling

\[
\phi_t(c,r)=SDF_{M_t}(q(c,r))
\]

\[
\hat S_t(c,r)=\sigma(-\alpha\phi_t(c,r))
\]

Explicit triangle-plane intersection보다 optimization gradient가 안정적일 수 있다.

---

## 9. Synthetic-to-Real Domain Randomization

Synthetic contour는 실제 MRI보다 너무 깨끗하므로, MRI observation process 자체를 randomize해야 한다.

```python
plane.position += random_translation()
plane.orientation += random_rotation()
plane.slice_thickness = random_thickness()
contour = add_spatial_noise(contour)
contour = drop_random_segments(contour)
contour = temporal_blur(contour)
contour = random_resample(contour)
contour = add_segmentation_bias(contour)
```

권장 augmentation:

- slice position variation
- slice orientation variation
- slice thickness variation
- partial contour dropout
- boundary ambiguity
- temporal sampling variation
- motion blur
- segmentation noise
- missing frames
- tongue-palate contact로 인한 contour 소실
- global rigid motion
- variable MRI resolution

---

## 10. VAE vs VQ-VAE

### Recommended Baseline: Conditional VAE

3D tongue motion은 연속적이며, 2D 관측의 ambiguity도 표현해야 하므로 continuous latent가 첫 모델에 적합하다.

장점:

- multiple plausible motions sampling
- uncertainty propagation
- smooth temporal interpolation
- latent refinement 가능

### VQ-VAE Extension

VQ-VAE는 반복되는 motion pattern 또는 articulatory gesture를 discrete token으로 표현하는 데 유용할 수 있다.

하지만 frame 단위의 hard quantization은 code switching과 motion snapping을 유발할 수 있다.

권장 구조:

\[
\Delta V_t=D(M_0,q_t,r_t)
\]

- \(q_t\): discrete gesture 또는 muscle-synergy token
- \(r_t\): continuous residual

```text
Discrete gesture/synergy code
            +
Continuous motion residual
            ↓
Smooth 3D tongue motion
```

초기 연구에서는 다음 순서를 권장한다.

1. Deterministic autoencoder baseline
2. Conditional VAE
3. VQ-VAE
4. VQ token + continuous residual

---

## 11. Activation Retrieval and Inverse Biomechanics

## 11.1 Biomechanical Simulation Library

activation이 알려진 FEM 또는 differentiable biomechanical simulation library를 만든다.

\[
\mathcal D_A=
\{A_j,\eta_j,M_j\}_{j=1}^{N}
\]

- \(A_j\): muscle activation sequence
- \(\eta_j\): material property, fiber orientation, attachment condition
- \(M_j=\mathcal B(M_0,A_j,\eta_j)\): simulated volumetric tongue motion

라이브러리는 activation뿐 아니라 다음 변이를 포함해야 한다.

- subject anatomy
- tissue stiffness
- fiber orientation
- jaw and hyoid pose
- palate contact
- attachment locations
- co-contraction
- activation timing and strength

## 11.2 Motion Retrieval

예측 3D motion의 embedding을 이용해 가장 유사한 simulation을 검색한다.

```text
Predicted 3D motion
        ↓ Motion Encoder
Query embedding
        ↓ Approximate nearest-neighbor search
Top-K simulated motions
        ↓
Corresponding activation hypotheses
```

단일 Top-1 activation보다 Top-K posterior를 반환한다.

\[
p(A,\eta\mid \hat M)
\]

## 11.3 Physics-Based Reranking

검색된 activation 후보를 biomechanical simulator에 다시 입력한다.

\[
M_j^{sim}=\mathcal B(M_0,A_j,\eta_j)
\]

후보 score:

\[
\begin{aligned}
s_j={}&
\lambda_{3D}d_{3D}(M_j^{sim},\hat M)\\
&+\lambda_{2D}d_{SDF}
(\mathcal R_\pi(M_j^{sim}),C^{obs})\\
&+\lambda_eR_{effort}(A_j)\\
&+\lambda_tR_{temporal}(A_j)
\end{aligned}
\]

## 11.4 Activation Refinement

retrieval 결과를 초기값으로 사용해 activation을 최적화한다.

\[
(A^*,\eta^*)=
\arg\min_{A,\eta}
\left[
 d_{3D}(\mathcal B(M_0,A,\eta),\hat M)
 +\lambda_c d_{2D}(\mathcal R_\pi(\mathcal B(M_0,A,\eta)),C^{obs})
 +R(A,\eta)
\right]
\]

FEM이 미분 가능하지 않다면 다음 대안이 있다.

- differentiable FEM
- adjoint differentiation
- FEM surrogate network
- finite differences
- derivative-free optimization

---

## 12. Why Use a Volumetric Mesh

근육 activation과 내부 strain을 논하려면 단순 surface mesh만으로는 부족하다.

권장 representation:

```text
Tetrahedral volumetric tongue mesh
+ anatomically labeled muscle regions
+ muscle fiber directions
+ active-stress 또는 active-strain model
+ near-incompressibility
+ palate contact
+ jaw/hyoid boundary conditions
```

activation 검색에 사용할 feature:

\[
f(M)=
[
\text{surface displacement},
\text{internal displacement},
\text{velocity},
\text{strain},
\text{fiber-direction strain},
\text{contact}
]
\]

---

## 13. Uncertainty Propagation

MRI posterior에서 여러 motion sample을 생성한다.

\[
z^{(s)}\sim q_I(z\mid I),\quad s=1,\ldots,S
\]

\[
M^{(s)}=D(z^{(s)},M_0)
\]

각 motion sample에 대해 activation 후보를 검색하고 통합한다.

```python
all_candidates = []

for z_sample in image_posterior.sample(num_samples):
    motion_sample = motion_decoder(z_sample, reference_mesh)
    candidates = activation_index.retrieve_top_k(motion_sample)
    all_candidates.extend(candidates)

activation_posterior = aggregate_and_calibrate(all_candidates)
```

최종 출력 예시:

```text
Muscle-synergy hypothesis A: 0.61
Muscle-synergy hypothesis B: 0.24
Alternative co-contraction pattern: 0.15
```

이 확률은 생리학적 진실의 확률이 아니라, 현재 model, prior, simulator와 관측에 기반한 posterior이다.

---

## 14. Explainability Strategy

단순히 mesh 또는 근육 부위를 색칠하는 것은 visualization이지 충분한 XAI는 아니다.

본 프로젝트에서는 activation을 **intervenable anatomical concept bottleneck**으로 사용한다.

```text
MRI
 ↓
3D motion posterior
 ↓
Muscle activation posterior
 ↓
Biomechanical forward model
 ↓
3D tongue motion
 ↓
MRI contour / constriction outcome
```

### Counterfactual Intervention

```python
activation_cf = activation.copy()
activation_cf["target_muscle"] *= 0.5

motion_cf = biomechanical_forward(
    reference_mesh,
    activation_cf,
    material_parameters,
)

contour_cf = slice_mesh(
    motion_cf,
    mri_geometry,
)
```

설명 예시:

> 선택한 muscle-control component를 줄이면, 현재 biomechanical model에서는 tongue dorsum 상승과 특정 constriction이 감소한다.

검증 전에는 이를 **model-based counterfactual**이라고 부른다. 실제 생리적 causal effect라고 주장하려면 EMG, tagged MRI, 또는 별도의 실험적 검증이 필요하다.

---

## 15. Full Training Pseudocode

```python
# ==================================================
# Phase A: Learn 3D tongue-motion prior
# ==================================================

for mesh_sequence, reference_mesh in motion_dataset:
    displacement_gt = (
        mesh_sequence.vertices
        - reference_mesh.vertices
    )

    mesh_posterior = mesh_motion_encoder(
        displacement_gt,
        reference_mesh,
    )

    z_mesh = mesh_posterior.rsample()

    displacement_pred = motion_decoder(
        z_mesh,
        reference_mesh,
    )

    loss = (
        vertex_loss(displacement_pred, displacement_gt)
        + lambda_normal * normal_loss(displacement_pred, displacement_gt)
        + lambda_edge * edge_loss(displacement_pred, displacement_gt)
        + lambda_lap * laplacian_loss(displacement_pred)
        + lambda_temporal * acceleration_loss(displacement_pred)
        + beta * kl_loss(mesh_posterior)
    )

    optimize(loss)


# ==================================================
# Phase B: Synthetic contour-to-3D pretraining
# ==================================================

freeze_or_partially_freeze(motion_decoder)

for mesh_sequence, reference_mesh in motion_dataset:
    slice_geometry = sample_random_mri_geometry()

    contour_sequence = differentiable_slice_contour(
        mesh_sequence,
        slice_geometry,
    )

    contour_sequence = simulate_mri_observation_noise(
        contour_sequence,
    )

    contour_posterior = contour_encoder(
        contour_sequence,
        slice_geometry,
        reference_mesh,
    )

    with no_grad():
        mesh_posterior = mesh_motion_encoder(
            mesh_sequence.vertices - reference_mesh.vertices,
            reference_mesh,
        )

    z_contour = contour_posterior.rsample()

    displacement_pred = motion_decoder(
        z_contour,
        reference_mesh,
    )

    mesh_pred = reference_mesh + displacement_pred

    contour_pred = differentiable_slice_contour(
        mesh_pred,
        slice_geometry,
    )

    loss = (
        lambda_3d * mesh_reconstruction_loss(
            mesh_pred,
            mesh_sequence,
        )
        + lambda_contour * contour_sdf_loss(
            contour_pred,
            contour_sequence,
        )
        + lambda_latent * posterior_alignment_loss(
            contour_posterior,
            mesh_posterior,
        )
        + lambda_temporal * acceleration_loss(
            displacement_pred,
        )
        + beta * kl_loss(contour_posterior)
    )

    optimize(loss)


# ==================================================
# Phase C: Real MRI adaptation
# ==================================================

freeze(motion_decoder)
freeze(contour_encoder)

for sample in real_mri_dataset:
    mri_video = sample.mri_video
    observed_contours = sample.contours
    geometry = sample.slice_geometry
    reference_mesh = sample.reference_mesh

    image_posterior = mri_video_encoder(
        mri_video,
        geometry,
        reference_mesh,
    )

    with no_grad():
        contour_teacher = contour_encoder(
            observed_contours,
            geometry,
            reference_mesh,
        )

    z_image = image_posterior.rsample()

    displacement_pred = motion_decoder(
        z_image,
        reference_mesh,
    )

    mesh_pred = reference_mesh + displacement_pred

    contour_pred = differentiable_slice_contour(
        mesh_pred,
        geometry,
    )

    loss = (
        lambda_contour * contour_sdf_loss(
            contour_pred,
            observed_contours,
        )
        + lambda_teacher * posterior_alignment_loss(
            image_posterior,
            contour_teacher,
        )
        + lambda_prior * kl_loss(image_posterior)
        + lambda_temporal * acceleration_loss(
            displacement_pred,
        )
        + lambda_seg * auxiliary_segmentation_loss(
            mri_video,
            observed_contours,
        )
    )

    optimize(loss)


# ==================================================
# Phase D: Activation retrieval and refinement
# ==================================================

motion_samples = []

for z_sample in image_posterior.sample(num_motion_samples):
    motion_samples.append(
        reference_mesh
        + motion_decoder(z_sample, reference_mesh)
    )

activation_solutions = []

for motion_sample in motion_samples:
    candidates = activation_database.retrieve_top_k(
        motion_embedding(motion_sample),
        k=20,
    )

    for candidate in candidates:
        activation_refined, material_refined = optimize_activation(
            initial_activation=candidate.activation,
            initial_material=candidate.material,
            objective=lambda activation, material: (
                lambda_motion * motion_distance(
                    biomechanical_forward(
                        reference_mesh,
                        activation,
                        material,
                    ),
                    motion_sample,
                )
                + lambda_contour * contour_sdf_loss(
                    differentiable_slice_contour(
                        biomechanical_forward(
                            reference_mesh,
                            activation,
                            material,
                        ),
                        geometry,
                    ),
                    observed_contours,
                )
                + lambda_effort
                * activation_effort_regularization(activation)
                + lambda_temporal
                * activation_temporal_regularization(activation)
            ),
        )

        activation_solutions.append({
            "activation": activation_refined,
            "material": material_refined,
        })

activation_posterior = rank_and_calibrate(
    activation_solutions,
)
```

---

## 16. Dataset Requirements

### 16.1 3D Motion Dataset

각 sequence는 다음 조건을 만족해야 한다.

- fixed topology
- vertex correspondence across time
- subject-level reference mesh
- normalized coordinate convention
- known or estimable rigid motion
- preferably volumetric tetrahedral mesh

### 16.2 Biomechanical Dataset

각 sample은 다음을 포함한다.

- reference anatomy
- activation sequence
- muscle labels
- fiber orientation
- material parameters
- boundary conditions
- simulated volumetric motion
- optional strain and contact information

### 16.3 Real MRI Dataset

- MRI video
- synchronized contour or segmentation
- DICOM slice geometry
- subject reference mesh
- optional orthogonal MRI views
- optional tagged MRI
- optional EMG
- speaker identity for subject-level split

---

## 17. Evaluation Protocol

2D contour reconstruction만으로 3D 정확성을 주장할 수 없다. 반드시 관측되지 않은 3D 정보에 대한 검증이 필요하다.

### 17.1 Synthetic 3D Evaluation

- vertex error
- surface Chamfer distance
- Hausdorff distance
- normal consistency
- volume error
- internal displacement error
- strain error
- temporal acceleration and jerk

### 17.2 Real-Data Evaluation

#### Held-Out View Evaluation

```text
Midsagittal MRI만 입력
        ↓
3D motion reconstruction
        ↓
Coronal 또는 axial contour 생성
        ↓
실제 held-out view와 비교
```

이는 single-view 입력으로 추정한 3D motion의 실제 cross-view consistency를 평가하는 강한 실험이다.

#### Tagged MRI Evaluation

- internal displacement
- deformation gradient
- Lagrangian strain
- fiber-direction strain

#### Activation Evaluation

- synthetic activation recovery
- unseen activation combination
- activation onset/offset timing
- waveform correlation
- synergy ranking
- forward-simulated motion consistency
- posterior calibration

### 17.3 Generalization Split

frame 단위가 아니라 subject 또는 speaker 단위로 분할한다.

```text
Train speakers
Validation speakers
Unseen test speakers
```

추가 평가:

- unseen speaker
- unseen phoneme
- unseen syllable
- different speaking rate
- unseen anatomy
- slice-pose perturbation
- pathology 또는 atypical articulation

---

## 18. Baselines

최소 비교 대상:

1. Nearest-neighbor motion retrieval
2. PCA / Statistical Shape Model
3. Contour-driven deformation graph
4. Contour-driven FEM
5. Deterministic autoencoder
6. Conditional VAE
7. VQ-VAE
8. VQ token + continuous residual
9. Image-to-mesh direct regression
10. Physics-free displacement decoder
11. Physics-grounded activation decoder
12. Proposed retrieval + biomechanical refinement

필수 ablation:

- without motion prior
- without contour consistency
- without posterior alignment
- without temporal regularization
- decoder frozen vs unfrozen
- single-view vs multi-view
- surface-only vs volumetric representation
- retrieval only vs retrieval + forward refinement
- deterministic output vs posterior sampling

---

## 19. CVPR Positioning

교육용 visualization 자체보다 다음 기술적 문제를 중심 contribution으로 둔다.

> **Uncertainty-aware inverse biomechanics from sparse MRI observations**

추천 framing:

> We introduce a probabilistic framework that reconstructs a subject-specific 3D tongue-motion sequence from sparse real-time MRI and infers biomechanically plausible muscle-control hypotheses through retrieval-augmented inverse simulation.

> The model is pretrained using automatically generated contour–mesh pairs, adapted to real MRI via differentiable slice consistency, and evaluated through held-out MRI views, volumetric motion references, and biomechanical forward validation.

> Anatomically named activation concepts support intervention-based, model-grounded counterfactual explanations for articulatory education.

### 예상 Weak-Reject 요인

- single-view ambiguity를 무시
- real 3D reference 부족
- activation을 true physiological signal처럼 표현
- contour reprojection만으로 3D accuracy 주장
- generic mesh만 사용하고 subject anatomy 미반영
- XAI가 heatmap visualization에 머묾
- 기존 contour-to-FEM 또는 image-to-mesh 접근과 차별성 불충분

### 경쟁력을 높이는 요소

- subject-specific volumetric mesh
- probabilistic posterior
- held-out view validation
- tagged MRI 또는 independent 3D reference
- activation forward-cycle validation
- intervention 가능한 muscle concept bottleneck
- 공개 가능한 multimodal benchmark

---

## 20. Educational Application

기술 모델 위에 다음 교육 interface를 구축할 수 있다.

```text
MRI video
+ observed 2D contour
+ reconstructed 3D tongue mesh
+ muscle/synergy posterior
+ counterfactual controls
```

교육적 설명 예시:

- 현재 발음에서 어느 tongue region이 어떻게 이동했는지
- 가능한 muscle synergy 후보
- activation을 감소 또는 증가시켰을 때 tongue shape 변화
- constriction location과 degree 변화
- 서로 다른 발음의 3D motion 비교

교육 효과는 모델 정확성과 별도로 검증한다.

가능한 A/B test:

```text
A: 2D MRI + contour only
B: 3D motion + activation posterior + counterfactual explanation
```

평가:

- articulatory anatomy 이해도
- place/manner 식별 정확도
- 새로운 phoneme에 대한 transfer
- retention
- 전문가가 평가한 anatomical plausibility

---

## 21. Proposed Repository Structure

```text
.
├── README.md
├── configs/
│   ├── phase_a_motion_vae.yaml
│   ├── phase_b_contour_pretrain.yaml
│   ├── phase_c_mri_adaptation.yaml
│   └── phase_d_activation_retrieval.yaml
├── data/
│   ├── motion_sequences/
│   ├── biomechanical_library/
│   ├── real_mri/
│   └── metadata/
├── models/
│   ├── mesh_motion_encoder.py
│   ├── contour_encoder.py
│   ├── mri_video_encoder.py
│   ├── motion_decoder.py
│   ├── activation_retriever.py
│   └── biomechanical_surrogate.py
├── geometry/
│   ├── dicom_geometry.py
│   ├── mesh_slice_operator.py
│   ├── sdf_renderer.py
│   └── mesh_regularization.py
├── training/
│   ├── train_motion_prior.py
│   ├── train_contour_encoder.py
│   ├── adapt_real_mri.py
│   └── train_biomechanical_surrogate.py
├── inference/
│   ├── reconstruct_motion.py
│   ├── retrieve_activation.py
│   └── counterfactual.py
├── evaluation/
│   ├── evaluate_3d_motion.py
│   ├── evaluate_heldout_view.py
│   ├── evaluate_activation.py
│   └── calibration.py
└── visualization/
    ├── render_mesh_motion.py
    ├── render_muscle_activation.py
    └── educational_viewer.py
```

---

## 22. Recommended Milestones

### Milestone 1. Deterministic Synthetic Baseline

- fixed-topology mesh-motion dataset 준비
- 3D motion autoencoder
- synthetic contour slicing
- contour-to-3D deterministic reconstruction
- synthetic hold-out 평가

### Milestone 2. Probabilistic Motion Model

- conditional VAE
- posterior sampling
- calibration
- temporal latent model

### Milestone 3. Real MRI Adaptation

- DICOM geometry 적용
- differentiable slice renderer
- contour consistency fine-tuning
- decoder freeze 전략 검증
- held-out view evaluation

### Milestone 4. Biomechanical Retrieval

- activation simulation library 구축
- motion embedding retrieval
- Top-K activation posterior
- physics-based reranking

### Milestone 5. Inverse Refinement and XAI

- activation refinement
- material-parameter marginalization
- forward-cycle validation
- counterfactual intervention viewer

### Milestone 6. Paper-Level Validation

- unseen-speaker split
- tagged MRI 또는 independent 3D reference
- activation timing validation
- educational user study
- full ablation and baseline study

---

## 23. Minimal Viable Experiment

첫 논리 검증은 다음처럼 단순화한다.

```text
1. FEM 또는 기존 3D motion dataset에서 mesh sequence 수집
2. 동일 topology로 정규화
3. Random MRI plane에서 synthetic contour sequence 생성
4. Contour Temporal Encoder + Conditional VAE Decoder 학습
5. Synthetic test에서 3D reconstruction error 측정
6. 같은 input contour에서 posterior sample들의 out-of-plane variation 분석
7. Activation library에서 Top-K 검색
8. 검색 activation의 forward simulation이 query motion을 재현하는지 평가
```

이 단계에서 증명해야 할 핵심 질문:

1. 2D contour motion이 3D motion latent를 어느 정도 식별하는가?
2. VAE posterior가 실제 ambiguity를 반영하는가?
3. contour consistency가 3D reconstruction을 개선하는가?
4. activation retrieval이 단순 nearest-neighbor보다 나은가?
5. forward biomechanical refinement가 activation 및 motion 정확도를 높이는가?

---

## 24. Final Summary

본 프로젝트의 핵심은 다음 세 요소의 결합이다.

\[
\boxed{
\text{Simulation-supervised 3D motion prior}
+
\text{Contour-consistent real-MRI adaptation}
+
\text{Retrieval-augmented inverse biomechanics}
}
\]

최종 파이프라인:

```text
Real 2D MRI video
        ↓
Probabilistic MRI motion encoder
        ↓
3D tongue-motion posterior
        ↓
Differentiable MRI contour consistency
        ↓
Top-K muscle activation retrieval
        ↓
Biomechanical forward simulation
        ↓
Activation refinement and posterior
        ↓
Counterfactual 3D articulatory explanation
```

가장 중요한 해석은 다음과 같다.

> 본 시스템은 MRI에서 실제 muscle activation을 직접 측정하는 모델이 아니다. Sparse MRI observation과 학습된 3D motion prior, biomechanical simulator를 결합해 관측과 일관되는 3D tongue motion 및 muscle-control hypotheses를 추론하는 모델이다.

이 점을 명확히 유지하면서 3D reference, held-out MRI views, tagged MRI, forward biomechanics로 검증한다면, 음성해부학·의료영상·3D vision·physics-based XAI를 연결하는 경쟁력 있는 연구가 될 수 있다.
