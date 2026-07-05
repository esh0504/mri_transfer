# mri_transfer

MRI 혀 분할 마스크 → 3D 메쉬 retarget (Stage 1) → ArtiSynth FEM forward 시뮬레이션 (Stage 2) 파이프라인.

Docker 이미지에 JDK 21, ArtiSynth, Python 의존성이 포함되어 있습니다.

---

## 1. Docker 컨테이너 설치

저장소 루트(`docker-compose.yml`이 있는 디렉터리)에서 실행합니다.

```bash
cp docker.env.example .env
```

```bash
docker compose build
```

```bash
docker compose up -d workspace
```

컨테이너에 접속:

```bash
docker compose exec workspace bash
```

중지 / 재시작:

```bash
docker compose stop workspace
docker compose start workspace
docker compose exec workspace bash
```

완전히 내릴 때:

```bash
docker compose down
```

| 항목 | 값 |
|------|-----|
| Compose project | `xai` |
| 이미지 | `xai/mri_transfer:latest` |
| 컨테이너 | `xai_mri_transfer` |
| 작업 디렉터리 (컨테이너 내부) | `/workspace` |

---

## 2. 데이터셋 다운로드

데이터셋은 저장소에 포함되지 않습니다. Hugging Face에서 별도로 받습니다.

- **데이터셋:** https://huggingface.co/datasets/SeunghoEum/mri-tongue-dataset  
- gated 데이터셋이므로 사전에 접근 권한을 요청·승인받아야 합니다.
- 컨테이너 안에서 Hugging Face 로그인: `hf auth login`

컨테이너 내부 (`/workspace`):

```bash
cd datasets && bash ./dataset_download.sh
```

DICOM 없이 마스크·메쉬만 받을 때:

```bash
cd datasets && bash ./dataset_download.sh --skip-dicom
```

다운로드 후 레이아웃:

```
datasets/
  GT_Segmentations/Subject{1-5}/mask_*.mat
  MRI_SSFP_10fps/Subject{1-5}/image_*.dcm
  tongue_model/tongue_rest_m.obj
```

---

## 3. 파이프라인 실행 (`main.py`)

컨테이너 내부 `/workspace`에서 실행합니다.

### 전체 파이프라인 (retarget + FEM)

```bash
python main.py
```

### Stage별 실행

```bash
# Stage 1 — retarget (2D 마스크 → 3D 메쉬)
python main.py stage=retarget

# Stage 2 — FEM (ArtiSynth forward)
python main.py stage=fem
```

### 예시

```bash
# 근육 활성값 변경 (11D: GGP, GGM, GGA, STY, GH, MH, HG, VERT, TRANS, IL, SL)
python main.py stage=fem artisynth.activations.HG=0.3

# 특정 마스크 프레임만 retarget
python main.py stage=retarget paths.target_mask=datasets/GT_Segmentations/Subject3/mask_51.mat

# 마스크 폴더 전체 시퀀스 retarget
python main.py stage=retarget paths.target_mask=datasets/GT_Segmentations/Subject3
```

### 출력 (`_test_out/`)

| 파일 | 설명 |
|------|------|
| `registration.csv` | 2D↔3D 앵커 매핑 |
| `tongue_lift_3d.npy` | Lift된 3D 마스크 시퀀스 |
| `retargeted.obj` | Retarget된 3D 메쉬 |
| `retargeted_objs/frame_*.obj` | 시퀀스별 메쉬 (폴더 target 시) |
| `rest.png` / `rest.obj` | Rest pose |
| `fem.png` / `fem.obj` | FEM 변형 결과 |

---

## 프로젝트 구조

```
main.py              진입점 (Hydra)
configs/             YAML 설정
retarget/            Stage 1: register, lift, retarget
artisynth/           Stage 2: JPype + ArtiSynth FEM
modules/             I/O, 시각화, 경로
datasets/            데이터셋 (다운로드 후)
```
