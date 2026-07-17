#!/usr/bin/env bash
# ============================================================================
# run_fem_db_full.sh — FEM 궤적 DB 생성 (조합 최대 커버리지 버전)
#
#   combo_max_size = 11  → 비어있지 않은 근육 부분집합 2047개를 **전부** 열거
#   (= 가능한 모든 근육 조합. 속도보다 조합 다양성 우선.)
#
# 사용:
#   bash scripts/run_fem_db_full.sh              # 전체 파이프라인
#   bash scripts/run_fem_db_full.sh sample       # 특정 단계만 (characterize|sample|run|index)
#
# ▼▼▼ 값은 아래 여기서 직접 수정하세요 (환경변수 필요 없음) ▼▼▼
# ============================================================================
set -euo pipefail

# --- 설정 (이 값들을 직접 고쳐서 쓰세요) ------------------------------------
PY=python                          # 파이썬 실행기 (python / python3)
GEN=datasets/build_fem_db.py       # 생성기 경로 (repo 루트 기준)
OUTDIR=datasets/fem_db_full        # 산출 폴더 (기존 fem_db 와 분리)

COMBO_MAX_SIZE=11                  # 11 = 모든 부분집합 2047개 전수 (최대 조합)
COMBO_REPS=4                       # 조합당 진폭 변이 수
N_GLOBAL=20000                     # 랜덤 co-activation (고차·진폭 다양성)
MAX_ACTIVE=5                       # walk 최대 동시활성 근육 수
N_KEYFRAMES=3                      # 궤적당 통과 settled 수 (K)
N_FRAMES=200                       # 궤적당 프레임 수 T (frame-spacing 사용 시 상한)
FRAME_SPACING=0.05                 # 프레임 간 목표 Σ|Δactivation| (0=끔; >0이면 T 자동 결정)
PCA_K=48                           # PCA 성분 수 (k)
AMAX=0.5                           # 활성 공통 상한(근육당)
EFFORT_BUDGET=1.8                  # 총 활성 budget (근육 많을수록 각 근육 낮게 → 고차 조합 feasible)
MAX_NRAMP=300                      # forward 재시도 NRAMP 상한 (초과 시 포기 → doomed 조합 빨리 판정)
PILOT_LIMIT=15                     # 파일럿 궤적 수 (0=파일럿 건너뛰기)
SEED=0
# ---------------------------------------------------------------------------

# --- repo 루트로 이동 (스크립트가 scripts/ 안에 있어도 동작) -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

mkdir -p "$OUTDIR"
LOG="$OUTDIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo " FEM 궤적 DB (조합 최대) | $(date)"
echo " root=$ROOT  OUTDIR=$OUTDIR"
echo " combo_max_size=$COMBO_MAX_SIZE (2047개 전수) combo_reps=$COMBO_REPS"
echo " n_global=$N_GLOBAL max_active=$MAX_ACTIVE K=$N_KEYFRAMES T=$N_FRAMES amax=$AMAX"
echo " log=$LOG"
echo "=========================================================="

STAGE="${1:-all}"

do_characterize() {
  echo; echo "### [1/4] characterize — feasible cap 스캔 + 길항/중복 행렬"
  "$PY" "$GEN" --stage characterize --outdir "$OUTDIR"
}

do_sample() {
  echo; echo "### [2/4] sample — anchor + 조합 전수(≤$COMBO_MAX_SIZE) + walk"
  "$PY" "$GEN" --stage sample --outdir "$OUTDIR" \
    --combo-max-size "$COMBO_MAX_SIZE" --combo-reps "$COMBO_REPS" \
    --n-global "$N_GLOBAL" --max-active "$MAX_ACTIVE" \
    --n-keyframes "$N_KEYFRAMES" --amax "$AMAX" --effort-budget "$EFFORT_BUDGET" --seed "$SEED"
}

do_run() {
  if [ "$PILOT_LIMIT" -gt 0 ]; then
    echo; echo "### [3/4] run(파일럿) — 먼저 $PILOT_LIMIT 궤적으로 수율/용량 확인"
    "$PY" "$GEN" --stage run --outdir "$OUTDIR" --n-frames "$N_FRAMES" --frame-spacing "$FRAME_SPACING" --max-nramp "$MAX_NRAMP" --limit "$PILOT_LIMIT"
    echo ">>> 파일럿 완료. full/partial/fail·GB 확인. 본실행은 이어서 진행."
  fi
  echo; echo "### [3/4] run(본실행) — 나머지 전체 (중단돼도 재실행하면 이어감)"
  "$PY" "$GEN" --stage run --outdir "$OUTDIR" --n-frames "$N_FRAMES" --frame-spacing "$FRAME_SPACING" --max-nramp "$MAX_NRAMP"
}

do_index() {
  echo; echo "### [4/4] index — 프레임별 변위 PCA → 궤적 계수"
  "$PY" "$GEN" --stage index --outdir "$OUTDIR" --k "$PCA_K"
}

case "$STAGE" in
  characterize) do_characterize ;;
  sample)       do_sample ;;
  run)          do_run ;;
  index)        do_index ;;
  all)          do_characterize; do_sample; do_run; do_index ;;
  *) echo "알 수 없는 단계: $STAGE  (characterize|sample|run|index|all)"; exit 1 ;;
esac

echo; echo "=========================================================="
echo " 완료: $STAGE | $(date) | 로그: $LOG"
echo "=========================================================="
