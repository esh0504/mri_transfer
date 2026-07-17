#!/usr/bin/env bash
# ============================================================================
# run_fem_db.sh — FEM 궤적 DB 생성 파이프라인 실행 스크립트
#
#   characterize → sample → run(파일럿) → run(본실행) → index
#
# 사용:
#   scripts/run_fem_db.sh              # 전체 파이프라인 (파일럿 후 본실행)
#   scripts/run_fem_db.sh characterize # 특정 단계만
#   scripts/run_fem_db.sh sample
#   scripts/run_fem_db.sh run
#   scripts/run_fem_db.sh index
#
# 상단 변수를 바꾸거나 환경변수로 덮어쓸 수 있음:
#   N_GLOBAL=8000 N_KEYFRAMES=4 scripts/run_fem_db.sh
# ============================================================================
set -euo pipefail

# --- 설정 (환경변수로 오버라이드 가능) --------------------------------------
PY="${PY:-python}"                         # 파이썬 실행기 (python / python3)
GEN="${GEN:-datasets/build_fem_db.py}"     # 생성기 경로 (repo 루트 기준)
OUTDIR="${OUTDIR:-datasets/fem_db}"        # 산출 폴더
N_GLOBAL="${N_GLOBAL:-4000}"               # 전역 chain 궤적 수 (M 의 대부분)
N_KEYFRAMES="${N_KEYFRAMES:-3}"            # 궤적당 통과 settled 수 (K)
N_FRAMES="${N_FRAMES:-48}"                 # 궤적 리샘플 프레임 수 (T)
PCA_K="${PCA_K:-48}"                       # PCA 성분 수 (k)
PILOT_LIMIT="${PILOT_LIMIT:-15}"           # 파일럿에서 돌릴 궤적 수 (0=건너뜀)
SEED="${SEED:-0}"

# --- repo 루트로 이동 (스크립트가 scripts/ 안에 있어도 동작) -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

mkdir -p "$OUTDIR"
LOG="$OUTDIR/run_$(date +%Y%m%d_%H%M%S).log"

# 모든 출력을 콘솔+로그 동시 기록
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo " FEM 궤적 DB 생성  |  $(date)"
echo " root=$ROOT"
echo " OUTDIR=$OUTDIR  N_GLOBAL=$N_GLOBAL  K=$N_KEYFRAMES  T=$N_FRAMES  k=$PCA_K"
echo " log=$LOG"
echo "=========================================================="

STAGE="${1:-all}"

do_characterize() {
  echo; echo "### [1/4] characterize — 단일근육 변위 + 길항/중복 행렬"
  "$PY" "$GEN" --stage characterize --outdir "$OUTDIR"
}

do_sample() {
  echo; echo "### [2/4] sample — keyframe 시퀀스 생성 (M x K x 11)"
  "$PY" "$GEN" --stage sample --outdir "$OUTDIR" \
    --n-global "$N_GLOBAL" --n-keyframes "$N_KEYFRAMES" --seed "$SEED"
}

do_run() {
  if [ "$PILOT_LIMIT" -gt 0 ]; then
    echo; echo "### [3/4] run(파일럿) — 먼저 $PILOT_LIMIT 궤적으로 속도/용량 측정"
    "$PY" "$GEN" --stage run --outdir "$OUTDIR" \
      --n-frames "$N_FRAMES" --limit "$PILOT_LIMIT"
    echo ">>> 파일럿 완료. 위의 초/it·GB 추정 확인. 본실행은 이어서 진행됩니다."
  fi
  echo; echo "### [3/4] run(본실행) — 나머지 전체 (중단돼도 재실행하면 이어감)"
  "$PY" "$GEN" --stage run --outdir "$OUTDIR" --n-frames "$N_FRAMES"
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
echo " 완료: $STAGE  |  $(date)  |  로그: $LOG"
echo "=========================================================="
