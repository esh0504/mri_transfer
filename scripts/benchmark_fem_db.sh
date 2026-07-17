#!/usr/bin/env bash
# ============================================================================
# benchmark_fem_db.sh — 여러 세팅을 순서대로 시험 실행해 비교 (주말용)
#
#   각 세팅마다: sample(전체) → run(무작위 PILOT개, 시간 측정)
#   → 궤적당 시간 · 커버범위(수율 %) · 총 프레임 수(추정) 출력.
#
#   전체를 다 돌리지 않고 대표 PILOT 궤적만 재서 전체를 '추정'한다
#   (전체는 수만 궤적이라 몇 주 걸리므로).
#
# 사용:
#   bash scripts/benchmark_fem_db.sh
#
# 결과:
#   datasets/fem_bench/summary_<시각>.csv   ← 표 (엑셀에서 열기)
#   datasets/fem_bench/<label>/run.log      ← 세팅값 + 전체 로그
# ============================================================================
set -uo pipefail   # -e 는 안 씀: 한 세팅이 실패해도 다음 세팅 계속

# --- 공통 설정 (모든 세팅이 공유; 여기 직접 수정) --------------------------
PY=python
GEN=datasets/build_fem_db.py
BENCH_DIR=datasets/fem_bench
PILOT=50                    # 세팅당 시험 궤적 수 (많을수록 정확·오래 걸림)
# '전체 실행'의 규모(추정에 쓰임) — 실제로 나중에 돌릴 값과 맞추세요
COMBO_MAX_SIZE=11
COMBO_REPS=4
N_GLOBAL=20000
MAX_ACTIVE=5
K=3
FRAME_SPACING=0.05
N_FRAMES=200
SEED=0

# --- 시험할 세팅 목록: "label AMAX EFFORT_BUDGET MAX_NRAMP" -----------------
#     여기에 줄을 추가/수정하면 그 조합도 벤치마크됩니다.
CONFIGS=(
  "A_base       0.4 2.5 200"
  "B_fast       0.4 1.8 50"
  "C_faster     0.3 1.5 50"
  "D_balanced   0.4 2.0 100"
  "E_lowact     0.3 1.5 100"
  "F_hiyield    0.5 2.5 200"
)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
mkdir -p "$BENCH_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="$BENCH_DIR/summary_$STAMP.csv"
echo "label,amax,effort_budget,max_nramp,pilot_done,full,partial,fail,yield_pct,sec_per_traj,M_total,T,est_total_hours,est_total_days,est_total_frames,pilot_frames" > "$SUMMARY"

echo "=========================================================="
echo " FEM DB 세팅 벤치마크 | $(date)"
echo " PILOT=$PILOT  (공통) combo≤$COMBO_MAX_SIZE×$COMBO_REPS  n_global=$N_GLOBAL  K=$K  spacing=$FRAME_SPACING  T상한=$N_FRAMES"
echo " 세팅 수: ${#CONFIGS[@]}   summary: $SUMMARY"
echo "=========================================================="

printf "%-12s %5s %6s %6s | %6s %6s %5s | %7s %8s | %10s %8s\n" \
  "label" "amax" "budg" "nramp" "yield%" "s/traj" "done" "days" "Mtotal" "estFrames" "pilotFr"
echo "-------------------------------------------------------------------------------------------------"

for cfg in "${CONFIGS[@]}"; do
  read -r LABEL AMAX BUDGET NRAMP <<< "$cfg"
  OUT="$BENCH_DIR/$LABEL"
  rm -rf "$OUT"; mkdir -p "$OUT"
  LOG="$OUT/run.log"

  # --- 세팅값을 로그 맨 위에 기록 ---
  {
    echo "=========================================================="
    echo " 세팅 $LABEL | $(date)"
    echo "   AMAX=$AMAX  EFFORT_BUDGET=$BUDGET  MAX_NRAMP=$NRAMP  MAX_ACTIVE=$MAX_ACTIVE"
    echo "   COMBO_MAX_SIZE=$COMBO_MAX_SIZE  COMBO_REPS=$COMBO_REPS  N_GLOBAL=$N_GLOBAL"
    echo "   K=$K  FRAME_SPACING=$FRAME_SPACING  N_FRAMES=$N_FRAMES  PILOT=$PILOT  SEED=$SEED"
    echo "=========================================================="
  } | tee -a "$LOG"

  # --- sample (FEM 없음, 빠름) ---
  "$PY" "$GEN" --stage sample --outdir "$OUT" \
    --combo-max-size "$COMBO_MAX_SIZE" --combo-reps "$COMBO_REPS" \
    --n-global "$N_GLOBAL" --max-active "$MAX_ACTIVE" \
    --n-keyframes "$K" --amax "$AMAX" --effort-budget "$BUDGET" --seed "$SEED" >> "$LOG" 2>&1

  # --- run: 무작위 PILOT 궤적, 시간 측정 ---
  T0=$(date +%s)
  "$PY" "$GEN" --stage run --outdir "$OUT" \
    --n-frames "$N_FRAMES" --frame-spacing "$FRAME_SPACING" \
    --max-nramp "$NRAMP" --limit "$PILOT" --shuffle >> "$LOG" 2>&1
  T1=$(date +%s)
  WALL=$((T1 - T0))

  # --- 로그에서 결과 파싱 ---
  RUNLINE=$(grep 'run: M=' "$LOG" | tail -1)
  MVAL=$(echo "$RUNLINE" | sed -n 's/.*run: M=\([0-9]*\).*/\1/p')
  TVAL=$(echo "$RUNLINE" | sed -n 's/.*T=\([0-9]*\).*/\1/p')
  DONELINE=$(grep 'run done:' "$LOG" | tail -1)
  FULL=$(echo "$DONELINE" | sed -n 's/.*full=\([0-9]*\).*/\1/p')
  PART=$(echo "$DONELINE" | sed -n 's/.*partial=\([0-9]*\).*/\1/p')
  FAIL=$(echo "$DONELINE" | sed -n 's/.*fail=\([0-9]*\).*/\1/p')
  MVAL=${MVAL:-0}; TVAL=${TVAL:-0}; FULL=${FULL:-0}; PART=${PART:-0}; FAIL=${FAIL:-0}

  # --- 계산 (awk, 실수) ---
  read Y SP EH ED EF PF DONE < <(awk -v w="$WALL" -v f="$FULL" -v p="$PART" -v x="$FAIL" -v M="$MVAL" -v T="$TVAL" 'BEGIN{
    done=f+p+x; use=f+p;
    y  = done>0 ? 100*use/done : 0;
    sp = done>0 ? w/done : 0;
    et = sp*M;                       # 전체 예상 시간(초)
    ef = done>0 ? M*(use/done)*T : 0;# 전체 예상 usable 프레임
    pf = use*T;                      # 이번 PILOT 이 만든 프레임
    printf "%.1f %.2f %.2f %.2f %.0f %.0f %d", y, sp, et/3600.0, et/86400.0, ef, pf, done
  }')

  # --- 요약 CSV + 콘솔 ---
  echo "$LABEL,$AMAX,$BUDGET,$NRAMP,$DONE,$FULL,$PART,$FAIL,$Y,$SP,$MVAL,$TVAL,$EH,$ED,$EF,$PF" >> "$SUMMARY"
  printf "%-12s %5s %6s %6s | %5s%% %6s %5s | %6sd %8s | %10s %8s\n" \
    "$LABEL" "$AMAX" "$BUDGET" "$NRAMP" "$Y" "$SP" "$DONE" "$ED" "$MVAL" "$EF" "$PF"

  # --- 큰 파일 정리(디스크 절약; 요약/로그만 남김) ---
  rm -f "$OUT/verts.npy" "$OUT/activation.npy" "$OUT/keyframes.npy" 2>/dev/null || true
done

echo "-------------------------------------------------------------------------------------------------"
echo "완료. 요약표: $SUMMARY"
echo "  yield% = (full+partial)/done  (커버범위)"
echo "  days   = 전체 M 궤적 예상 소요일  |  estFrames = 전체 예상 usable 프레임(=M×yield×T)"
