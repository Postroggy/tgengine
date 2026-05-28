#!/bin/bash
# Overnight accuracy benchmark: DyGFormer + GraphMixer + TGN on wiki/reddit/lastfm
# Must run with CUDA_VISIBLE_DEVICES=0 (RTX 4080 only on scnu)
# Usage: nohup bash benchmarks/run_overnight.sh > logs/overnight_$(date +%Y%m%d_%H%M).log 2>&1 &

set -e

LOGDIR="$(dirname "$0")/../logs"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/overnight_${TIMESTAMP}.log"

echo "===================================================================" | tee -a "$LOG"
echo "  TGEngine Overnight Accuracy Benchmark  $(date)" | tee -a "$LOG"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
echo "===================================================================" | tee -a "$LOG"

# Check GPU
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "$LOG"
echo "" | tee -a "$LOG"

cd "$(dirname "$0")/.."

run_model() {
    local MODEL=$1
    local EPOCHS=$2
    local PATIENCE=$3
    echo "" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"
    echo "  MODEL: $MODEL  epochs=$EPOCHS  patience=$PATIENCE  $(date)" | tee -a "$LOG"
    echo "================================================================" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 python benchmarks/bench_accuracy.py \
        --model "$MODEL" \
        --datasets wiki reddit lastfm \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --device cuda \
        2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "  [DONE] $MODEL at $(date)" | tee -a "$LOG"
}

# DyGFormer: the main model with all fixes
run_model dygformer 100 20

# GraphMixer: fixed time encoding + unmasked mean
run_model graphmixer 100 20

# TGN: fixed message direction + delta time
run_model tgn 50 15

echo "" | tee -a "$LOG"
echo "===================================================================" | tee -a "$LOG"
echo "  ALL DONE  $(date)" | tee -a "$LOG"
echo "===================================================================" | tee -a "$LOG"
echo "  Log saved to: $LOG" | tee -a "$LOG"
