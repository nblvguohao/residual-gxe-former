#!/usr/bin/env bash
# =============================================================================
# DNNGP only — run on all 3 deployment-shift splits (3 seeds each = 9 tasks)
# with NaN detection, progress logging, and early stopping enabled.
# GPU0 only (DNNGP is small, one GPU is plenty).
# =============================================================================
set -euo pipefail

PYTHON=/home/amax/Downloads/yes/bin/python
D=/opt/data/lgh/gwas1/data/processed/g2f
SP=/opt/data/lgh/gwas1/outputs/v2_stage/splits
O=/opt/data/lgh/gwas1/outputs/ablation_study
SC=/opt/data/lgh/gwas1/scripts/06_run_ablations.py

export PATH="/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:$PATH"

echo "============================================"
echo "  DNNGP — all splits, seeds 2020/2021/2022"
echo "  $(date)"
echo "============================================"

# Kill old DNNGP processes
pkill -9 -f "dnngp" 2>/dev/null || true
sleep 1
find /opt/data/lgh/gwas1/src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$O"

TASK_COUNT=0
GPU=0  # single GPU, run sequentially to avoid contention

for split in leave_environment leave_ge leave_year; do
    TASK_COUNT=$((TASK_COUNT + 1))
    METRICS_FILE="$O/task_dnngp2_${split}.csv"
    LOGFILE="$O/task_dnngp2_${split}.log"

    # Skip if already done (3 seeds)
    if [ -f "$METRICS_FILE" ]; then
        seeds=$(tail -n +2 "$METRICS_FILE" 2>/dev/null | wc -l)
        if [ "$seeds" -ge 3 ]; then
            echo "  SKIP $split (already has $seeds seeds)"
            continue
        fi
    fi

    echo ""
    echo "=== DNNGP $split [GPU$GPU] ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SC" \
        --data-dir "$D" \
        --split-dir "$SP" \
        --out-dir "$O" \
        --device cuda \
        --model dnngp \
        --split-type "$split" \
        --metrics-file "$METRICS_FILE" \
        2>&1 | tee "$LOGFILE"
    echo "  exit=$?"
done

echo ""
echo "============================================"
echo "  DNNGP ALL DONE at $(date)"
echo "============================================"

# Print summary
echo ""
echo "=== Results ==="
for split in leave_environment leave_ge leave_year; do
    f="$O/task_dnngp2_${split}.csv"
    if [ -f "$f" ]; then
        echo -n "  $split: "
        tail -n +2 "$f" | wc -l
        echo " seeds"
        grep "pearson=" "$O/task_dnngp2_${split}.log" 2>/dev/null | tail -3 || true
    else
        echo "  $split: MISSING"
    fi
done
