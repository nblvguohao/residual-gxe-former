#!/usr/bin/env bash
# =============================================================================
# Run ablation study + DNNGP completion + Cropformer on 2 GPUs
# Server: amax@100.66.246.20
# =============================================================================
set -euo pipefail

PYTHON=/home/amax/Downloads/yes/bin/python
PROJ=/opt/data/lgh/gwas1
DATA=$PROJ/data/processed/g2f
SPLITS=$PROJ/data/processed/g2f/splits
OUT=$PROJ/outputs/ablation_study
SCRIPT=$PROJ/scripts/06_run_ablations.py

echo "Using python: $PYTHON"
echo "Script: $SCRIPT"
echo "Data: $DATA"
echo "Splits: $SPLITS"
echo "Output: $OUT"
echo "============================================"
echo "  Ablation Study Launch"
echo "  $(date)"
echo "============================================"

# Kill any existing ablation processes
pkill -9 -f "06_run_ablations" 2>/dev/null || true
sleep 2

# Clear pycache
find $PROJ/src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
find $PROJ/src -name "*.pyc" -delete 2>/dev/null || true

mkdir -p $OUT

# =============================================================================
# GPU 0: genotype_only + weather_only + cropformer
# =============================================================================
echo "=== Launching GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    export PATH=/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:\$PATH
    for split in leave_environment leave_ge leave_year; do
        echo \"[GPU0] genotype_only \$split\"
        $PYTHON $SCRIPT --data-dir $DATA --split-dir $SPLITS --out-dir $OUT --device cuda --skip-existing --model genotype_only --split-type \$split
        echo \"[GPU0] weather_only \$split\"
        $PYTHON $SCRIPT --data-dir $DATA --split-dir $SPLITS --out-dir $OUT --device cuda --skip-existing --model weather_only --split-type \$split
    done
    for split in leave_environment leave_ge leave_year; do
        echo \"[GPU0] cropformer \$split\"
        $PYTHON $SCRIPT --data-dir $DATA --split-dir $SPLITS --out-dir $OUT --device cuda --skip-existing --model cropformer --split-type \$split
    done
    echo '[GPU0] ALL DONE'
" > $OUT/gpu0.log 2>&1 &
echo "  GPU0 PID: $!"

# =============================================================================
# GPU 1: static_env_only + geno_env + dnngp
# =============================================================================
echo "=== Launching GPU 1 ==="
CUDA_VISIBLE_DEVICES=1 nohup bash -c "
    export PATH=/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:\$PATH
    for split in leave_environment leave_ge leave_year; do
        echo \"[GPU1] static_env_only \$split\"
        $PYTHON $SCRIPT --data-dir $DATA --split-dir $SPLITS --out-dir $OUT --device cuda --skip-existing --model static_env_only --split-type \$split
        echo \"[GPU1] geno_env \$split\"
        $PYTHON $SCRIPT --data-dir $DATA --split-dir $SPLITS --out-dir $OUT --device cuda --skip-existing --model geno_env --split-type \$split
    done
    echo '[GPU1] dnngp leave_ge'
    $PYTHON $SCRIPT --data-dir $DATA --split-dir $SPLITS --out-dir $OUT --device cuda --skip-existing --model dnngp --split-type leave_ge
    echo '[GPU1] dnngp leave_year'
    $PYTHON $SCRIPT --data-dir $DATA --split-dir $SPLITS --out-dir $OUT --device cuda --skip-existing --model dnngp --split-type leave_year
    echo '[GPU1] ALL DONE'
" > $OUT/gpu1.log 2>&1 &
echo "  GPU1 PID: $!"

echo ""
echo "All GPUs launched. Monitor with:"
echo "  grep -E '(pearson=|DONE|FAILED)' $OUT/gpu*.log"
