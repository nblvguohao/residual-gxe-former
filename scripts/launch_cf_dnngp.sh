#!/usr/bin/env bash
# Run cropformer (fixed) + dnngp: one per GPU, no competition
set -euo pipefail

PYTHON=/home/amax/Downloads/yes/bin/python
SC=/opt/data/lgh/gwas1/scripts/06_run_ablations.py
D=/opt/data/lgh/gwas1/data/processed/g2f
SP=/opt/data/lgh/gwas1/outputs/v2_stage/splits
O=/opt/data/lgh/gwas1/outputs/ablation_study

export PATH="/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:$PATH"

echo "=== cropformer (GPU0) + dnngp (GPU1) ==="
echo ""

# Kill old
pkill -9 -f "06_run_ablations" 2>/dev/null || true
sleep 2
find /opt/data/lgh/gwas1/src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# GPU0: cropformer on all 3 splits
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    export PATH=/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:\$PATH
    for split in leave_environment leave_ge leave_year; do
        echo \"[GPU0] cropformer \$split\"
        $PYTHON $SC --data-dir $D --split-dir $SP --out-dir $O --device cuda --model cropformer --split-type \$split --metrics-file $O/task_cropformer_\${split}.csv
    done
    echo '[GPU0] ALL DONE'
" > $O/gpu_cropformer.log 2>&1 &
echo "GPU0 PID: $! (cropformer × 3 splits)"

sleep 1

# GPU1: dnngp on leave_ge and leave_year
CUDA_VISIBLE_DEVICES=1 nohup bash -c "
    export PATH=/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:\$PATH
    echo '[GPU1] dnngp leave_ge'
    $PYTHON $SC --data-dir $D --split-dir $SP --out-dir $O --device cuda --model dnngp --split-type leave_ge --metrics-file $O/task_dnngp_leave_ge.csv
    echo '[GPU1] dnngp leave_year'
    $PYTHON $SC --data-dir $D --split-dir $SP --out-dir $O --device cuda --model dnngp --split-type leave_year --metrics-file $O/task_dnngp_leave_year.csv
    echo '[GPU1] ALL DONE'
" > $O/gpu_dnngp.log 2>&1 &
echo "GPU1 PID: $! (dnngp × 2 splits)"

echo ""
echo "Monitor: grep pearson= $O/gpu_{cropformer,dnngp}.log"
