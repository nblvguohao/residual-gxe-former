#!/usr/bin/env bash
# =============================================================================
# PARALLEL launch v2: all model x split combos with correct splits directory
# SPLITS = /opt/data/lgh/gwas1/outputs/v2_stage/splits (includes leave_ge!)
# =============================================================================
set -euo pipefail

PYTHON=/home/amax/Downloads/yes/bin/python
PROJ=/opt/data/lgh/gwas1
DATA=$PROJ/data/processed/g2f
SPLITS=$PROJ/outputs/v2_stage/splits
OUT=$PROJ/outputs/ablation_study
SCRIPT=$PROJ/scripts/06_run_ablations.py

export PATH="/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:$PATH"

echo "============================================"
echo "  PARALLEL Ablation v2"
echo "  Splits: $SPLITS"
echo "  $(date)"
echo "============================================"

# Kill old
pkill -9 -f "06_run_ablations" 2>/dev/null || true
sleep 2

# Clean
find $PROJ/src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
rm -f $OUT/task_*.log $OUT/task_*.csv $OUT/task_*.json $OUT/metrics.csv $OUT/MASTER.log
mkdir -p $OUT

# Generate task list
TASK_COUNT=0
gpu=0
for model in genotype_only weather_only static_env_only geno_env cropformer; do
    for split in leave_environment leave_ge leave_year; do
        echo "$gpu|$model|$split" >> /tmp/task_list.txt
        gpu=$(( (gpu + 1) % 2 ))
        TASK_COUNT=$((TASK_COUNT + 1))
    done
done
echo "0|dnngp|leave_ge" >> /tmp/task_list.txt
echo "1|dnngp|leave_year" >> /tmp/task_list.txt
TASK_COUNT=$((TASK_COUNT + 2))

echo "Launching $TASK_COUNT tasks across 2 GPUs"
echo ""

# Launch all tasks
task_num=0
while IFS='|' read -r gpu_id model split; do
    task_num=$((task_num + 1))
    logfile="$OUT/task_${task_num}_${model}_${split}.log"
    metricsfile="$OUT/task_${task_num}_${model}_${split}.csv"

    CUDA_VISIBLE_DEVICES=$gpu_id nohup bash -c "
        export PATH=/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:\$PATH
        echo \"START GPU\$CUDA_VISIBLE_DEVICES $model $split\"
        $PYTHON $SCRIPT \
            --data-dir $DATA \
            --split-dir $SPLITS \
            --out-dir $OUT \
            --device cuda \
            --model $model \
            --split-type $split \
            --metrics-file $metricsfile
        echo \"DONE GPU\$CUDA_VISIBLE_DEVICES $model $split (exit=\$?)\"
    " > "$logfile" 2>&1 &

    printf "  [%2d/%2d] GPU%d %-25s %-20s PID=%s\n" $task_num $TASK_COUNT $gpu_id "$model" "$split" "$!"
    sleep 0.3
done < /tmp/task_list.txt

rm -f /tmp/task_list.txt

echo ""
echo "All $TASK_COUNT tasks launched. Waiting for completion..."
echo "Monitor: grep pearson= $OUT/task_*.log"
wait

echo ""
echo "============================================"
echo "  ALL TASKS COMPLETE at $(date)"
echo "============================================"

# Merge results
echo ""
echo "=== Merging results ==="
$PYTHON -c "
import pandas as pd, glob, os

csv_files = sorted(glob.glob('$OUT/task_*.csv'))
dfs = []
for f in csv_files:
    try:
        if os.path.getsize(f) > 50:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
                print(f'  {os.path.basename(f)}: {len(df)} records')
    except Exception as e:
        print(f'  Skip {f}: {e}')

if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv('$OUT/metrics.csv', index=False)
    print(f'\nMerged {len(merged)} total records')

    print('\n' + '='*60)
    print('RESULTS SUMMARY')
    print('='*60)
    for model in sorted(merged['model'].unique()):
        sub = merged[merged['model'] == model]
        print(f'\n{model}:')
        for st in sorted(sub['split_type'].unique()):
            ss = sub[sub['split_type'] == st]
            p = ss['pearson'].dropna()
            if len(p) > 0:
                print(f'  {st:20s}: pearson={p.mean():.4f}+/-{p.std():.4f} (n={len(p)})')
"

echo ""
echo "=== Errors (if any) ==="
grep -l "Traceback" $OUT/task_*.log 2>/dev/null | while read f; do
    echo "  $(basename $f):"
    grep -A3 "Traceback" "$f" | tail -4
    echo ""
done || echo "  No errors found"
