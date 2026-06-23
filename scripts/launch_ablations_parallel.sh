#!/usr/bin/env bash
# =============================================================================
# PARALLEL launch: all model x split combos run simultaneously across 2 GPUs
# Each task writes to its own metrics file to avoid race conditions.
# A merge step combines all results at the end.
#
# With H200 143GB VRAM: multiple small models can coexist per GPU.
# CUDA handles serialization automatically.
# =============================================================================
set -euo pipefail

PYTHON=/home/amax/Downloads/yes/bin/python
PROJ=/opt/data/lgh/gwas1
DATA=$PROJ/data/processed/g2f
SPLITS=$PROJ/data/processed/g2f/splits
OUT=$PROJ/outputs/ablation_study
SCRIPT=$PROJ/scripts/06_run_ablations.py

export PATH="/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:$PATH"

echo "============================================"
echo "  PARALLEL Ablation Study Launch"
echo "  $(date)"
echo "============================================"

# Kill old processes
pkill -9 -f "06_run_ablations" 2>/dev/null || true
sleep 2

# Clean slate
find $PROJ/src -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf $OUT/task_*.log $OUT/task_*.csv $OUT/task_*.json
mkdir -p $OUT

# All model x split combos
# GPU distribution: alternate between 0 and 1
declare -a TASKS=()
gpu=0
for model in genotype_only weather_only static_env_only geno_env cropformer; do
    for split in leave_environment leave_ge leave_year; do
        TASKS+=("$gpu|$model|$split")
        gpu=$(( (gpu + 1) % 2 ))
    done
done
# DNNGP on remaining splits
TASKS+=("0|dnngp|leave_ge")
TASKS+=("1|dnngp|leave_year")

TOTAL=${#TASKS[@]}
echo "Launching $TOTAL tasks across 2 GPUs (H200, 143GB each)"
echo ""

# Launch all tasks in background
task_num=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r gpu_id model split <<< "$task"
    task_num=$((task_num + 1))
    logfile="$OUT/task_${task_num}_${model}_${split}.log"
    metricsfile="$OUT/task_${task_num}_${model}_${split}.csv"

    CUDA_VISIBLE_DEVICES=$gpu_id nohup bash -c "
        export PATH=/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:\$PATH
        echo \"[\$(( \$(date +%s) - \$(date -d '2026-06-09' +%s) ))] START GPU$gpu_id $model $split\"
        $PYTHON $SCRIPT \
            --data-dir $DATA \
            --split-dir $SPLITS \
            --out-dir $OUT \
            --device cuda \
            --model $model \
            --split-type $split \
            --metrics-file $metricsfile
        rc=\$?
        echo \"[\$(( \$(date +%s) - \$(date -d '2026-06-09' +%s) ))] DONE GPU$gpu_id $model $split (exit=\$rc)\"
    " > "$logfile" 2>&1 &

    printf "  [%2d/%2d] GPU%d %-22s %-20s PID=%-8s\n" $task_num $TOTAL $gpu_id $model $split $!
    sleep 0.3
done

echo ""
echo "All $TOTAL tasks launched. Waiting for completion..."
echo "Monitor: grep pearson= $OUT/task_*.log"

# Wait for all background jobs
wait

echo ""
echo "============================================"
echo "  ALL TASKS COMPLETE at $(date)"
echo "============================================"

# Merge all task CSV files
echo ""
echo "=== Merging results ==="
$PYTHON -c "
import pandas as pd, glob, os

csv_files = sorted(glob.glob('$OUT/task_*.csv'))
if not csv_files:
    print('ERROR: No CSV files found!')
    exit(1)

dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        if len(df) > 0:
            dfs.append(df)
            print(f'  Loaded {len(df)} records from {os.path.basename(f)}')
    except Exception as e:
        print(f'  Skipping {f}: {e}')

if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv('$OUT/metrics.csv', index=False)
    print(f'\nMerged {len(merged)} total records -> $OUT/metrics.csv')

    # Print summary
    print('\n=== RESULTS SUMMARY ===')
    for model in sorted(merged['model'].unique()):
        sub = merged[merged['model'] == model]
        print(f'\n{model}:')
        for st in sorted(sub['split_type'].unique()):
            ss = sub[sub['split_type'] == st]
            if 'pearson' in ss.columns and len(ss) > 0:
                p_vals = ss['pearson'].dropna()
                if len(p_vals) > 0:
                    print(f'  {st}: pearson={p_vals.mean():.4f}+/-{p_vals.std():.4f} (n={len(p_vals)})')
                else:
                    print(f'  {st}: NO VALID PEARSON (n={len(ss)})')
else:
    print('No results found! Check task_*.log files for errors.')
"

echo ""
echo "=== Checking for errors ==="
grep -l "Traceback\|Error\|FAILED" $OUT/task_*.log 2>/dev/null | while read f; do
    echo "ERROR in $(basename $f):"
    grep -A5 "Traceback\|Error" "$f" | tail -6
    echo ""
done || echo "No errors found."

echo ""
echo "=== All pearson results ==="
grep -h "pearson=" $OUT/task_*.log 2>/dev/null | grep -v "+/-" | while read line; do
    echo "  ${line//[/}"
done
