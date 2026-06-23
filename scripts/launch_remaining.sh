#!/usr/bin/env bash
# =============================================================================
# Launch remaining 11 model×split combos — 3 tasks/GPU max to avoid contention
# =============================================================================
set -euo pipefail

PYTHON=/home/amax/Downloads/yes/bin/python
D=/opt/data/lgh/gwas1/data/processed/g2f
SP=/opt/data/lgh/gwas1/outputs/v2_stage/splits
O=/opt/data/lgh/gwas1/outputs/ablation_study
SC=/opt/data/lgh/gwas1/scripts/06_run_ablations.py

export PATH="/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:$PATH"

echo "=== Launching remaining 11 tasks (3/GPU max) ==="
echo ""

# Define remaining tasks as "GPU|MODEL|SPLIT"
# Distribute evenly: GPU0 and GPU1 each get ~5-6 tasks
# Launch ALL simultaneously with max 3 per GPU
# CUDA will queue the excess

TASKS=(
    "0|genotype_only|leave_environment"
    "1|genotype_only|leave_year"
    "0|weather_only|leave_environment"
    "1|static_env_only|leave_year"
    "0|geno_env|leave_environment"
    "1|geno_env|leave_year"
    "0|cropformer|leave_environment"
    "1|cropformer|leave_ge"
    "0|cropformer|leave_year"
    "1|dnngp|leave_ge"
    "0|dnngp|leave_year"
)

task_num=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r gpu_id model split <<< "$task"
    task_num=$((task_num + 1))
    logfile="$O/task_r_${model}_${split}.log"
    metricsfile="$O/task_r_${model}_${split}.csv"

    # If already done (CSV exists with 3 seeds), skip
    if [ -f "$metricsfile" ]; then
        seeds=$(tail -n +2 "$metricsfile" 2>/dev/null | wc -l)
        if [ "$seeds" -ge 3 ]; then
            echo "  SKIP GPU$gpu_id $model $split (already has $seeds seeds)"
            continue
        fi
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id nohup bash -c "
        export PATH=/home/amax/Downloads/yes/bin:/usr/local/cuda/bin:\$PATH
        echo \"START GPU$gpu_id $model $split\"
        $PYTHON $SC --data-dir $D --split-dir $SP --out-dir $O --device cuda --model $model --split-type $split --metrics-file $metricsfile
        echo \"DONE GPU$gpu_id $model $split\"
    " > "$logfile" 2>&1 &

    printf "  LAUNCH GPU%d %-25s %-20s PID=%s\n" $gpu_id "$model" "$split" "$!"
    sleep 0.5
done

echo ""
echo "All tasks launched. Waiting..."
wait

echo ""
echo "=== ALL DONE. Merging results ==="
$PYTHON -c "
import pandas as pd, glob, os

csv_files = sorted(glob.glob('$O/task_*.csv') + glob.glob('$O/task_r_*.csv'))
dfs = []
for f in csv_files:
    try:
        if os.path.getsize(f) > 50:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
    except Exception as e:
        print(f'  Skip {os.path.basename(f)}: {e}')

if dfs:
    merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['model','split_type','seed'])
    merged.to_csv('$O/metrics.csv', index=False)
    print(f'Merged {len(merged)} records\n')

    print('='*70)
    print('FINAL RESULTS SUMMARY')
    print('='*70)

    # Also print comparison with full model
    full_model = {
        'leave_environment': 0.380,
        'leave_ge': 0.365,
        'leave_year': 0.325,
    }

    for model in sorted(merged['model'].unique()):
        sub = merged[merged['model'] == model]
        print(f'\n{model}:')
        for st in ['leave_environment', 'leave_ge', 'leave_year']:
            ss = sub[sub['split_type'] == st]
            p = ss['pearson'].dropna()
            if len(p) > 0:
                gap = ''
                if model != 'residual_gxe_former' and st in full_model:
                    gap = f' (gap={full_model[st]-p.mean():+.3f})'
                print(f'  {st:20s}: p={p.mean():.4f}+/-{p.std():.4f} n={len(p)}{gap}')

    print()
    # Reference
    print('Reference (ResidualGxEFormer full model):')
    for st, p in full_model.items():
        print(f'  {st:20s}: p={p:.3f}')
"
