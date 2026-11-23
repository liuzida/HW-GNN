#!/bin/bash
# Train MGTAB models on a fixed random split and evaluate ensembles of their
# saved predictions. Select checkpoints and thresholds on the validation set.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/fixed_sota_configs.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
PROBDIR="${LOGDIR}/mgtab_probs_${TIMESTAMP}"
mkdir -p "$LOGDIR" "$PROBDIR"

run_exp() {
    local name="$1"
    shift
    local logfile="${LOGDIR}/${name}_${TIMESTAMP}.log"
    echo "[$(date '+%H:%M:%S')] START: $name -> $logfile"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u main.py "$@" > "$logfile" 2>&1
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE:  $name (exit 0)"
    else
        echo "[$(date '+%H:%M:%S')] FAIL:  $name (exit $exit_code)"
    fi
    echo ""
}

BASE_RANDOM=("${MGTAB_RANDOM_ALPHA078_ARGS[@]}")

echo "========================================"
echo "MGTAB ensemble optimization"
echo "Timestamp: $TIMESTAMP"
echo "Start: $(date)"
echo "========================================"
echo "Saved probability dir: $PROBDIR"
echo "Base: random split2 + alpha=0.78"
echo "========================================"
echo ""

# Same fixed command, separate processes. Non-determinism creates model diversity;
# the ensemble analysis uses only validation labels to select thresholds.
for r in 01 02 03 04 05 06 07 08 09 10; do
    run_exp "MGTAB_ens_alpha078_repeat${r}" \
        "${BASE_RANDOM[@]}" --save_probs_path "${PROBDIR}/alpha078_repeat${r}.npz"
done

echo "[$(date '+%H:%M:%S')] START: MGTAB ensemble analysis"
python -u analyze_mgtab_ensemble.py \
    --pattern "${PROBDIR}/alpha078_repeat*.npz" \
    --output "${LOGDIR}/MGTAB_ensemble_analysis_${TIMESTAMP}.txt" \
    > "${LOGDIR}/MGTAB_ensemble_analysis_${TIMESTAMP}.log" 2>&1
analysis_exit=$?
if [ $analysis_exit -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] DONE:  MGTAB ensemble analysis (exit 0)"
else
    echo "[$(date '+%H:%M:%S')] FAIL:  MGTAB ensemble analysis (exit $analysis_exit)"
fi
echo ""

echo "========================================"
echo "MGTAB ENSEMBLE OPTIMIZATION FINISHED"
echo "End: $(date)"
echo "========================================"
echo ""

echo "=== MGTAB Ensemble Optimization Results Summary ==="
for f in ${LOGDIR}/MGTAB_ens_*_${TIMESTAMP}.log; do
    name=$(basename "$f" .log)
    if grep -q "Test:" "$f" 2>/dev/null; then
        status="DONE"
        result=$(grep "Test:" "$f" | tail -1)
    elif grep -q "Traceback\|Error\|Killed\|Segmentation fault\|OutOfMemoryError\|CUDA out" "$f" 2>/dev/null; then
        status="FAIL"
        result=$(grep -E "OutOfMemoryError|RuntimeError|Traceback|Segmentation fault|Killed|CUDA out" "$f" | tail -1 | head -c 180)
    else
        status="UNKNOWN"
        result=""
    fi
    echo "$status | $name | $result"
done

if [ -f "${LOGDIR}/MGTAB_ensemble_analysis_${TIMESTAMP}.txt" ]; then
    echo ""
    cat "${LOGDIR}/MGTAB_ensemble_analysis_${TIMESTAMP}.txt"
fi
