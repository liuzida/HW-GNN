#!/bin/bash
# Run the single-model configuration for each benchmark.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/fixed_sota_configs.sh"

# Check the dataset paths before starting training.
DATA_ROOT="${BOT_DATA_ROOT:-$SCRIPT_DIR/dataset/raw}"
PRETRAINED_ROOT_PATH="${BOT_PRETRAINED_ROOT:-$SCRIPT_DIR/pretrained}"
for graph_file in tfinance tsocial Twibot-22/graph.bin; do
    if [ ! -f "$DATA_ROOT/$graph_file" ]; then
        echo "Missing graph file: $DATA_ROOT/$graph_file" >&2
        echo "Prepare the benchmark data as described in README.md and rerun." >&2
        exit 2
    fi
done
for required_dir in Twibot-20_raw MGTAB; do
    if [ ! -d "$DATA_ROOT/$required_dir" ]; then
        echo "Missing dataset directory: $DATA_ROOT/$required_dir" >&2
        echo "Download the benchmark data from the links in README.md, prepare the loader layout, and rerun." >&2
        exit 2
    fi
done
if [ ! -d "$PRETRAINED_ROOT_PATH" ]; then
    echo "Missing pretrained-artifact directory: $PRETRAINED_ROOT_PATH" >&2
    echo "Set BOT_PRETRAINED_ROOT or prepare the pretrained files described in README.md." >&2
    exit 2
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
mkdir -p "$LOGDIR"

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

echo "========================================"
echo "Best reproduction for five datasets"
echo "Timestamp: $TIMESTAMP"
echo "Start: $(date)"
echo "========================================"
echo ""

run_exp "BEST_tfinance_ppr_topk32_h32" \
    "${TFINANCE_SOTA_ARGS[@]}"

run_exp "BEST_tsocial_ppr_topk64_h10" \
    "${TSOCIAL_SOTA_ARGS[@]}"

run_exp "BEST_Twibot22_raw2_h64_K3S5_lr003_fp16" \
    "${TWIBOT22_SOTA_ARGS[@]}"

run_exp "BEST_Twibot20_strat_seed1_h192_lf02_fp16" \
    "${TWIBOT20_SOTA_ARGS[@]}"

run_exp "BEST_MGTAB_stable_stratified" \
    "${MGTAB_BEST_REPORTABLE_STABLE_ARGS[@]}"

echo "========================================"
echo "BEST REPRODUCTION FINISHED"
echo "End: $(date)"
echo "========================================"
echo ""

echo "=== Five-Dataset Best Reproduction Results Summary ==="
for f in ${LOGDIR}/*_${TIMESTAMP}.log; do
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
