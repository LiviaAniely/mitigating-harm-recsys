#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
RUNS="${RUNS:-10}"
CORES="${CORES:-8}"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/reproduction/results/main}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/reproduction/logs}"

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNS must be a positive integer" >&2
    exit 2
fi

LOCK_FILE="$WORKSPACE/.reproduction-${RUNS}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "A reproduction with RUNS=$RUNS is already running in $WORKSPACE" >&2
    exit 9
fi

RESULT_NAME="results_kuairand_hybrid_lightgcl_harm_${RUNS}_0.0_0.3_100_False_all.csv"
TEMP_RESULT="$WORKSPACE/$RESULT_NAME"
FINAL_RESULT="$OUTPUT_DIR/$RESULT_NAME"

if [[ -e "$FINAL_RESULT" ]]; then
    echo "Refusing to overwrite existing result: $FINAL_RESULT" >&2
    exit 3
fi
if [[ -e "$TEMP_RESULT" ]]; then
    echo "Refusing to overwrite stale temporary result: $TEMP_RESULT" >&2
    exit 4
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/docker-reproduction-runs-${RUNS}.log}"
set -o noclobber
if ! : > "$LOG_FILE"; then
    echo "Refusing to overwrite existing log: $LOG_FILE" >&2
    exit 5
fi
set +o noclobber

prepare_and_evaluate() {
    cd "$WORKSPACE"

    SOURCE_DATA="$WORKSPACE/KuaiRand-Harm/training/single_and_repeated_interactions_is_click.csv.gzip"
    SPLIT_DIR="$WORKSPACE/methods/kuairand/training"
    [[ -s "$SOURCE_DATA" ]] || { echo "Missing source data: $SOURCE_DATA"; return 6; }

    "$PYTHON" reproduction/docker/generate_splits.py \
        --input "$SOURCE_DATA" \
        --output-dir "$SPLIT_DIR" \
        --runs "$RUNS"

    for ((run_id = 0; run_id < RUNS; run_id++)); do
        for model in lightgcl ncf; do
            prediction="$WORKSPACE/methods/kuairand/results/test_${run_id}_${model}_False_0.3_100.zlib.pickle"
            [[ -s "$prediction" ]] || { echo "Missing prediction: $prediction"; return 7; }
        done
    done

    echo "Starting ${RUNS}-run CPU reproduction with ${CORES} workers"
    /usr/bin/time -v "$PYTHON" ranker_kuairand.py \
        --runs "$RUNS" \
        --score-model lightgcl \
        --score-type harm \
        --method hybrid \
        --cores "$CORES" \
        --beta 0.0 \
        --epoch 100
}

set +e
prepare_and_evaluate 2>&1 | tee -a "$LOG_FILE"
pipeline_status=("${PIPESTATUS[@]}")
set -e
if ((pipeline_status[1] != 0)); then
    echo "Persistent logger failed with status ${pipeline_status[1]}" >&2
    exit 10
fi
if ((pipeline_status[0] != 0)); then
    echo "Reproduction failed with status ${pipeline_status[0]}; see $LOG_FILE" >&2
    exit "${pipeline_status[0]}"
fi

[[ -s "$TEMP_RESULT" ]] || { echo "Evaluator did not create $TEMP_RESULT"; exit 8; }
STAGED_RESULT="$(mktemp "$OUTPUT_DIR/.${RESULT_NAME}.XXXXXX")"
trap 'rm -f "${STAGED_RESULT:-}"' EXIT
cp "$TEMP_RESULT" "$STAGED_RESULT"
if ! ln "$STAGED_RESULT" "$FINAL_RESULT"; then
    echo "Refusing to overwrite result created during execution: $FINAL_RESULT" >&2
    exit 11
fi
rm -f "$STAGED_RESULT" "$TEMP_RESULT"
trap - EXIT
echo "Result saved to $FINAL_RESULT"
