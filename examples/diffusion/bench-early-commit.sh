#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

BIN=${BIN:-./build/bin/llama-diffusion-cli}
HF_MODEL=${HF_MODEL:-keisuke-miyako/Dream-v0-Instruct-7B-gguf-q4_k_m:Q4_K_M}
MODEL_PATH=${MODEL_PATH:-}
PROMPT=${PROMPT:-Write a short Python function that adds two numbers.}
REPEATS=${REPEATS:-3}
THRESHOLD=${THRESHOLD:-0.999}
GENERATED_BLOCK_SCHEDULE=${GENERATED_BLOCK_SCHEDULE:-1}
LOG_DIR=${LOG_DIR:-/tmp/llama-diffusion-early-commit-$$}

if [[ ! -x "$BIN" ]]; then
    echo "error: diffusion binary is not executable: $BIN" >&2
    exit 1
fi

if [[ ! "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: REPEATS must be a positive integer" >&2
    exit 1
fi

if [[ ! "$THRESHOLD" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "error: THRESHOLD must be a number in [0, 1)" >&2
    exit 1
fi

if ! awk -v threshold="$THRESHOLD" 'BEGIN { exit !(threshold >= 0 && threshold < 1) }'; then
    echo "error: THRESHOLD must be in [0, 1)" >&2
    exit 1
fi

if [[ "$GENERATED_BLOCK_SCHEDULE" != 0 && "$GENERATED_BLOCK_SCHEDULE" != 1 ]]; then
    echo "error: GENERATED_BLOCK_SCHEDULE must be 0 or 1" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
printf "variant\trun\ttotal_ms\tmain_forwards\tlogit_rows\ttokens_committed\tthreshold_extra_selections\tforced_final_selections\tremaining_masks\tblocks_started\tblocks_completed\tblocks_finished_early\tscheduled_steps_skipped\tscheduled_forwards_skipped\tearly_enabled\tgenerated_block_schedule\n" > "$SUMMARY"

if [[ -n "$MODEL_PATH" ]]; then
    model_args=(-m "$MODEL_PATH")
else
    model_args=(-hf "$HF_MODEL")
fi

common_args=(
    "${model_args[@]}"
    -p "$PROMPT"
    -ngl "${NGL:-99}"
    -c "${CTX:-128}"
    -b "${BATCH:-128}"
    -ub "${UBATCH:-128}"
    -fa "${FLASH_ATTN:-on}"
    --seed "${SEED:-1234}"
    --temp "${TEMP:-0.2}"
    --top-p "${TOP_P:-0.95}"
    --diffusion-block-length "${BLOCK_LENGTH:-32}"
    --diffusion-algorithm 4
    --diffusion-alg-temp 0
    --diffusion-steps "${STEPS:-32}"
)

if [[ "$GENERATED_BLOCK_SCHEDULE" == 1 ]]; then
    common_args+=(--diffusion-generated-block-schedule)
fi

extract_metric() {
    local expression=$1
    local log_file=$2
    sed -nE "$expression" "$log_file" | tail -n 1
}

run_variant() {
    local variant=$1
    local run=$2
    shift 2

    local log_file="$LOG_DIR/${variant}-${run}.log"
    echo "running $variant iteration $run"
    "$BIN" "${common_args[@]}" "$@" 2>&1 | tee "$log_file"

    local total_ms main_forwards logit_rows tokens_committed threshold_extra forced_final remaining_masks blocks_started blocks_completed blocks_early steps_skipped forwards_skipped early_enabled generated_schedule
    total_ms=$(extract_metric 's/.*total time: ([0-9.]+)ms.*/\1/p' "$log_file")
    main_forwards=$(extract_metric 's/.*conditional\/main = ([0-9]+).*/\1/p' "$log_file")
    logit_rows=$(extract_metric 's/.*logits: rows = ([0-9]+).*/\1/p' "$log_file")
    tokens_committed=$(extract_metric 's/.*token commits: committed = ([0-9]+).*/\1/p' "$log_file")
    threshold_extra=$(extract_metric 's/.*token selections: base = [0-9]+, threshold extra = ([0-9]+).*/\1/p' "$log_file")
    forced_final=$(extract_metric 's/.*forced final = ([0-9]+).*/\1/p' "$log_file")
    remaining_masks=$(extract_metric 's/.*remaining masks = ([0-9]+).*/\1/p' "$log_file")
    blocks_started=$(extract_metric 's/.*blocks: started = ([0-9]+).*/\1/p' "$log_file")
    blocks_completed=$(extract_metric 's/.*blocks: started = [0-9]+, completed = ([0-9]+).*/\1/p' "$log_file")
    blocks_early=$(extract_metric 's/.*finished before last step = ([0-9]+).*/\1/p' "$log_file")
    steps_skipped=$(extract_metric 's/.*scheduled steps skipped = ([0-9]+).*/\1/p' "$log_file")
    forwards_skipped=$(extract_metric 's/.*scheduled forwards skipped = ([0-9]+).*/\1/p' "$log_file")
    early_enabled=$(extract_metric 's/.*early commit: enabled = (true|false).*/\1/p' "$log_file")
    generated_schedule=$(extract_metric 's/.*block schedule: generated-aware = (true|false).*/\1/p' "$log_file")

    if [[ -z "$total_ms" || -z "$main_forwards" || -z "$logit_rows" || -z "$tokens_committed" ||
          -z "$threshold_extra" || -z "$forced_final" || -z "$remaining_masks" ||
          -z "$blocks_started" || -z "$blocks_completed" || -z "$blocks_early" ||
          -z "$steps_skipped" || -z "$forwards_skipped" || -z "$early_enabled" ||
          -z "$generated_schedule" ]]; then
        echo "error: failed to parse performance metrics from $log_file" >&2
        exit 1
    fi

    if [[ "$remaining_masks" != 0 || "$blocks_completed" != "$blocks_started" ]]; then
        echo "error: incomplete generation in $log_file (remaining masks: $remaining_masks, blocks: $blocks_completed/$blocks_started)" >&2
        exit 1
    fi

    if [[ "$variant" == legacy && "$early_enabled" != false ]] ||
       [[ "$variant" != legacy && "$early_enabled" != true ]]; then
        echo "error: unexpected early-commit state in $log_file: $early_enabled" >&2
        exit 1
    fi

    if [[ "$variant" == control && "$threshold_extra" != 0 ]]; then
        echo "error: control run made threshold-extra selections in $log_file" >&2
        exit 1
    fi

    local expected_generated_schedule=false
    if [[ "$GENERATED_BLOCK_SCHEDULE" == 1 ]]; then
        expected_generated_schedule=true
    fi
    if [[ "$generated_schedule" != "$expected_generated_schedule" ]]; then
        echo "error: unexpected generated block schedule in $log_file: $generated_schedule" >&2
        exit 1
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$variant" "$run" "$total_ms" "$main_forwards" "$logit_rows" \
        "$tokens_committed" "$threshold_extra" "$forced_final" "$remaining_masks" \
        "$blocks_started" "$blocks_completed" "$blocks_early" "$steps_skipped" \
        "$forwards_skipped" "$early_enabled" "$generated_schedule" >> "$SUMMARY"
}

for ((run = 1; run <= REPEATS; run++)); do
    case $((run % 3)) in
        1)
            run_variant legacy "$run"
            run_variant control "$run" --diffusion-early-commit-threshold 1.0
            run_variant early "$run" --diffusion-early-commit-threshold "$THRESHOLD"
            ;;
        2)
            run_variant control "$run" --diffusion-early-commit-threshold 1.0
            run_variant early "$run" --diffusion-early-commit-threshold "$THRESHOLD"
            run_variant legacy "$run"
            ;;
        0)
            run_variant early "$run" --diffusion-early-commit-threshold "$THRESHOLD"
            run_variant legacy "$run"
            run_variant control "$run" --diffusion-early-commit-threshold 1.0
            ;;
    esac
done

median_column() {
    local variant=$1
    local column=$2
    awk -F '\t' -v variant="$variant" -v column="$column" \
        'NR > 1 && $1 == variant { print $column }' "$SUMMARY" |
        sort -n |
        awk '{ values[NR] = $1 } END {
            if (NR % 2 == 1) {
                print values[(NR + 1) / 2]
            } else {
                printf "%.3f\n", (values[NR / 2] + values[NR / 2 + 1]) / 2
            }
        }'
}

legacy_total=$(median_column legacy 3)
control_total=$(median_column control 3)
early_total=$(median_column early 3)
legacy_forwards=$(median_column legacy 4)
control_forwards=$(median_column control 4)
early_forwards=$(median_column early 4)
threshold_speedup=$(awk -v control="$control_total" -v early="$early_total" \
    'BEGIN { printf "%.2f", 100.0 * (control - early) / control }')
overall_speedup=$(awk -v legacy="$legacy_total" -v early="$early_total" \
    'BEGIN { printf "%.2f", 100.0 * (legacy - early) / legacy }')
forward_reduction=$(awk -v control="$control_forwards" -v early="$early_forwards" \
    'BEGIN { printf "%.0f", control - early }')

echo
echo "summary: $SUMMARY"
printf "legacy median:  %s ms, %s main forwards\n" "$legacy_total" "$legacy_forwards"
printf "control median: %s ms, %s main forwards\n" "$control_total" "$control_forwards"
printf "early median:   %s ms, %s main forwards\n" "$early_total" "$early_forwards"
printf "threshold-vs-control total reduction: %s%%\n" "$threshold_speedup"
printf "early-vs-legacy total reduction: %s%%\n" "$overall_speedup"
printf "threshold-vs-control main-forward reduction: %s\n" "$forward_reduction"
if awk -v control="$control_forwards" -v early="$early_forwards" 'BEGIN { exit !(early >= control) }'; then
    echo "warning: threshold arm did not reduce the measured main forward count versus control" >&2
fi
echo "inspect legacy-*.log, control-*.log, and early-*.log in $LOG_DIR for generated text and full metrics"
