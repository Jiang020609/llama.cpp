#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

BIN=${BIN:-./build/bin/llama-diffusion-cli}
HF_MODEL=${HF_MODEL:-keisuke-miyako/Dream-v0-Instruct-7B-gguf-q4_k_m:Q4_K_M}
MODEL_PATH=${MODEL_PATH:-}
PROMPT=${PROMPT:-Write a short Python function that adds two numbers.}
REPEATS=${REPEATS:-2}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
MBSD_TRIGGER=${MBSD_TRIGGER:-8}
SINGLE_UBATCH=${SINGLE_UBATCH:-32}
MULTI_UBATCH=${MULTI_UBATCH:-128}
SINGLE_STEPS=${SINGLE_STEPS:-16}
MULTI_STEPS=${MULTI_STEPS:-32}
LOG_DIR=${LOG_DIR:-${TMPDIR:-/tmp}/llama-diffusion-mbsd-$$}

die() {
    echo "error: $*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

if [[ ! -x "$BIN" ]]; then
    die "diffusion binary is not executable: $BIN"
fi

is_positive_integer "$REPEATS" || die "REPEATS must be a positive integer"
is_positive_integer "$BLOCK_LENGTH" || die "BLOCK_LENGTH must be a positive integer"
is_nonnegative_integer "$MBSD_TRIGGER" || die "MBSD_TRIGGER must be a non-negative integer"
is_positive_integer "$SINGLE_UBATCH" || die "SINGLE_UBATCH must be a positive integer"
is_positive_integer "$MULTI_UBATCH" || die "MULTI_UBATCH must be a positive integer"
is_positive_integer "$SINGLE_STEPS" || die "SINGLE_STEPS must be a positive integer"
is_positive_integer "$MULTI_STEPS" || die "MULTI_STEPS must be a positive integer"

if (( SINGLE_UBATCH > BLOCK_LENGTH )); then
    die "SINGLE_UBATCH must not exceed BLOCK_LENGTH"
fi
if (( MULTI_UBATCH <= BLOCK_LENGTH )); then
    die "MULTI_UBATCH must exceed BLOCK_LENGTH"
fi

if [[ -n "$MODEL_PATH" ]]; then
    [[ -f "$MODEL_PATH" ]] || die "MODEL_PATH does not exist: $MODEL_PATH"
    model_args=(-m "$MODEL_PATH")
else
    model_args=(-hf "$HF_MODEL")
fi

mkdir -p "$LOG_DIR/invalid"
SUMMARY="$LOG_DIR/summary.tsv"

write_tsv_row() {
    local IFS=$'\t'
    printf '%s\n' "$*"
}

header=(
    case variant lookahead run total_ms main_forwards transformer_rows logit_rows
    generated_tokens id_hash generated_masks invalid_tokens remaining_masks planned_blocks
    blocks_started blocks_completed mbsd_enabled trigger max_lookahead policy execution trigger_checks
    lookahead_expansions slides slide_distance first_start first_end last_start last_end max_end
    draft_introduced draft_updates draft_prediction_rows draft_reevaluated draft_accepted
    draft_reconfirmed draft_replacements draft_rejected draft_pending role_current role_future
    role_context role_total logits_current logits_future steps_with_future extra_steps
    extra_forwards main_steps steps_saved forwards_saved nominal_zero_tail_slots
    nominal_tail_skipped final_forced_selections
    future_semantic_commits prefix_kv_disabled
    block_order_violations verification_input_errors bounds_errors trajectory_hash
)
write_tsv_row "${header[@]}" > "$SUMMARY"

line_count() {
    local marker=$1
    local log_file=$2
    awk -v marker="$marker" 'index($0, marker) { count++ } END { print count + 0 }' "$log_file"
}

extract_value() {
    local prefix=$1
    local key=$2
    local log_file=$3
    awk -v prefix="$prefix" -v key="$key" '
        index($0, prefix) {
            marker = key " = "
            at = index($0, marker)
            if (at > 0) {
                text = substr($0, at + length(marker))
                sub(/[,[:space:]].*$/, "", text)
                value = text
            }
        }
        END {
            if (value != "") {
                print value
            }
        }
    ' "$log_file"
}

extract_range_value() {
    local prefix=$1
    local key=$2
    local endpoint=$3
    local log_file=$4
    awk -v prefix="$prefix" -v key="$key" -v endpoint="$endpoint" '
        index($0, prefix) {
            marker = key " = ["
            at = index($0, marker)
            if (at > 0) {
                text = substr($0, at + length(marker))
                comma = index(text, ",")
                if (endpoint == "start") {
                    value = substr(text, 1, comma - 1)
                } else {
                    value = substr(text, comma + 1)
                    sub(/\).*/, "", value)
                    sub(/^[[:space:]]+/, "", value)
                }
            }
        }
        END {
            if (value != "") {
                print value
            }
        }
    ' "$log_file"
}

extract_total_ms() {
    sed -nE 's/.*total time: ([0-9.]+)ms.*/\1/p' "$1" | tail -n 1
}

require_metric() {
    local name=$1
    local value=$2
    local log_file=$3
    [[ -n "$value" ]] || die "missing metric '$name' in $log_file"
}

require_line_count() {
    local marker=$1
    local expected=$2
    local log_file=$3
    local actual
    actual=$(line_count "$marker" "$log_file")
    [[ "$actual" == "$expected" ]] ||
        die "expected $expected '$marker' line(s), found $actual in $log_file"
}

expect_fail() {
    local name=$1
    local expected=$2
    shift 2

    local log_file="$LOG_DIR/invalid/$name.log"
    if "$BIN" "$@" > "$log_file" 2>&1; then
        die "invalid-argument test '$name' unexpectedly succeeded"
    fi
    grep -Fq -- "$expected" "$log_file" ||
        die "invalid-argument test '$name' did not report: $expected"
    printf 'invalid test %-22s PASS\n' "$name"
}

run_invalid_tests() {
    local common=(
        "${model_args[@]}"
        -p "$PROMPT"
        -ngl "${NGL:-99}"
        -c 64
        -b 64
        -ub 64
        -fa "${FLASH_ATTN:-on}"
        --seed "${SEED:-1234}"
        --temp "${TEMP:-0}"
        --diffusion-algorithm 4
        --diffusion-alg-temp 0
        --diffusion-steps 16
    )
    local valid=(
        "${common[@]}"
        --diffusion-block-length "$BLOCK_LENGTH"
        --diffusion-generated-block-schedule
        --diffusion-mbsd
        --diffusion-mbsd-trigger "$MBSD_TRIGGER"
        --diffusion-mbsd-lookahead 16
    )

    expect_fail negative-trigger \
        "--diffusion-mbsd-trigger must be non-negative" \
        "${valid[@]}" --diffusion-mbsd-trigger -1
    expect_fail negative-lookahead \
        "--diffusion-mbsd-lookahead must be non-negative" \
        "${valid[@]}" --diffusion-mbsd-lookahead -1
    expect_fail missing-generated-schedule \
        "--diffusion-mbsd requires block scheduling and --diffusion-generated-block-schedule" \
        "${common[@]}" --diffusion-block-length "$BLOCK_LENGTH" --diffusion-mbsd
    expect_fail timestep-schedule \
        "--diffusion-mbsd requires block scheduling and --diffusion-generated-block-schedule" \
        "${common[@]}" --diffusion-eps 0.001 --diffusion-mbsd
    expect_fail wrong-algorithm \
        "--diffusion-mbsd requires --diffusion-algorithm 4 and --diffusion-alg-temp 0" \
        "${valid[@]}" --diffusion-algorithm 3

    local unsupported="--diffusion-mbsd does not yet support early commit, prefix KV, the full-sequence KV oracle, staged token stabilization, CFG, or Gumbel noise"
    expect_fail early-commit "$unsupported" \
        "${valid[@]}" --diffusion-early-commit-threshold 0.9
    expect_fail prefix-kv "$unsupported" \
        "${valid[@]}" --diffusion-prefix-kv
    expect_fail full-sequence-oracle "$unsupported" \
        "${valid[@]}" --diffusion-full-sequence-kv-oracle
    expect_fail staged "$unsupported" \
        "${valid[@]}" --diffusion-staged-token-stabilization
    expect_fail cfg "$unsupported" \
        "${valid[@]}" --diffusion-cfg-scale 1
    expect_fail gumbel "$unsupported" \
        "${valid[@]}" --diffusion-add-gumbel-noise 1
}

run_variant() {
    local case_name=$1
    local ubatch=$2
    local steps=$3
    local variant=$4
    local run=$5
    local lookahead=-1
    local expected_enabled=false

    if [[ "$variant" != baseline ]]; then
        lookahead=${variant#mbsd-la}
        expected_enabled=true
    fi

    local log_file="$LOG_DIR/${case_name}-${variant}-${run}.log"
    local args=(
        "${model_args[@]}"
        -p "$PROMPT"
        -ngl "${NGL:-99}"
        -c "$ubatch"
        -b "$ubatch"
        -ub "$ubatch"
        -fa "${FLASH_ATTN:-on}"
        --seed "${SEED:-1234}"
        --temp "${TEMP:-0}"
        --top-p "${TOP_P:-0.95}"
        --diffusion-block-length "$BLOCK_LENGTH"
        --diffusion-generated-block-schedule
        --diffusion-algorithm 4
        --diffusion-alg-temp 0
        --diffusion-steps "$steps"
        --diffusion-dump-generated-tokens
    )
    if [[ "$variant" != baseline ]]; then
        args+=(
            --diffusion-mbsd
            --diffusion-mbsd-trigger "$MBSD_TRIGGER"
            --diffusion-mbsd-lookahead "$lookahead"
        )
    fi

    echo
    echo "RUN case=$case_name variant=$variant iteration=$run"
    "$BIN" "${args[@]}" 2>&1 | tee "$log_file"

    require_line_count "MBSD: enabled" 1 "$log_file"

    local total_ms main_forwards transformer_rows logit_rows
    local generated_tokens id_hash generated_masks invalid_tokens remaining_masks
    local planned_blocks blocks_started blocks_completed
    local mbsd_enabled trigger max_lookahead policy execution

    total_ms=$(extract_total_ms "$log_file")
    main_forwards=$(extract_value "forwards:" "conditional/main" "$log_file")
    transformer_rows=$(extract_value "transformer rows: main" "main" "$log_file")
    logit_rows=$(extract_value "logits: rows" "rows" "$log_file")
    generated_tokens=$(extract_value "diffusion generated tokens:" "count" "$log_file")
    id_hash=$(extract_value "diffusion generated tokens:" "id hash" "$log_file")
    generated_masks=$(extract_value "diffusion generated tokens:" "mask" "$log_file")
    invalid_tokens=$(extract_value "diffusion generated tokens:" "invalid" "$log_file")
    remaining_masks=$(extract_value "token commits:" "remaining masks" "$log_file")
    planned_blocks=$(extract_value "block schedule:" "planned blocks" "$log_file")
    blocks_started=$(extract_value "blocks:" "started" "$log_file")
    blocks_completed=$(extract_value "blocks:" "completed" "$log_file")
    mbsd_enabled=$(extract_value "MBSD: enabled" "enabled" "$log_file")
    trigger=$(extract_value "MBSD: enabled" "trigger" "$log_file")
    max_lookahead=$(extract_value "MBSD: enabled" "max lookahead" "$log_file")
    policy=$(extract_value "MBSD: enabled" "policy" "$log_file")
    execution=$(extract_value "MBSD: enabled" "execution" "$log_file")

    local metric
    for metric in \
        "total_ms:$total_ms" \
        "main_forwards:$main_forwards" \
        "transformer_rows:$transformer_rows" \
        "logit_rows:$logit_rows" \
        "generated_tokens:$generated_tokens" \
        "id_hash:$id_hash" \
        "generated_masks:$generated_masks" \
        "invalid_tokens:$invalid_tokens" \
        "remaining_masks:$remaining_masks" \
        "planned_blocks:$planned_blocks" \
        "blocks_started:$blocks_started" \
        "blocks_completed:$blocks_completed" \
        "mbsd_enabled:$mbsd_enabled" \
        "trigger:$trigger" \
        "max_lookahead:$max_lookahead" \
        "policy:$policy" \
        "execution:$execution"
    do
        require_metric "${metric%%:*}" "${metric#*:}" "$log_file"
    done

    [[ "$generated_masks" == 0 ]] || die "generated masks remain in $log_file"
    [[ "$invalid_tokens" == 0 ]] || die "invalid generated tokens in $log_file"
    [[ "$remaining_masks" == 0 ]] || die "remaining masks in $log_file"
    [[ "$blocks_started" == "$planned_blocks" ]] || die "not all blocks started in $log_file"
    [[ "$blocks_completed" == "$planned_blocks" ]] || die "not all blocks completed in $log_file"
    [[ "$mbsd_enabled" == "$expected_enabled" ]] || die "unexpected MBSD state in $log_file"
    [[ "$policy" == fixed-budget ]] || die "unexpected MBSD policy in $log_file: $policy"
    [[ "$execution" == full-sequence-reference ]] ||
        die "unexpected MBSD execution mode in $log_file: $execution"

    if [[ "$case_name" == single ]]; then
        [[ "$planned_blocks" == 1 ]] ||
            die "single case produced $planned_blocks blocks; shorten PROMPT or adjust SINGLE_UBATCH"
    else
        (( planned_blocks >= 2 )) ||
            die "multi case produced fewer than two blocks; shorten PROMPT or increase MULTI_UBATCH"
    fi

    local trigger_checks=0 lookahead_expansions=0 slides=0 slide_distance=0
    local first_start=-1 first_end=-1 last_start=-1 last_end=-1 max_end=-1
    local draft_introduced=0 draft_updates=0 draft_prediction_rows=0 draft_reevaluated=0
    local draft_accepted=0 draft_reconfirmed=0 draft_replacements=0 draft_rejected=0 draft_pending=0
    local role_current=0 role_future=0 role_context=0 role_total=0
    local logits_current=0 logits_future=0 steps_with_future=0 extra_steps=0 extra_forwards=0
    local main_steps=0 steps_saved=0 forwards_saved=0 nominal_zero_tail_slots=0
    local nominal_tail_skipped=0 final_forced_selections=0
    local future_semantic_commits=0 prefix_kv_disabled=false
    local block_order_violations=0 verification_input_errors=0 bounds_errors=0 trajectory_hash=0

    local detail_markers=(
        "MBSD windows:"
        "MBSD drafts:"
        "MBSD logical row roles:"
        "MBSD logits:"
        "MBSD schedule:"
        "MBSD invariants:"
    )
    local marker
    if [[ "$mbsd_enabled" == false ]]; then
        for marker in "${detail_markers[@]}"; do
            require_line_count "$marker" 0 "$log_file"
        done
    else
        for marker in "${detail_markers[@]}"; do
            require_line_count "$marker" 1 "$log_file"
        done

        [[ "$trigger" == "$MBSD_TRIGGER" ]] || die "unexpected trigger in $log_file"
        [[ "$max_lookahead" == "$lookahead" ]] || die "unexpected lookahead in $log_file"

        trigger_checks=$(extract_value "MBSD windows:" "trigger checks" "$log_file")
        lookahead_expansions=$(extract_value "MBSD windows:" "lookahead expansions" "$log_file")
        slides=$(extract_value "MBSD windows:" "slides" "$log_file")
        slide_distance=$(extract_value "MBSD windows:" "slide distance" "$log_file")
        first_start=$(extract_range_value "MBSD windows:" "first" start "$log_file")
        first_end=$(extract_range_value "MBSD windows:" "first" end "$log_file")
        last_start=$(extract_range_value "MBSD windows:" "last" start "$log_file")
        last_end=$(extract_range_value "MBSD windows:" "last" end "$log_file")
        max_end=$(extract_value "MBSD windows:" "max end" "$log_file")

        draft_introduced=$(extract_value "MBSD drafts:" "introduced" "$log_file")
        draft_updates=$(extract_value "MBSD drafts:" "updates" "$log_file")
        draft_prediction_rows=$(extract_value "MBSD drafts:" "prediction rows" "$log_file")
        draft_reevaluated=$(extract_value "MBSD drafts:" "re-evaluated" "$log_file")
        draft_accepted=$(extract_value "MBSD drafts:" "accepted" "$log_file")
        draft_reconfirmed=$(extract_value "MBSD drafts:" "reconfirmed" "$log_file")
        draft_replacements=$(extract_value "MBSD drafts:" "replacements" "$log_file")
        draft_rejected=$(extract_value "MBSD drafts:" "rejected" "$log_file")
        draft_pending=$(extract_value "MBSD drafts:" "pending" "$log_file")

        role_current=$(extract_value "MBSD logical row roles:" "current window" "$log_file")
        role_future=$(extract_value "MBSD logical row roles:" "future window" "$log_file")
        role_context=$(extract_value "MBSD logical row roles:" "dense context/outside" "$log_file")
        role_total=$(extract_value "MBSD logical row roles:" "dense total" "$log_file")
        logits_current=$(extract_value "MBSD logits:" "current rows" "$log_file")
        logits_future=$(extract_value "MBSD logits:" "future rows" "$log_file")

        steps_with_future=$(extract_value "MBSD schedule:" "steps with future work" "$log_file")
        extra_steps=$(extract_value "MBSD schedule:" "extra steps" "$log_file")
        extra_forwards=$(extract_value "MBSD schedule:" "extra forwards" "$log_file")
        main_steps=$(extract_value "MBSD schedule:" "main steps" "$log_file")
        steps_saved=$(extract_value "MBSD schedule:" "scheduled steps saved after draft acceptance" "$log_file")
        forwards_saved=$(extract_value "MBSD schedule:" "scheduled forwards saved" "$log_file")
        nominal_zero_tail_slots=$(extract_value "MBSD schedule:" "nominal zero-tail slots" "$log_file")
        nominal_tail_skipped=$(extract_value "MBSD schedule:" "nominal tail skipped" "$log_file")
        final_forced_selections=$(extract_value "MBSD schedule:" "final forced selections" "$log_file")

        future_semantic_commits=$(extract_value "MBSD invariants:" "future semantic commits" "$log_file")
        prefix_kv_disabled=$(extract_value "MBSD invariants:" "prefix KV disabled" "$log_file")
        block_order_violations=$(extract_value "MBSD invariants:" "block-order violations" "$log_file")
        verification_input_errors=$(extract_value "MBSD invariants:" "verification input errors" "$log_file")
        bounds_errors=$(extract_value "MBSD invariants:" "bounds errors" "$log_file")
        trajectory_hash=$(extract_value "MBSD invariants:" "trajectory hash" "$log_file")

        for metric in \
            "trigger_checks:$trigger_checks" \
            "lookahead_expansions:$lookahead_expansions" \
            "slides:$slides" \
            "slide_distance:$slide_distance" \
            "first_start:$first_start" \
            "first_end:$first_end" \
            "last_start:$last_start" \
            "last_end:$last_end" \
            "max_end:$max_end" \
            "draft_introduced:$draft_introduced" \
            "draft_updates:$draft_updates" \
            "draft_prediction_rows:$draft_prediction_rows" \
            "draft_reevaluated:$draft_reevaluated" \
            "draft_accepted:$draft_accepted" \
            "draft_reconfirmed:$draft_reconfirmed" \
            "draft_replacements:$draft_replacements" \
            "draft_rejected:$draft_rejected" \
            "draft_pending:$draft_pending" \
            "role_current:$role_current" \
            "role_future:$role_future" \
            "role_context:$role_context" \
            "role_total:$role_total" \
            "logits_current:$logits_current" \
            "logits_future:$logits_future" \
            "steps_with_future:$steps_with_future" \
            "extra_steps:$extra_steps" \
            "extra_forwards:$extra_forwards" \
            "main_steps:$main_steps" \
            "steps_saved:$steps_saved" \
            "forwards_saved:$forwards_saved" \
            "nominal_zero_tail_slots:$nominal_zero_tail_slots" \
            "nominal_tail_skipped:$nominal_tail_skipped" \
            "final_forced_selections:$final_forced_selections" \
            "future_semantic_commits:$future_semantic_commits" \
            "prefix_kv_disabled:$prefix_kv_disabled" \
            "block_order_violations:$block_order_violations" \
            "verification_input_errors:$verification_input_errors" \
            "bounds_errors:$bounds_errors" \
            "trajectory_hash:$trajectory_hash"
        do
            require_metric "${metric%%:*}" "${metric#*:}" "$log_file"
        done

        [[ "$draft_pending" == 0 ]] || die "pending drafts in $log_file"
        [[ "$future_semantic_commits" == 0 ]] || die "future semantic commit in $log_file"
        [[ "$prefix_kv_disabled" == true ]] || die "prefix KV was not disabled in $log_file"
        [[ "$block_order_violations" == 0 ]] || die "block-order violation in $log_file"
        [[ "$verification_input_errors" == 0 ]] || die "unmasked draft verification in $log_file"
        [[ "$bounds_errors" == 0 ]] || die "MBSD bounds error in $log_file"
        (( draft_introduced == draft_reevaluated )) || die "draft lifecycle mismatch in $log_file"
        (( draft_reevaluated == draft_accepted + draft_rejected )) ||
            die "draft acceptance accounting mismatch in $log_file"
        (( draft_accepted == draft_reconfirmed )) ||
            die "accepted draft accounting mismatch in $log_file"
        (( draft_replacements <= draft_updates )) ||
            die "draft replacement accounting mismatch in $log_file"
        (( role_current + role_future + role_context == role_total )) ||
            die "MBSD transformer role accounting mismatch in $log_file"
        (( role_total == transformer_rows )) ||
            die "MBSD role rows do not match transformer rows in $log_file"
        (( logits_current + logits_future == logit_rows )) ||
            die "MBSD logit role accounting mismatch in $log_file"
        [[ "$extra_steps" == 0 && "$extra_forwards" == 0 ]] ||
            die "MBSD unexpectedly added steps or forwards in $log_file"
        (( main_steps + steps_saved + nominal_tail_skipped == steps )) ||
            die "MBSD step accounting mismatch in $log_file"
        (( main_steps == main_forwards && steps_saved == forwards_saved )) ||
            die "MBSD forward accounting mismatch in $log_file"
        (( nominal_tail_skipped <= nominal_zero_tail_slots )) ||
            die "MBSD nominal tail accounting mismatch in $log_file"

        local n_input=$((ubatch - generated_tokens))
        (( first_start >= n_input && first_start <= first_end && first_end <= ubatch )) ||
            die "invalid first MBSD window in $log_file"
        (( last_start >= first_start && last_start <= last_end && last_end <= ubatch )) ||
            die "invalid last MBSD window in $log_file"
        (( max_end >= first_end && max_end >= last_end && max_end <= ubatch )) ||
            die "invalid maximum MBSD window end in $log_file"

        if [[ "$case_name" == single ]]; then
            [[ "$lookahead_expansions" == 0 ]] || die "single block expanded lookahead in $log_file"
            [[ "$draft_introduced" == 0 && "$draft_updates" == 0 && "$draft_prediction_rows" == 0 ]] ||
                die "single block produced draft work in $log_file"
            [[ "$draft_reevaluated" == 0 && "$draft_accepted" == 0 && "$draft_rejected" == 0 ]] ||
                die "single block verified drafts in $log_file"
            [[ "$role_future" == 0 && "$logits_future" == 0 && "$steps_with_future" == 0 ]] ||
                die "single block reported future work in $log_file"
            [[ "$steps_saved" == 0 && "$forwards_saved" == 0 ]] ||
                die "single block reported speculative savings in $log_file"
        else
            if [[ "$lookahead" == 0 ]]; then
                [[ "$lookahead_expansions" == 0 ]] ||
                    die "lookahead 0 performed proactive expansion in $log_file"
            else
                (( lookahead_expansions > 0 )) ||
                    die "positive lookahead did not perform proactive expansion in $log_file"
                (( draft_introduced > 0 && role_future > 0 && logits_future > 0 && steps_with_future > 0 )) ||
                    die "multi-block proactive MBSD path was not exercised in $log_file"
            fi
        fi
    fi

    row=(
        "$case_name" "$variant" "$lookahead" "$run" "$total_ms" "$main_forwards"
        "$transformer_rows" "$logit_rows" "$generated_tokens" "$id_hash" "$generated_masks"
        "$invalid_tokens" "$remaining_masks" "$planned_blocks" "$blocks_started" "$blocks_completed"
        "$mbsd_enabled" "$trigger" "$max_lookahead" "$policy" "$execution"
        "$trigger_checks" "$lookahead_expansions"
        "$slides" "$slide_distance" "$first_start" "$first_end" "$last_start" "$last_end" "$max_end"
        "$draft_introduced" "$draft_updates" "$draft_prediction_rows" "$draft_reevaluated"
        "$draft_accepted" "$draft_reconfirmed" "$draft_replacements" "$draft_rejected" "$draft_pending"
        "$role_current" "$role_future" "$role_context" "$role_total" "$logits_current" "$logits_future"
        "$steps_with_future" "$extra_steps" "$extra_forwards" "$main_steps" "$steps_saved"
        "$forwards_saved" "$nominal_zero_tail_slots" "$nominal_tail_skipped"
        "$final_forced_selections" "$future_semantic_commits" "$prefix_kv_disabled"
        "$block_order_violations" "$verification_input_errors" "$bounds_errors"
        "$trajectory_hash"
    )
    write_tsv_row "${row[@]}" >> "$SUMMARY"

    printf 'PASS case=%s variant=%s run=%s ms=%s forwards=%s drafts=%s accepted=%s saved=%s\n' \
        "$case_name" "$variant" "$run" "$total_ms" "$main_forwards" \
        "$draft_introduced" "$draft_accepted" "$forwards_saved"
}

column_values() {
    local case_name=$1
    local variant=$2
    local column=$3
    awk -F '\t' -v case_name="$case_name" -v variant="$variant" -v column="$column" '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                index_by_name[$i] = i
            }
            next
        }
        $1 == case_name && $2 == variant {
            print $index_by_name[column]
        }
    ' "$SUMMARY"
}

distinct_count() {
    sort -u | awk 'END { print NR + 0 }'
}

median_values() {
    sort -n | awk '
        { values[NR] = $1 }
        END {
            if (NR == 0) {
                exit 1
            }
            if (NR % 2 == 1) {
                printf "%.2f", values[(NR + 1) / 2]
            } else {
                printf "%.2f", (values[NR / 2] + values[NR / 2 + 1]) / 2
            }
        }
    '
}

verify_determinism() {
    local case_name=$1
    local variant=$2
    local rows id_hashes forward_counts trajectory_hashes

    rows=$(column_values "$case_name" "$variant" run | awk 'END { print NR + 0 }')
    [[ "$rows" == "$REPEATS" ]] || die "missing repeated rows for $case_name/$variant"

    id_hashes=$(column_values "$case_name" "$variant" id_hash | distinct_count)
    [[ "$id_hashes" == 1 ]] || die "generated token hashes are not deterministic for $case_name/$variant"

    forward_counts=$(column_values "$case_name" "$variant" main_forwards | distinct_count)
    [[ "$forward_counts" == 1 ]] || die "forward counts are not deterministic for $case_name/$variant"

    if [[ "$variant" != baseline ]]; then
        trajectory_hashes=$(column_values "$case_name" "$variant" trajectory_hash | distinct_count)
        [[ "$trajectory_hashes" == 1 ]] || die "MBSD trajectory hashes are not deterministic for $case_name/$variant"
    fi
}

echo "===== invalid parameter tests ====="
run_invalid_tests

variants=(baseline mbsd-la0 mbsd-la16 mbsd-la32)
for case_name in single multi; do
    if [[ "$case_name" == single ]]; then
        ubatch=$SINGLE_UBATCH
        steps=$SINGLE_STEPS
    else
        ubatch=$MULTI_UBATCH
        steps=$MULTI_STEPS
    fi

    for ((run = 1; run <= REPEATS; run++)); do
        offset=$(((run - 1) % ${#variants[@]}))
        for ((slot = 0; slot < ${#variants[@]}; slot++)); do
            index=$(((offset + slot) % ${#variants[@]}))
            run_variant "$case_name" "$ubatch" "$steps" "${variants[$index]}" "$run"
        done
    done
done

echo
echo "===== determinism gates ====="
for case_name in single multi; do
    for variant in "${variants[@]}"; do
        verify_determinism "$case_name" "$variant"
        echo "PASS deterministic case=$case_name variant=$variant"
    done
done

echo
echo "===== medians ====="
printf '%-8s %-12s %12s %12s %12s\n' case variant median_ms forwards draft_tokens
for case_name in single multi; do
    for variant in "${variants[@]}"; do
        median_ms=$(column_values "$case_name" "$variant" total_ms | median_values)
        median_forwards=$(column_values "$case_name" "$variant" main_forwards | median_values)
        median_drafts=$(column_values "$case_name" "$variant" draft_introduced | median_values)
        printf '%-8s %-12s %12s %12s %12s\n' \
            "$case_name" "$variant" "$median_ms" "$median_forwards" "$median_drafts"
    done
done

echo
echo "MBSD_BENCH_GATE=PASS"
echo "SUMMARY=$SUMMARY"
echo "LOG_DIR=$LOG_DIR"
