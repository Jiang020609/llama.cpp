#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

BIN=${BIN:-./build/bin/llama-diffusion-cli}
HF_MODEL=${HF_MODEL:-keisuke-miyako/Dream-v0-Instruct-7B-gguf-q4_k_m:Q4_K_M}
MODEL_PATH=${MODEL_PATH:-}
PROMPT=${PROMPT:-Write a short Python function that adds two numbers.}
TEMP=${TEMP:-0}
CACHE_TYPE_K=${CACHE_TYPE_K:-f32}
CACHE_TYPE_V=${CACHE_TYPE_V:-f32}
REPEATS=${REPEATS:-2}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
MBSD_TRIGGER=${MBSD_TRIGGER:-8}
MBSD_BASELINE_PARITY=${MBSD_BASELINE_PARITY:-require}
LIFECYCLE_MATRIX=${LIFECYCLE_MATRIX:-0}
LIFECYCLE_VISIBILITY_THRESHOLD=0.7
LIFECYCLE_STABILITY_THRESHOLD=0.9
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
[[ "$TEMP" =~ ^0([.]0+)?$ ]] || die "TEMP must be 0 for deterministic parity gates"
[[ "$MBSD_BASELINE_PARITY" == report || "$MBSD_BASELINE_PARITY" == require ]] ||
    die "MBSD_BASELINE_PARITY must be report or require"
[[ "$LIFECYCLE_MATRIX" == 0 || "$LIFECYCLE_MATRIX" == 1 ]] ||
    die "LIFECYCLE_MATRIX must be 0 or 1"
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
LIFECYCLE_SUMMARY="$LOG_DIR/lifecycle-summary.tsv"

write_tsv_row() {
    local IFS=$'\t'
    printf '%s\n' "$*"
}

header=(
    case variant lookahead run total_ms main_forwards transformer_rows logit_rows
    generated_tokens id_hash generated_masks invalid_tokens post_eog_nonterminal remaining_masks planned_blocks
    blocks_started blocks_completed mbsd_enabled trigger max_lookahead policy execution
    fresh_kv_requested fresh_kv_active physical_compact_active fresh_kv_pre_forward_clears
    cache_invariant_errors
    lifecycle_enabled lifecycle_observer_only lifecycle_staged_integration
    lifecycle_visibility_threshold lifecycle_stability_threshold
    lifecycle_invisible lifecycle_visible lifecycle_stable lifecycle_state_total
    lifecycle_remaining_masks lifecycle_mask_parity_errors
    lifecycle_iv_to_v lifecycle_iv_to_s lifecycle_v_to_s lifecycle_forced_visible
    lifecycle_deferred_stable lifecycle_illegal_transitions
    lifecycle_confidence_current_updates lifecycle_confidence_draft_updates
    lifecycle_confidence_invalidations lifecycle_confidence_errors
    lifecycle_residency_none lifecycle_residency_mutable lifecycle_residency_stable
    lifecycle_residency_total lifecycle_duplicate_ownership lifecycle_ownership_errors
    lifecycle_stable_mutations lifecycle_version_errors lifecycle_future_semantic_commits
    lifecycle_future_cache_writes lifecycle_pending_entries lifecycle_state_accounting_errors
    lifecycle_refresh_enqueued lifecycle_refresh_drained lifecycle_refresh_max_depth
    lifecycle_refresh_snapshots_pending lifecycle_refresh_duplicate_errors
    lifecycle_refresh_ineligible_errors lifecycle_refresh_future_errors lifecycle_refresh_stable_errors
    lifecycle_last_refresh_updates lifecycle_last_refresh_errors lifecycle_queue_hash
    lifecycle_state_hash lifecycle_ownership_hash lifecycle_snapshots
    lifecycle_actual_cache_reads lifecycle_actual_cache_writes lifecycle_actual_refreshes
    lifecycle_actual_merges lifecycle_actual_async_tasks lifecycle_actual_row_saving
    trigger_checks lookahead_expansions slides slide_distance first_start first_end last_start last_end max_end
    draft_introduced draft_updates draft_prediction_rows draft_reevaluated draft_accepted
    draft_reconfirmed draft_replacements draft_rejected draft_pending role_current role_future
    role_context role_total logits_current logits_future steps_with_future extra_steps
    extra_forwards main_steps steps_saved forwards_saved nominal_zero_tail_slots
    nominal_tail_skipped final_forced_selections
    future_semantic_commits explicit_prefix_kv_disabled
    block_order_violations verification_input_errors bounds_errors post_eog_corrections trajectory_hash
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

extract_scoped_value() {
    local prefix=$1
    local key=$2
    local log_file=$3
    awk -v prefix="$prefix" -v key="$key" '
        index($0, prefix) {
            text = substr($0, index($0, prefix) + length(prefix))
            count = split(text, fields, /,[[:space:]]*/)
            marker = key " = "
            for (i = 1; i <= count; i++) {
                sub(/^[[:space:]]+/, "", fields[i])
                if (index(fields[i], marker) == 1) {
                    value = substr(fields[i], length(marker) + 1)
                    sub(/[[:space:]]+$/, "", value)
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

extract_generated_token_ids() {
    local log_file=$1
    awk '
        BEGIN {
            marker = "diffusion generated token ids: ["
        }
        index($0, marker) {
            lines++
            value = substr($0, index($0, marker) + length(marker))
            sub(/\].*$/, "", value)
            gsub(/[[:space:]]/, "", value)
        }
        END {
            if (lines != 1 || value == "") {
                exit 1
            }
            print value
        }
    ' "$log_file"
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

verify_fresh_physical() {
    local log_file=$1
    local ubatch=$2
    local marker

    for marker in \
        "MBSD fresh KV:" \
        "MBSD physical rows:" \
        "MBSD physical batches:" \
        "completed-prefix cache:" \
        "diffusion KV:"
    do
        require_line_count "$marker" 1 "$log_file"
    done

    local main_forwards requested execution_active physical_compact fresh_forwards cache_invariant_errors
    local dense_equivalent main_submitted cache_maintenance total_submitted main_saved net_saved
    local cache_forwards
    local physical_batches min_rows max_rows prefix_rows_reused fresh_kv_pre_forward_clears mapping_errors
    local cache_enabled cache_owner prompt_prefills transition_seals kv_mode kv_clears

    main_forwards=$(extract_value "forwards:" "conditional/main" "$log_file")
    requested=$(extract_value "MBSD fresh KV:" "requested" "$log_file")
    execution_active=$(extract_value "MBSD fresh KV:" "execution active" "$log_file")
    physical_compact=$(extract_value "MBSD fresh KV:" "physical compact active" "$log_file")
    fresh_forwards=$(extract_value "MBSD fresh KV:" "forwards" "$log_file")
    cache_invariant_errors=$(extract_value "MBSD fresh KV:" "cache invariant errors" "$log_file")

    dense_equivalent=$(extract_value "MBSD physical rows:" "dense equivalent" "$log_file")
    main_submitted=$(extract_value "MBSD physical rows:" "main submitted" "$log_file")
    cache_maintenance=$(extract_value "MBSD physical rows:" "cache maintenance" "$log_file")
    total_submitted=$(extract_value "MBSD physical rows:" "total submitted" "$log_file")
    main_saved=$(extract_value "MBSD physical rows:" "main saved" "$log_file")
    net_saved=$(extract_value "MBSD physical rows:" "net saved" "$log_file")
    cache_forwards=$(extract_value "forwards:" "cache maintenance" "$log_file")

    physical_batches=$(extract_value "MBSD physical batches:" "main" "$log_file")
    min_rows=$(extract_value "MBSD physical batches:" "min rows" "$log_file")
    max_rows=$(extract_value "MBSD physical batches:" "max rows" "$log_file")
    prefix_rows_reused=$(extract_value "MBSD physical batches:" "prefix rows reused" "$log_file")
    fresh_kv_pre_forward_clears=$(
        extract_value "MBSD physical batches:" "fresh KV pre-forward clears" "$log_file"
    )
    mapping_errors=$(extract_value "MBSD physical batches:" "mapping errors" "$log_file")

    cache_enabled=$(extract_value "completed-prefix cache:" "enabled" "$log_file")
    cache_owner=$(extract_value "completed-prefix cache:" "owner" "$log_file")
    prompt_prefills=$(extract_value "completed-prefix cache:" "prompt prefills" "$log_file")
    transition_seals=$(extract_value "completed-prefix cache:" "transition seals" "$log_file")
    kv_mode=$(extract_value "diffusion KV:" "mode" "$log_file")
    kv_clears=$(extract_value "diffusion KV:" "full-sequence pre-forward clears" "$log_file")

    local metric
    for metric in \
        "main_forwards:$main_forwards" \
        "requested:$requested" \
        "execution_active:$execution_active" \
        "physical_compact:$physical_compact" \
        "fresh_forwards:$fresh_forwards" \
        "cache_invariant_errors:$cache_invariant_errors" \
        "dense_equivalent:$dense_equivalent" \
        "main_submitted:$main_submitted" \
        "cache_maintenance:$cache_maintenance" \
        "total_submitted:$total_submitted" \
        "cache_forwards:$cache_forwards" \
        "main_saved:$main_saved" \
        "net_saved:$net_saved" \
        "physical_batches:$physical_batches" \
        "min_rows:$min_rows" \
        "max_rows:$max_rows" \
        "prefix_rows_reused:$prefix_rows_reused" \
        "fresh_kv_pre_forward_clears:$fresh_kv_pre_forward_clears" \
        "mapping_errors:$mapping_errors" \
        "cache_enabled:$cache_enabled" \
        "cache_owner:$cache_owner" \
        "prompt_prefills:$prompt_prefills" \
        "transition_seals:$transition_seals" \
        "kv_mode:$kv_mode" \
        "kv_clears:$kv_clears"
    do
        require_metric "${metric%%:*}" "${metric#*:}" "$log_file"
    done

    [[ "$requested" == true && "$execution_active" == true && "$physical_compact" == false ]] ||
        die "fresh full-sequence path was not selected in $log_file"
    [[ "$fresh_forwards" == "$main_forwards" ]] ||
        die "fresh path forward count mismatch in $log_file"
    [[ "$cache_invariant_errors" == 0 ]] ||
        die "fresh path cache invariant failure in $log_file"
    (( dense_equivalent == main_forwards * ubatch )) ||
        die "dense-equivalent row count mismatch in $log_file"
    [[ "$main_submitted" == "$dense_equivalent" && "$total_submitted" == "$dense_equivalent" &&
       "$cache_maintenance" == 0 && "$cache_forwards" == 0 ]] ||
        die "fresh path did not submit exactly the dense rows in $log_file"
    [[ "$main_saved" == 0 && "$net_saved" == 0 ]] ||
        die "fresh path unexpectedly reported row savings in $log_file"
    [[ "$physical_batches" == "$main_forwards" && "$min_rows" == "$ubatch" && "$max_rows" == "$ubatch" ]] ||
        die "fresh path used a truncated physical batch in $log_file"
    [[ "$prefix_rows_reused" == 0 && "$fresh_kv_pre_forward_clears" == "$main_forwards" &&
       "$mapping_errors" == 0 ]] ||
        die "fresh path reused prefix rows or missed a KV clear in $log_file"
    [[ "$cache_enabled" == false && "$cache_owner" == none && "$prompt_prefills" == 0 && "$transition_seals" == 0 ]] ||
        die "completed-prefix cache was active in $log_file"
    [[ "$kv_mode" == mbsd-fresh-full-sequence && "$kv_clears" == "$main_forwards" ]] ||
        die "unexpected diffusion KV mode in $log_file"
}

verify_lifecycle_bookkeeping() {
    local log_file=$1
    local generated_tokens=$2
    local expected=$3
    local marker completed_iterations
    local detail_markers=(
        "MBSD lifecycle states:"
        "MBSD lifecycle transitions:"
        "MBSD lifecycle confidence:"
        "MBSD lifecycle residency:"
        "MBSD lifecycle invariants:"
        "MBSD lifecycle refresh queue:"
        "MBSD lifecycle refresh state:"
        "MBSD lifecycle hashes:"
        "MBSD lifecycle actual:"
    )

    lifecycle_enabled=false
    lifecycle_observer_only=false
    lifecycle_staged_integration=false
    lifecycle_visibility_threshold=0
    lifecycle_stability_threshold=0
    lifecycle_invisible=0
    lifecycle_visible=0
    lifecycle_stable=0
    lifecycle_state_total=0
    lifecycle_remaining_masks=0
    lifecycle_mask_parity_errors=0
    lifecycle_iv_to_v=0
    lifecycle_iv_to_s=0
    lifecycle_v_to_s=0
    lifecycle_forced_visible=0
    lifecycle_deferred_stable=0
    lifecycle_illegal_transitions=0
    lifecycle_confidence_current_updates=0
    lifecycle_confidence_draft_updates=0
    lifecycle_confidence_invalidations=0
    lifecycle_confidence_errors=0
    lifecycle_residency_none=0
    lifecycle_residency_mutable=0
    lifecycle_residency_stable=0
    lifecycle_residency_total=0
    lifecycle_duplicate_ownership=0
    lifecycle_ownership_errors=0
    lifecycle_stable_mutations=0
    lifecycle_version_errors=0
    lifecycle_future_semantic_commits=0
    lifecycle_future_cache_writes=0
    lifecycle_pending_entries=0
    lifecycle_state_accounting_errors=0
    lifecycle_refresh_enqueued=0
    lifecycle_refresh_drained=0
    lifecycle_refresh_max_depth=0
    lifecycle_refresh_snapshots_pending=0
    lifecycle_refresh_duplicate_errors=0
    lifecycle_refresh_ineligible_errors=0
    lifecycle_refresh_future_errors=0
    lifecycle_refresh_stable_errors=0
    lifecycle_last_refresh_updates=0
    lifecycle_last_refresh_errors=0
    lifecycle_queue_hash=0
    lifecycle_state_hash=0
    lifecycle_ownership_hash=0
    lifecycle_snapshots=0
    lifecycle_actual_cache_reads=0
    lifecycle_actual_cache_writes=0
    lifecycle_actual_refreshes=0
    lifecycle_actual_merges=0
    lifecycle_actual_async_tasks=0
    lifecycle_actual_row_saving=0

    if [[ "$expected" == absent ]]; then
        require_line_count "MBSD lifecycle:" 0 "$log_file"
        for marker in "${detail_markers[@]}"; do
            require_line_count "$marker" 0 "$log_file"
        done
        return
    fi

    require_line_count "MBSD lifecycle:" 1 "$log_file"
    lifecycle_enabled=$(extract_scoped_value "MBSD lifecycle:" "enabled" "$log_file")
    lifecycle_observer_only=$(extract_scoped_value "MBSD lifecycle:" "observer only" "$log_file")
    lifecycle_staged_integration=$(extract_scoped_value "MBSD lifecycle:" "staged integration" "$log_file")
    lifecycle_visibility_threshold=$(extract_scoped_value "MBSD lifecycle:" "visibility threshold" "$log_file")
    lifecycle_stability_threshold=$(extract_scoped_value "MBSD lifecycle:" "stability threshold" "$log_file")
    require_metric lifecycle_enabled "$lifecycle_enabled" "$log_file"
    require_metric lifecycle_observer_only "$lifecycle_observer_only" "$log_file"
    require_metric lifecycle_staged_integration "$lifecycle_staged_integration" "$log_file"
    require_metric lifecycle_visibility_threshold "$lifecycle_visibility_threshold" "$log_file"
    require_metric lifecycle_stability_threshold "$lifecycle_stability_threshold" "$log_file"

    if [[ "$expected" == disabled ]]; then
        [[ "$lifecycle_enabled" == false && "$lifecycle_observer_only" == false &&
           "$lifecycle_staged_integration" == false ]] ||
            die "lifecycle bookkeeping unexpectedly active in $log_file"
        for marker in "${detail_markers[@]}"; do
            require_line_count "$marker" 0 "$log_file"
        done
        return
    fi

    [[ "$expected" == enabled ]] || die "invalid lifecycle expectation: $expected"
    [[ "$lifecycle_enabled" == true && "$lifecycle_observer_only" == true &&
       "$lifecycle_staged_integration" == true ]] ||
        die "lifecycle bookkeeping observer was not active in $log_file"
    [[ "$lifecycle_visibility_threshold" == 0.700 && "$lifecycle_stability_threshold" == 0.900 ]] ||
        die "unexpected lifecycle thresholds in $log_file"
    for marker in "${detail_markers[@]}"; do
        require_line_count "$marker" 1 "$log_file"
    done

    lifecycle_invisible=$(extract_scoped_value "MBSD lifecycle states:" "invisible" "$log_file")
    lifecycle_visible=$(extract_scoped_value "MBSD lifecycle states:" "visible" "$log_file")
    lifecycle_stable=$(extract_scoped_value "MBSD lifecycle states:" "stable" "$log_file")
    lifecycle_state_total=$(extract_scoped_value "MBSD lifecycle states:" "total" "$log_file")
    lifecycle_remaining_masks=$(extract_scoped_value "MBSD lifecycle states:" "remaining masks" "$log_file")
    lifecycle_mask_parity_errors=$(extract_scoped_value "MBSD lifecycle states:" "mask parity errors" "$log_file")

    lifecycle_iv_to_v=$(extract_scoped_value "MBSD lifecycle transitions:" "invisible to visible" "$log_file")
    lifecycle_iv_to_s=$(extract_scoped_value "MBSD lifecycle transitions:" "invisible to stable" "$log_file")
    lifecycle_v_to_s=$(extract_scoped_value "MBSD lifecycle transitions:" "visible to stable" "$log_file")
    lifecycle_forced_visible=$(extract_scoped_value "MBSD lifecycle transitions:" "forced visible" "$log_file")
    lifecycle_deferred_stable=$(extract_scoped_value "MBSD lifecycle transitions:" "deferred stable" "$log_file")
    lifecycle_illegal_transitions=$(extract_scoped_value "MBSD lifecycle transitions:" "illegal transitions" "$log_file")

    lifecycle_confidence_current_updates=$(extract_scoped_value "MBSD lifecycle confidence:" "current updates" "$log_file")
    lifecycle_confidence_draft_updates=$(extract_scoped_value "MBSD lifecycle confidence:" "draft updates" "$log_file")
    lifecycle_confidence_invalidations=$(extract_scoped_value "MBSD lifecycle confidence:" "invalidations" "$log_file")
    lifecycle_confidence_errors=$(extract_scoped_value "MBSD lifecycle confidence:" "errors" "$log_file")

    lifecycle_residency_none=$(extract_scoped_value "MBSD lifecycle residency:" "none" "$log_file")
    lifecycle_residency_mutable=$(extract_scoped_value "MBSD lifecycle residency:" "mutable" "$log_file")
    lifecycle_residency_stable=$(extract_scoped_value "MBSD lifecycle residency:" "long-term stable" "$log_file")
    lifecycle_residency_total=$(extract_scoped_value "MBSD lifecycle residency:" "total" "$log_file")
    lifecycle_duplicate_ownership=$(extract_scoped_value "MBSD lifecycle residency:" "duplicate ownership" "$log_file")
    lifecycle_ownership_errors=$(extract_scoped_value "MBSD lifecycle residency:" "ownership errors" "$log_file")

    lifecycle_stable_mutations=$(extract_scoped_value "MBSD lifecycle invariants:" "stable mutations" "$log_file")
    lifecycle_version_errors=$(extract_scoped_value "MBSD lifecycle invariants:" "version errors" "$log_file")
    lifecycle_future_semantic_commits=$(extract_scoped_value "MBSD lifecycle invariants:" "future semantic commits" "$log_file")
    lifecycle_future_cache_writes=$(extract_scoped_value "MBSD lifecycle invariants:" "future cache writes" "$log_file")
    lifecycle_pending_entries=$(extract_scoped_value "MBSD lifecycle invariants:" "pending entries" "$log_file")
    lifecycle_state_accounting_errors=$(extract_scoped_value "MBSD lifecycle invariants:" "state accounting errors" "$log_file")

    lifecycle_refresh_enqueued=$(extract_scoped_value "MBSD lifecycle refresh queue:" "enqueued" "$log_file")
    lifecycle_refresh_drained=$(extract_scoped_value "MBSD lifecycle refresh queue:" "observer drained" "$log_file")
    lifecycle_refresh_max_depth=$(extract_scoped_value "MBSD lifecycle refresh queue:" "max depth" "$log_file")
    lifecycle_refresh_snapshots_pending=$(extract_scoped_value "MBSD lifecycle refresh queue:" "snapshots with pending" "$log_file")
    lifecycle_refresh_duplicate_errors=$(extract_scoped_value "MBSD lifecycle refresh queue:" "duplicate errors" "$log_file")
    lifecycle_refresh_ineligible_errors=$(extract_scoped_value "MBSD lifecycle refresh queue:" "ineligible errors" "$log_file")
    lifecycle_refresh_future_errors=$(extract_scoped_value "MBSD lifecycle refresh queue:" "future errors" "$log_file")
    lifecycle_refresh_stable_errors=$(extract_scoped_value "MBSD lifecycle refresh queue:" "stable errors" "$log_file")

    lifecycle_last_refresh_updates=$(extract_scoped_value "MBSD lifecycle refresh state:" "last refresh updates" "$log_file")
    lifecycle_last_refresh_errors=$(extract_scoped_value "MBSD lifecycle refresh state:" "last refresh errors" "$log_file")
    lifecycle_queue_hash=$(extract_scoped_value "MBSD lifecycle refresh state:" "queue hash" "$log_file")

    lifecycle_state_hash=$(extract_scoped_value "MBSD lifecycle hashes:" "state hash" "$log_file")
    lifecycle_ownership_hash=$(extract_scoped_value "MBSD lifecycle hashes:" "ownership hash" "$log_file")
    lifecycle_snapshots=$(extract_scoped_value "MBSD lifecycle hashes:" "snapshots" "$log_file")

    lifecycle_actual_cache_reads=$(extract_scoped_value "MBSD lifecycle actual:" "cache reads" "$log_file")
    lifecycle_actual_cache_writes=$(extract_scoped_value "MBSD lifecycle actual:" "cache writes" "$log_file")
    lifecycle_actual_refreshes=$(extract_scoped_value "MBSD lifecycle actual:" "refreshes" "$log_file")
    lifecycle_actual_merges=$(extract_scoped_value "MBSD lifecycle actual:" "merges" "$log_file")
    lifecycle_actual_async_tasks=$(extract_scoped_value "MBSD lifecycle actual:" "async tasks" "$log_file")
    lifecycle_actual_row_saving=$(extract_scoped_value "MBSD lifecycle actual:" "row saving" "$log_file")
    completed_iterations=$(extract_scoped_value "iterations:" "completed" "$log_file")

    local metric
    for metric in \
        "invisible:$lifecycle_invisible" \
        "visible:$lifecycle_visible" \
        "stable:$lifecycle_stable" \
        "state_total:$lifecycle_state_total" \
        "lifecycle_remaining_masks:$lifecycle_remaining_masks" \
        "mask_parity_errors:$lifecycle_mask_parity_errors" \
        "iv_to_v:$lifecycle_iv_to_v" \
        "iv_to_s:$lifecycle_iv_to_s" \
        "v_to_s:$lifecycle_v_to_s" \
        "forced_visible:$lifecycle_forced_visible" \
        "deferred_stable:$lifecycle_deferred_stable" \
        "illegal_transitions:$lifecycle_illegal_transitions" \
        "confidence_current_updates:$lifecycle_confidence_current_updates" \
        "confidence_draft_updates:$lifecycle_confidence_draft_updates" \
        "confidence_invalidations:$lifecycle_confidence_invalidations" \
        "confidence_errors:$lifecycle_confidence_errors" \
        "residency_none:$lifecycle_residency_none" \
        "residency_mutable:$lifecycle_residency_mutable" \
        "residency_stable:$lifecycle_residency_stable" \
        "residency_total:$lifecycle_residency_total" \
        "duplicate_ownership:$lifecycle_duplicate_ownership" \
        "ownership_errors:$lifecycle_ownership_errors" \
        "stable_mutations:$lifecycle_stable_mutations" \
        "version_errors:$lifecycle_version_errors" \
        "future_semantic_commits:$lifecycle_future_semantic_commits" \
        "future_cache_writes:$lifecycle_future_cache_writes" \
        "pending_entries:$lifecycle_pending_entries" \
        "state_accounting_errors:$lifecycle_state_accounting_errors" \
        "refresh_enqueued:$lifecycle_refresh_enqueued" \
        "refresh_drained:$lifecycle_refresh_drained" \
        "refresh_max_depth:$lifecycle_refresh_max_depth" \
        "refresh_snapshots_pending:$lifecycle_refresh_snapshots_pending" \
        "refresh_duplicate_errors:$lifecycle_refresh_duplicate_errors" \
        "refresh_ineligible_errors:$lifecycle_refresh_ineligible_errors" \
        "refresh_future_errors:$lifecycle_refresh_future_errors" \
        "refresh_stable_errors:$lifecycle_refresh_stable_errors" \
        "last_refresh_updates:$lifecycle_last_refresh_updates" \
        "last_refresh_errors:$lifecycle_last_refresh_errors" \
        "queue_hash:$lifecycle_queue_hash" \
        "state_hash:$lifecycle_state_hash" \
        "ownership_hash:$lifecycle_ownership_hash" \
        "snapshots:$lifecycle_snapshots" \
        "actual_cache_reads:$lifecycle_actual_cache_reads" \
        "actual_cache_writes:$lifecycle_actual_cache_writes" \
        "actual_refreshes:$lifecycle_actual_refreshes" \
        "actual_merges:$lifecycle_actual_merges" \
        "actual_async_tasks:$lifecycle_actual_async_tasks" \
        "actual_row_saving:$lifecycle_actual_row_saving" \
        "completed_iterations:$completed_iterations"
    do
        require_metric "${metric%%:*}" "${metric#*:}" "$log_file"
    done

    (( lifecycle_invisible + lifecycle_visible + lifecycle_stable == generated_tokens )) ||
        die "lifecycle state count does not match generated tokens in $log_file"
    [[ "$lifecycle_state_total" == "$generated_tokens" ]] ||
        die "lifecycle state total does not match generated tokens in $log_file"
    [[ "$lifecycle_invisible" == "$lifecycle_remaining_masks" &&
       "$lifecycle_remaining_masks" == 0 && "$lifecycle_mask_parity_errors" == 0 ]] ||
        die "lifecycle invisible/mask parity invariant failed in $log_file"
    (( lifecycle_residency_none + lifecycle_residency_mutable + lifecycle_residency_stable == generated_tokens )) ||
        die "lifecycle residency count does not match generated tokens in $log_file"
    [[ "$lifecycle_residency_total" == "$generated_tokens" ]] ||
        die "lifecycle residency total does not match generated tokens in $log_file"
    [[ "$lifecycle_residency_none" == "$lifecycle_invisible" &&
       "$lifecycle_residency_mutable" == "$lifecycle_visible" &&
       "$lifecycle_residency_stable" == "$lifecycle_stable" ]] ||
        die "lifecycle state/residency mapping failed in $log_file"
    [[ "$lifecycle_duplicate_ownership" == 0 && "$lifecycle_ownership_errors" == 0 ]] ||
        die "lifecycle ownership invariant failed in $log_file"
    [[ "$lifecycle_illegal_transitions" == 0 && "$lifecycle_stable_mutations" == 0 &&
       "$lifecycle_version_errors" == 0 && "$lifecycle_future_semantic_commits" == 0 &&
       "$lifecycle_future_cache_writes" == 0 && "$lifecycle_pending_entries" == 0 &&
       "$lifecycle_state_accounting_errors" == 0 && "$lifecycle_confidence_errors" == 0 ]] ||
        die "lifecycle state invariant failed in $log_file"
    (( lifecycle_confidence_current_updates > 0 )) ||
        die "lifecycle observer did not consume current-token confidence in $log_file"
    [[ "$lifecycle_refresh_enqueued" == "$lifecycle_refresh_drained" &&
       "$lifecycle_refresh_duplicate_errors" == 0 &&
       "$lifecycle_refresh_ineligible_errors" == 0 &&
       "$lifecycle_refresh_future_errors" == 0 && "$lifecycle_refresh_stable_errors" == 0 &&
       "$lifecycle_last_refresh_updates" == 0 && "$lifecycle_last_refresh_errors" == 0 ]] ||
        die "lifecycle observer refresh queue invariant failed in $log_file"
    [[ "$lifecycle_state_hash" != 0 && "$lifecycle_ownership_hash" != 0 &&
       "$lifecycle_queue_hash" != 0 ]] ||
        die "lifecycle observer produced a zero hash in $log_file"
    (( lifecycle_snapshots == completed_iterations + 2 )) ||
        die "lifecycle snapshot count does not match completed iterations in $log_file"
    [[ "$lifecycle_actual_cache_reads" == 0 && "$lifecycle_actual_cache_writes" == 0 &&
       "$lifecycle_actual_refreshes" == 0 && "$lifecycle_actual_merges" == 0 &&
       "$lifecycle_actual_async_tasks" == 0 && "$lifecycle_actual_row_saving" == 0 ]] ||
        die "lifecycle observer performed physical cache work in $log_file"
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
        -ctk "$CACHE_TYPE_K"
        -ctv "$CACHE_TYPE_V"
        --seed "${SEED:-1234}"
        --temp "$TEMP"
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

    expect_fail fresh-kv-requires-mbsd \
        "--diffusion-mbsd-fresh-kv requires --diffusion-mbsd" \
        "${common[@]}" --diffusion-block-length "$BLOCK_LENGTH" \
        --diffusion-generated-block-schedule --diffusion-mbsd-fresh-kv
    expect_fail lifecycle-requires-mbsd \
        "--diffusion-mbsd-lifecycle-bookkeeping requires --diffusion-mbsd" \
        "${common[@]}" --diffusion-block-length "$BLOCK_LENGTH" \
        --diffusion-generated-block-schedule --diffusion-mbsd-lifecycle-bookkeeping
    expect_fail compact-requires-mbsd \
        "--diffusion-mbsd-compact requires --diffusion-mbsd" \
        "${common[@]}" --diffusion-block-length "$BLOCK_LENGTH" \
        --diffusion-generated-block-schedule --diffusion-mbsd-compact
    expect_fail fresh-compact-mutual \
        "--diffusion-mbsd-fresh-kv and --diffusion-mbsd-compact are mutually exclusive" \
        "${valid[@]}" --diffusion-mbsd-fresh-kv --diffusion-mbsd-compact
    expect_fail lifecycle-fresh-mutual \
        "--diffusion-mbsd-lifecycle-bookkeeping and --diffusion-mbsd-fresh-kv are mutually exclusive" \
        "${valid[@]}" --diffusion-mbsd-lifecycle-bookkeeping --diffusion-mbsd-fresh-kv
    expect_fail compact-unavailable \
        "--diffusion-mbsd-compact is unavailable until paper-aligned compact KV refresh and step-boundary merge are implemented" \
        "${valid[@]}" --diffusion-mbsd-compact
    if grep -Eq "diffusion_params:|diffusion performance:" "$LOG_DIR/invalid/compact-unavailable.log"; then
        die "compact unavailable test entered model execution"
    fi

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

    local unsupported="--diffusion-mbsd does not yet support early commit, prefix KV, the full-sequence KV oracle, CFG, or Gumbel noise"
    expect_fail early-commit "$unsupported" \
        "${valid[@]}" --diffusion-early-commit-threshold 0.9
    expect_fail prefix-kv "$unsupported" \
        "${valid[@]}" --diffusion-prefix-kv
    expect_fail full-sequence-oracle "$unsupported" \
        "${valid[@]}" --diffusion-full-sequence-kv-oracle
    expect_fail staged-requires-lifecycle \
        "MBSD staged stabilization requires --diffusion-mbsd-lifecycle-bookkeeping" \
        "${valid[@]}" --diffusion-staged-token-stabilization \
        --diffusion-visibility-threshold "$LIFECYCLE_VISIBILITY_THRESHOLD" \
        --diffusion-stability-threshold "$LIFECYCLE_STABILITY_THRESHOLD"
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
    local fresh_kv=false
    local lifecycle=false

    if [[ "$variant" == mbsd-la32-fresh-kv ]]; then
        lookahead=32
        expected_enabled=true
        fresh_kv=true
    elif [[ "$variant" == mbsd-la32-lifecycle ]]; then
        lookahead=32
        expected_enabled=true
        lifecycle=true
    elif [[ "$variant" != baseline ]]; then
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
        -ctk "$CACHE_TYPE_K"
        -ctv "$CACHE_TYPE_V"
        --seed "${SEED:-1234}"
        --temp "$TEMP"
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
    if [[ "$fresh_kv" == true ]]; then
        args+=(--diffusion-mbsd-fresh-kv)
    fi
    if [[ "$lifecycle" == true ]]; then
        args+=(
            --diffusion-mbsd-lifecycle-bookkeeping
            --diffusion-staged-token-stabilization
            --diffusion-visibility-threshold "$LIFECYCLE_VISIBILITY_THRESHOLD"
            --diffusion-stability-threshold "$LIFECYCLE_STABILITY_THRESHOLD"
        )
    fi

    echo
    echo "RUN case=$case_name variant=$variant iteration=$run"
    "$BIN" "${args[@]}" 2>&1 | tee "$log_file"

    require_line_count "MBSD: enabled" 1 "$log_file"
    require_line_count "diffusion generated token ids:" 1 "$log_file"

    local total_ms main_forwards transformer_rows logit_rows
    local generated_tokens id_hash generated_masks invalid_tokens post_eog_nonterminal remaining_masks
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
    post_eog_nonterminal=$(extract_value "diffusion generated tokens:" "post-eog non-terminal" "$log_file")
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
        "post_eog_nonterminal:$post_eog_nonterminal" \
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
    [[ "$post_eog_nonterminal" == 0 ]] || die "non-terminal token after first EOG in $log_file"
    [[ "$remaining_masks" == 0 ]] || die "remaining masks in $log_file"
    [[ "$blocks_started" == "$planned_blocks" ]] || die "not all blocks started in $log_file"
    [[ "$blocks_completed" == "$planned_blocks" ]] || die "not all blocks completed in $log_file"
    [[ "$mbsd_enabled" == "$expected_enabled" ]] || die "unexpected MBSD state in $log_file"
    [[ "$policy" == fixed-budget ]] || die "unexpected MBSD policy in $log_file: $policy"
    local expected_execution=full-sequence-reference
    if [[ "$fresh_kv" == true ]]; then
        expected_execution=fresh-full-sequence-kv
    fi
    [[ "$execution" == "$expected_execution" ]] ||
        die "unexpected MBSD execution mode in $log_file: $execution"

    if [[ "$case_name" == single ]]; then
        [[ "$planned_blocks" == 1 ]] ||
            die "single case produced $planned_blocks blocks; shorten PROMPT or adjust SINGLE_UBATCH"
    else
        (( planned_blocks >= 2 )) ||
            die "multi case produced fewer than two blocks; shorten PROMPT or increase MULTI_UBATCH"
    fi

    local fresh_kv_requested=false fresh_kv_active=false physical_compact_active=false
    local fresh_kv_pre_forward_clears=0 cache_invariant_errors=0 fresh_kv_forwards=0
    local trigger_checks=0 lookahead_expansions=0 slides=0 slide_distance=0
    local first_start=-1 first_end=-1 last_start=-1 last_end=-1 max_end=-1
    local draft_introduced=0 draft_updates=0 draft_prediction_rows=0 draft_reevaluated=0
    local draft_accepted=0 draft_reconfirmed=0 draft_replacements=0 draft_rejected=0 draft_pending=0
    local role_current=0 role_future=0 role_context=0 role_total=0
    local logits_current=0 logits_future=0 steps_with_future=0 extra_steps=0 extra_forwards=0
    local main_steps=0 steps_saved=0 forwards_saved=0 nominal_zero_tail_slots=0
    local nominal_tail_skipped=0 final_forced_selections=0
    local future_semantic_commits=0 explicit_prefix_kv_disabled=false
    local block_order_violations=0 verification_input_errors=0 bounds_errors=0
    local post_eog_corrections=0 trajectory_hash=0

    local detail_markers=(
        "MBSD windows:"
        "MBSD fresh KV:"
        "MBSD physical batches:"
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

        fresh_kv_requested=$(extract_value "MBSD fresh KV:" "requested" "$log_file")
        fresh_kv_active=$(extract_value "MBSD fresh KV:" "execution active" "$log_file")
        physical_compact_active=$(extract_value "MBSD fresh KV:" "physical compact active" "$log_file")
        fresh_kv_forwards=$(extract_value "MBSD fresh KV:" "forwards" "$log_file")
        cache_invariant_errors=$(extract_value "MBSD fresh KV:" "cache invariant errors" "$log_file")
        fresh_kv_pre_forward_clears=$(
            extract_value "MBSD physical batches:" "fresh KV pre-forward clears" "$log_file"
        )

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
        explicit_prefix_kv_disabled=$(extract_value "MBSD invariants:" "explicit prefix KV disabled" "$log_file")
        block_order_violations=$(extract_value "MBSD invariants:" "block-order violations" "$log_file")
        verification_input_errors=$(extract_value "MBSD invariants:" "verification input errors" "$log_file")
        bounds_errors=$(extract_value "MBSD invariants:" "bounds errors" "$log_file")
        post_eog_corrections=$(extract_value "MBSD invariants:" "post-EOG corrections" "$log_file")
        trajectory_hash=$(extract_value "MBSD invariants:" "trajectory hash" "$log_file")

        for metric in \
            "fresh_kv_requested:$fresh_kv_requested" \
            "fresh_kv_active:$fresh_kv_active" \
            "physical_compact_active:$physical_compact_active" \
            "fresh_kv_forwards:$fresh_kv_forwards" \
            "fresh_kv_pre_forward_clears:$fresh_kv_pre_forward_clears" \
            "cache_invariant_errors:$cache_invariant_errors" \
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
            "explicit_prefix_kv_disabled:$explicit_prefix_kv_disabled" \
            "block_order_violations:$block_order_violations" \
            "verification_input_errors:$verification_input_errors" \
            "bounds_errors:$bounds_errors" \
            "post_eog_corrections:$post_eog_corrections" \
            "trajectory_hash:$trajectory_hash"
        do
            require_metric "${metric%%:*}" "${metric#*:}" "$log_file"
        done

        if [[ "$fresh_kv" == true ]]; then
            [[ "$fresh_kv_requested" == true && "$fresh_kv_active" == true ]] ||
                die "fresh-KV mode was not active in $log_file"
            [[ "$fresh_kv_forwards" == "$main_forwards" &&
               "$fresh_kv_pre_forward_clears" == "$main_forwards" ]] ||
                die "fresh-KV forward or clear accounting mismatch in $log_file"
        else
            [[ "$fresh_kv_requested" == false && "$fresh_kv_active" == false ]] ||
                die "fresh-KV mode unexpectedly active in $log_file"
            [[ "$fresh_kv_forwards" == 0 && "$fresh_kv_pre_forward_clears" == 0 ]] ||
                die "dense MBSD unexpectedly used the KV graph in $log_file"
        fi
        [[ "$physical_compact_active" == false && "$cache_invariant_errors" == 0 ]] ||
            die "invalid physical compact or cache invariant state in $log_file"

        [[ "$draft_pending" == 0 ]] || die "pending drafts in $log_file"
        [[ "$future_semantic_commits" == 0 ]] || die "future semantic commit in $log_file"
        [[ "$explicit_prefix_kv_disabled" == true ]] ||
            die "explicit prefix KV was not disabled in $log_file"
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

    if [[ "$fresh_kv" == true ]]; then
        verify_fresh_physical "$log_file" "$ubatch"
    fi

    local lifecycle_expectation=absent
    if [[ "$mbsd_enabled" == true ]]; then
        lifecycle_expectation=disabled
    fi
    if [[ "$lifecycle" == true ]]; then
        lifecycle_expectation=enabled
    fi
    verify_lifecycle_bookkeeping "$log_file" "$generated_tokens" "$lifecycle_expectation"
    if [[ "$lifecycle" == true && "$case_name" == multi ]]; then
        (( lifecycle_confidence_draft_updates > 0 )) ||
            die "multi-block lifecycle observer did not consume future-draft confidence in $log_file"
    fi

    row=(
        "$case_name" "$variant" "$lookahead" "$run" "$total_ms" "$main_forwards"
        "$transformer_rows" "$logit_rows" "$generated_tokens" "$id_hash" "$generated_masks"
        "$invalid_tokens" "$post_eog_nonterminal" "$remaining_masks" "$planned_blocks"
        "$blocks_started" "$blocks_completed"
        "$mbsd_enabled" "$trigger" "$max_lookahead" "$policy" "$execution"
        "$fresh_kv_requested" "$fresh_kv_active" "$physical_compact_active"
        "$fresh_kv_pre_forward_clears" "$cache_invariant_errors"
        "$lifecycle_enabled" "$lifecycle_observer_only" "$lifecycle_staged_integration"
        "$lifecycle_visibility_threshold" "$lifecycle_stability_threshold"
        "$lifecycle_invisible" "$lifecycle_visible" "$lifecycle_stable" "$lifecycle_state_total"
        "$lifecycle_remaining_masks" "$lifecycle_mask_parity_errors"
        "$lifecycle_iv_to_v" "$lifecycle_iv_to_s" "$lifecycle_v_to_s"
        "$lifecycle_forced_visible" "$lifecycle_deferred_stable" "$lifecycle_illegal_transitions"
        "$lifecycle_confidence_current_updates" "$lifecycle_confidence_draft_updates"
        "$lifecycle_confidence_invalidations" "$lifecycle_confidence_errors"
        "$lifecycle_residency_none" "$lifecycle_residency_mutable"
        "$lifecycle_residency_stable" "$lifecycle_residency_total"
        "$lifecycle_duplicate_ownership" "$lifecycle_ownership_errors"
        "$lifecycle_stable_mutations" "$lifecycle_version_errors"
        "$lifecycle_future_semantic_commits" "$lifecycle_future_cache_writes"
        "$lifecycle_pending_entries"
        "$lifecycle_state_accounting_errors"
        "$lifecycle_refresh_enqueued" "$lifecycle_refresh_drained"
        "$lifecycle_refresh_max_depth" "$lifecycle_refresh_snapshots_pending"
        "$lifecycle_refresh_duplicate_errors" "$lifecycle_refresh_ineligible_errors"
        "$lifecycle_refresh_future_errors" "$lifecycle_refresh_stable_errors"
        "$lifecycle_last_refresh_updates" "$lifecycle_last_refresh_errors" "$lifecycle_queue_hash"
        "$lifecycle_state_hash" "$lifecycle_ownership_hash" "$lifecycle_snapshots"
        "$lifecycle_actual_cache_reads" "$lifecycle_actual_cache_writes"
        "$lifecycle_actual_refreshes" "$lifecycle_actual_merges"
        "$lifecycle_actual_async_tasks" "$lifecycle_actual_row_saving"
        "$trigger_checks" "$lookahead_expansions"
        "$slides" "$slide_distance" "$first_start" "$first_end" "$last_start" "$last_end" "$max_end"
        "$draft_introduced" "$draft_updates" "$draft_prediction_rows" "$draft_reevaluated"
        "$draft_accepted" "$draft_reconfirmed" "$draft_replacements" "$draft_rejected" "$draft_pending"
        "$role_current" "$role_future" "$role_context" "$role_total" "$logits_current" "$logits_future"
        "$steps_with_future" "$extra_steps" "$extra_forwards" "$main_steps" "$steps_saved"
        "$forwards_saved" "$nominal_zero_tail_slots" "$nominal_tail_skipped"
        "$final_forced_selections" "$future_semantic_commits" "$explicit_prefix_kv_disabled"
        "$block_order_violations" "$verification_input_errors" "$bounds_errors"
        "$post_eog_corrections" "$trajectory_hash"
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
    local rows id_hashes forward_counts trajectory_hashes state_hashes ownership_hashes queue_hashes

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

    if [[ "$variant" == mbsd-la32-lifecycle ]]; then
        state_hashes=$(column_values "$case_name" "$variant" lifecycle_state_hash | distinct_count)
        [[ "$state_hashes" == 1 ]] || die "lifecycle state hashes are not deterministic for $case_name/$variant"
        ownership_hashes=$(column_values "$case_name" "$variant" lifecycle_ownership_hash | distinct_count)
        [[ "$ownership_hashes" == 1 ]] || die "lifecycle ownership hashes are not deterministic for $case_name/$variant"
        queue_hashes=$(column_values "$case_name" "$variant" lifecycle_queue_hash | distinct_count)
        [[ "$queue_hashes" == 1 ]] || die "lifecycle refresh queue hashes are not deterministic for $case_name/$variant"
    fi
}

verify_baseline_parity() {
    local case_name=$1
    local run baseline_log baseline_ids baseline_hash
    local variant candidate_log candidate_ids candidate_hash

    for ((run = 1; run <= REPEATS; run++)); do
        baseline_log="$LOG_DIR/${case_name}-baseline-${run}.log"
        baseline_ids=$(extract_generated_token_ids "$baseline_log") ||
            die "could not parse generated token IDs from $baseline_log"
        baseline_hash=$(extract_value "diffusion generated tokens:" "id hash" "$baseline_log")

        for variant in mbsd-la0 mbsd-la16 mbsd-la32; do
            candidate_log="$LOG_DIR/${case_name}-${variant}-${run}.log"
            candidate_ids=$(extract_generated_token_ids "$candidate_log") ||
                die "could not parse generated token IDs from $candidate_log"
            candidate_hash=$(extract_value "diffusion generated tokens:" "id hash" "$candidate_log")

            if [[ "$candidate_ids" == "$baseline_ids" ]]; then
                echo "PASS baseline parity case=$case_name variant=$variant run=$run exact=true"
            else
                baseline_parity_mismatches=$((baseline_parity_mismatches + 1))
                if [[ "$MBSD_BASELINE_PARITY" == require ]]; then
                    die "baseline token mismatch for $case_name/$variant run $run "\
                        "(baseline hash = $baseline_hash, candidate hash = $candidate_hash)"
                fi
                echo "REPORT baseline parity case=$case_name variant=$variant run=$run exact=false "\
                    "baseline_hash=$baseline_hash candidate_hash=$candidate_hash"
            fi
        done
    done
}

verify_fresh_parity() {
    local case_name run reference_log fresh_log reference_ids fresh_ids
    local reference_hash fresh_hash reference_trajectory fresh_trajectory
    local reference_forwards fresh_forwards

    for case_name in single multi; do
        for ((run = 1; run <= REPEATS; run++)); do
            reference_log="$LOG_DIR/${case_name}-mbsd-la32-${run}.log"
            fresh_log="$LOG_DIR/${case_name}-mbsd-la32-fresh-kv-${run}.log"

            reference_ids=$(extract_generated_token_ids "$reference_log") ||
                die "could not parse generated token IDs from $reference_log"
            fresh_ids=$(extract_generated_token_ids "$fresh_log") ||
                die "could not parse generated token IDs from $fresh_log"
            reference_hash=$(extract_value "diffusion generated tokens:" "id hash" "$reference_log")
            fresh_hash=$(extract_value "diffusion generated tokens:" "id hash" "$fresh_log")
            reference_trajectory=$(extract_value "MBSD invariants:" "trajectory hash" "$reference_log")
            fresh_trajectory=$(extract_value "MBSD invariants:" "trajectory hash" "$fresh_log")
            reference_forwards=$(extract_value "forwards:" "conditional/main" "$reference_log")
            fresh_forwards=$(extract_value "forwards:" "conditional/main" "$fresh_log")

            [[ "$fresh_ids" == "$reference_ids" ]] ||
                die "fresh-KV generated token mismatch for $case_name run $run"
            [[ "$fresh_hash" == "$reference_hash" ]] ||
                die "fresh-KV generated token hash mismatch for $case_name run $run"
            [[ "$fresh_trajectory" == "$reference_trajectory" ]] ||
                die "fresh-KV trajectory mismatch for $case_name run $run"
            [[ "$fresh_forwards" == "$reference_forwards" ]] ||
                die "fresh-KV main forward mismatch for $case_name run $run"

            echo "PASS fresh-KV parity case=$case_name variant=mbsd-la32-fresh-kv run=$run exact=true"
        done
    done
}

verify_lifecycle_pair() {
    local reference_log=$1
    local lifecycle_log=$2
    local label=$3
    local reference_ids lifecycle_ids reference_hash lifecycle_hash
    local reference_trajectory lifecycle_trajectory reference_forwards lifecycle_forwards
    local reference_transformer_rows lifecycle_transformer_rows reference_logit_rows lifecycle_logit_rows
    local reference_generated lifecycle_generated

    reference_ids=$(extract_generated_token_ids "$reference_log") ||
        die "could not parse generated token IDs from $reference_log"
    lifecycle_ids=$(extract_generated_token_ids "$lifecycle_log") ||
        die "could not parse generated token IDs from $lifecycle_log"
    reference_hash=$(extract_value "diffusion generated tokens:" "id hash" "$reference_log")
    lifecycle_hash=$(extract_value "diffusion generated tokens:" "id hash" "$lifecycle_log")
    reference_trajectory=$(extract_value "MBSD invariants:" "trajectory hash" "$reference_log")
    lifecycle_trajectory=$(extract_value "MBSD invariants:" "trajectory hash" "$lifecycle_log")
    reference_forwards=$(extract_value "forwards:" "conditional/main" "$reference_log")
    lifecycle_forwards=$(extract_value "forwards:" "conditional/main" "$lifecycle_log")
    reference_transformer_rows=$(extract_value "transformer rows: main" "main" "$reference_log")
    lifecycle_transformer_rows=$(extract_value "transformer rows: main" "main" "$lifecycle_log")
    reference_logit_rows=$(extract_value "logits: rows" "rows" "$reference_log")
    lifecycle_logit_rows=$(extract_value "logits: rows" "rows" "$lifecycle_log")
    reference_generated=$(extract_value "diffusion generated tokens:" "count" "$reference_log")
    lifecycle_generated=$(extract_value "diffusion generated tokens:" "count" "$lifecycle_log")

    local metric
    for metric in \
        "reference_hash:$reference_hash" \
        "lifecycle_hash:$lifecycle_hash" \
        "reference_trajectory:$reference_trajectory" \
        "lifecycle_trajectory:$lifecycle_trajectory" \
        "reference_forwards:$reference_forwards" \
        "lifecycle_forwards:$lifecycle_forwards" \
        "reference_transformer_rows:$reference_transformer_rows" \
        "lifecycle_transformer_rows:$lifecycle_transformer_rows" \
        "reference_logit_rows:$reference_logit_rows" \
        "lifecycle_logit_rows:$lifecycle_logit_rows" \
        "reference_generated:$reference_generated" \
        "lifecycle_generated:$lifecycle_generated"
    do
        require_metric "${metric%%:*}" "${metric#*:}" "$lifecycle_log"
    done

    verify_lifecycle_bookkeeping "$reference_log" "$reference_generated" disabled
    verify_lifecycle_bookkeeping "$lifecycle_log" "$lifecycle_generated" enabled

    [[ "$lifecycle_ids" == "$reference_ids" ]] ||
        die "lifecycle generated token mismatch for $label"
    [[ "$lifecycle_hash" == "$reference_hash" ]] ||
        die "lifecycle generated token hash mismatch for $label"
    [[ "$lifecycle_generated" == "$reference_generated" ]] ||
        die "lifecycle generated token count mismatch for $label"
    [[ "$lifecycle_trajectory" == "$reference_trajectory" ]] ||
        die "lifecycle trajectory mismatch for $label"
    [[ "$lifecycle_forwards" == "$reference_forwards" ]] ||
        die "lifecycle main forward mismatch for $label"
    [[ "$lifecycle_transformer_rows" == "$reference_transformer_rows" ]] ||
        die "lifecycle transformer row mismatch for $label"
    [[ "$lifecycle_logit_rows" == "$reference_logit_rows" ]] ||
        die "lifecycle logit row mismatch for $label"

    echo "PASS lifecycle parity $label exact=true"
}

verify_lifecycle_parity() {
    local case_name run

    for case_name in single multi; do
        for ((run = 1; run <= REPEATS; run++)); do
            verify_lifecycle_pair \
                "$LOG_DIR/${case_name}-mbsd-la32-${run}.log" \
                "$LOG_DIR/${case_name}-mbsd-la32-lifecycle-${run}.log" \
                "case=$case_name variant=mbsd-la32-lifecycle run=$run"
        done
    done
}

run_lifecycle_matrix_once() {
    local prompt_index=$1
    local matrix_prompt=$2
    local matrix_seed=$3
    local case_name=$4
    local ubatch=$5
    local steps=$6
    local variant=$7
    local run=$8
    local log_file="$LOG_DIR/lifecycle-matrix/p${prompt_index}-s${matrix_seed}-${case_name}-${variant}-${run}.log"
    local args=(
        "${model_args[@]}"
        -p "$matrix_prompt"
        -ngl "${NGL:-99}"
        -c "$ubatch"
        -b "$ubatch"
        -ub "$ubatch"
        -fa "${FLASH_ATTN:-on}"
        -ctk "$CACHE_TYPE_K"
        -ctv "$CACHE_TYPE_V"
        --seed "$matrix_seed"
        --temp "$TEMP"
        --top-p "${TOP_P:-0.95}"
        --diffusion-block-length "$BLOCK_LENGTH"
        --diffusion-generated-block-schedule
        --diffusion-algorithm 4
        --diffusion-alg-temp 0
        --diffusion-steps "$steps"
        --diffusion-dump-generated-tokens
        --diffusion-mbsd
        --diffusion-mbsd-trigger "$MBSD_TRIGGER"
        --diffusion-mbsd-lookahead 32
    )
    if [[ "$variant" == lifecycle ]]; then
        args+=(
            --diffusion-mbsd-lifecycle-bookkeeping
            --diffusion-staged-token-stabilization
            --diffusion-visibility-threshold "$LIFECYCLE_VISIBILITY_THRESHOLD"
            --diffusion-stability-threshold "$LIFECYCLE_STABILITY_THRESHOLD"
        )
    fi

    echo
    echo "RUN lifecycle-matrix prompt=$prompt_index seed=$matrix_seed case=$case_name variant=$variant run=$run"
    "$BIN" "${args[@]}" 2>&1 | tee "$log_file"

    require_line_count "diffusion generated token ids:" 1 "$log_file"
    require_line_count "MBSD: enabled" 1 "$log_file"
    require_line_count "MBSD invariants:" 1 "$log_file"

    local generated_tokens generated_masks invalid_tokens post_eog_nonterminal remaining_masks
    local planned_blocks blocks_started blocks_completed mbsd_enabled execution
    local main_forwards transformer_rows logit_rows id_hash trajectory_hash
    local draft_pending future_semantic_commits
    local block_order_violations verification_input_errors bounds_errors

    generated_tokens=$(extract_value "diffusion generated tokens:" "count" "$log_file")
    generated_masks=$(extract_value "diffusion generated tokens:" "mask" "$log_file")
    invalid_tokens=$(extract_value "diffusion generated tokens:" "invalid" "$log_file")
    post_eog_nonterminal=$(extract_value "diffusion generated tokens:" "post-eog non-terminal" "$log_file")
    remaining_masks=$(extract_value "token commits:" "remaining masks" "$log_file")
    planned_blocks=$(extract_value "block schedule:" "planned blocks" "$log_file")
    blocks_started=$(extract_value "blocks:" "started" "$log_file")
    blocks_completed=$(extract_value "blocks:" "completed" "$log_file")
    mbsd_enabled=$(extract_value "MBSD: enabled" "enabled" "$log_file")
    execution=$(extract_value "MBSD: enabled" "execution" "$log_file")
    main_forwards=$(extract_value "forwards:" "conditional/main" "$log_file")
    transformer_rows=$(extract_value "transformer rows: main" "main" "$log_file")
    logit_rows=$(extract_value "logits: rows" "rows" "$log_file")
    id_hash=$(extract_value "diffusion generated tokens:" "id hash" "$log_file")
    trajectory_hash=$(extract_value "MBSD invariants:" "trajectory hash" "$log_file")
    draft_pending=$(extract_value "MBSD drafts:" "pending" "$log_file")
    future_semantic_commits=$(extract_value "MBSD invariants:" "future semantic commits" "$log_file")
    block_order_violations=$(extract_value "MBSD invariants:" "block-order violations" "$log_file")
    verification_input_errors=$(extract_value "MBSD invariants:" "verification input errors" "$log_file")
    bounds_errors=$(extract_value "MBSD invariants:" "bounds errors" "$log_file")

    local metric
    for metric in \
        "generated_tokens:$generated_tokens" \
        "generated_masks:$generated_masks" \
        "invalid_tokens:$invalid_tokens" \
        "post_eog_nonterminal:$post_eog_nonterminal" \
        "remaining_masks:$remaining_masks" \
        "planned_blocks:$planned_blocks" \
        "blocks_started:$blocks_started" \
        "blocks_completed:$blocks_completed" \
        "mbsd_enabled:$mbsd_enabled" \
        "execution:$execution" \
        "main_forwards:$main_forwards" \
        "transformer_rows:$transformer_rows" \
        "logit_rows:$logit_rows" \
        "id_hash:$id_hash" \
        "trajectory_hash:$trajectory_hash" \
        "draft_pending:$draft_pending" \
        "future_semantic_commits:$future_semantic_commits" \
        "block_order_violations:$block_order_violations" \
        "verification_input_errors:$verification_input_errors" \
        "bounds_errors:$bounds_errors"
    do
        require_metric "${metric%%:*}" "${metric#*:}" "$log_file"
    done

    [[ "$generated_masks" == 0 && "$invalid_tokens" == 0 && "$post_eog_nonterminal" == 0 ]] ||
        die "invalid generated output in lifecycle matrix log $log_file"
    [[ "$remaining_masks" == 0 && "$draft_pending" == 0 ]] ||
        die "unfinished lifecycle matrix run in $log_file"
    [[ "$blocks_started" == "$planned_blocks" && "$blocks_completed" == "$planned_blocks" ]] ||
        die "incomplete block accounting in $log_file"
    [[ "$mbsd_enabled" == true && "$execution" == full-sequence-reference ]] ||
        die "unexpected lifecycle matrix execution in $log_file"
    (( transformer_rows == main_forwards * ubatch )) ||
        die "lifecycle matrix did not use full-sequence transformer rows in $log_file"
    [[ "$future_semantic_commits" == 0 && "$block_order_violations" == 0 &&
       "$verification_input_errors" == 0 && "$bounds_errors" == 0 ]] ||
        die "MBSD invariant failed in lifecycle matrix log $log_file"

    if [[ "$case_name" == single ]]; then
        [[ "$planned_blocks" == 1 ]] ||
            die "lifecycle matrix single case produced $planned_blocks blocks in $log_file"
    else
        (( planned_blocks >= 2 )) ||
            die "lifecycle matrix multi case produced fewer than two blocks in $log_file"
    fi

    if [[ "$variant" == lifecycle ]]; then
        verify_lifecycle_bookkeeping "$log_file" "$generated_tokens" enabled
        if [[ "$case_name" == multi ]]; then
            (( lifecycle_confidence_draft_updates > 0 )) ||
                die "multi-block lifecycle matrix did not consume future-draft confidence in $log_file"
        fi
    else
        verify_lifecycle_bookkeeping "$log_file" "$generated_tokens" disabled
    fi

    write_tsv_row \
        "$prompt_index" "$matrix_seed" "$case_name" "$variant" "$run" \
        "$generated_tokens" "$id_hash" "$trajectory_hash" "$main_forwards" \
        "$transformer_rows" "$logit_rows" "$lifecycle_state_hash" "$lifecycle_ownership_hash" \
        "$lifecycle_queue_hash" "$lifecycle_refresh_enqueued" "$lifecycle_refresh_max_depth" \
        "$lifecycle_refresh_snapshots_pending" \
        >> "$LIFECYCLE_SUMMARY"

    matrix_last_log=$log_file
}

run_lifecycle_matrix() {
    local prompts=(
        "Return exactly OK."
        "What is the capital of France? Answer with one word."
    )
    local seeds=(42 1234 2026)
    local prompt_index matrix_prompt matrix_seed case_name ubatch steps
    local reference_log lifecycle_log repeat_log state_hash repeat_state_hash
    local ownership_hash repeat_ownership_hash queue_hash repeat_queue_hash
    local refresh_enqueued_total refresh_max_depth refresh_snapshots_pending_total

    mkdir -p "$LOG_DIR/lifecycle-matrix"
    write_tsv_row \
        prompt seed case variant run generated_tokens id_hash trajectory_hash main_forwards \
        transformer_rows logit_rows state_hash ownership_hash queue_hash \
        refresh_enqueued refresh_max_depth refresh_snapshots_pending \
        > "$LIFECYCLE_SUMMARY"

    prompt_index=0
    for matrix_prompt in "${prompts[@]}"; do
        prompt_index=$((prompt_index + 1))
        for matrix_seed in "${seeds[@]}"; do
            for case_name in single multi; do
                if [[ "$case_name" == single ]]; then
                    ubatch=$SINGLE_UBATCH
                    steps=$SINGLE_STEPS
                else
                    ubatch=$MULTI_UBATCH
                    steps=$MULTI_STEPS
                fi

                run_lifecycle_matrix_once \
                    "$prompt_index" "$matrix_prompt" "$matrix_seed" \
                    "$case_name" "$ubatch" "$steps" reference 1
                reference_log=$matrix_last_log
                run_lifecycle_matrix_once \
                    "$prompt_index" "$matrix_prompt" "$matrix_seed" \
                    "$case_name" "$ubatch" "$steps" lifecycle 1
                lifecycle_log=$matrix_last_log
                verify_lifecycle_pair "$reference_log" "$lifecycle_log" \
                    "matrix_prompt=$prompt_index seed=$matrix_seed case=$case_name"
            done
        done
    done

    matrix_prompt=${prompts[0]}
    matrix_seed=${seeds[0]}
    for case_name in single multi; do
        if [[ "$case_name" == single ]]; then
            ubatch=$SINGLE_UBATCH
            steps=$SINGLE_STEPS
        else
            ubatch=$MULTI_UBATCH
            steps=$MULTI_STEPS
        fi

        reference_log="$LOG_DIR/lifecycle-matrix/p1-s${matrix_seed}-${case_name}-reference-1.log"
        lifecycle_log="$LOG_DIR/lifecycle-matrix/p1-s${matrix_seed}-${case_name}-lifecycle-1.log"
        run_lifecycle_matrix_once \
            1 "$matrix_prompt" "$matrix_seed" "$case_name" "$ubatch" "$steps" lifecycle 2
        repeat_log=$matrix_last_log
        verify_lifecycle_pair "$reference_log" "$repeat_log" \
            "matrix_repeat prompt=1 seed=$matrix_seed case=$case_name run=2"

        state_hash=$(extract_scoped_value "MBSD lifecycle hashes:" "state hash" "$lifecycle_log")
        repeat_state_hash=$(extract_scoped_value "MBSD lifecycle hashes:" "state hash" "$repeat_log")
        ownership_hash=$(extract_scoped_value "MBSD lifecycle hashes:" "ownership hash" "$lifecycle_log")
        repeat_ownership_hash=$(extract_scoped_value "MBSD lifecycle hashes:" "ownership hash" "$repeat_log")
        queue_hash=$(extract_scoped_value "MBSD lifecycle refresh state:" "queue hash" "$lifecycle_log")
        repeat_queue_hash=$(extract_scoped_value "MBSD lifecycle refresh state:" "queue hash" "$repeat_log")
        [[ "$state_hash" == "$repeat_state_hash" && "$ownership_hash" == "$repeat_ownership_hash" &&
           "$queue_hash" == "$repeat_queue_hash" ]] ||
            die "lifecycle bookkeeping hashes are not deterministic for matrix $case_name case"
        echo "PASS lifecycle hash determinism matrix_prompt=1 seed=$matrix_seed case=$case_name"
    done

    read -r refresh_enqueued_total refresh_max_depth refresh_snapshots_pending_total < <(
        awk -F '\t' '
            NR == 1 {
                for (i = 1; i <= NF; i++) {
                    column[$i] = i
                }
                next
            }
            $(column["case"]) == "multi" && $(column["variant"]) == "lifecycle" {
                enqueued += $(column["refresh_enqueued"])
                if ($(column["refresh_max_depth"]) > max_depth) {
                    max_depth = $(column["refresh_max_depth"])
                }
                pending += $(column["refresh_snapshots_pending"])
            }
            END { print enqueued + 0, max_depth + 0, pending + 0 }
        ' "$LIFECYCLE_SUMMARY"
    )
    (( refresh_enqueued_total > 0 && refresh_max_depth > 0 &&
       refresh_snapshots_pending_total > 0 )) ||
        die "full lifecycle matrix did not exercise a visible-prefix refresh queue"
    echo "LIFECYCLE_REFRESH_QUEUE_COVERAGE=PASS"

    echo "LIFECYCLE_MATRIX_GATE=PASS"
    echo "LIFECYCLE_SUMMARY=$LIFECYCLE_SUMMARY"
}

echo "===== invalid parameter tests ====="
run_invalid_tests

variants=(baseline mbsd-la0 mbsd-la16 mbsd-la32 mbsd-la32-fresh-kv mbsd-la32-lifecycle)
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
echo "===== baseline token parity ====="
baseline_parity_mismatches=0
for case_name in single multi; do
    verify_baseline_parity "$case_name"
done
echo "BASELINE_PARITY_MODE=$MBSD_BASELINE_PARITY"
echo "BASELINE_PARITY_MISMATCHES=$baseline_parity_mismatches"

echo
echo "===== fresh-KV token and trajectory parity ====="
verify_fresh_parity

echo
echo "===== lifecycle observer exact parity ====="
verify_lifecycle_parity

if [[ "$LIFECYCLE_MATRIX" == 1 ]]; then
    echo
    echo "===== full lifecycle prompt and seed matrix ====="
    run_lifecycle_matrix
else
    echo
    echo "LIFECYCLE_MATRIX_GATE=SKIP"
fi

echo
echo "===== medians ====="
printf '%-8s %-21s %12s %12s %12s\n' case variant median_ms forwards draft_tokens
for case_name in single multi; do
    for variant in "${variants[@]}"; do
        median_ms=$(column_values "$case_name" "$variant" total_ms | median_values)
        median_forwards=$(column_values "$case_name" "$variant" main_forwards | median_values)
        median_drafts=$(column_values "$case_name" "$variant" draft_introduced | median_values)
        printf '%-8s %-21s %12s %12s %12s\n' \
            "$case_name" "$variant" "$median_ms" "$median_forwards" "$median_drafts"
    done
done

echo
echo "MBSD_BENCH_GATE=PASS"
echo "SUMMARY=$SUMMARY"
echo "LOG_DIR=$LOG_DIR"
