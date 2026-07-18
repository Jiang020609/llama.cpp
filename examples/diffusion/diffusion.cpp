#include "diffusion.h"

#include "log.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

enum class diffusion_token_state {
    invisible,
    visible,
    stable,
};

static float calculate_confidence(const llama_token_data_array & cur_p,
                                  diffusion_algorithm            algorithm,
                                  std::mt19937 &                 rng) {
    switch (algorithm) {
        case DIFFUSION_ALGORITHM_CONFIDENCE_BASED:
            return cur_p.data[cur_p.selected].p;  // Selected token probability

        case DIFFUSION_ALGORITHM_ENTROPY_BASED:
            {
                float       entropy = 0.0f;
                const float epsilon = 1e-10f;
                for (size_t i = 0; i < cur_p.size; i++) {
                    float prob = cur_p.data[i].p;
                    entropy += prob * logf(prob + epsilon);
                }
                return -entropy;  // Higher entropy = lower confidence
            }

        case DIFFUSION_ALGORITHM_MARGIN_BASED:
            return (cur_p.size > 1) ? cur_p.data[0].p - cur_p.data[1].p : cur_p.data[0].p;

        case DIFFUSION_ALGORITHM_RANDOM:
            {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                return uniform(rng);  // Random confidence
            }

        case DIFFUSION_ALGORITHM_ORIGIN:
            return cur_p.data[cur_p.selected].p;

        default:
            return 0.0f;
    }
}

// Unified transfer count calculation function
static int32_t calculate_transfer_count(int32_t                      step,
                                        int32_t                      total_steps,
                                        int32_t                      remaining_masked,
                                        diffusion_transfer_schedule  schedule,
                                        float                        eps,
                                        const std::vector<int32_t> & num_transfer_tokens = {}) {
    switch (schedule) {
        case DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED:
            {
                float t          = 1.0f - (float) step / total_steps * (1.0f - eps);
                float s          = 1.0f - (float) (step + 1) / total_steps * (1.0f - eps);
                float p_transfer = (step < total_steps - 1) ? (1.0f - s / t) : 1.0f;
                return (int32_t) (remaining_masked * p_transfer);
            }

        case DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED:
            if (!num_transfer_tokens.empty() && step < (int32_t) num_transfer_tokens.size()) {
                return num_transfer_tokens[step];
            }
            return remaining_masked / (total_steps - step);  // Fallback

        default:
            return remaining_masked / (total_steps - step);
    }
}

static void add_gumbel_noise(float * logits, int32_t n_vocab, float temperature, std::mt19937 & rng) {
    if (temperature == 0.0f) {
        return;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int32_t i = 0; i < n_vocab; i++) {
        double noise        = uniform(rng);
        // Prevent log(0)
        noise               = std::max(noise, 1e-20);
        double gumbel_noise = std::pow(-std::log(noise), temperature);
        logits[i]           = std::exp(logits[i]) / gumbel_noise;
    }
}

static std::vector<int32_t> get_num_transfer_tokens(int32_t mask_count, int32_t steps) {
    std::vector<int32_t> num_transfer_tokens(steps);

    int32_t base      = mask_count / steps;
    int32_t remainder = mask_count % steps;

    for (int32_t i = 0; i < steps; i++) {
        num_transfer_tokens[i] = base + (i < remainder ? 1 : 0);
    }

    return num_transfer_tokens;
}

void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    const bool early_commit_enabled       = params.early_commit_threshold >= 0.0f;
    const bool mbsd_enabled               = params.mbsd;
    const bool mbsd_fresh_kv_enabled      = params.mbsd_fresh_kv;
    const bool prefix_kv_enabled          = params.prefix_kv;
    const bool full_sequence_kv_oracle    = params.full_sequence_kv_oracle;
    const bool staged_token_stabilization = params.staged_token_stabilization;
    const bool cache_reuse_enabled        = prefix_kv_enabled;
    const bool fresh_full_sequence_kv     = full_sequence_kv_oracle || mbsd_fresh_kv_enabled;
    const bool diffusion_kv_graph_enabled = cache_reuse_enabled || fresh_full_sequence_kv;
    if (params.steps <= 0 || !std::isfinite(params.early_commit_threshold) ||
        params.early_commit_threshold > 1.0f || !std::isfinite(params.visibility_threshold) ||
        params.visibility_threshold < 0.0f || params.visibility_threshold > 1.0f ||
        !std::isfinite(params.stability_threshold) || params.stability_threshold < params.visibility_threshold ||
        params.stability_threshold > 1.0f ||
        params.staged_revision_policy < DIFFUSION_STAGED_REVISION_OLDEST ||
        params.staged_revision_policy > DIFFUSION_STAGED_REVISION_BALANCED_LOW_CONFIDENCE ||
        params.staged_final_revision_steps < 0 || !std::isfinite(params.staged_final_visible_ratio) ||
        params.staged_final_visible_ratio < 0.0f || params.staged_final_visible_ratio > 1.0f) {
        LOG_ERR("%s: invalid diffusion parameters\n", __func__);
        return;
    }

    if (!staged_token_stabilization &&
        (params.staged_revision_policy != DIFFUSION_STAGED_REVISION_OLDEST ||
         params.staged_final_revision_steps > 0)) {
        LOG_ERR("%s: staged revision options require staged token stabilization\n", __func__);
        return;
    }

    if (early_commit_enabled &&
        (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED ||
         params.algorithm != DIFFUSION_ALGORITHM_CONFIDENCE_BASED || params.alg_temp != 0.0f)) {
        LOG_ERR("%s: early commit requires block scheduling, confidence selection, and alg-temp 0\n", __func__);
        return;
    }

    if (params.generated_block_schedule && params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        LOG_ERR("%s: generated block scheduling requires block scheduling\n", __func__);
        return;
    }

    if (mbsd_enabled && (params.mbsd_trigger < 0 || params.mbsd_lookahead < 0)) {
        LOG_ERR("%s: MBSD trigger and lookahead must be non-negative\n", __func__);
        return;
    }

    if (mbsd_enabled && params.block_length <= 0) {
        LOG_ERR("%s: MBSD block length must be positive\n", __func__);
        return;
    }

    if (mbsd_fresh_kv_enabled && !mbsd_enabled) {
        LOG_ERR("%s: MBSD fresh KV requires MBSD\n", __func__);
        return;
    }

    if (mbsd_enabled &&
        (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || !params.generated_block_schedule ||
         params.algorithm != DIFFUSION_ALGORITHM_CONFIDENCE_BASED || params.alg_temp != 0.0f ||
         early_commit_enabled || prefix_kv_enabled || full_sequence_kv_oracle ||
         staged_token_stabilization || params.cfg_scale != 0.0f || params.add_gumbel_noise)) {
        LOG_ERR("%s: MBSD requires generated block scheduling, confidence selection, alg-temp 0, "
                "and no early commit, prefix KV, full-sequence KV oracle, staged stabilization, CFG, "
                "or Gumbel noise\n",
                __func__);
        return;
    }

    if (prefix_kv_enabled && full_sequence_kv_oracle) {
        LOG_ERR("%s: prefix KV and the full-sequence KV oracle are mutually exclusive\n", __func__);
        return;
    }

    if (staged_token_stabilization &&
        (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || !params.generated_block_schedule ||
         params.algorithm != DIFFUSION_ALGORITHM_CONFIDENCE_BASED || params.alg_temp != 0.0f ||
         params.temperature != 0.0f || params.steps != params.max_length - n_input || early_commit_enabled ||
         prefix_kv_enabled ||
         params.cfg_scale != 0.0f || params.add_gumbel_noise)) {
        LOG_ERR("%s: staged token stabilization requires generated block scheduling, confidence selection, "
                "alg-temp 0, temp 0, one step per generated token, and no early commit, prefix KV, CFG, "
                "or Gumbel noise\n",
                __func__);
        return;
    }

    if (prefix_kv_enabled &&
        (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || !params.generated_block_schedule ||
         params.algorithm != DIFFUSION_ALGORITHM_CONFIDENCE_BASED || params.alg_temp != 0.0f ||
         early_commit_enabled || params.cfg_scale != 0.0f || params.add_gumbel_noise)) {
        LOG_ERR("%s: prefix KV requires generated block scheduling, confidence selection, alg-temp 0, "
                "and no early commit, CFG, or Gumbel noise\n", __func__);
        return;
    }

    const llama_model * model  = llama_get_model(ctx);
    const llama_vocab * vocab  = llama_model_get_vocab(model);
    llama_memory_t      memory = llama_get_memory(ctx);

    if ((memory != nullptr) != diffusion_kv_graph_enabled) {
        LOG_ERR("%s: diffusion KV context and generation mode must be enabled together\n", __func__);
        return;
    }

    if (diffusion_kv_graph_enabled) {
        int32_t max_window_tokens = params.max_length;
        if (prefix_kv_enabled) {
            const int32_t generated_tokens = params.max_length - n_input;
            const int32_t max_block_tokens = std::min(params.block_length, generated_tokens);
            max_window_tokens = max_block_tokens + (params.shift_logits ? 1 : 0);
        }
        if ((uint32_t) params.max_length > llama_n_ctx_seq(ctx) ||
            (uint32_t) n_input > llama_n_batch(ctx) || (uint32_t) n_input > llama_n_ubatch(ctx) ||
            (uint32_t) max_window_tokens > llama_n_batch(ctx) ||
            (uint32_t) max_window_tokens > llama_n_ubatch(ctx)) {
            LOG_ERR("%s: diffusion KV mode exceeds context or batch capacity "
                    "(length = %d, prompt = %d, max window = %d, ctx = %u, batch = %u, ubatch = %u)\n",
                    __func__, params.max_length, n_input, max_window_tokens,
                    llama_n_ctx_seq(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx));
            return;
        }
    }

    // Initialize with input and pad with mask tokens
    std::copy(input_tokens, input_tokens + n_input, output_tokens);
    std::fill(output_tokens + n_input, output_tokens + params.max_length, params.mask_token_id);

    std::vector<diffusion_token_state> token_states;
    std::vector<int32_t>               sts_last_revision_step;
    std::vector<int32_t>               sts_revision_count;
    std::vector<float>                 sts_latest_confidence;
    if (staged_token_stabilization) {
        token_states.resize(params.max_length, diffusion_token_state::stable);
        std::fill(token_states.begin() + n_input, token_states.end(), diffusion_token_state::invisible);
        sts_last_revision_step.resize(params.max_length, -1);
        sts_revision_count.resize(params.max_length, 0);
        sts_latest_confidence.resize(params.max_length, 0.0f);
    }

    std::vector<llama_token> mbsd_draft_tokens;
    std::vector<float>       mbsd_draft_confidences;
    std::vector<uint8_t>     mbsd_draft_valid;
    std::vector<uint8_t>     mbsd_draft_ever;
    if (mbsd_enabled) {
        mbsd_draft_tokens.resize(params.max_length, params.mask_token_id);
        mbsd_draft_confidences.resize(params.max_length, 0.0f);
        mbsd_draft_valid.resize(params.max_length, 0);
        mbsd_draft_ever.resize(params.max_length, 0);
    }

    std::mt19937 rng(params.seed);

    llama_set_causal_attn(ctx, false);

    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    if (params.mask_token_id < 0 || params.mask_token_id >= n_vocab) {
        LOG_ERR("%s: invalid mask token id %d\n", __func__, params.mask_token_id);
        return;
    }

    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<llama_token_data> conf_candidates;
    conf_candidates.reserve(params.max_length);
    std::vector<int32_t> mask_positions;
    mask_positions.reserve(params.max_length);
    std::vector<int32_t> mbsd_future_positions;
    std::vector<int32_t> mbsd_verification_positions;
    std::vector<int32_t> mbsd_logit_positions;
    struct mbsd_future_plan {
        int32_t              block = -1;
        int32_t              quota = 0;
        std::vector<int32_t> positions;
    };
    std::vector<mbsd_future_plan> mbsd_future_plans;
    if (mbsd_enabled) {
        mbsd_future_positions.reserve(params.max_length);
        mbsd_verification_positions.reserve(params.max_length);
        mbsd_logit_positions.reserve(params.max_length);
        mbsd_future_plans.reserve(params.max_length / std::max(params.block_length, 1));
    }
    std::vector<int32_t> active_positions;
    std::vector<llama_token> sts_sampled_tokens;
    std::vector<float> sts_sampled_confidences;
    if (staged_token_stabilization) {
        active_positions.reserve(params.max_length);
        sts_sampled_tokens.reserve(params.max_length);
        sts_sampled_confidences.reserve(params.max_length);
    }
    std::vector<int32_t> logits_row_by_pos(params.max_length, -1);
    std::vector<int32_t> mbsd_future_index_by_pos;
    if (mbsd_enabled) {
        mbsd_future_index_by_pos.resize(params.max_length, -1);
    }

    // Setup sampler chain
    struct llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));

    struct llama_sampler * mbsd_sampler = nullptr;
    if (mbsd_enabled) {
        mbsd_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (params.top_k > 0) {
            llama_sampler_chain_add(mbsd_sampler, llama_sampler_init_top_k(params.top_k));
        }
        if (params.top_p < 1.0f) {
            llama_sampler_chain_add(mbsd_sampler, llama_sampler_init_top_p(params.top_p, 1));
        }
        if (params.temperature > 0.0f) {
            llama_sampler_chain_add(mbsd_sampler, llama_sampler_init_temp(params.temperature));
        }
        const uint32_t mbsd_seed = (uint32_t) params.seed ^ 0x6d627364U;
        llama_sampler_chain_add(mbsd_sampler, llama_sampler_init_dist(mbsd_seed));
    }

    struct llama_sampler * dist_sampler = llama_sampler_init_dist(params.seed);

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);
    batch.n_tokens    = params.max_length;

    // Pre-allocate buffers for CFG if needed
    std::vector<float>       cond_logits_buffer;
    std::vector<llama_token> un_x_buffer;
    if (params.cfg_scale > 0.0f) {
        un_x_buffer.resize(params.max_length);
    }

    // For block-based processing
    std::vector<int32_t> num_transfer_tokens;
    int32_t              num_blocks           = 1;
    int32_t              base_steps_per_block = params.steps;
    int32_t              extra_step_blocks    = 0;

    if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        if (params.block_length <= 0) {
            LOG_ERR("%s: block length must be positive\n", __func__);
            llama_batch_free(batch);
            llama_sampler_free(sampler);
            llama_sampler_free(mbsd_sampler);
            llama_sampler_free(dist_sampler);
            return;
        }

        if (params.generated_block_schedule) {
            const int32_t generated_tokens = params.max_length - n_input;
            num_blocks = 1 + (generated_tokens - 1) / params.block_length;
            if (params.steps < num_blocks) {
                LOG_ERR("%s: diffusion steps must be at least the number of generated blocks\n", __func__);
                llama_batch_free(batch);
                llama_sampler_free(sampler);
                llama_sampler_free(mbsd_sampler);
                llama_sampler_free(dist_sampler);
                return;
            }
            base_steps_per_block = params.steps / num_blocks;
            extra_step_blocks    = params.steps % num_blocks;
        } else {
            if (params.max_length % params.block_length != 0) {
                LOG_ERR("%s: max length must be divisible by block length\n", __func__);
                llama_batch_free(batch);
                llama_sampler_free(sampler);
                llama_sampler_free(mbsd_sampler);
                llama_sampler_free(dist_sampler);
                return;
            }
            num_blocks = params.max_length / params.block_length;
            if (params.steps % num_blocks != 0) {
                LOG_ERR("%s: diffusion steps must be divisible by the number of blocks\n", __func__);
                llama_batch_free(batch);
                llama_sampler_free(sampler);
                llama_sampler_free(mbsd_sampler);
                llama_sampler_free(dist_sampler);
                return;
            }
            base_steps_per_block = params.steps / num_blocks;
        }
    }

    auto get_block_bounds = [&](int32_t block_num) {
        const int64_t start = (int64_t) n_input + (int64_t) block_num * params.block_length;
        return std::make_pair(
            (int32_t) std::min<int64_t>(start, params.max_length),
            (int32_t) std::min<int64_t>(start + params.block_length, params.max_length));
    };

    std::vector<std::vector<int32_t>> mbsd_draft_transfer_tokens;
    std::vector<int32_t>              mbsd_draft_steps;
    std::vector<int32_t>              mbsd_draft_quota;
    if (mbsd_enabled) {
        mbsd_draft_transfer_tokens.resize(num_blocks);
        mbsd_draft_steps.resize(num_blocks, 0);
        mbsd_draft_quota.resize(num_blocks, 0);
        for (int32_t block = 0; block < num_blocks; block++) {
            const auto bounds      = get_block_bounds(block);
            const int32_t n_tokens = bounds.second - bounds.first;
            const int32_t n_steps  = base_steps_per_block + (block < extra_step_blocks ? 1 : 0);
            mbsd_draft_transfer_tokens[block] = get_num_transfer_tokens(n_tokens, n_steps);
            if (!mbsd_draft_transfer_tokens[block].empty()) {
                mbsd_draft_quota[block] = mbsd_draft_transfer_tokens[block][0];
            }
        }
    }

    struct forward_perf {
        int32_t  calls           = 0;
        int32_t  completed       = 0;
        int64_t  decode_time     = 0;
        int64_t  completion_time = 0;
        uint64_t input_tokens    = 0;
        uint64_t active_masks    = 0;
        uint64_t output_rows     = 0;
        uint64_t logits_bytes    = 0;
    };

    forward_perf conditional_perf;
    forward_perf unconditional_perf;
    forward_perf revision_perf;
    forward_perf cache_perf;

    int32_t iterations_started      = 0;
    int32_t iterations_with_forward = 0;
    int32_t iterations_completed    = 0;
    int32_t sampling_passes         = 0;

    int32_t blocks_started              = 0;
    int32_t blocks_completed            = 0;
    int32_t blocks_finished_early       = 0;
    int32_t scheduled_steps_skipped     = 0;
    int32_t scheduled_forwards_skipped  = 0;
    int32_t sts_scheduled_steps_skipped = 0;

    uint64_t base_token_selections      = 0;
    uint64_t threshold_extra_selections = 0;
    uint64_t forced_final_selections    = 0;
    uint64_t tokens_committed           = 0;

    int32_t  mbsd_trigger_checks          = 0;
    int32_t  mbsd_lookahead_expansions    = 0;
    int32_t  mbsd_window_slides           = 0;
    int32_t  mbsd_window_slide_distance   = 0;
    int32_t  mbsd_window_first_start      = -1;
    int32_t  mbsd_window_first_end        = -1;
    int32_t  mbsd_window_last_start       = -1;
    int32_t  mbsd_window_last_end         = -1;
    int32_t  mbsd_window_max_end          = -1;
    int32_t  mbsd_steps_with_future_work  = 0;
    int32_t  mbsd_main_steps_total         = 0;
    int32_t  mbsd_scheduled_steps_saved   = 0;
    int32_t  mbsd_scheduled_forwards_saved = 0;
    int32_t  mbsd_nominal_zero_tail_slots  = 0;
    int32_t  mbsd_nominal_tail_steps_skipped = 0;
    uint64_t mbsd_final_forced_selections  = 0;
    uint64_t mbsd_current_transformer_rows = 0;
    uint64_t mbsd_future_transformer_rows  = 0;
    uint64_t mbsd_context_transformer_rows = 0;
    uint64_t mbsd_current_logit_rows        = 0;
    uint64_t mbsd_future_logit_rows         = 0;
    uint64_t mbsd_draft_prediction_rows     = 0;
    uint64_t mbsd_drafts_introduced         = 0;
    uint64_t mbsd_draft_updates             = 0;
    uint64_t mbsd_drafts_reevaluated        = 0;
    uint64_t mbsd_drafts_accepted           = 0;
    uint64_t mbsd_drafts_reconfirmed        = 0;
    uint64_t mbsd_drafts_replaced           = 0;
    uint64_t mbsd_drafts_rejected           = 0;
    uint64_t mbsd_future_semantic_commits   = 0;
    uint64_t mbsd_block_order_violations    = 0;
    uint64_t mbsd_verification_input_errors = 0;
    uint64_t mbsd_bounds_errors             = 0;
    uint64_t mbsd_post_eog_corrections      = 0;
    uint64_t mbsd_trajectory_hash            = 14695981039346656037ULL;
    uint64_t mbsd_fresh_prefix_rows_reused   = 0;
    uint64_t mbsd_fresh_cache_invariant_errors = 0;
    uint64_t mbsd_physical_mapping_errors    = 0;
    int32_t  mbsd_fresh_main_batches         = 0;
    int32_t  mbsd_fresh_main_rows_min        = std::numeric_limits<int32_t>::max();
    int32_t  mbsd_fresh_main_rows_max        = 0;

    uint64_t sts_visibility_promotions          = 0;
    uint64_t sts_direct_stable_promotions       = 0;
    uint64_t sts_visible_to_stable              = 0;
    uint64_t sts_forced_visible                 = 0;
    uint64_t sts_revision_candidates            = 0;
    uint64_t sts_token_revisions                = 0;
    uint64_t sts_stable_logits_skipped          = 0;
    uint64_t sts_unstable_at_block_completion = 0;
    uint64_t sts_trajectory_hash                = 14695981039346656037ULL;
    uint64_t sts_revision_target_hash           = 14695981039346656037ULL;
    int32_t  sts_revision_passes                = 0;
    int32_t  sts_ordinary_revision_passes       = 0;
    int32_t  sts_max_visible                    = 0;

    bool     sts_final_revision_triggered        = false;
    int32_t  sts_final_available_steps           = 0;
    int32_t  sts_final_revision_budget           = 0;
    int32_t  sts_final_revision_passes           = 0;
    int32_t  sts_final_revision_sweeps_started   = 0;
    int32_t  sts_final_revision_sweeps_completed = 0;
    int32_t  sts_final_visible_before            = 0;
    int32_t  sts_final_visible_after             = 0;
    float    sts_final_visible_ratio_observed     = 0.0f;
    uint64_t sts_final_visible_to_stable          = 0;
    uint64_t sts_final_token_revisions            = 0;
    const char * sts_final_stop_reason = params.staged_final_revision_steps > 0 ? "not-reached" : "disabled";

    int32_t prompt_prefills                  = 0;
    int32_t transition_seals                 = 0;
    int32_t full_sequence_pre_forward_clears = 0;
    bool    generation_failed                 = false;

    int64_t total_callback_time    = 0;
    int64_t total_batch_time       = 0;
    int64_t total_cache_clear_time = 0;
    int64_t total_cfg_cpu_time     = 0;
    int64_t total_sampling_time    = 0;
    int64_t total_time             = 0;

    if (diffusion_kv_graph_enabled && !mbsd_fresh_kv_enabled) {
        llama_synchronize(ctx);
        llama_memory_clear(memory, false);
    }

    const int32_t graph_reuses_start = llama_perf_context(ctx).n_reused;
    const int64_t time_start         = ggml_time_us();

    auto run_forward = [&](forward_perf & perf, int32_t active_masks, int32_t output_rows) -> std::pair<int, float *> {
        perf.calls++;

        if (fresh_full_sequence_kv) {
            const int64_t clear_start = ggml_time_us();
            llama_synchronize(ctx);
            llama_memory_clear(memory, false);
            total_cache_clear_time += ggml_time_us() - clear_start;
            full_sequence_pre_forward_clears++;
        }

        const int64_t decode_start = ggml_time_us();
        const int     ret          = llama_decode(ctx, batch);
        perf.decode_time += ggml_time_us() - decode_start;

        if (ret != 0) {
            return { ret, nullptr };
        }

        // Includes outstanding backend compute and logits readback.
        const int64_t completion_start = ggml_time_us();
        float *       result           = llama_get_logits(ctx);
        perf.completion_time += ggml_time_us() - completion_start;

        if (!result) {
            return { -1, nullptr };
        }

        perf.completed++;
        perf.input_tokens += batch.n_tokens;
        perf.active_masks += active_masks;
        perf.output_rows  += output_rows;
        perf.logits_bytes += (uint64_t) output_rows * n_vocab * sizeof(float);

        return { 0, result };
    };

    auto setup_cache_batch = [&](int32_t abs_start, int32_t abs_end) {
        GGML_ASSERT(abs_start >= 0 && abs_start < abs_end && abs_end <= params.max_length);

        batch.n_tokens = abs_end - abs_start;
        for (int32_t local = 0; local < batch.n_tokens; local++) {
            const int32_t pos = abs_start + local;
            batch.token[local]     = output_tokens[pos];
            batch.pos[local]       = pos;
            batch.n_seq_id[local]  = 1;
            batch.seq_id[local][0] = 0;
            batch.logits[local]    = 0;
        }

        batch.logits[batch.n_tokens - 1] = 1;
    };

    auto sample_staged_logits = [&](const float * pos_logits) -> std::pair<llama_token, float> {
        llama_token selected  = LLAMA_TOKEN_NULL;
        float       max_logit = -std::numeric_limits<float>::infinity();
        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
            if (token_id != params.mask_token_id && pos_logits[token_id] > max_logit) {
                selected = token_id;
                max_logit = pos_logits[token_id];
            }
        }
        GGML_ASSERT(selected != LLAMA_TOKEN_NULL);

        double probability_sum = 0.0;
        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
            if (token_id != params.mask_token_id) {
                probability_sum += std::exp((double) pos_logits[token_id] - max_logit);
            }
        }
        const float confidence = probability_sum > 0.0 ? (float) (1.0 / probability_sum) : 0.0f;
        return { selected, confidence };
    };

    auto select_staged_revision_target = [&](int32_t end_pos,
                                             const std::vector<uint8_t> * excluded,
                                             bool final_sweep) {
        int32_t target = -1;
        for (int32_t pos = n_input; pos < end_pos; pos++) {
            if (token_states[pos] != diffusion_token_state::visible ||
                (excluded && (*excluded)[pos])) {
                continue;
            }

            bool prefer = target < 0;
            if (!prefer) {
                if (params.staged_revision_policy == DIFFUSION_STAGED_REVISION_OLDEST) {
                    prefer = sts_last_revision_step[pos] < sts_last_revision_step[target] ||
                             (sts_last_revision_step[pos] == sts_last_revision_step[target] && pos < target);
                } else if (!final_sweep && sts_revision_count[pos] != sts_revision_count[target]) {
                    prefer = sts_revision_count[pos] < sts_revision_count[target];
                } else if (sts_latest_confidence[pos] != sts_latest_confidence[target]) {
                    prefer = sts_latest_confidence[pos] < sts_latest_confidence[target];
                } else if (sts_last_revision_step[pos] != sts_last_revision_step[target]) {
                    prefer = sts_last_revision_step[pos] < sts_last_revision_step[target];
                } else {
                    prefer = pos < target;
                }
            }

            if (prefer) {
                target = pos;
            }
        }
        return target;
    };

    auto run_staged_revision = [&](int32_t target, llama_token & token, float & confidence) {
        const int64_t batch_start = ggml_time_us();
        batch.n_tokens = params.max_length;
        for (int32_t pos = 0; pos < params.max_length; pos++) {
            batch.token[pos]     = output_tokens[pos];
            batch.pos[pos]       = pos;
            batch.n_seq_id[pos]  = 1;
            batch.seq_id[pos][0] = 0;
            batch.logits[pos]    = 0;
        }
        batch.token[target] = params.mask_token_id;
        const int32_t source = params.shift_logits ? std::max(target - 1, 0) : target;
        batch.logits[source] = 1;
        total_batch_time += ggml_time_us() - batch_start;

        auto [ret, revision_logits] = run_forward(revision_perf, 0, 1);
        if (ret != 0 || !revision_logits) {
            LOG_ERR("%s: failed to revise visible token at position %d, ret = %d\n",
                    __func__, target, ret);
            return false;
        }

        const int64_t sample_start = ggml_time_us();
        const auto revised = sample_staged_logits(revision_logits);
        total_sampling_time += ggml_time_us() - sample_start;
        token = revised.first;
        confidence = revised.second;
        return true;
    };

    auto record_staged_revision_target = [&](int32_t target, bool final_revision) {
        sts_revision_target_hash ^= final_revision ? 0x5441494cU : 0x4d41494eU;
        sts_revision_target_hash *= 1099511628211ULL;
        sts_revision_target_hash ^= (uint32_t) target;
        sts_revision_target_hash *= 1099511628211ULL;
    };

    if (cache_reuse_enabled) {
        const int64_t batch_start = ggml_time_us();
        setup_cache_batch(0, n_input);
        total_batch_time += ggml_time_us() - batch_start;

        auto [ret, ignored] = run_forward(cache_perf, 0, 1);
        GGML_UNUSED(ignored);
        if (ret != 0) {
            LOG_ERR("%s: failed to prefill the prompt cache, ret = %d\n", __func__, ret);
            generation_failed = true;
        } else {
            prompt_prefills++;
            const int32_t keep_end = params.shift_logits ? n_input - 1 : n_input;
            if (!llama_memory_seq_rm(memory, 0, keep_end, -1)) {
                LOG_ERR("%s: failed to trim the prompt cache at position %d\n", __func__, keep_end);
                generation_failed = true;
            }
        }
    }

    for (int32_t block_num = 0; block_num < num_blocks && !generation_failed; block_num++) {
        int32_t block_start = 0;
        int32_t block_end   = params.max_length;
        if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
            const auto bounds = get_block_bounds(block_num);
            block_start = bounds.first;
            block_end   = bounds.second;
        }

        if (params.generated_block_schedule) {
            GGML_ASSERT(block_start < block_end);
            if (block_num == num_blocks - 1) {
                GGML_ASSERT(block_end == params.max_length);
            }
        }

        int32_t steps_this_block =
            base_steps_per_block + (block_num < extra_step_blocks ? 1 : 0);
        int32_t block_step_offset =
            block_num * base_steps_per_block + std::min(block_num, extra_step_blocks);
        if (staged_token_stabilization) {
            steps_this_block = block_end - block_start;
            block_step_offset = block_start - n_input;
        }

        GGML_ASSERT(steps_this_block > 0);
        GGML_ASSERT(block_step_offset + steps_this_block <= params.steps);
        if (block_num == num_blocks - 1) {
            GGML_ASSERT(block_step_offset + steps_this_block == params.steps);
        }

        // Count masked tokens in current block for block-based processing
        bool block_completion_recorded = false;
        int32_t sts_unused_steps_this_block = 0;
        int32_t sts_unused_step_start = 0;
        int32_t mbsd_main_steps_this_block = 0;
        int32_t mbsd_baseline_effective_steps = steps_this_block;
        int32_t mbsd_draft_accepts_this_block = 0;
        bool    mbsd_verification_pending = false;
        if (mbsd_enabled) {
            for (int32_t pos = n_input; pos < block_start; pos++) {
                if (output_tokens[pos] == params.mask_token_id) {
                    mbsd_block_order_violations++;
                }
            }
            for (int32_t pos = block_start; pos < params.max_length; pos++) {
                if (output_tokens[pos] != params.mask_token_id) {
                    mbsd_future_semantic_commits++;
                }
            }
            for (int32_t pos = block_start; pos < block_end; pos++) {
                mbsd_verification_pending |= mbsd_draft_valid[pos] != 0;
            }
            if (mbsd_block_order_violations > 0 || mbsd_future_semantic_commits > 0) {
                LOG_ERR("%s: MBSD block-order invariant failed before block %d\n", __func__, block_num + 1);
                generation_failed = true;
                break;
            }
        }
        if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
            int32_t block_mask_count = 0;
            for (int i = block_start; i < block_end; i++) {
                if (staged_token_stabilization ?
                        token_states[i] == diffusion_token_state::invisible :
                        output_tokens[i] == params.mask_token_id) {
                    block_mask_count++;
                }
            }
            if (!staged_token_stabilization) {
                num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps_this_block);
                if (mbsd_enabled) {
                    mbsd_baseline_effective_steps = 0;
                    for (int32_t i = 0; i < (int32_t) num_transfer_tokens.size(); i++) {
                        if (num_transfer_tokens[i] > 0) {
                            mbsd_baseline_effective_steps = i + 1;
                        }
                    }
                    mbsd_nominal_zero_tail_slots +=
                        steps_this_block - mbsd_baseline_effective_steps;
                }
            }
            if (block_mask_count > 0) {
                blocks_started++;
            }
        }

        for (int32_t step = 0; step < steps_this_block; step++) {
            int32_t global_step = block_step_offset + step;
            iterations_started++;

            if (params.step_callback) {
                const int64_t callback_start = ggml_time_us();
                if (!params.step_callback(
                        global_step, params.steps, output_tokens, params.max_length, params.step_callback_user_data)) {
                    total_callback_time += ggml_time_us() - callback_start;
                    if (diffusion_kv_graph_enabled || staged_token_stabilization || mbsd_enabled) {
                        generation_failed = true;
                    }
                    break;
                }
                total_callback_time += ggml_time_us() - callback_start;
            }

            // Setup batch
            const int64_t batch_start = ggml_time_us();

            mask_positions.clear();
            active_positions.clear();
            if (staged_token_stabilization) {
                int32_t visible_this_step = 0;
                int32_t stable_this_step  = 0;
                for (int32_t i = n_input; i < block_end; i++) {
                    if (token_states[i] == diffusion_token_state::invisible) {
                        if (i >= block_start) {
                            mask_positions.push_back(i);
                            active_positions.push_back(i);
                        }
                    } else if (token_states[i] == diffusion_token_state::visible) {
                        active_positions.push_back(i);
                        visible_this_step++;
                    } else {
                        stable_this_step++;
                    }
                }
                sts_max_visible = std::max(sts_max_visible, visible_this_step);
                sts_revision_candidates += visible_this_step;
                if (!active_positions.empty()) {
                    sts_stable_logits_skipped += stable_this_step;
                }
            } else {
                for (int32_t i = 0; i < params.max_length; i++) {
                    if (output_tokens[i] == params.mask_token_id &&
                        (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED ||
                         (i >= block_start && i < block_end))) {
                        mask_positions.push_back(i);
                    }
                }
            }

            if (mask_positions.empty()) {
                total_batch_time += ggml_time_us() - batch_start;
                break;
            }

            const int32_t active_masks = (int32_t) mask_positions.size();
            int32_t       mbsd_window_start = block_start;
            int32_t       mbsd_window_end   = block_end;
            bool          mbsd_proactive_lookahead = false;
            const std::vector<int32_t> * logit_positions = &mask_positions;
            mbsd_future_positions.clear();
            mbsd_verification_positions.clear();
            mbsd_future_plans.clear();

            if (mbsd_enabled) {
                mbsd_logit_positions.clear();
                mbsd_logit_positions.insert(
                    mbsd_logit_positions.end(), mask_positions.begin(), mask_positions.end());
                logit_positions = &mbsd_logit_positions;
                while (mbsd_window_start < block_end &&
                       output_tokens[mbsd_window_start] != params.mask_token_id) {
                    mbsd_window_start++;
                }

                const int32_t base_window_end = (int32_t) std::min<int64_t>(
                    (int64_t) mbsd_window_start + params.block_length, params.max_length);
                mbsd_window_end = base_window_end;
                mbsd_trigger_checks++;
                if (active_masks < params.mbsd_trigger && params.mbsd_lookahead > 0) {
                    mbsd_window_end = (int32_t) std::min<int64_t>(
                        (int64_t) base_window_end + params.mbsd_lookahead, params.max_length);
                    mbsd_proactive_lookahead = mbsd_window_end > base_window_end;
                    mbsd_lookahead_expansions += mbsd_proactive_lookahead;
                }

                if (mbsd_window_start < block_start || mbsd_window_start > block_end ||
                    mbsd_window_end < block_end || mbsd_window_end > params.max_length) {
                    mbsd_bounds_errors++;
                    LOG_ERR("%s: MBSD window bounds invalid at block %d step %d "
                            "(start = %d, end = %d, block = [%d, %d), max = %d)\n",
                            __func__, block_num + 1, step + 1, mbsd_window_start, mbsd_window_end,
                            block_start, block_end, params.max_length);
                    total_batch_time += ggml_time_us() - batch_start;
                    generation_failed = true;
                    break;
                }

                if (mbsd_window_first_start < 0) {
                    mbsd_window_first_start = mbsd_window_start;
                    mbsd_window_first_end   = mbsd_window_end;
                }
                if (mbsd_window_last_start >= 0 && mbsd_window_start > mbsd_window_last_start) {
                    mbsd_window_slides++;
                    mbsd_window_slide_distance += mbsd_window_start - mbsd_window_last_start;
                }
                mbsd_window_last_start = mbsd_window_start;
                mbsd_window_last_end   = mbsd_window_end;
                mbsd_window_max_end    = std::max(mbsd_window_max_end, mbsd_window_end);

                if (mbsd_verification_pending) {
                    for (int32_t pos = block_start; pos < block_end; pos++) {
                        if (mbsd_draft_valid[pos]) {
                            mbsd_verification_positions.push_back(pos);
                        }
                    }
                }

                for (int32_t future_block = block_num + 1; future_block < num_blocks; future_block++) {
                    const auto future_bounds = get_block_bounds(future_block);
                    if (future_bounds.first >= mbsd_window_end) {
                        break;
                    }

                    const int32_t future_start = std::max(future_bounds.first, block_end);
                    const int32_t future_end   = std::min(future_bounds.second, mbsd_window_end);
                    if (future_start >= future_end) {
                        continue;
                    }

                    const bool can_promote = mbsd_draft_steps[future_block] <
                        (int32_t) mbsd_draft_transfer_tokens[future_block].size();
                    mbsd_future_plan plan;
                    plan.block = future_block;
                    plan.quota = can_promote ? mbsd_draft_quota[future_block] : 0;
                    for (int32_t pos = future_start; pos < future_end; pos++) {
                        if (mbsd_draft_valid[pos]) {
                            mbsd_future_positions.push_back(pos);
                            mbsd_logit_positions.push_back(pos);
                        } else if (can_promote && plan.quota > 0) {
                            plan.positions.push_back(pos);
                            mbsd_future_positions.push_back(pos);
                            mbsd_logit_positions.push_back(pos);
                        }
                    }
                    if (can_promote) {
                        mbsd_future_plans.push_back(std::move(plan));
                    }
                }

                if (!mbsd_future_positions.empty()) {
                    mbsd_steps_with_future_work++;
                }

                LOG_DBG("%s: MBSD window step = %d, block = %d, range = [%d, %d), "
                        "current = [%d, %d), remaining masks = %d, future logits = %d, proactive = %s\n",
                        __func__, global_step + 1, block_num + 1, mbsd_window_start, mbsd_window_end,
                        block_start, block_end, active_masks, (int32_t) mbsd_future_positions.size(),
                        mbsd_proactive_lookahead ? "true" : "false");
            }

            int32_t batch_abs_start = 0;
            int32_t batch_abs_end   = params.max_length;
            if (prefix_kv_enabled) {
                batch_abs_start = params.shift_logits ? block_start - 1 : block_start;
                batch_abs_end   = block_end;
            }

            if (batch_abs_start < 0 || batch_abs_start >= batch_abs_end ||
                batch_abs_end > params.max_length) {
                mbsd_physical_mapping_errors += mbsd_fresh_kv_enabled;
                LOG_ERR("%s: invalid physical batch bounds [%d, %d) for length %d\n",
                        __func__, batch_abs_start, batch_abs_end, params.max_length);
                total_batch_time += ggml_time_us() - batch_start;
                generation_failed = true;
                break;
            }

            if (cache_reuse_enabled && !llama_memory_seq_rm(memory, 0, batch_abs_start, -1)) {
                LOG_ERR("%s: failed to reset the cache tail at position %d\n", __func__, batch_abs_start);
                total_batch_time += ggml_time_us() - batch_start;
                generation_failed = true;
                break;
            }
            batch.n_tokens = batch_abs_end - batch_abs_start;
            for (int32_t local = 0; local < batch.n_tokens; local++) {
                const int32_t pos = batch_abs_start + local;
                batch.token[local]     = output_tokens[pos];
                batch.pos[local]       = pos;
                batch.n_seq_id[local]  = 1;
                batch.seq_id[local][0] = 0;
                batch.logits[local]    = 0;
            }

            if (mbsd_enabled) {
                for (int32_t pos = block_end; pos < mbsd_window_end; pos++) {
                    if (mbsd_draft_valid[pos]) {
                        batch.token[pos - batch_abs_start] = mbsd_draft_tokens[pos];
                    }
                }
                for (int32_t pos : mbsd_verification_positions) {
                    if (batch.token[pos - batch_abs_start] != params.mask_token_id) {
                        mbsd_verification_input_errors++;
                    }
                }
                if (mbsd_verification_input_errors > 0) {
                    LOG_ERR("%s: MBSD verification input was not masked at block %d step %d\n",
                            __func__, block_num + 1, step + 1);
                    total_batch_time += ggml_time_us() - batch_start;
                    generation_failed = true;
                    break;
                }
            }

            bool physical_mapping_failed = false;
            for (int32_t pos : *logit_positions) {
                const int32_t source_pos = params.shift_logits ? std::max(pos - 1, 0) : pos;
                if (source_pos < batch_abs_start || source_pos >= batch_abs_end) {
                    mbsd_physical_mapping_errors += mbsd_fresh_kv_enabled;
                    LOG_ERR("%s: logit source %d for target %d is outside physical batch [%d, %d)\n",
                            __func__, source_pos, pos, batch_abs_start, batch_abs_end);
                    physical_mapping_failed = true;
                    break;
                }
                batch.logits[source_pos - batch_abs_start] = 1;
            }
            if (physical_mapping_failed) {
                total_batch_time += ggml_time_us() - batch_start;
                generation_failed = true;
                break;
            }

            if (params.add_gumbel_noise && params.temperature > 0.0f) {
                batch.logits[0] = 1;
            }

            std::fill(logits_row_by_pos.begin(), logits_row_by_pos.end(), -1);
            int32_t output_rows = 0;
            for (int32_t local = 0; local < batch.n_tokens; local++) {
                if (batch.logits[local]) {
                    logits_row_by_pos[batch_abs_start + local] = output_rows++;
                }
            }
            GGML_ASSERT(output_rows > 0);
            if (mbsd_enabled) {
                const int32_t current_window_rows = block_end - mbsd_window_start;
                const int32_t future_window_rows  = mbsd_window_end - block_end;
                const int32_t context_rows = batch.n_tokens - current_window_rows - future_window_rows;
                if (current_window_rows < 0 || future_window_rows < 0 || context_rows < 0) {
                    mbsd_bounds_errors++;
                    LOG_ERR("%s: MBSD row accounting failed at block %d step %d\n",
                            __func__, block_num + 1, step + 1);
                    total_batch_time += ggml_time_us() - batch_start;
                    generation_failed = true;
                    break;
                }
                mbsd_current_transformer_rows += current_window_rows;
                mbsd_future_transformer_rows  += future_window_rows;
                mbsd_context_transformer_rows += context_rows;
                mbsd_current_logit_rows += mask_positions.size();
                mbsd_future_logit_rows  += mbsd_future_positions.size();
                mbsd_draft_prediction_rows += mbsd_future_positions.size();
                mbsd_main_steps_this_block++;
                mbsd_main_steps_total++;
                if (mbsd_fresh_kv_enabled) {
                    mbsd_fresh_main_batches++;
                    mbsd_fresh_main_rows_min = std::min(mbsd_fresh_main_rows_min, batch.n_tokens);
                    mbsd_fresh_main_rows_max = std::max(mbsd_fresh_main_rows_max, batch.n_tokens);
                    mbsd_fresh_prefix_rows_reused += batch_abs_start;
                }
            }
            total_batch_time += ggml_time_us() - batch_start;
            iterations_with_forward++;

            LOG_DBG("%s: step %d/%d, block %d/%d, active masks = %d, output rows = %d\n",
                    __func__, global_step + 1, params.steps, block_num + 1, num_blocks, active_masks, output_rows);

            float * logits = nullptr;
            const size_t logits_size = (size_t) output_rows * n_vocab;

            if (params.cfg_scale > 0.0f) {
                auto [ret, cond_logits_ptr] = run_forward(conditional_perf, active_masks, output_rows);
                if (ret != 0) {
                    LOG_ERR("Failed to generate conditional");
                    if (diffusion_kv_graph_enabled || staged_token_stabilization || mbsd_enabled) {
                        generation_failed = true;
                    }
                    break;
                }

                const int64_t cfg_copy_start = ggml_time_us();
                cond_logits_buffer.resize(logits_size);
                std::memcpy(cond_logits_buffer.data(), cond_logits_ptr, logits_size * sizeof(float));

                // Unconditional generation (mask input)
                std::copy(output_tokens, output_tokens + params.max_length, un_x_buffer.begin());
                for (int32_t i = 0; i < n_input; i++) {
                    un_x_buffer[i] = params.mask_token_id;
                }

                for (int32_t i = 0; i < batch.n_tokens; i++) {
                    batch.token[i] = un_x_buffer[batch_abs_start + i];
                }
                total_cfg_cpu_time += ggml_time_us() - cfg_copy_start;

                auto uncond_result = run_forward(unconditional_perf, active_masks, output_rows);
                ret                = uncond_result.first;
                if (ret != 0) {
                    LOG_ERR("Failed to generate unconditional");
                    if (diffusion_kv_graph_enabled || staged_token_stabilization || mbsd_enabled) {
                        generation_failed = true;
                    }
                    break;
                }
                float * uncond_logits = uncond_result.second;

                // Apply CFG
                const int64_t cfg_mix_start = ggml_time_us();
                for (size_t i = 0; i < logits_size; i++) {
                    cond_logits_buffer[i] =
                        uncond_logits[i] + (params.cfg_scale + 1.0f) * (cond_logits_buffer[i] - uncond_logits[i]);
                }
                total_cfg_cpu_time += ggml_time_us() - cfg_mix_start;
                logits = cond_logits_buffer.data();
            } else {
                auto [ret, result] = run_forward(conditional_perf, active_masks, output_rows);
                if (ret != 0) {
                    LOG_ERR("%s: failed to decode at step %d, ret = %d\n", __func__, global_step, ret);
                    if (diffusion_kv_graph_enabled || staged_token_stabilization || mbsd_enabled) {
                        generation_failed = true;
                    }
                    break;
                }
                logits = result;
            }

            if (!logits) {
                LOG_ERR("%s: failed to get logits at step %d\n", __func__, global_step);
                if (diffusion_kv_graph_enabled || staged_token_stabilization || mbsd_enabled) {
                    generation_failed = true;
                }
                break;
            }

            auto get_logits_for_pos = [&](int32_t pos) -> const float * {
                const int32_t source_pos = params.shift_logits ? std::max(pos - 1, 0) : pos;
                const int32_t row        = logits_row_by_pos[source_pos];
                GGML_ASSERT(row >= 0);
                return logits + (size_t) row * n_vocab;
            };

            int64_t time_start_sampling = ggml_time_us();
            sampling_passes++;

            int32_t base_selections_this_step      = 0;
            int32_t threshold_selections_this_step = 0;
            int32_t forced_selections_this_step    = 0;
            const bool mbsd_force_final =
                mbsd_enabled && step + 1 == mbsd_baseline_effective_steps;

            if (params.add_gumbel_noise && params.temperature > 0.0f) {
                add_gumbel_noise(logits, n_vocab, params.temperature, rng);
            }

            if (staged_token_stabilization) {
                sts_sampled_tokens.resize(mask_positions.size());
                sts_sampled_confidences.resize(mask_positions.size());
                for (size_t i = 0; i < mask_positions.size(); i++) {
                    const auto sampled = sample_staged_logits(get_logits_for_pos(mask_positions[i]));
                    sts_sampled_tokens[i]      = sampled.first;
                    sts_sampled_confidences[i] = sampled.second;
                }

                const int32_t revision_target =
                    select_staged_revision_target(block_end, nullptr, false);

                llama_token revised_token      = LLAMA_TOKEN_NULL;
                float       revised_confidence = 0.0f;
                if (revision_target >= 0) {
                    total_sampling_time += ggml_time_us() - time_start_sampling;
                    if (!run_staged_revision(revision_target, revised_token, revised_confidence)) {
                        generation_failed = true;
                        break;
                    }
                    time_start_sampling = ggml_time_us();
                }

                int32_t promoted_this_step       = 0;
                int32_t best_invisible_idx       = -1;
                float   best_invisible_confidence = -std::numeric_limits<float>::infinity();

                for (size_t i = 0; i < mask_positions.size(); i++) {
                    const int32_t pos  = mask_positions[i];
                    const float   conf = sts_sampled_confidences[i];

                    GGML_ASSERT(token_states[pos] == diffusion_token_state::invisible);
                    if (conf > best_invisible_confidence) {
                        best_invisible_idx        = (int32_t) i;
                        best_invisible_confidence = conf;
                    }

                    if (conf >= params.visibility_threshold) {
                        output_tokens[pos] = sts_sampled_tokens[i];
                        sts_latest_confidence[pos] = conf;
                        promoted_this_step++;
                        threshold_selections_this_step++;
                        if (conf >= params.stability_threshold) {
                            token_states[pos] = diffusion_token_state::stable;
                            sts_direct_stable_promotions++;
                        } else {
                            token_states[pos] = diffusion_token_state::visible;
                            sts_visibility_promotions++;
                        }
                    }
                }

                if (promoted_this_step == 0 && best_invisible_idx >= 0) {
                    const int32_t pos = mask_positions[best_invisible_idx];
                    output_tokens[pos] = sts_sampled_tokens[best_invisible_idx];
                    token_states[pos] = diffusion_token_state::visible;
                    sts_latest_confidence[pos] = sts_sampled_confidences[best_invisible_idx];
                    base_selections_this_step++;
                    sts_forced_visible++;
                }

                if (revision_target >= 0) {
                    if (output_tokens[revision_target] != revised_token) {
                        sts_token_revisions++;
                    }
                    output_tokens[revision_target] = revised_token;
                    sts_latest_confidence[revision_target] = revised_confidence;
                    sts_last_revision_step[revision_target] = global_step;
                    sts_revision_count[revision_target]++;
                    sts_revision_passes++;
                    sts_ordinary_revision_passes++;
                    record_staged_revision_target(revision_target, false);
                    if (revised_confidence >= params.stability_threshold) {
                        token_states[revision_target] = diffusion_token_state::stable;
                        sts_visible_to_stable++;
                    }
                }

                int32_t visible_after_step = 0;
                for (int32_t pos = n_input; pos < block_end; pos++) {
                    visible_after_step += token_states[pos] == diffusion_token_state::visible;
                }
                sts_max_visible = std::max(sts_max_visible, visible_after_step);
            } else if (params.algorithm == DIFFUSION_ALGORITHM_ORIGIN) {
                int32_t transfer_count = calculate_transfer_count(
                    step, steps_this_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);
                float p_transfer = (float) transfer_count / mask_positions.size();

                for (int32_t pos : mask_positions) {
                    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < p_transfer) {
                        base_selections_this_step++;
                        const float * pos_logits = get_logits_for_pos(pos);
                        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                            candidates[token_id].id    = token_id;
                            candidates[token_id].logit = pos_logits[token_id];
                            candidates[token_id].p     = 0.0f;
                        }

                        llama_token_data_array cur_p = {
                            candidates.data(),
                            (size_t) n_vocab,
                            -1,
                            false,
                        };

                        llama_sampler_apply(sampler, &cur_p);
                        output_tokens[pos] = cur_p.data[cur_p.selected].id;
                    }
                }
            } else {
                auto confidence_greater = [](const std::pair<float, int32_t> & a,
                                             const std::pair<float, int32_t> & b) {
                    if (a.first != b.first) {
                        return a.first > b.first;
                    }
                    return a.second < b.second;
                };

                std::vector<std::pair<float, int32_t>> confidences;
                std::vector<llama_token>               sampled_tokens(mask_positions.size());

                for (size_t i = 0; i < mask_positions.size(); i++) {
                    int32_t       pos        = mask_positions[i];
                    const float * pos_logits = get_logits_for_pos(pos);

                    for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                        candidates[token_id].logit = pos_logits[token_id];
                        candidates[token_id].p     = 0.0f;
                        candidates[token_id].id    = token_id;
                    }

                    if (mbsd_force_final ||
                        (early_commit_enabled && step == steps_this_block - 1)) {
                        candidates[params.mask_token_id].logit = -std::numeric_limits<float>::infinity();
                    }

                    llama_token_data_array cur_p = {
                        candidates.data(),
                        candidates.size(),
                        -1,
                        false,
                    };

                    llama_sampler_apply(sampler, &cur_p);
                    const llama_token sampled_token = cur_p.data[cur_p.selected].id;
                    const float conf = calculate_confidence(cur_p, params.algorithm, rng);

                    sampled_tokens[i] = sampled_token;
                    confidences.emplace_back(conf, (int32_t) i);
                }

                std::vector<uint8_t> mbsd_preaccepted;
                if (mbsd_enabled) {
                    mbsd_preaccepted.resize(mask_positions.size(), 0);
                    const int32_t draft_credit = (int32_t) mbsd_verification_positions.size();
                    std::vector<std::pair<float, int32_t>> verification_ranking = confidences;
                    const int32_t ranked = std::min(draft_credit, (int32_t) verification_ranking.size());
                    if (ranked > 0) {
                        std::partial_sort(verification_ranking.begin(),
                                          verification_ranking.begin() + ranked,
                                          verification_ranking.end(),
                                          confidence_greater);
                    }

                    for (int32_t rank = 0; rank < ranked; rank++) {
                        const int32_t mask_idx = verification_ranking[rank].second;
                        const int32_t pos      = mask_positions[mask_idx];
                        if (mbsd_draft_valid[pos] && sampled_tokens[mask_idx] == mbsd_draft_tokens[pos]) {
                            output_tokens[pos] = sampled_tokens[mask_idx];
                            mbsd_preaccepted[mask_idx] = 1;
                        }
                    }

                    for (int32_t pos : mbsd_verification_positions) {
                        int32_t mask_idx = -1;
                        for (size_t i = 0; i < mask_positions.size(); i++) {
                            if (mask_positions[i] == pos) {
                                mask_idx = (int32_t) i;
                                break;
                            }
                        }
                        GGML_ASSERT(mask_idx >= 0);
                        mbsd_drafts_reevaluated++;
                        if (mbsd_preaccepted[mask_idx]) {
                            mbsd_drafts_accepted++;
                            mbsd_drafts_reconfirmed++;
                            mbsd_draft_accepts_this_block++;
                        } else {
                            mbsd_drafts_rejected++;
                        }
                        mbsd_draft_valid[pos]       = 0;
                        mbsd_draft_tokens[pos]      = params.mask_token_id;
                        mbsd_draft_confidences[pos] = 0.0f;
                    }
                    mbsd_verification_pending = false;

                    if (!mbsd_preaccepted.empty()) {
                        confidences.erase(
                            std::remove_if(confidences.begin(), confidences.end(),
                                [&](const std::pair<float, int32_t> & item) {
                                    return mbsd_preaccepted[item.second] != 0;
                                }),
                            confidences.end());
                    }
                }

                int32_t transfer_count = calculate_transfer_count(
                    step, steps_this_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);

                const int32_t scheduled_selection_count =
                    std::min(std::max(transfer_count, 0), (int32_t) confidences.size());

                if (early_commit_enabled && step == steps_this_block - 1) {
                    transfer_count = (int32_t) mask_positions.size();
                    forced_selections_this_step = (int32_t) mask_positions.size() - scheduled_selection_count;
                } else if (mbsd_force_final) {
                    transfer_count = (int32_t) confidences.size();
                    mbsd_final_forced_selections +=
                        (uint64_t) std::max(0, transfer_count - scheduled_selection_count);
                }

                const int32_t base_selection_count =
                    std::min(std::max(transfer_count, 0), (int32_t) confidences.size());

                if (early_commit_enabled) {
                    std::sort(confidences.begin(), confidences.end(), confidence_greater);

                    int32_t selection_count = base_selection_count;
                    while (selection_count < (int32_t) confidences.size() &&
                           confidences[selection_count].first > params.early_commit_threshold) {
                        selection_count++;
                    }

                    for (int32_t i = 0; i < selection_count; i++) {
                        const int32_t mask_idx = confidences[i].second;
                        const int32_t pos      = mask_positions[mask_idx];
                        output_tokens[pos]     = sampled_tokens[mask_idx];
                    }

                    base_selections_this_step      = scheduled_selection_count;
                    threshold_selections_this_step = selection_count - base_selection_count;
                } else if (transfer_count > 0) {
                    base_selections_this_step =
                        mbsd_force_final ?
                            scheduled_selection_count : base_selection_count;
                    if (params.alg_temp == 0.0f) {
                        std::partial_sort(confidences.begin(),
                                          confidences.begin() + base_selection_count,
                                          confidences.end(),
                                          confidence_greater);

                        for (int32_t i = 0; i < base_selection_count; i++) {
                            const int32_t mask_idx = confidences[i].second;
                            const int32_t pos      = mask_positions[mask_idx];
                            output_tokens[pos]     = sampled_tokens[mask_idx];
                        }
                    } else {
                        conf_candidates.clear();
                        for (size_t i = 0; i < confidences.size(); i++) {
                            const float conf_logit = confidences[i].first / params.alg_temp;
                            conf_candidates.emplace_back(llama_token_data{ (int32_t) i, conf_logit, 0.0f });
                        }

                        llama_token_data_array conf_array = {
                            conf_candidates.data(),
                            conf_candidates.size(),
                            -1,
                            false,
                        };

                        for (int32_t i = 0; i < base_selection_count; i++) {
                            llama_sampler_apply(dist_sampler, &conf_array);
                            const int32_t selected_idx = conf_array.selected;
                            const int32_t mask_idx     = confidences[selected_idx].second;
                            const int32_t pos          = mask_positions[mask_idx];
                            output_tokens[pos]         = sampled_tokens[mask_idx];

                            conf_candidates[selected_idx].p = 0.0f;
                            conf_array.selected             = -1;
                        }
                    }
                }

                if (mbsd_enabled) {
                    std::vector<llama_token> future_sampled_tokens(mbsd_future_positions.size());
                    std::vector<float>       future_confidences(mbsd_future_positions.size());
                    std::fill(mbsd_future_index_by_pos.begin(), mbsd_future_index_by_pos.end(), -1);

                    for (size_t i = 0; i < mbsd_future_positions.size(); i++) {
                        const int32_t pos = mbsd_future_positions[i];
                        const float * pos_logits = get_logits_for_pos(pos);
                        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                            candidates[token_id].logit = pos_logits[token_id];
                            candidates[token_id].p     = 0.0f;
                            candidates[token_id].id    = token_id;
                        }
                        candidates[params.mask_token_id].logit = -std::numeric_limits<float>::infinity();

                        llama_token_data_array cur_p = {
                            candidates.data(),
                            candidates.size(),
                            -1,
                            false,
                        };
                        llama_sampler_apply(mbsd_sampler, &cur_p);
                        future_sampled_tokens[i] = cur_p.data[cur_p.selected].id;
                        future_confidences[i] = calculate_confidence(cur_p, params.algorithm, rng);
                        mbsd_future_index_by_pos[pos] = (int32_t) i;
                    }

                    for (int32_t pos : mbsd_future_positions) {
                        if (mbsd_draft_valid[pos]) {
                            const int32_t sample_idx = mbsd_future_index_by_pos[pos];
                            GGML_ASSERT(sample_idx >= 0);
                            mbsd_drafts_replaced +=
                                mbsd_draft_tokens[pos] != future_sampled_tokens[sample_idx];
                            mbsd_draft_tokens[pos]      = future_sampled_tokens[sample_idx];
                            mbsd_draft_confidences[pos] = future_confidences[sample_idx];
                            mbsd_draft_updates++;
                        }
                    }

                    for (mbsd_future_plan & plan : mbsd_future_plans) {
                        if (plan.quota == 0) {
                            mbsd_draft_steps[plan.block]++;
                        } else if (!plan.positions.empty()) {
                            std::vector<std::pair<float, int32_t>> draft_confidences;
                            draft_confidences.reserve(plan.positions.size());
                            for (int32_t pos : plan.positions) {
                                const int32_t sample_idx = mbsd_future_index_by_pos[pos];
                                GGML_ASSERT(sample_idx >= 0);
                                draft_confidences.emplace_back(future_confidences[sample_idx], pos);
                            }
                            const int32_t selected = std::min(plan.quota, (int32_t) draft_confidences.size());
                            std::partial_sort(draft_confidences.begin(),
                                              draft_confidences.begin() + selected,
                                              draft_confidences.end(),
                                              confidence_greater);
                            for (int32_t i = 0; i < selected; i++) {
                                const int32_t pos        = draft_confidences[i].second;
                                const int32_t sample_idx = mbsd_future_index_by_pos[pos];
                                GGML_ASSERT(!mbsd_draft_valid[pos]);
                                mbsd_drafts_introduced++;
                                mbsd_draft_ever[pos] = 1;
                                mbsd_draft_valid[pos]       = 1;
                                mbsd_draft_tokens[pos]      = future_sampled_tokens[sample_idx];
                                mbsd_draft_confidences[pos] = future_confidences[sample_idx];
                            }
                            mbsd_draft_quota[plan.block] -= selected;
                            if (mbsd_draft_quota[plan.block] == 0) {
                                mbsd_draft_steps[plan.block]++;
                            }
                        }

                        if (mbsd_draft_quota[plan.block] == 0) {
                            if (mbsd_draft_steps[plan.block] <
                                (int32_t) mbsd_draft_transfer_tokens[plan.block].size()) {
                                mbsd_draft_quota[plan.block] =
                                    mbsd_draft_transfer_tokens[plan.block][mbsd_draft_steps[plan.block]];
                            } else {
                                mbsd_draft_quota[plan.block] = 0;
                            }
                        }
                    }
                }
            }

            base_token_selections += base_selections_this_step;
            threshold_extra_selections += threshold_selections_this_step;
            forced_final_selections += forced_selections_this_step;

            int32_t remaining_masks = 0;
            for (int32_t pos : mask_positions) {
                if (staged_token_stabilization ?
                        token_states[pos] == diffusion_token_state::invisible :
                        output_tokens[pos] == params.mask_token_id) {
                    remaining_masks++;
                }
            }
            tokens_committed += active_masks - remaining_masks;

            bool finish_block = false;
            if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED && remaining_masks == 0) {
                if (!block_completion_recorded) {
                    blocks_completed++;
                    block_completion_recorded = true;
                    if (staged_token_stabilization) {
                        for (int32_t pos = block_start; pos < block_end; pos++) {
                            sts_unstable_at_block_completion +=
                                token_states[pos] == diffusion_token_state::visible;
                        }
                    }
                }

                if (staged_token_stabilization) {
                    finish_block = true;
                } else {
                    finish_block = early_commit_enabled || mbsd_enabled;
                }

                if (finish_block && step + 1 < steps_this_block &&
                    (early_commit_enabled || staged_token_stabilization || mbsd_enabled)) {
                    const int32_t skipped_steps = steps_this_block - step - 1;
                    blocks_finished_early++;
                    if (staged_token_stabilization) {
                        sts_unused_steps_this_block = skipped_steps;
                        sts_unused_step_start = global_step + 1;
                    } else if (early_commit_enabled) {
                        scheduled_steps_skipped += skipped_steps;
                        scheduled_forwards_skipped += skipped_steps * (params.cfg_scale > 0.0f ? 2 : 1);
                    }
                }
            }

            if (staged_token_stabilization) {
                sts_trajectory_hash ^= (uint32_t) global_step;
                sts_trajectory_hash *= 1099511628211ULL;
                for (int32_t pos = n_input; pos < params.max_length; pos++) {
                    sts_trajectory_hash ^= (uint32_t) output_tokens[pos];
                    sts_trajectory_hash *= 1099511628211ULL;
                    sts_trajectory_hash ^= (uint32_t) token_states[pos];
                    sts_trajectory_hash *= 1099511628211ULL;
                }
            }

            if (mbsd_enabled) {
                for (int32_t pos = block_end; pos < params.max_length; pos++) {
                    if (output_tokens[pos] != params.mask_token_id) {
                        mbsd_future_semantic_commits++;
                    }
                }
                if (mbsd_future_semantic_commits > 0) {
                    LOG_ERR("%s: MBSD committed a future-block token at block %d step %d\n",
                            __func__, block_num + 1, step + 1);
                    generation_failed = true;
                }

                mbsd_trajectory_hash ^= (uint32_t) global_step;
                mbsd_trajectory_hash *= 1099511628211ULL;
                mbsd_trajectory_hash ^= (uint32_t) mbsd_window_start;
                mbsd_trajectory_hash *= 1099511628211ULL;
                mbsd_trajectory_hash ^= (uint32_t) mbsd_window_end;
                mbsd_trajectory_hash *= 1099511628211ULL;
                for (int32_t pos = n_input; pos < params.max_length; pos++) {
                    mbsd_trajectory_hash ^= (uint32_t) output_tokens[pos];
                    mbsd_trajectory_hash *= 1099511628211ULL;
                    mbsd_trajectory_hash ^= (uint32_t) mbsd_draft_valid[pos];
                    mbsd_trajectory_hash *= 1099511628211ULL;
                    if (mbsd_draft_valid[pos]) {
                        mbsd_trajectory_hash ^= (uint32_t) mbsd_draft_tokens[pos];
                        mbsd_trajectory_hash *= 1099511628211ULL;
                    }
                }
            }

            const int64_t time_end_sampling = ggml_time_us();
            total_sampling_time += time_end_sampling - time_start_sampling;
            iterations_completed++;

            if (generation_failed || finish_block) {
                break;
            }
        }

        if (mbsd_enabled && !generation_failed) {
            int32_t block_masks_remaining = 0;
            int32_t block_drafts_remaining = 0;
            for (int32_t pos = block_start; pos < block_end; pos++) {
                block_masks_remaining += output_tokens[pos] == params.mask_token_id;
                block_drafts_remaining += mbsd_draft_valid[pos] != 0;
            }
            if (!block_completion_recorded || block_masks_remaining != 0 || block_drafts_remaining != 0 ||
                blocks_completed != block_num + 1) {
                mbsd_block_order_violations++;
                LOG_ERR("%s: MBSD block completion invariant failed for block %d "
                        "(recorded = %s, masks = %d, drafts = %d, completed = %d)\n",
                        __func__, block_num + 1, block_completion_recorded ? "true" : "false",
                        block_masks_remaining, block_drafts_remaining, blocks_completed);
                generation_failed = true;
            }
            const int32_t mbsd_steps_saved_this_block =
                std::max(0, mbsd_baseline_effective_steps - mbsd_main_steps_this_block);
            const int32_t mbsd_nominal_tail_skipped_this_block =
                steps_this_block - std::max(mbsd_main_steps_this_block, mbsd_baseline_effective_steps);
            mbsd_scheduled_steps_saved += mbsd_steps_saved_this_block;
            mbsd_scheduled_forwards_saved += mbsd_steps_saved_this_block;
            mbsd_nominal_tail_steps_skipped += mbsd_nominal_tail_skipped_this_block;
            if (mbsd_main_steps_this_block < 0 ||
                mbsd_nominal_tail_skipped_this_block < 0 ||
                mbsd_main_steps_this_block + mbsd_steps_saved_this_block +
                    mbsd_nominal_tail_skipped_this_block != steps_this_block ||
                (mbsd_steps_saved_this_block > 0 && mbsd_draft_accepts_this_block == 0)) {
                mbsd_bounds_errors++;
                LOG_ERR("%s: MBSD step accounting failed for block %d "
                        "(main = %d, effective = %d, scheduled = %d, saved = %d, "
                        "nominal tail skipped = %d, accepted drafts = %d)\n",
                        __func__, block_num + 1, mbsd_main_steps_this_block,
                        mbsd_baseline_effective_steps, steps_this_block, mbsd_steps_saved_this_block,
                        mbsd_nominal_tail_skipped_this_block, mbsd_draft_accepts_this_block);
                generation_failed = true;
            }
        }

        if (staged_token_stabilization) {
            int32_t final_tail_steps_used = 0;
            if (block_num == num_blocks - 1 && block_completion_recorded) {
                const int32_t generated_tokens = params.max_length - n_input;
                for (int32_t pos = n_input; pos < params.max_length; pos++) {
                    sts_final_visible_before += token_states[pos] == diffusion_token_state::visible;
                }
                sts_final_visible_after = sts_final_visible_before;
                sts_final_available_steps = sts_unused_steps_this_block;
                sts_final_revision_budget = std::min(
                    params.staged_final_revision_steps, sts_final_available_steps);

                const float visible_ratio = generated_tokens > 0 ?
                    (float) sts_final_visible_before / generated_tokens : 0.0f;
                sts_final_visible_ratio_observed = visible_ratio;
                if (params.staged_final_revision_steps <= 0) {
                    sts_final_stop_reason = "disabled";
                } else if (sts_final_visible_before == 0) {
                    sts_final_stop_reason = "all-stable";
                } else if (visible_ratio < params.staged_final_visible_ratio) {
                    sts_final_stop_reason = "below-ratio";
                } else if (sts_final_revision_budget == 0) {
                    sts_final_stop_reason = "no-schedule-slots";
                } else {
                    sts_final_revision_triggered = true;
                    sts_final_stop_reason = "budget";

                    std::vector<uint8_t> revised_this_sweep(params.max_length, 0);
                    bool sweep_token_changed = false;
                    bool start_new_sweep = true;

                    while (final_tail_steps_used < sts_final_revision_budget && !generation_failed) {
                        int32_t visible_now = 0;
                        for (int32_t pos = n_input; pos < params.max_length; pos++) {
                            visible_now += token_states[pos] == diffusion_token_state::visible;
                        }
                        if (visible_now == 0) {
                            sts_final_stop_reason = "all-stable";
                            break;
                        }
                        if ((float) visible_now / generated_tokens < params.staged_final_visible_ratio) {
                            sts_final_stop_reason = "below-ratio";
                            break;
                        }

                        if (start_new_sweep) {
                            std::fill(revised_this_sweep.begin(), revised_this_sweep.end(), 0);
                            sweep_token_changed = false;
                            start_new_sweep = false;
                            sts_final_revision_sweeps_started++;
                        }

                        const int64_t sampling_start = ggml_time_us();
                        const int32_t target = select_staged_revision_target(
                            params.max_length, &revised_this_sweep, true);
                        GGML_ASSERT(target >= 0);
                        total_sampling_time += ggml_time_us() - sampling_start;

                        const int32_t global_step = sts_unused_step_start + final_tail_steps_used;
                        iterations_started++;
                        if (params.step_callback) {
                            const int64_t callback_start = ggml_time_us();
                            if (!params.step_callback(
                                    global_step, params.steps, output_tokens, params.max_length,
                                    params.step_callback_user_data)) {
                                total_callback_time += ggml_time_us() - callback_start;
                                sts_final_stop_reason = "cancelled";
                                generation_failed = true;
                                break;
                            }
                            total_callback_time += ggml_time_us() - callback_start;
                        }

                        sampling_passes++;
                        sts_revision_candidates += visible_now;

                        llama_token revised_token      = LLAMA_TOKEN_NULL;
                        float       revised_confidence = 0.0f;
                        iterations_with_forward++;
                        if (!run_staged_revision(target, revised_token, revised_confidence)) {
                            sts_final_stop_reason = "error";
                            generation_failed = true;
                            break;
                        }

                        const int64_t update_start  = ggml_time_us();
                        const bool    token_changed = output_tokens[target] != revised_token;
                        const bool    became_stable = revised_confidence >= params.stability_threshold;
                        if (token_changed) {
                            sts_token_revisions++;
                            sts_final_token_revisions++;
                            sweep_token_changed = true;
                        }
                        output_tokens[target] = revised_token;
                        sts_latest_confidence[target] = revised_confidence;
                        sts_last_revision_step[target] = global_step;
                        sts_revision_count[target]++;
                        revised_this_sweep[target] = 1;
                        sts_revision_passes++;
                        sts_final_revision_passes++;
                        final_tail_steps_used++;
                        record_staged_revision_target(target, true);

                        if (became_stable) {
                            token_states[target] = diffusion_token_state::stable;
                            sts_visible_to_stable++;
                            sts_final_visible_to_stable++;
                        }

                        sts_trajectory_hash ^= 0x5441494cU;
                        sts_trajectory_hash *= 1099511628211ULL;
                        sts_trajectory_hash ^= (uint32_t) global_step;
                        sts_trajectory_hash *= 1099511628211ULL;
                        sts_trajectory_hash ^= (uint32_t) target;
                        sts_trajectory_hash *= 1099511628211ULL;
                        for (int32_t pos = n_input; pos < params.max_length; pos++) {
                            sts_trajectory_hash ^= (uint32_t) output_tokens[pos];
                            sts_trajectory_hash *= 1099511628211ULL;
                            sts_trajectory_hash ^= (uint32_t) token_states[pos];
                            sts_trajectory_hash *= 1099511628211ULL;
                        }

                        total_sampling_time += ggml_time_us() - update_start;
                        iterations_completed++;

                        int32_t visible_after_pass    = 0;
                        bool    has_unrevised_visible = false;
                        for (int32_t pos = n_input; pos < params.max_length; pos++) {
                            if (token_states[pos] == diffusion_token_state::visible) {
                                visible_after_pass++;
                                has_unrevised_visible |= !revised_this_sweep[pos];
                            }
                        }

                        if (visible_after_pass == 0) {
                            sts_final_revision_sweeps_completed++;
                            sts_final_stop_reason = "all-stable";
                            break;
                        }
                        if (!has_unrevised_visible) {
                            sts_final_revision_sweeps_completed++;
                            if (!sweep_token_changed) {
                                sts_final_stop_reason = "fixed-point";
                                break;
                            }
                            start_new_sweep = true;
                        }
                    }

                    sts_final_visible_after = 0;
                    for (int32_t pos = n_input; pos < params.max_length; pos++) {
                        sts_final_visible_after += token_states[pos] == diffusion_token_state::visible;
                    }
                }
            }

            sts_scheduled_steps_skipped += sts_unused_steps_this_block - final_tail_steps_used;
        }

        if (cache_reuse_enabled && !generation_failed) {
            int32_t block_masks_remaining = 0;
            for (int32_t pos = block_start; pos < block_end; pos++) {
                block_masks_remaining += output_tokens[pos] == params.mask_token_id;
            }

            if (block_masks_remaining > 0) {
                LOG_ERR("%s: block %d finished with %d masks; refusing to advance the prefix cache\n",
                        __func__, block_num + 1, block_masks_remaining);
                generation_failed = true;
                break;
            }

            if (block_num + 1 < num_blocks) {
                const int32_t seal_start = params.shift_logits ? block_start - 1 : block_start;
                if (!llama_memory_seq_rm(memory, 0, seal_start, -1)) {
                    LOG_ERR("%s: failed to reset the cache before sealing block %d\n", __func__, block_num + 1);
                    generation_failed = true;
                    break;
                }
                const int64_t batch_start = ggml_time_us();
                setup_cache_batch(seal_start, block_end);
                total_batch_time += ggml_time_us() - batch_start;

                auto [ret, ignored] = run_forward(cache_perf, 0, 1);
                GGML_UNUSED(ignored);
                if (ret != 0) {
                    LOG_ERR("%s: failed to seal block %d, ret = %d\n", __func__, block_num + 1, ret);
                    generation_failed = true;
                    break;
                }
                transition_seals++;

                const int32_t next_window_start = params.shift_logits ? block_end - 1 : block_end;
                if (!llama_memory_seq_rm(memory, 0, next_window_start, -1)) {
                    LOG_ERR("%s: failed to trim the sealed cache at position %d\n", __func__, next_window_start);
                    generation_failed = true;
                    break;
                }
            }
        }
    }

    if (diffusion_kv_graph_enabled) {
        llama_synchronize(ctx);
    }
    const int64_t time_end = ggml_time_us();
    const llama_pos final_cache_pos = diffusion_kv_graph_enabled ? llama_memory_seq_pos_max(memory, 0) : -1;
    if (diffusion_kv_graph_enabled) {
        llama_memory_clear(memory, false);
    }
    total_time += time_end - time_start;

    const int32_t graph_reuses_end = llama_perf_context(ctx).n_reused;
    const int32_t graph_reuses     = std::max(0, graph_reuses_end - graph_reuses_start);

    const int32_t total_forward_calls =
        conditional_perf.calls + unconditional_perf.calls + revision_perf.calls + cache_perf.calls;
    const int32_t total_forwards =
        conditional_perf.completed + unconditional_perf.completed + revision_perf.completed + cache_perf.completed;
    const int64_t total_decode_time =
        conditional_perf.decode_time + unconditional_perf.decode_time + revision_perf.decode_time +
        cache_perf.decode_time;
    const int64_t total_completion_time =
        conditional_perf.completion_time + unconditional_perf.completion_time + revision_perf.completion_time +
        cache_perf.completion_time;

    const uint64_t total_active_masks = conditional_perf.active_masks + unconditional_perf.active_masks;
    const uint64_t total_output_rows  =
        conditional_perf.output_rows + unconditional_perf.output_rows + revision_perf.output_rows;
    const uint64_t total_logits_bytes =
        conditional_perf.logits_bytes + unconditional_perf.logits_bytes + revision_perf.logits_bytes;
    const uint64_t main_input_tokens  = conditional_perf.input_tokens + unconditional_perf.input_tokens;
    const uint64_t revision_input_tokens = revision_perf.input_tokens;
    const uint64_t total_input_tokens = main_input_tokens + revision_input_tokens + cache_perf.input_tokens;
    const int32_t  main_forwards_completed = conditional_perf.completed + unconditional_perf.completed;
    const uint64_t mbsd_reference_dense_rows =
        (uint64_t) main_forwards_completed * params.max_length;
    const uint64_t mbsd_main_rows_saved =
        mbsd_reference_dense_rows >= main_input_tokens ?
            mbsd_reference_dense_rows - main_input_tokens : 0;
    const int64_t mbsd_net_rows_saved =
        (int64_t) mbsd_reference_dense_rows - (int64_t) main_input_tokens -
        (int64_t) cache_perf.input_tokens;
    const int32_t mbsd_fresh_main_rows_min_report =
        mbsd_fresh_main_rows_min == std::numeric_limits<int32_t>::max() ?
            0 : mbsd_fresh_main_rows_min;

    if (fresh_full_sequence_kv) {
        const int32_t oracle_forward_calls =
            conditional_perf.calls + unconditional_perf.calls + revision_perf.calls;
        const int32_t oracle_forwards =
            conditional_perf.completed + unconditional_perf.completed + revision_perf.completed;
        const uint64_t oracle_input_tokens = main_input_tokens + revision_input_tokens;
        const uint64_t expected_rows       = (uint64_t) oracle_forwards * params.max_length;
        if (cache_perf.calls != 0 || prompt_prefills != 0 || transition_seals != 0 ||
            full_sequence_pre_forward_clears != oracle_forward_calls || oracle_input_tokens != expected_rows) {
            LOG_ERR("%s: %s invariant failed "
                    "(clears = %d/%d, rows = %llu/%llu, cache forwards = %d, prefills = %d, seals = %d)\n",
                    __func__,
                    mbsd_fresh_kv_enabled ? "MBSD fresh KV" : "full-sequence KV oracle",
                    full_sequence_pre_forward_clears,
                    oracle_forward_calls,
                    (unsigned long long) oracle_input_tokens,
                    (unsigned long long) expected_rows,
                    cache_perf.calls,
                    prompt_prefills,
                    transition_seals);
            mbsd_fresh_cache_invariant_errors += mbsd_fresh_kv_enabled;
            generation_failed = true;
        }
    }

    if (mbsd_enabled && !generation_failed) {
        const uint64_t logical_transformer_rows =
            mbsd_current_transformer_rows + mbsd_future_transformer_rows +
            mbsd_context_transformer_rows;
        bool physical_invariants_ok =
            mbsd_physical_mapping_errors == 0 && logical_transformer_rows == main_input_tokens;

        if (mbsd_fresh_kv_enabled) {
            const int32_t expected_full_sequence_clears =
                conditional_perf.calls + unconditional_perf.calls + revision_perf.calls;
            const bool fresh_invariants_ok =
                prompt_prefills == 0 && transition_seals == 0 &&
                cache_perf.calls == 0 && cache_perf.completed == 0 &&
                full_sequence_pre_forward_clears == expected_full_sequence_clears &&
                mbsd_fresh_main_batches == main_forwards_completed &&
                mbsd_fresh_prefix_rows_reused == 0 &&
                mbsd_fresh_main_rows_min_report == params.max_length &&
                mbsd_fresh_main_rows_max == params.max_length &&
                main_input_tokens == mbsd_reference_dense_rows &&
                mbsd_main_rows_saved == 0 && mbsd_net_rows_saved == 0;
            physical_invariants_ok = physical_invariants_ok && fresh_invariants_ok;
            mbsd_fresh_cache_invariant_errors += !fresh_invariants_ok;
        } else {
            physical_invariants_ok = physical_invariants_ok &&
                cache_perf.calls == 0 && cache_perf.completed == 0 &&
                prompt_prefills == 0 && transition_seals == 0 &&
                full_sequence_pre_forward_clears == 0 &&
                main_input_tokens == mbsd_reference_dense_rows;
        }

        if (!physical_invariants_ok) {
            LOG_ERR("%s: MBSD physical execution invariant failed "
                    "(fresh KV = %s, dense/main/cache/net-saved = %llu/%llu/%llu/%lld, "
                    "main batches/min/max = %d/%d/%d, full-sequence clears = %d, cache calls/completed = %d/%d, "
                    "prefills/seals = %d/%d, logical rows = %llu, mapping errors = %llu)\n",
                    __func__,
                    mbsd_fresh_kv_enabled ? "true" : "false",
                    (unsigned long long) mbsd_reference_dense_rows,
                    (unsigned long long) main_input_tokens,
                    (unsigned long long) cache_perf.input_tokens,
                    (long long) mbsd_net_rows_saved,
                    mbsd_fresh_main_batches,
                    mbsd_fresh_main_rows_min_report,
                    mbsd_fresh_main_rows_max,
                    full_sequence_pre_forward_clears,
                    cache_perf.calls,
                    cache_perf.completed,
                    prompt_prefills,
                    transition_seals,
                    (unsigned long long) logical_transformer_rows,
                    (unsigned long long) mbsd_physical_mapping_errors);
            generation_failed = true;
        }
    }

    int32_t output_masks_remaining = 0;
    for (int32_t i = n_input; i < params.max_length; i++) {
        if (output_tokens[i] == params.mask_token_id) {
            output_masks_remaining++;
        }
    }

    if (mbsd_enabled && !generation_failed && output_masks_remaining == 0) {
        const llama_token pad_token      = llama_vocab_pad(vocab);
        const bool        has_valid_pad  =
            pad_token >= 0 && pad_token < n_vocab && pad_token != params.mask_token_id;
        llama_token       terminal_token = LLAMA_TOKEN_NULL;
        bool              after_eog      = false;

        for (int32_t pos = n_input; pos < params.max_length; pos++) {
            const llama_token token = output_tokens[pos];
            const bool token_is_eog =
                token >= 0 && token < n_vocab && llama_vocab_is_eog(vocab, token);
            if (!after_eog) {
                if (token_is_eog) {
                    after_eog = true;
                    terminal_token = has_valid_pad ? pad_token : token;
                }
                continue;
            }

            if (!token_is_eog && (!has_valid_pad || token != pad_token)) {
                output_tokens[pos] = terminal_token;
                mbsd_post_eog_corrections++;
            }
        }
    }

    int32_t mbsd_drafts_pending = 0;
    int32_t mbsd_distinct_drafts = 0;
    if (mbsd_enabled) {
        for (int32_t pos = n_input; pos < params.max_length; pos++) {
            mbsd_drafts_pending += mbsd_draft_valid[pos] != 0;
            mbsd_distinct_drafts += mbsd_draft_ever[pos] != 0;
        }
        const bool mbsd_invariants_ok =
            output_masks_remaining == 0 && mbsd_drafts_pending == 0 &&
            mbsd_drafts_reevaluated == mbsd_drafts_accepted + mbsd_drafts_rejected &&
            mbsd_drafts_introduced == (uint64_t) mbsd_distinct_drafts &&
            mbsd_drafts_reevaluated == (uint64_t) mbsd_distinct_drafts &&
            mbsd_future_semantic_commits == 0 && !prefix_kv_enabled &&
            mbsd_block_order_violations == 0 && mbsd_verification_input_errors == 0 &&
            mbsd_bounds_errors == 0 && mbsd_physical_mapping_errors == 0 &&
            blocks_started == num_blocks && blocks_completed == num_blocks &&
            tokens_committed == (uint64_t) (params.max_length - n_input) &&
            mbsd_main_steps_total + mbsd_scheduled_steps_saved +
                mbsd_nominal_tail_steps_skipped == params.steps &&
            mbsd_scheduled_forwards_saved == mbsd_scheduled_steps_saved &&
            conditional_perf.completed == mbsd_main_steps_total;
        if (!mbsd_invariants_ok) {
            LOG_ERR("%s: MBSD final invariant failed "
                    "(masks = %d, pending = %d, introduced/distinct/reevaluated = %llu/%d/%llu, "
                    "accepted/rejected = %llu/%llu, future commits = %llu, prefix KV = %s, "
                    "block/verification/bounds/physical errors = %llu/%llu/%llu/%llu, blocks = %d/%d/%d, "
                    "committed/generated = %llu/%d, main/saved/nominal-tail/scheduled = %d/%d/%d/%d, "
                    "completed forwards = %d)\n",
                    __func__, output_masks_remaining, mbsd_drafts_pending,
                    (unsigned long long) mbsd_drafts_introduced, mbsd_distinct_drafts,
                    (unsigned long long) mbsd_drafts_reevaluated,
                    (unsigned long long) mbsd_drafts_accepted,
                    (unsigned long long) mbsd_drafts_rejected,
                    (unsigned long long) mbsd_future_semantic_commits,
                    prefix_kv_enabled ? "enabled" : "disabled",
                    (unsigned long long) mbsd_block_order_violations,
                    (unsigned long long) mbsd_verification_input_errors,
                    (unsigned long long) mbsd_bounds_errors,
                    (unsigned long long) mbsd_physical_mapping_errors,
                    blocks_started, blocks_completed, num_blocks,
                    (unsigned long long) tokens_committed, params.max_length - n_input,
                    mbsd_main_steps_total, mbsd_scheduled_steps_saved,
                    mbsd_nominal_tail_steps_skipped, params.steps,
                    conditional_perf.completed);
            generation_failed = true;
        }
    }

    int32_t  sts_invisible              = 0;
    int32_t  sts_visible                = 0;
    int32_t  sts_stable                 = 0;
    int32_t  sts_never_revised_visible  = 0;
    int32_t  sts_visible_revision_min   = 0;
    int32_t  sts_visible_revision_max   = 0;
    uint64_t sts_visible_revision_sum   = 0;
    float    sts_visible_confidence_min = 0.0f;
    float    sts_visible_confidence_max = 0.0f;
    double   sts_visible_confidence_sum = 0.0;
    if (staged_token_stabilization) {
        for (int32_t i = n_input; i < params.max_length; i++) {
            switch (token_states[i]) {
                case diffusion_token_state::invisible:
                    sts_invisible++;
                    break;
                case diffusion_token_state::visible:
                    sts_visible++;
                    break;
                case diffusion_token_state::stable:
                    sts_stable++;
                    break;
            }
        }

        const int32_t generated_tokens = params.max_length - n_input;
        if (sts_invisible + sts_visible + sts_stable != generated_tokens ||
            sts_invisible != output_masks_remaining ||
            tokens_committed != (uint64_t) (sts_visible + sts_stable)) {
            LOG_ERR("%s: staged token state invariant failed "
                    "(IV = %d, V = %d, S = %d, masks = %d, commits = %llu, generated = %d)\n",
                    __func__, sts_invisible, sts_visible, sts_stable, output_masks_remaining,
                    (unsigned long long) tokens_committed, generated_tokens);
            generation_failed = true;
        }

        bool first_visible = true;
        for (int32_t i = n_input; i < params.max_length; i++) {
            if (token_states[i] != diffusion_token_state::visible) {
                continue;
            }
            sts_never_revised_visible += sts_revision_count[i] == 0;
            sts_visible_revision_sum += sts_revision_count[i];
            sts_visible_confidence_sum += sts_latest_confidence[i];
            if (first_visible) {
                sts_visible_revision_min = sts_revision_count[i];
                sts_visible_revision_max = sts_revision_count[i];
                sts_visible_confidence_min = sts_latest_confidence[i];
                sts_visible_confidence_max = sts_latest_confidence[i];
                first_visible = false;
            } else {
                sts_visible_revision_min = std::min(sts_visible_revision_min, sts_revision_count[i]);
                sts_visible_revision_max = std::max(sts_visible_revision_max, sts_revision_count[i]);
                sts_visible_confidence_min = std::min(sts_visible_confidence_min, sts_latest_confidence[i]);
                sts_visible_confidence_max = std::max(sts_visible_confidence_max, sts_latest_confidence[i]);
            }
        }
    }

    const int64_t accounted_time = total_callback_time + total_batch_time + total_cache_clear_time + total_decode_time +
                                   total_completion_time + total_cfg_cpu_time + total_sampling_time;
    const int64_t other_time = std::max<int64_t>(0, total_time - accounted_time);

    const double step_divisor     = iterations_completed > 0 ? iterations_completed : 1;
    const double sampling_divisor = sampling_passes > 0 ? sampling_passes : 1;
    const double forward_divisor  = total_forwards > 0 ? total_forwards : 1;
    const double main_forward_divisor =
        conditional_perf.completed + unconditional_perf.completed > 0 ?
        conditional_perf.completed + unconditional_perf.completed : 1;

    LOG_INF("\ntotal time: %0.2fms, time per completed iteration: %0.2fms, "
            "sampling time per completed iteration: %0.2fms\n",
            total_time / 1000.0,
            total_time / 1000.0 / step_divisor,
            total_sampling_time / 1000.0 / step_divisor);

    LOG_INF("diffusion performance:\n");
    LOG_INF("  iterations: started = %d, with forward = %d, sampling passes = %d, completed = %d\n",
            iterations_started, iterations_with_forward, sampling_passes, iterations_completed);
    if (staged_token_stabilization) {
        LOG_INF("  forwards: calls = %d, completed = %d, conditional/main = %d, unconditional = %d, "
                "revision = %d, cache maintenance = %d\n",
                total_forward_calls, total_forwards, conditional_perf.completed, unconditional_perf.completed,
                revision_perf.completed, cache_perf.completed);
    } else {
        LOG_INF("  forwards: calls = %d, completed = %d, conditional/main = %d, unconditional = %d, "
                "cache maintenance = %d\n",
                total_forward_calls, total_forwards, conditional_perf.completed, unconditional_perf.completed,
                cache_perf.completed);
    }
    LOG_INF("  active masks: total = %llu, average per forward = %.2f\n",
            (unsigned long long) total_active_masks, total_active_masks / main_forward_divisor);
    if (staged_token_stabilization) {
        LOG_INF("  transformer rows: main = %llu, revision = %llu, cache maintenance = %llu, total = %llu\n",
                (unsigned long long) main_input_tokens,
                (unsigned long long) revision_input_tokens,
                (unsigned long long) cache_perf.input_tokens,
                (unsigned long long) total_input_tokens);
    } else {
        LOG_INF("  transformer rows: main = %llu, cache maintenance = %llu, total = %llu\n",
                (unsigned long long) main_input_tokens,
                (unsigned long long) cache_perf.input_tokens,
                (unsigned long long) total_input_tokens);
    }
    LOG_INF("  logits: rows = %llu, bytes = %llu (%.2f MiB)\n",
            (unsigned long long) total_output_rows,
            (unsigned long long) total_logits_bytes,
            total_logits_bytes / (1024.0 * 1024.0));
    LOG_INF("  cache maintenance output: rows = %llu, logits bytes = %llu (%.2f MiB)\n",
            (unsigned long long) cache_perf.output_rows,
            (unsigned long long) cache_perf.logits_bytes,
            cache_perf.logits_bytes / (1024.0 * 1024.0));
    if (staged_token_stabilization) {
        LOG_INF("  revision output: rows = %llu, logits bytes = %llu (%.2f MiB)\n",
                (unsigned long long) revision_perf.output_rows,
                (unsigned long long) revision_perf.logits_bytes,
                revision_perf.logits_bytes / (1024.0 * 1024.0));
    }
    LOG_INF("  token commits: committed = %llu, remaining masks = %d\n",
            (unsigned long long) tokens_committed,
            output_masks_remaining);
    if (!staged_token_stabilization) {
        LOG_INF("  token selections: base = %llu, threshold extra = %llu, forced final = %llu\n",
                (unsigned long long) base_token_selections,
                (unsigned long long) threshold_extra_selections,
                (unsigned long long) forced_final_selections);
    }
    if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        if (staged_token_stabilization) {
            LOG_INF("  block schedule: generated-aware = %s, generated tokens = %d, planned blocks = %d, "
                    "step allocation = block token count\n",
                    params.generated_block_schedule ? "true" : "false",
                    params.max_length - n_input,
                    num_blocks);
        } else {
            LOG_INF("  block schedule: generated-aware = %s, generated tokens = %d, planned blocks = %d, "
                    "base steps = %d, extra-step blocks = %d\n",
                    params.generated_block_schedule ? "true" : "false",
                    params.max_length - n_input,
                    num_blocks,
                    base_steps_per_block,
                    extra_step_blocks);
        }
        LOG_INF("  blocks: started = %d, completed = %d, finished before last step = %d\n",
                blocks_started, blocks_completed, blocks_finished_early);
        LOG_INF("  early commit: enabled = %s, threshold = %.3f, scheduled steps skipped = %d, "
                "scheduled forwards skipped = %d\n",
                early_commit_enabled ? "true" : "false",
                params.early_commit_threshold,
                scheduled_steps_skipped,
                scheduled_forwards_skipped);
        LOG_INF("  prefix KV: enabled = %s, prompt prefills = %d, transition seals = %d, "
                 "last allocated pos = %d, cleared = %s\n",
                 prefix_kv_enabled ? "true" : "false",
                 prefix_kv_enabled ? prompt_prefills : 0,
                 prefix_kv_enabled ? transition_seals : 0,
                 prefix_kv_enabled ? final_cache_pos : -1,
                 prefix_kv_enabled ? "true" : "false");
        LOG_INF("  completed-prefix cache: enabled = %s, owner = %s, prompt prefills = %d, "
                "transition seals = %d, last allocated pos = %d, cleared = %s\n",
                cache_reuse_enabled ? "true" : "false",
                prefix_kv_enabled ? "prefix-kv" : "none",
                prompt_prefills,
                transition_seals,
                cache_reuse_enabled ? final_cache_pos : -1,
                cache_reuse_enabled ? "true" : "false");
        LOG_INF("  MBSD: enabled = %s, trigger = %d, max lookahead = %d, policy = fixed-budget, "
                "execution = %s\n",
                mbsd_enabled ? "true" : "false", params.mbsd_trigger, params.mbsd_lookahead,
                mbsd_fresh_kv_enabled ? "fresh-full-sequence-kv" : "full-sequence-reference");
        if (mbsd_enabled) {
            const double mbsd_main_row_reduction = mbsd_reference_dense_rows > 0 ?
                100.0 * mbsd_main_rows_saved / mbsd_reference_dense_rows : 0.0;
            const double mbsd_net_row_reduction = mbsd_reference_dense_rows > 0 ?
                100.0 * mbsd_net_rows_saved / mbsd_reference_dense_rows : 0.0;
            const int32_t mbsd_physical_main_batches =
                mbsd_fresh_kv_enabled ? mbsd_fresh_main_batches : main_forwards_completed;
            const int32_t mbsd_physical_main_rows_min =
                mbsd_fresh_kv_enabled ? mbsd_fresh_main_rows_min_report :
                    (main_forwards_completed > 0 ? params.max_length : 0);
            const int32_t mbsd_physical_main_rows_max =
                mbsd_fresh_kv_enabled ? mbsd_fresh_main_rows_max :
                    (main_forwards_completed > 0 ? params.max_length : 0);
            LOG_INF("  MBSD fresh KV: requested = %s, execution active = %s, "
                    "physical compact active = false, forwards = %d, cache invariant errors = %llu\n",
                    mbsd_fresh_kv_enabled ? "true" : "false",
                    mbsd_fresh_kv_enabled ? "true" : "false",
                    mbsd_fresh_kv_enabled ? main_forwards_completed : 0,
                    (unsigned long long) mbsd_fresh_cache_invariant_errors);
            LOG_INF("  MBSD physical rows: dense equivalent = %llu, main submitted = %llu, "
                    "cache maintenance = %llu, total submitted = %llu, main saved = %llu, "
                    "main reduction = %.2f%%, net saved = %lld, net reduction = %.2f%%\n",
                    (unsigned long long) mbsd_reference_dense_rows,
                    (unsigned long long) main_input_tokens,
                    (unsigned long long) cache_perf.input_tokens,
                    (unsigned long long) (main_input_tokens + cache_perf.input_tokens),
                    (unsigned long long) mbsd_main_rows_saved,
                    mbsd_main_row_reduction,
                    (long long) mbsd_net_rows_saved,
                    mbsd_net_row_reduction);
            LOG_INF("  MBSD physical batches: main = %d, min rows = %d, max rows = %d, "
                    "prefix rows reused = %llu, fresh KV pre-forward clears = %d, mapping errors = %llu\n",
                    mbsd_physical_main_batches,
                    mbsd_physical_main_rows_min,
                    mbsd_physical_main_rows_max,
                    (unsigned long long) mbsd_fresh_prefix_rows_reused,
                    full_sequence_pre_forward_clears,
                    (unsigned long long) mbsd_physical_mapping_errors);
            LOG_INF("  MBSD backend completion + logits readback: main = %.2f ms, "
                    "cache maintenance = %.2f ms\n",
                    (conditional_perf.completion_time + unconditional_perf.completion_time) / 1000.0,
                    cache_perf.completion_time / 1000.0);
            LOG_INF("  MBSD windows: trigger checks = %d, lookahead expansions = %d, slides = %d, "
                    "slide distance = %d, first = [%d, %d), last = [%d, %d), max end = %d\n",
                    mbsd_trigger_checks, mbsd_lookahead_expansions, mbsd_window_slides,
                    mbsd_window_slide_distance, mbsd_window_first_start, mbsd_window_first_end,
                    mbsd_window_last_start, mbsd_window_last_end, mbsd_window_max_end);
            LOG_INF("  MBSD drafts: introduced = %llu, updates = %llu, prediction rows = %llu, "
                    "re-evaluated = %llu, accepted = %llu, reconfirmed = %llu, replacements = %llu, "
                    "rejected = %llu, pending = %d\n",
                    (unsigned long long) mbsd_drafts_introduced,
                    (unsigned long long) mbsd_draft_updates,
                    (unsigned long long) mbsd_draft_prediction_rows,
                    (unsigned long long) mbsd_drafts_reevaluated,
                    (unsigned long long) mbsd_drafts_accepted,
                    (unsigned long long) mbsd_drafts_reconfirmed,
                    (unsigned long long) mbsd_drafts_replaced,
                    (unsigned long long) mbsd_drafts_rejected,
                    mbsd_drafts_pending);
            LOG_INF("  MBSD logical row roles: current window = %llu, future window = %llu, "
                    "dense context/outside = %llu, dense total = %llu\n",
                    (unsigned long long) mbsd_current_transformer_rows,
                    (unsigned long long) mbsd_future_transformer_rows,
                    (unsigned long long) mbsd_context_transformer_rows,
                    (unsigned long long) (mbsd_current_transformer_rows + mbsd_future_transformer_rows +
                                          mbsd_context_transformer_rows));
            LOG_INF("  MBSD logits: current rows = %llu, future rows = %llu\n",
                    (unsigned long long) mbsd_current_logit_rows,
                    (unsigned long long) mbsd_future_logit_rows);
            LOG_INF("  MBSD schedule: steps with future work = %d, extra steps = 0, extra forwards = 0, "
                    "main steps = %d, scheduled steps saved after draft acceptance = %d, "
                    "scheduled forwards saved = %d, nominal zero-tail slots = %d, "
                    "nominal tail skipped = %d, final forced selections = %llu\n",
                    mbsd_steps_with_future_work, mbsd_main_steps_total, mbsd_scheduled_steps_saved,
                    mbsd_scheduled_forwards_saved, mbsd_nominal_zero_tail_slots,
                    mbsd_nominal_tail_steps_skipped,
                    (unsigned long long) mbsd_final_forced_selections);
            LOG_INF("  MBSD invariants: future semantic commits = %llu, explicit prefix KV disabled = %s, "
                    "block-order violations = %llu, verification input errors = %llu, "
                    "bounds errors = %llu, physical mapping errors = %llu, "
                    "post-EOG corrections = %llu, trajectory hash = %llu\n",
                    (unsigned long long) mbsd_future_semantic_commits,
                    prefix_kv_enabled ? "false" : "true",
                    (unsigned long long) mbsd_block_order_violations,
                    (unsigned long long) mbsd_verification_input_errors,
                    (unsigned long long) mbsd_bounds_errors,
                    (unsigned long long) mbsd_physical_mapping_errors,
                    (unsigned long long) mbsd_post_eog_corrections,
                    (unsigned long long) mbsd_trajectory_hash);
        }
    }
    if (staged_token_stabilization) {
        const char * revision_policy_name =
            params.staged_revision_policy == DIFFUSION_STAGED_REVISION_OLDEST ?
                "oldest" : "balanced-low-confidence";
        LOG_INF("  staged token stabilization: enabled = true, visibility threshold = %.3f, "
                "stability threshold = %.3f, confidence = greedy-non-mask-softmax, "
                "visible revision policy = %s-one-masked, dual path = false\n",
                params.visibility_threshold,
                params.stability_threshold,
                revision_policy_name);
        LOG_INF("  staged states: invisible = %d, visible = %d, stable = %d, "
                "unstable at block completion = %llu, trajectory hash = %llu, revision target hash = %llu\n",
                sts_invisible, sts_visible, sts_stable,
                (unsigned long long) sts_unstable_at_block_completion,
                (unsigned long long) sts_trajectory_hash,
                (unsigned long long) sts_revision_target_hash);
        LOG_INF("  staged transitions: IV-to-V = %llu, IV-to-S = %llu, V-to-S = %llu, forced V = %llu\n",
                (unsigned long long) sts_visibility_promotions,
                (unsigned long long) sts_direct_stable_promotions,
                (unsigned long long) sts_visible_to_stable,
                (unsigned long long) sts_forced_visible);
        LOG_INF("  staged revision: passes = %d, candidates = %llu, token changes = %llu, "
                "max visible = %d, stable decisions skipped = %llu, scheduled steps skipped = %d\n",
                sts_revision_passes,
                (unsigned long long) sts_revision_candidates,
                (unsigned long long) sts_token_revisions,
                sts_max_visible,
                (unsigned long long) sts_stable_logits_skipped,
                sts_scheduled_steps_skipped);
        LOG_INF("  staged revision phases: ordinary = %d, final = %d\n",
                sts_ordinary_revision_passes,
                sts_final_revision_passes);
        LOG_INF("  staged final revision: enabled = %s, triggered = %s, configured passes = %d, "
                "available schedule slots = %d, budget = %d, used = %d, sweeps started/completed = %d/%d, "
                "visible before = %d, visible after = %d, observed ratio = %.3f, trigger ratio = %.3f, "
                "V-to-S = %llu, "
                "token changes = %llu, stop = %s\n",
                params.staged_final_revision_steps > 0 ? "true" : "false",
                sts_final_revision_triggered ? "true" : "false",
                params.staged_final_revision_steps,
                sts_final_available_steps,
                sts_final_revision_budget,
                sts_final_revision_passes,
                sts_final_revision_sweeps_started,
                sts_final_revision_sweeps_completed,
                sts_final_visible_before,
                sts_final_visible_after,
                sts_final_visible_ratio_observed,
                params.staged_final_visible_ratio,
                (unsigned long long) sts_final_visible_to_stable,
                (unsigned long long) sts_final_token_revisions,
                sts_final_stop_reason);
        LOG_INF("  staged final visible diagnostics: never revised = %d, revision count min/mean/max = "
                "%d/%.2f/%d, latest confidence min/mean/max = %.6f/%.6f/%.6f\n",
                sts_never_revised_visible,
                sts_visible_revision_min,
                sts_visible > 0 ? (double) sts_visible_revision_sum / sts_visible : 0.0,
                sts_visible_revision_max,
                sts_visible_confidence_min,
                sts_visible > 0 ? sts_visible_confidence_sum / sts_visible : 0.0,
                sts_visible_confidence_max);
    }
    LOG_INF("  diffusion KV: mode = %s, full-sequence pre-forward clears = %d, "
            "pre-forward clear time = %.2f ms, "
            "last allocated pos = %d, cleared = %s\n",
            mbsd_fresh_kv_enabled ? "mbsd-fresh-full-sequence" :
                (prefix_kv_enabled ? "prefix" :
                    (full_sequence_kv_oracle ? "full-sequence-oracle" : "none")),
            full_sequence_pre_forward_clears,
            total_cache_clear_time / 1000.0,
            final_cache_pos,
            diffusion_kv_graph_enabled ? "true" : "false");
    LOG_INF("  callback time: %.2f ms\n", total_callback_time / 1000.0);
    LOG_INF("  batch setup time: %.2f ms\n", total_batch_time / 1000.0);
    LOG_INF("  decode call time: %.2f ms, %.2f ms per forward\n",
            total_decode_time / 1000.0, total_decode_time / 1000.0 / forward_divisor);
    LOG_INF("  completion + logits wait: %.2f ms, %.2f ms per forward\n",
            total_completion_time / 1000.0, total_completion_time / 1000.0 / forward_divisor);
    LOG_INF("  conditional/main forward: decode = %.2f ms, completion + logits wait = %.2f ms\n",
            conditional_perf.decode_time / 1000.0, conditional_perf.completion_time / 1000.0);
    LOG_INF("  unconditional forward: decode = %.2f ms, completion + logits wait = %.2f ms\n",
            unconditional_perf.decode_time / 1000.0, unconditional_perf.completion_time / 1000.0);
    if (staged_token_stabilization) {
        LOG_INF("  revision forward: decode = %.2f ms, completion + logits wait = %.2f ms\n",
                revision_perf.decode_time / 1000.0, revision_perf.completion_time / 1000.0);
    }
    LOG_INF("  cache maintenance forward: decode = %.2f ms, completion + logits wait = %.2f ms\n",
            cache_perf.decode_time / 1000.0, cache_perf.completion_time / 1000.0);
    LOG_INF("  CFG CPU time: %.2f ms\n", total_cfg_cpu_time / 1000.0);
    LOG_INF("  sampling/update time: %.2f ms, %.2f ms per sampling pass, %.2f ms per completed iteration\n",
            total_sampling_time / 1000.0,
            total_sampling_time / 1000.0 / sampling_divisor,
            total_sampling_time / 1000.0 / step_divisor);
    LOG_INF("  other loop time: %.2f ms\n", other_time / 1000.0);
    LOG_INF("  graph reuses: %d\n", graph_reuses);
    LOG_INF("  throughput: %.2f completed iterations/s, %.2f forwards/s\n",
            iterations_completed * 1000000.0 / std::max<int64_t>(1, total_time),
            total_forwards * 1000000.0 / std::max<int64_t>(1, total_time));

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_sampler_free(mbsd_sampler);
    llama_sampler_free(dist_sampler);

    n_generated = ((early_commit_enabled && output_masks_remaining > 0) ||
                   ((diffusion_kv_graph_enabled || staged_token_stabilization || mbsd_enabled) &&
                    (generation_failed || output_masks_remaining > 0))) ?
                  0 : params.max_length;
}
