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

    const bool early_commit_enabled      = params.early_commit_threshold >= 0.0f;
    const bool prefix_kv_enabled          = params.prefix_kv;
    const bool full_sequence_kv_oracle    = params.full_sequence_kv_oracle;
    const bool diffusion_kv_graph_enabled = prefix_kv_enabled || full_sequence_kv_oracle;
    if (params.steps <= 0 || !std::isfinite(params.early_commit_threshold) ||
        params.early_commit_threshold > 1.0f) {
        LOG_ERR("%s: invalid diffusion parameters\n", __func__);
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

    if (prefix_kv_enabled && full_sequence_kv_oracle) {
        LOG_ERR("%s: prefix KV and the full-sequence KV oracle are mutually exclusive\n", __func__);
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

    const llama_model * model = llama_get_model(ctx);
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

    std::mt19937 rng(params.seed);

    llama_set_causal_attn(ctx, false);

    int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    if (params.mask_token_id < 0 || params.mask_token_id >= n_vocab) {
        LOG_ERR("%s: invalid mask token id %d\n", __func__, params.mask_token_id);
        return;
    }

    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<llama_token_data> conf_candidates;
    conf_candidates.reserve(params.max_length);
    std::vector<int32_t> mask_positions;
    mask_positions.reserve(params.max_length);
    std::vector<int32_t> logits_row_by_pos(params.max_length, -1);

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
                llama_sampler_free(dist_sampler);
                return;
            }
            num_blocks = params.max_length / params.block_length;
            if (params.steps % num_blocks != 0) {
                LOG_ERR("%s: diffusion steps must be divisible by the number of blocks\n", __func__);
                llama_batch_free(batch);
                llama_sampler_free(sampler);
                llama_sampler_free(dist_sampler);
                return;
            }
            base_steps_per_block = params.steps / num_blocks;
        }
    }

    std::vector<float> confidence(params.max_length);

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
    forward_perf cache_perf;

    int32_t iterations_started      = 0;
    int32_t iterations_with_forward = 0;
    int32_t iterations_completed    = 0;
    int32_t sampling_passes         = 0;

    int32_t blocks_started             = 0;
    int32_t blocks_completed           = 0;
    int32_t blocks_finished_early      = 0;
    int32_t scheduled_steps_skipped    = 0;
    int32_t scheduled_forwards_skipped = 0;

    uint64_t base_token_selections      = 0;
    uint64_t threshold_extra_selections = 0;
    uint64_t forced_final_selections    = 0;
    uint64_t tokens_committed           = 0;

    int32_t prompt_prefills           = 0;
    int32_t transition_seals          = 0;
    int32_t oracle_pre_forward_clears = 0;
    bool    generation_failed          = false;

    int64_t total_callback_time    = 0;
    int64_t total_batch_time       = 0;
    int64_t total_cache_clear_time = 0;
    int64_t total_cfg_cpu_time     = 0;
    int64_t total_sampling_time    = 0;
    int64_t total_time             = 0;

    if (diffusion_kv_graph_enabled) {
        llama_synchronize(ctx);
        llama_memory_clear(memory, false);
    }

    const int32_t graph_reuses_start = llama_perf_context(ctx).n_reused;
    const int64_t time_start         = ggml_time_us();

    auto run_forward = [&](forward_perf & perf, int32_t active_masks, int32_t output_rows) -> std::pair<int, float *> {
        perf.calls++;

        if (full_sequence_kv_oracle) {
            const int64_t clear_start = ggml_time_us();
            llama_synchronize(ctx);
            llama_memory_clear(memory, false);
            total_cache_clear_time += ggml_time_us() - clear_start;
            oracle_pre_forward_clears++;
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

    if (prefix_kv_enabled) {
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
        const int32_t steps_this_block =
            base_steps_per_block + (block_num < extra_step_blocks ? 1 : 0);
        const int32_t block_step_offset =
            block_num * base_steps_per_block + std::min(block_num, extra_step_blocks);

        GGML_ASSERT(steps_this_block > 0);
        GGML_ASSERT(block_step_offset + steps_this_block <= params.steps);
        if (block_num == num_blocks - 1) {
            GGML_ASSERT(block_step_offset + steps_this_block == params.steps);
        }

        int32_t block_start = 0;
        int32_t block_end   = params.max_length;
        if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
            const int64_t scheduled_block_start =
                (int64_t) n_input + (int64_t) block_num * params.block_length;
            block_start = (int32_t) std::min<int64_t>(scheduled_block_start, params.max_length);
            block_end   = (int32_t) std::min<int64_t>(
                scheduled_block_start + params.block_length, params.max_length);
        }

        if (params.generated_block_schedule) {
            GGML_ASSERT(block_start < block_end);
            if (block_num == num_blocks - 1) {
                GGML_ASSERT(block_end == params.max_length);
            }
        }

        // Count masked tokens in current block for block-based processing
        if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
            int32_t block_mask_count = 0;
            for (int i = block_start; i < block_end; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    block_mask_count++;
                }
            }
            num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps_this_block);
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
                    if (diffusion_kv_graph_enabled) {
                        generation_failed = true;
                    }
                    break;
                }
                total_callback_time += ggml_time_us() - callback_start;
            }

            // Setup batch
            const int64_t batch_start = ggml_time_us();

            mask_positions.clear();
            for (int32_t i = 0; i < params.max_length; i++) {
                if (output_tokens[i] == params.mask_token_id &&
                    (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || (i >= block_start && i < block_end))) {
                    mask_positions.push_back(i);
                }
            }

            if (mask_positions.empty()) {
                total_batch_time += ggml_time_us() - batch_start;
                break;
            }

            const int32_t active_masks = (int32_t) mask_positions.size();
            const int32_t batch_abs_start = prefix_kv_enabled ?
                (params.shift_logits ? block_start - 1 : block_start) : 0;
            const int32_t batch_abs_end = prefix_kv_enabled ? block_end : params.max_length;

            if (prefix_kv_enabled && !llama_memory_seq_rm(memory, 0, batch_abs_start, -1)) {
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

            for (int32_t pos : mask_positions) {
                const int32_t source_pos = params.shift_logits ? std::max(pos - 1, 0) : pos;
                GGML_ASSERT(source_pos >= batch_abs_start && source_pos < batch_abs_end);
                batch.logits[source_pos - batch_abs_start] = 1;
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
                    if (diffusion_kv_graph_enabled) {
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
                    if (diffusion_kv_graph_enabled) {
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
                    if (diffusion_kv_graph_enabled) {
                        generation_failed = true;
                    }
                    break;
                }
                logits = result;
            }

            if (!logits) {
                LOG_ERR("%s: failed to get logits at step %d\n", __func__, global_step);
                if (diffusion_kv_graph_enabled) {
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

            if (params.add_gumbel_noise && params.temperature > 0.0f) {
                add_gumbel_noise(logits, n_vocab, params.temperature, rng);
            }

            if (params.algorithm == DIFFUSION_ALGORITHM_ORIGIN) {
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

                    if (early_commit_enabled && step == steps_this_block - 1) {
                        candidates[params.mask_token_id].logit = -std::numeric_limits<float>::infinity();
                    }

                    llama_token_data_array cur_p = {
                        candidates.data(),
                        candidates.size(),
                        -1,
                        false,
                    };

                    llama_sampler_apply(sampler, &cur_p);
                    llama_token sampled_token = cur_p.data[cur_p.selected].id;

                    float conf = calculate_confidence(cur_p, params.algorithm, rng);

                    sampled_tokens[i] = sampled_token;
                    confidences.emplace_back(conf, i);
                }

                int32_t transfer_count = calculate_transfer_count(
                    step, steps_this_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);

                const int32_t scheduled_selection_count =
                    std::min(std::max(transfer_count, 0), (int32_t) confidences.size());

                if (early_commit_enabled && step == steps_this_block - 1) {
                    transfer_count = (int32_t) mask_positions.size();
                    forced_selections_this_step = (int32_t) mask_positions.size() - scheduled_selection_count;
                }

                const int32_t base_selection_count =
                    std::min(std::max(transfer_count, 0), (int32_t) confidences.size());

                if (early_commit_enabled) {
                    std::sort(confidences.begin(), confidences.end(),
                              [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                                  if (a.first != b.first) {
                                      return a.first > b.first;
                                  }
                                  return a.second < b.second;
                              });

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
                    base_selections_this_step = base_selection_count;
                    if (params.alg_temp == 0.0f) {
                        std::partial_sort(confidences.begin(),
                                          confidences.begin() + base_selection_count,
                                          confidences.end(),
                                          [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                                              if (a.first != b.first) {
                                                  return a.first > b.first;
                                              }
                                              return a.second < b.second;
                                          });

                        for (int32_t i = 0; i < base_selection_count; i++) {
                            int32_t mask_idx   = confidences[i].second;
                            int32_t pos        = mask_positions[mask_idx];
                            output_tokens[pos] = sampled_tokens[mask_idx];
                        }
                    } else {
                        conf_candidates.clear();
                        for (size_t i = 0; i < confidences.size(); i++) {
                            float conf_logit = confidences[i].first / params.alg_temp;
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
                            int32_t selected_idx = conf_array.selected;
                            int32_t mask_idx     = selected_idx;
                            int32_t pos          = mask_positions[mask_idx];
                            output_tokens[pos]   = sampled_tokens[mask_idx];

                            conf_candidates[selected_idx].p = 0.0f;
                            conf_array.selected             = -1;
                        }
                    }
                }
            }

            base_token_selections += base_selections_this_step;
            threshold_extra_selections += threshold_selections_this_step;
            forced_final_selections += forced_selections_this_step;

            int32_t remaining_masks = 0;
            for (int32_t pos : mask_positions) {
                if (output_tokens[pos] == params.mask_token_id) {
                    remaining_masks++;
                }
            }
            tokens_committed += active_masks - remaining_masks;

            bool finish_block = false;
            if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED && remaining_masks == 0) {
                blocks_completed++;
                if (early_commit_enabled && step + 1 < steps_this_block) {
                    const int32_t skipped_steps = steps_this_block - step - 1;
                    blocks_finished_early++;
                    scheduled_steps_skipped += skipped_steps;
                    scheduled_forwards_skipped += skipped_steps * (params.cfg_scale > 0.0f ? 2 : 1);
                }
                finish_block = early_commit_enabled;
            }

            const int64_t time_end_sampling = ggml_time_us();
            total_sampling_time += time_end_sampling - time_start_sampling;
            iterations_completed++;

            if (finish_block) {
                break;
            }
        }

        if (prefix_kv_enabled && !generation_failed) {
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
        conditional_perf.calls + unconditional_perf.calls + cache_perf.calls;
    const int32_t total_forwards =
        conditional_perf.completed + unconditional_perf.completed + cache_perf.completed;
    const int64_t total_decode_time =
        conditional_perf.decode_time + unconditional_perf.decode_time + cache_perf.decode_time;
    const int64_t total_completion_time =
        conditional_perf.completion_time + unconditional_perf.completion_time + cache_perf.completion_time;

    const uint64_t total_active_masks = conditional_perf.active_masks + unconditional_perf.active_masks;
    const uint64_t total_output_rows  = conditional_perf.output_rows + unconditional_perf.output_rows;
    const uint64_t total_logits_bytes = conditional_perf.logits_bytes + unconditional_perf.logits_bytes;
    const uint64_t main_input_tokens  = conditional_perf.input_tokens + unconditional_perf.input_tokens;
    const uint64_t total_input_tokens = main_input_tokens + cache_perf.input_tokens;

    if (full_sequence_kv_oracle) {
        const int32_t main_forward_calls = conditional_perf.calls + unconditional_perf.calls;
        const int32_t main_forwards      = conditional_perf.completed + unconditional_perf.completed;
        const uint64_t expected_rows     = (uint64_t) main_forwards * params.max_length;
        if (cache_perf.calls != 0 || prompt_prefills != 0 || transition_seals != 0 ||
            oracle_pre_forward_clears != main_forward_calls || main_input_tokens != expected_rows) {
            LOG_ERR("%s: full-sequence KV oracle invariant failed "
                    "(clears = %d/%d, rows = %llu/%llu, cache forwards = %d, prefills = %d, seals = %d)\n",
                    __func__,
                    oracle_pre_forward_clears,
                    main_forward_calls,
                    (unsigned long long) main_input_tokens,
                    (unsigned long long) expected_rows,
                    cache_perf.calls,
                    prompt_prefills,
                    transition_seals);
            generation_failed = true;
        }
    }

    int32_t output_masks_remaining = 0;
    for (int32_t i = n_input; i < params.max_length; i++) {
        if (output_tokens[i] == params.mask_token_id) {
            output_masks_remaining++;
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
    LOG_INF("  forwards: calls = %d, completed = %d, conditional/main = %d, unconditional = %d, "
            "cache maintenance = %d\n",
            total_forward_calls, total_forwards, conditional_perf.completed, unconditional_perf.completed,
            cache_perf.completed);
    LOG_INF("  active masks: total = %llu, average per forward = %.2f\n",
            (unsigned long long) total_active_masks, total_active_masks / main_forward_divisor);
    LOG_INF("  transformer rows: main = %llu, cache maintenance = %llu, total = %llu\n",
            (unsigned long long) main_input_tokens,
            (unsigned long long) cache_perf.input_tokens,
            (unsigned long long) total_input_tokens);
    LOG_INF("  logits: rows = %llu, bytes = %llu (%.2f MiB)\n",
            (unsigned long long) total_output_rows,
            (unsigned long long) total_logits_bytes,
            total_logits_bytes / (1024.0 * 1024.0));
    LOG_INF("  cache maintenance output: rows = %llu, logits bytes = %llu (%.2f MiB)\n",
            (unsigned long long) cache_perf.output_rows,
            (unsigned long long) cache_perf.logits_bytes,
            cache_perf.logits_bytes / (1024.0 * 1024.0));
    LOG_INF("  token commits: committed = %llu, remaining masks = %d\n",
            (unsigned long long) tokens_committed,
            output_masks_remaining);
    LOG_INF("  token selections: base = %llu, threshold extra = %llu, forced final = %llu\n",
            (unsigned long long) base_token_selections,
            (unsigned long long) threshold_extra_selections,
            (unsigned long long) forced_final_selections);
    if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        LOG_INF("  block schedule: generated-aware = %s, generated tokens = %d, planned blocks = %d, "
                "base steps = %d, extra-step blocks = %d\n",
                params.generated_block_schedule ? "true" : "false",
                params.max_length - n_input,
                num_blocks,
                base_steps_per_block,
                extra_step_blocks);
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
                prompt_prefills,
                transition_seals,
                final_cache_pos,
                prefix_kv_enabled ? "true" : "false");
    }
    LOG_INF("  diffusion KV: mode = %s, oracle pre-forward clears = %d, pre-forward clear time = %.2f ms, "
            "last allocated pos = %d, cleared = %s\n",
            prefix_kv_enabled ? "prefix" : (full_sequence_kv_oracle ? "full-sequence-oracle" : "none"),
            oracle_pre_forward_clears,
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
    llama_sampler_free(dist_sampler);

    n_generated = ((early_commit_enabled && output_masks_remaining > 0) ||
                   (diffusion_kv_graph_enabled && (generation_failed || output_masks_remaining > 0))) ?
                  0 : params.max_length;
}
