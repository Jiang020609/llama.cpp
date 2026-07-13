#include "arg.h"
#include "chat.h"
#include "common.h"
#include "diffusion.h"
#include "llama.h"
#include "log.h"

#include <limits.h>

#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct callback_data {
    diffusion_params *  diff_params;
    const llama_vocab * vocab;
    int32_t             n_input;
};

static bool diffusion_step_callback(int32_t             step,
                                    int32_t             total_steps,
                                    const llama_token * tokens,
                                    int32_t             n_tokens,
                                    void *              user_data) {
    (void) user_data;

    callback_data * data = static_cast<callback_data *>(user_data);

    auto print_progress_bar = [](int32_t step, int32_t total_steps) {
        int progress_percent = (step * 100) / total_steps;
        int progress_bars    = (step * 50) / total_steps;
        LOG_INF("\rdiffusion step: %d/%d [%s%s] %d%%",
                step,
                total_steps,
                std::string(progress_bars, '=').c_str(),
                std::string(50 - progress_bars, ' ').c_str(),
                progress_percent);
    };

    if (data->diff_params->visual_mode) {
        // Visual mode: clear
        LOG_INF("\033[2J\033[H");  // Clear screen and move cursor to top-left

        print_progress_bar(step, total_steps);

        LOG_INF("\n");

        std::string current_text = " ";

        for (int32_t i = data->n_input; i < n_tokens; i++) {
            std::string token_str;
            if (tokens[i] != llama_vocab_mask(data->vocab)) {
                char piece[256];
                int  n_chars = llama_token_to_piece(data->vocab, tokens[i], piece, sizeof(piece), 0, false);
                if (n_chars > 0) {
                    piece[n_chars] = '\0';
                    token_str      = piece;
                }
            } else {
                token_str = " ";
            }

            current_text += token_str;
        }

        LOG_INF("%s\n", current_text.c_str());
    } else {
        print_progress_bar(step, total_steps);
    }

    return true;
}

static std::string format_input_text(const std::string & prompt, const std::string & system_prompt, bool use_chat_template, llama_model * model) {
    if (!use_chat_template) {
        return prompt;
    }

    auto chat_templates = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;
    common_chat_msg system_msg;

    if (!system_prompt.empty()) {
        system_msg.role = "system";
        system_msg.content = system_prompt;
        inputs.messages.push_back(system_msg);
    }

    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = prompt;

    inputs.messages.push_back(user_msg);
    inputs.add_generation_prompt = true;

    auto result = common_chat_templates_apply(chat_templates.get(), inputs);

    return result.prompt;
}

static void dump_generated_tokens(const llama_vocab *              vocab,
                                  const std::vector<llama_token> & tokens,
                                  int32_t                          n_input) {
    if (!vocab || n_input < 0 || (size_t) n_input > tokens.size()) {
        return;
    }

    const int32_t     n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token eos     = llama_vocab_eos(vocab);
    const llama_token pad     = llama_vocab_pad(vocab);
    const llama_token mask    = llama_vocab_mask(vocab);

    int32_t  eog_count     = 0;
    int32_t  control_count = 0;
    int32_t  pad_count     = 0;
    int32_t  mask_count    = 0;
    int32_t  invalid_count = 0;
    int32_t  first_eog     = -1;
    uint64_t token_hash    = 14695981039346656037ULL;
    std::string token_ids;

    for (size_t pos = n_input; pos < tokens.size(); ++pos) {
        const llama_token token = tokens[pos];
        if (!token_ids.empty()) {
            token_ids += ", ";
        }
        token_ids += std::to_string(token);

        token_hash ^= (uint32_t) token;
        token_hash *= 1099511628211ULL;

        if (token < 0 || token >= n_vocab) {
            invalid_count++;
            continue;
        }
        if (llama_vocab_is_eog(vocab, token)) {
            if (first_eog < 0) {
                first_eog = (int32_t) pos - n_input;
            }
            eog_count++;
        }
        control_count += llama_vocab_is_control(vocab, token);
        pad_count     += token == pad;
        mask_count    += token == mask;
    }

    LOG_INF("diffusion generated tokens: count = %d, id hash = %llu, eog = %d, control = %d, "
            "pad = %d, mask = %d, invalid = %d, first eog = %d, eos id = %d, pad id = %d, mask id = %d\n",
            (int32_t) tokens.size() - n_input,
            (unsigned long long) token_hash,
            eog_count,
            control_count,
            pad_count,
            mask_count,
            invalid_count,
            first_eog,
            eos,
            pad,
            mask);
    LOG_INF("diffusion generated token ids: [%s]\n", token_ids.c_str());
}

static bool validate_diffusion_params(const common_params & params) {
    if (params.diffusion.steps <= 0) {
        LOG_ERR("error: --diffusion-steps must be greater than zero\n");
        return false;
    }

    if (!std::isfinite(params.diffusion.eps) || params.diffusion.eps < 0.0f) {
        LOG_ERR("error: --diffusion-eps must be finite and non-negative\n");
        return false;
    }

    if (params.diffusion.block_length < 0) {
        LOG_ERR("error: --diffusion-block-length must be non-negative\n");
        return false;
    }

    const bool has_timestep_schedule = params.diffusion.eps > 0.0f;
    const bool has_block_schedule    = params.diffusion.block_length > 0;
    if (has_timestep_schedule == has_block_schedule) {
        LOG_ERR("error: specify exactly one of --diffusion-eps or --diffusion-block-length\n");
        return false;
    }

    if (params.diffusion.generated_block_schedule && !has_block_schedule) {
        LOG_ERR("error: --diffusion-generated-block-schedule requires --diffusion-block-length\n");
        return false;
    }

    if (params.diffusion.algorithm < DIFFUSION_ALGORITHM_ORIGIN ||
        params.diffusion.algorithm > DIFFUSION_ALGORITHM_CONFIDENCE_BASED) {
        LOG_ERR("error: --diffusion-algorithm must be between 0 and 4\n");
        return false;
    }

    if (!std::isfinite(params.diffusion.alg_temp) || params.diffusion.alg_temp < 0.0f) {
        LOG_ERR("error: --diffusion-alg-temp must be finite and non-negative\n");
        return false;
    }

    const float early_commit_threshold = params.diffusion.early_commit_threshold;
    if (!std::isfinite(early_commit_threshold) || early_commit_threshold > 1.0f) {
        LOG_ERR("error: --diffusion-early-commit-threshold must be finite and at most 1\n");
        return false;
    }

    if (early_commit_threshold >= 0.0f) {
        if (!has_block_schedule) {
            LOG_ERR("error: early commit requires --diffusion-block-length\n");
            return false;
        }
        if (params.diffusion.algorithm != DIFFUSION_ALGORITHM_CONFIDENCE_BASED) {
            LOG_ERR("error: early commit requires --diffusion-algorithm 4\n");
            return false;
        }
        if (params.diffusion.alg_temp != 0.0f) {
            LOG_ERR("error: early commit requires --diffusion-alg-temp 0\n");
            return false;
        }
    }

    if (params.diffusion.prefix_kv && params.diffusion.full_sequence_kv_oracle) {
        LOG_ERR("error: --diffusion-prefix-kv and --diffusion-full-sequence-kv-oracle are mutually exclusive\n");
        return false;
    }

    if (params.diffusion.prefix_kv) {
        if (!has_block_schedule || !params.diffusion.generated_block_schedule) {
            LOG_ERR("error: --diffusion-prefix-kv requires block scheduling and --diffusion-generated-block-schedule\n");
            return false;
        }
        if (params.diffusion.algorithm != DIFFUSION_ALGORITHM_CONFIDENCE_BASED ||
            params.diffusion.alg_temp != 0.0f) {
            LOG_ERR("error: --diffusion-prefix-kv requires --diffusion-algorithm 4 and --diffusion-alg-temp 0\n");
            return false;
        }
        if (early_commit_threshold >= 0.0f || params.diffusion.cfg_scale != 0.0f ||
            params.diffusion.add_gumbel_noise) {
            LOG_ERR("error: --diffusion-prefix-kv does not yet support early commit, CFG, or Gumbel noise\n");
            return false;
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }

    if (!validate_diffusion_params(params)) {
        return 1;
    }

    const bool use_diffusion_kv = params.diffusion.prefix_kv || params.diffusion.full_sequence_kv_oracle;

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = params.n_gpu_layers;
    model_params.devices            = params.devices.data();
    model_params.use_mmap           = params.use_mmap;
    model_params.use_direct_io      = params.use_direct_io;
    model_params.use_mlock          = params.use_mlock;
    model_params.check_tensors      = params.check_tensors;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    if (!llama_model_is_diffusion(model)) {
        LOG_ERR("error: unsupported model for diffusion");
        llama_model_free(model);
        return 1;
    }

    if (use_diffusion_kv) {
        char architecture[32] = {};
        if (llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture)) < 0 ||
            strcmp(architecture, "dream") != 0) {
            LOG_ERR("error: diffusion KV modes currently support Dream models only\n");
            llama_model_free(model);
            return 1;
        }
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = params.n_ctx;
    ctx_params.n_batch              = params.n_batch;
    ctx_params.n_ubatch             = params.n_ubatch;
    ctx_params.flash_attn_type      = params.flash_attn_type;
    ctx_params.no_perf              = params.no_perf;
    ctx_params.type_k               = params.cache_type_k;
    ctx_params.type_v               = params.cache_type_v;
    ctx_params.offload_kqv          = !params.no_kv_offload;
    if (use_diffusion_kv) {
        ctx_params.ctx_type = LLAMA_CONTEXT_TYPE_DIFFUSION_KV;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);

    const llama_vocab * vocab            = llama_model_get_vocab(model);

    std::string         formatted_prompt = format_input_text(params.prompt, params.system_prompt, params.enable_chat_template, model);

    std::vector<llama_token> input_tokens = common_tokenize(vocab,
                                                            formatted_prompt,
                                                            /*add special tokens*/ true,
                                                            /*parse special*/ true);

    int n_input = input_tokens.size();

    if (static_cast<uint32_t>(n_input) >= llama_n_ctx(ctx)) {
        LOG_ERR("error: input too long (%d tokens), max context is %d\n", n_input, llama_n_ctx(ctx));
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (n_input >= params.n_ubatch) {
        LOG_ERR("error: input too long (%d tokens), max diffusion length is %d\n", n_input, params.n_ubatch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (params.diffusion.block_length > 0) {
        const int32_t block_length = params.diffusion.block_length;
        if (params.diffusion.generated_block_schedule) {
            const int32_t generated_tokens = params.n_ubatch - n_input;
            const int32_t generated_blocks = 1 + (generated_tokens - 1) / block_length;
            if (params.diffusion.steps < generated_blocks) {
                LOG_ERR("error: diffusion steps (%d) must be at least generated blocks (%d)\n",
                        params.diffusion.steps, generated_blocks);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
        } else {
            if (params.n_ubatch % block_length != 0) {
                LOG_ERR("error: diffusion max length (%d) must be divisible by block length (%d)\n",
                        params.n_ubatch, block_length);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }

            const int32_t scheduled_blocks = params.n_ubatch / block_length;
            if (params.diffusion.steps % scheduled_blocks != 0) {
                LOG_ERR("error: diffusion steps (%d) must be divisible by scheduled blocks (%d)\n",
                        params.diffusion.steps, scheduled_blocks);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }

            const int32_t generated_blocks =
                (params.n_ubatch - n_input + block_length - 1) / block_length;
            if (generated_blocks != scheduled_blocks) {
                LOG_WRN("block scheduler plans %d blocks for %d generated tokens; %d trailing block(s) will be empty\n",
                        scheduled_blocks, params.n_ubatch - n_input, scheduled_blocks - generated_blocks);
                if (params.diffusion.early_commit_threshold >= 0.0f) {
                    LOG_ERR("error: early commit requires the tokenized prompt to be shorter than block length\n");
                    llama_free(ctx);
                    llama_model_free(model);
                    return 1;
                }
            }
        }
    }

    llama_token mask_token_id = llama_vocab_mask(vocab);

    GGML_ASSERT(mask_token_id != LLAMA_TOKEN_NULL);

    bool visual_mode = params.diffusion.visual_mode;

    int32_t                  n_generated = 0;
    std::vector<llama_token> output_tokens(params.n_ubatch);

    struct diffusion_params diff_params;

    char shift_logits_str[8];
    if (llama_model_meta_val_str(model, "diffusion.shift_logits", shift_logits_str, sizeof(shift_logits_str)) >= 0) {
        diff_params.shift_logits = (strcmp(shift_logits_str, "true") == 0);
    } else {
        diff_params.shift_logits = true;
    }

    if (params.diffusion.eps) {
        diff_params.schedule = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;
        diff_params.eps      = params.diffusion.eps;
    } else if (params.diffusion.block_length) {
        diff_params.schedule     = DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED;
        diff_params.block_length = params.diffusion.block_length;
    }

    diff_params.mask_token_id    = mask_token_id;
    diff_params.seed             = params.sampling.seed;
    diff_params.temperature      = params.sampling.temp;
    diff_params.steps            = params.diffusion.steps;
    diff_params.algorithm        = static_cast<diffusion_algorithm>(params.diffusion.algorithm);
    diff_params.max_length       = params.n_ubatch;
    diff_params.top_p            = params.sampling.top_p;
    diff_params.top_k            = params.sampling.top_k;
    diff_params.visual_mode      = params.diffusion.visual_mode;
    diff_params.alg_temp         = params.diffusion.alg_temp;
    diff_params.generated_block_schedule = params.diffusion.generated_block_schedule;
    diff_params.early_commit_threshold = params.diffusion.early_commit_threshold;
    diff_params.prefix_kv         = params.diffusion.prefix_kv;
    diff_params.full_sequence_kv_oracle = params.diffusion.full_sequence_kv_oracle;
    diff_params.cfg_scale        = params.diffusion.cfg_scale;
    diff_params.add_gumbel_noise = params.diffusion.add_gumbel_noise;

    diff_params.step_callback           = diffusion_step_callback;
    callback_data cb_data               = { &diff_params, vocab, n_input };
    diff_params.step_callback_user_data = &cb_data;

    const char * alg_names[]   = {
        "DIFFUSION_ALGORITHM_ORIGIN",
        "DIFFUSION_ALGORITHM_ENTROPY_BASED",
        "DIFFUSION_ALGORITHM_MARGIN_BASED",
        "DIFFUSION_ALGORITHM_RANDOM",
        "DIFFUSION_ALGORITHM_CONFIDENCE_BASED",
    };
    const char * sched_names[] = {
        "DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED",
        "DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED",
    };
    const char * alg_name =
        (diff_params.algorithm >= 0 && diff_params.algorithm <= 4) ? alg_names[diff_params.algorithm] : "UNKNOWN";
    const char * sched_name =
        (diff_params.schedule >= 0 && diff_params.schedule <= 1) ? sched_names[diff_params.schedule] : "UNKNOWN";

    LOG_INF("diffusion_params: - %-25s llama_token      = %d\n", "mask_token_id", mask_token_id);
    LOG_INF("diffusion_params: - %-25s u32              = %d\n", "steps", diff_params.steps);
    LOG_INF("diffusion_params: - %-25s u32              = %d\n", "max_length", diff_params.max_length);
    LOG_INF("diffusion_params: - %-25s enum             = %d (%s)\n", "algorithm", diff_params.algorithm, alg_name);
    LOG_INF("diffusion_params: - %-25s enum             = %d (%s)\n", "schedule", diff_params.schedule, sched_name);
    LOG_INF("diffusion_params: - %-25s f32              = %.3f\n", "temperature", diff_params.temperature);
    LOG_INF("diffusion_params: - %-25s bool             = %s\n",
            "shift_logits", diff_params.shift_logits ? "true" : "false");
    LOG_INF("diffusion_params: - %-25s bool             = %s\n",
            "full_sequence_kv_oracle", diff_params.full_sequence_kv_oracle ? "true" : "false");
    if (diff_params.schedule == DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED) {
        LOG_INF("diffusion_params: - %-25s f32              = %.6f\n", "eps", diff_params.eps);
        LOG_INF("diffusion_params: - %-25s f32              = %.3f\n", "alg_temp", diff_params.alg_temp);
    }
    if (diff_params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        LOG_INF("diffusion_params: - %-25s u32              = %d\n", "block_length", diff_params.block_length);
        LOG_INF("diffusion_params: - %-25s bool             = %s\n",
                "generated_block_schedule", diff_params.generated_block_schedule ? "true" : "false");
        LOG_INF("diffusion_params: - %-25s bool             = %s\n",
                "prefix_kv", diff_params.prefix_kv ? "true" : "false");
        LOG_INF("diffusion_params: - %-25s f32              = %.3f\n", "cfg_scale", diff_params.cfg_scale);
        LOG_INF("diffusion_params: - %-25s f32              = %.3f\n",
                "early_commit_threshold", diff_params.early_commit_threshold);
    }

    diffusion_generate(ctx, input_tokens.data(), output_tokens.data(), n_input, diff_params, n_generated);

    if (params.diffusion.dump_generated_tokens) {
        if (n_generated > 0) {
            dump_generated_tokens(vocab, output_tokens, n_input);
        } else {
            LOG_WRN("diffusion generated token diagnostics unavailable because generation failed\n");
        }
    }

    int result = 0;
    if (n_generated > 0) {
        if (visual_mode) {
            //clear screen and move cursor to top-left
            LOG_INF("\033[2J\033[H");
        }

        output_tokens.erase(output_tokens.begin(), output_tokens.begin() + n_input);
        std::string output_data = common_detokenize(vocab, output_tokens, false);
        LOG_INF("\n%s\n", output_data.c_str());
    } else {
        LOG_ERR("error: diffusion generation failed\n");
        result = 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return result;
}
