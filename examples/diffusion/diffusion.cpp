#include "diffusion.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

enum class diffusion_cache_residency {
    none,
    mutable_planned,
    long_term_stable_planned,
};

enum class diffusion_token_role {
    prefix,
    current,
    future_draft,
};

struct diffusion_token_lifecycle {
    llama_token               semantic_token       = LLAMA_TOKEN_NULL;
    llama_token               stable_token         = LLAMA_TOKEN_NULL;
    llama_token               confidence_token     = LLAMA_TOKEN_NULL;
    float                     confidence           = 0.0f;
    uint32_t                  token_version        = 0;
    uint32_t                  cache_version        = 0;
    uint32_t                  stable_token_version = 0;
    uint32_t                  stable_cache_version = 0;
    diffusion_cache_residency residency            = diffusion_cache_residency::none;
    diffusion_token_role      role                 = diffusion_token_role::future_draft;
    int32_t                   owner_block          = -1;
    int32_t                   step_epoch           = 0;
    int32_t                   last_refresh_step    = -1;
    bool                      confidence_valid     = false;
    bool                      confidence_is_draft  = false;
    bool                      refresh_pending      = false;
};

static bool diffusion_token_transition_allowed(diffusion_token_state from, diffusion_token_state to) {
    return from == to ||
        (from == diffusion_token_state::invisible &&
         (to == diffusion_token_state::visible || to == diffusion_token_state::stable)) ||
        (from == diffusion_token_state::visible && to == diffusion_token_state::stable);
}

static void diffusion_hash_u32(uint64_t & hash, uint32_t value) {
    hash ^= value;
    hash *= 1099511628211ULL;
}

static void diffusion_hash_u64(uint64_t & hash, uint64_t value) {
    diffusion_hash_u32(hash, (uint32_t) value);
    diffusion_hash_u32(hash, (uint32_t) (value >> 32));
}

static uint32_t diffusion_float_bits(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float hash size mismatch");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct diffusion_logical_kv_entry {
    bool                  valid            = false;
    int32_t               position         = -1;
    llama_token           token_id         = LLAMA_TOKEN_NULL;
    diffusion_token_state state             = diffusion_token_state::invisible;
    uint32_t              token_version     = 0;
    uint32_t              cache_version     = 0;
    int32_t               owner_block       = -1;
    diffusion_token_role  role              = diffusion_token_role::future_draft;
    int32_t               admitted_step     = -1;
    int32_t               last_update_step  = -1;
};

enum class diffusion_logical_kv_action {
    upsert_mutable,
    insert_stable,
    promote_stable,
};

struct diffusion_logical_kv_update {
    diffusion_logical_kv_action action                 = diffusion_logical_kv_action::upsert_mutable;
    diffusion_logical_kv_entry  entry;
    uint32_t                    expected_token_version = 0;
    uint32_t                    expected_cache_version = 0;
};

struct diffusion_logical_kv_ledger {
    int32_t n_input      = 0;
    int32_t max_length   = 0;
    int32_t block_length = 0;
    int32_t num_blocks   = 0;

    std::vector<diffusion_logical_kv_entry> stable;
    std::vector<diffusion_logical_kv_entry> mutable_visible;
    std::vector<diffusion_logical_kv_entry> step_local;
    std::vector<diffusion_logical_kv_entry> pre_boundary_stable;
    std::vector<diffusion_logical_kv_entry> pre_boundary_mutable;
    std::vector<diffusion_logical_kv_update> pending;
    std::vector<uint8_t> block_committed;
    std::vector<uint8_t> block_sealed;
    std::vector<uint8_t> mutable_prefix_seen;

    bool    boundary_open  = false;
    int32_t boundary_epoch = -1;

    int32_t stable_entries          = 0;
    int32_t mutable_entries         = 0;
    int32_t step_local_current      = 0;
    int32_t step_local_future       = 0;
    int32_t step_local_current_max  = 0;
    int32_t step_local_future_max   = 0;
    int32_t pending_entries         = 0;

    uint64_t stable_inserts            = 0;
    uint64_t mutable_inserts           = 0;
    uint64_t mutable_replacements      = 0;
    uint64_t mutable_to_stable         = 0;
    uint64_t mutable_prefix_crossings  = 0;
    uint64_t stable_carryover_checks   = 0;
    uint64_t boundary_transactions     = 0;
    uint64_t step_local_rebuilds       = 0;
    uint64_t step_local_syncs          = 0;
    uint64_t step_local_clears         = 0;
    uint64_t mid_step_mutation_errors  = 0;
    uint64_t stable_mutation_errors    = 0;
    uint64_t stale_updates_dropped     = 0;
    uint64_t token_version_mismatches  = 0;
    uint64_t cache_version_mismatches  = 0;
    uint64_t duplicate_durable         = 0;
    uint64_t future_durable_inserts    = 0;
    uint64_t blocks_committed_count    = 0;
    uint64_t blocks_sealed_count       = 0;
    uint64_t seal_refused_visible      = 0;
    uint64_t seal_refused_invisible    = 0;
    uint64_t seal_refused_cache        = 0;
    uint64_t illegal_block_seals       = 0;
    uint64_t state_membership_errors   = 0;

    uint64_t stable_hash      = 14695981039346656037ULL;
    uint64_t mutable_hash     = 14695981039346656037ULL;
    uint64_t step_local_hash  = 14695981039346656037ULL;
    uint64_t transaction_hash = 14695981039346656037ULL;
    uint64_t combined_hash    = 14695981039346656037ULL;

    bool     stale_self_test_passed         = false;
    uint64_t stale_self_test_attempts       = 0;
    uint64_t stale_self_test_dropped        = 0;
    uint64_t stale_self_test_mutation_errors = 0;

    static bool entry_equal(const diffusion_logical_kv_entry & a,
                            const diffusion_logical_kv_entry & b) {
        return a.valid == b.valid && a.position == b.position && a.token_id == b.token_id &&
            a.state == b.state && a.token_version == b.token_version &&
            a.cache_version == b.cache_version && a.owner_block == b.owner_block &&
            a.role == b.role && a.admitted_step == b.admitted_step &&
            a.last_update_step == b.last_update_step;
    }

    static bool durable_content_equal(const diffusion_logical_kv_entry & a,
                                      const diffusion_logical_kv_entry & b) {
        return a.valid == b.valid && a.position == b.position && a.token_id == b.token_id &&
            a.state == b.state && a.token_version == b.token_version &&
            a.cache_version == b.cache_version && a.owner_block == b.owner_block &&
            a.role == b.role && a.admitted_step == b.admitted_step;
    }

    static void hash_entry(uint64_t & hash, const diffusion_logical_kv_entry & entry) {
        diffusion_hash_u32(hash, (uint32_t) entry.valid);
        if (!entry.valid) {
            return;
        }
        diffusion_hash_u32(hash, (uint32_t) entry.position);
        diffusion_hash_u32(hash, (uint32_t) entry.token_id);
        diffusion_hash_u32(hash, (uint32_t) entry.state);
        diffusion_hash_u32(hash, entry.token_version);
        diffusion_hash_u32(hash, entry.cache_version);
        diffusion_hash_u32(hash, (uint32_t) entry.owner_block);
        diffusion_hash_u32(hash, (uint32_t) entry.role);
        diffusion_hash_u32(hash, (uint32_t) entry.admitted_step);
        diffusion_hash_u32(hash, (uint32_t) entry.last_update_step);
    }

    static bool vector_equal(const std::vector<diffusion_logical_kv_entry> & a,
                             const std::vector<diffusion_logical_kv_entry> & b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); i++) {
            if (!entry_equal(a[i], b[i])) {
                return false;
            }
        }
        return true;
    }

    void reset(int32_t input, int32_t length, int32_t block_size, int32_t blocks) {
        *this = diffusion_logical_kv_ledger {};
        n_input      = input;
        max_length   = length;
        block_length = block_size;
        num_blocks   = blocks;
        stable.resize(max_length);
        mutable_visible.resize(max_length);
        step_local.resize(max_length);
        pre_boundary_stable.resize(max_length);
        pre_boundary_mutable.resize(max_length);
        pending.reserve(max_length - n_input);
        block_committed.resize(num_blocks, 0);
        block_sealed.resize(num_blocks, 0);
        mutable_prefix_seen.resize(max_length, 0);
    }

    uint64_t hard_errors() const {
        return mid_step_mutation_errors + stable_mutation_errors + token_version_mismatches +
            cache_version_mismatches + duplicate_durable + future_durable_inserts +
            illegal_block_seals + state_membership_errors;
    }

    void begin_boundary(int32_t epoch) {
        if (boundary_open) {
            mid_step_mutation_errors++;
            return;
        }
        boundary_open          = true;
        boundary_epoch         = epoch;
        pre_boundary_stable    = stable;
        pre_boundary_mutable   = mutable_visible;
        pending.clear();
        pending_entries = 0;
    }

    void clear_step_local(int32_t epoch) {
        for (diffusion_logical_kv_entry & entry : step_local) {
            entry = {};
        }
        step_local_current = 0;
        step_local_future  = 0;
        step_local_clears++;
        diffusion_hash_u32(step_local_hash, 0x57494e44U);
        diffusion_hash_u32(step_local_hash, (uint32_t) epoch);
    }

    void rebuild_step_local(int32_t window_start,
                            int32_t block_end,
                            int32_t window_end,
                            int32_t epoch,
                            const llama_token * output_tokens,
                            const std::vector<diffusion_token_state> & states,
                            const std::vector<diffusion_token_lifecycle> & lifecycle,
                            const std::vector<llama_token> & draft_tokens,
                            const std::vector<uint8_t> & draft_valid) {
        clear_step_local(epoch);
        step_local_rebuilds++;
        for (int32_t pos = window_start; pos < block_end; pos++) {
            diffusion_logical_kv_entry & cached = step_local[pos];
            const diffusion_token_lifecycle & live = lifecycle[pos];
            cached.valid            = true;
            cached.position         = pos;
            cached.token_id          = output_tokens[pos];
            cached.state             = states[pos];
            cached.token_version     = live.token_version;
            cached.cache_version     = live.cache_version;
            cached.owner_block       = live.owner_block;
            cached.role              = diffusion_token_role::current;
            cached.admitted_step     = epoch;
            cached.last_update_step  = epoch;
            step_local_current++;
        }
        for (int32_t pos = block_end; pos < window_end; pos++) {
            if (!draft_valid[pos]) {
                continue;
            }
            diffusion_logical_kv_entry & cached = step_local[pos];
            const diffusion_token_lifecycle & live = lifecycle[pos];
            cached.valid            = true;
            cached.position         = pos;
            cached.token_id          = draft_tokens[pos];
            cached.state             = diffusion_token_state::invisible;
            cached.token_version     = 0;
            cached.cache_version     = 0;
            cached.owner_block       = live.owner_block;
            cached.role              = diffusion_token_role::future_draft;
            cached.admitted_step     = epoch;
            cached.last_update_step  = epoch;
            step_local_future++;
        }
        for (int32_t pos = n_input; pos < max_length; pos++) {
            if (!draft_valid[pos] || lifecycle[pos].role != diffusion_token_role::future_draft ||
                step_local[pos].valid) {
                continue;
            }
            diffusion_logical_kv_entry & cached = step_local[pos];
            const diffusion_token_lifecycle & live = lifecycle[pos];
            cached.valid            = true;
            cached.position         = pos;
            cached.token_id          = draft_tokens[pos];
            cached.state             = diffusion_token_state::invisible;
            cached.token_version     = 0;
            cached.cache_version     = 0;
            cached.owner_block       = live.owner_block;
            cached.role              = diffusion_token_role::future_draft;
            cached.admitted_step     = epoch;
            cached.last_update_step  = epoch;
            step_local_future++;
        }
        step_local_current_max = std::max(step_local_current_max, step_local_current);
        step_local_future_max  = std::max(step_local_future_max, step_local_future);
        for (int32_t pos = n_input; pos < max_length; pos++) {
            hash_entry(step_local_hash, step_local[pos]);
        }
    }

    void sync_future_drafts(int32_t epoch,
                            const std::vector<diffusion_token_lifecycle> & lifecycle,
                            const std::vector<llama_token> & draft_tokens,
                            const std::vector<uint8_t> & draft_valid) {
        step_local_syncs++;
        diffusion_hash_u32(step_local_hash, 0x44524146U);
        diffusion_hash_u32(step_local_hash, (uint32_t) epoch);
        for (int32_t pos = n_input; pos < max_length; pos++) {
            diffusion_logical_kv_entry & cached = step_local[pos];
            const diffusion_token_lifecycle & live = lifecycle[pos];
            const bool future_valid =
                live.role == diffusion_token_role::future_draft && draft_valid[pos];
            if (!future_valid) {
                if (cached.valid && cached.role == diffusion_token_role::future_draft) {
                    cached = {};
                }
                continue;
            }

            const int32_t admitted_step =
                cached.valid && cached.role == diffusion_token_role::future_draft ?
                    cached.admitted_step : epoch;
            const int32_t last_update_step =
                cached.valid && cached.role == diffusion_token_role::future_draft &&
                cached.token_id == draft_tokens[pos] ? cached.last_update_step : epoch;
            cached.valid            = true;
            cached.position         = pos;
            cached.token_id          = draft_tokens[pos];
            cached.state             = diffusion_token_state::invisible;
            cached.token_version     = 0;
            cached.cache_version     = 0;
            cached.owner_block       = live.owner_block;
            cached.role              = diffusion_token_role::future_draft;
            cached.admitted_step     = admitted_step;
            cached.last_update_step  = last_update_step;
        }

        step_local_current = 0;
        step_local_future  = 0;
        for (int32_t pos = n_input; pos < max_length; pos++) {
            step_local_current += step_local[pos].valid &&
                step_local[pos].role == diffusion_token_role::current;
            step_local_future += step_local[pos].valid &&
                step_local[pos].role == diffusion_token_role::future_draft;
            hash_entry(step_local_hash, step_local[pos]);
        }
        step_local_current_max = std::max(step_local_current_max, step_local_current);
        step_local_future_max  = std::max(step_local_future_max, step_local_future);
    }

    diffusion_logical_kv_entry make_entry(int32_t pos,
                                          diffusion_token_state state,
                                          const diffusion_token_lifecycle & live,
                                          int32_t epoch,
                                          int32_t admitted_step) const {
        diffusion_logical_kv_entry result;
        result.valid            = true;
        result.position         = pos;
        result.token_id          = live.semantic_token;
        result.state             = state;
        result.token_version     = live.token_version;
        result.cache_version     = live.cache_version;
        result.owner_block       = live.owner_block;
        result.role              = live.role;
        result.admitted_step     = admitted_step >= 0 ? admitted_step : epoch;
        result.last_update_step  = epoch;
        return result;
    }

    void stage_from_lifecycle(const std::vector<diffusion_token_state> & states,
                              const std::vector<diffusion_token_lifecycle> & lifecycle,
                              int32_t current_block,
                              int32_t epoch) {
        GGML_ASSERT(boundary_open);
        for (int32_t pos = n_input; pos < max_length; pos++) {
            const diffusion_token_lifecycle & live = lifecycle[pos];
            const bool in_stable = stable[pos].valid;
            const bool in_mutable = mutable_visible[pos].valid;
            if (in_stable && in_mutable) {
                duplicate_durable++;
                continue;
            }
            if (in_mutable && live.role == diffusion_token_role::prefix &&
                live.owner_block < current_block && current_block < num_blocks &&
                !mutable_prefix_seen[pos]) {
                mutable_prefix_seen[pos] = 1;
                mutable_prefix_crossings++;
            }
            if (live.role == diffusion_token_role::future_draft) {
                if (in_stable || in_mutable) {
                    future_durable_inserts++;
                }
                continue;
            }

            if (states[pos] == diffusion_token_state::invisible) {
                if (in_stable || in_mutable) {
                    state_membership_errors++;
                }
                continue;
            }

            diffusion_logical_kv_update update;
            update.expected_token_version = live.token_version;
            update.expected_cache_version = live.cache_version;
            if (states[pos] == diffusion_token_state::visible) {
                if (in_stable) {
                    stable_mutation_errors++;
                    continue;
                }
                const int32_t admitted = in_mutable ? mutable_visible[pos].admitted_step : epoch;
                update.action = diffusion_logical_kv_action::upsert_mutable;
                update.entry  = make_entry(pos, diffusion_token_state::visible, live, epoch, admitted);
                if (!in_mutable || !durable_content_equal(mutable_visible[pos], update.entry)) {
                    pending.push_back(update);
                }
            } else {
                if (in_stable) {
                    const diffusion_logical_kv_entry & cached = stable[pos];
                    if (cached.token_id != live.semantic_token || cached.token_version != live.token_version ||
                        cached.cache_version != live.cache_version || cached.owner_block != live.owner_block ||
                        cached.state != diffusion_token_state::stable) {
                        stable_mutation_errors++;
                    }
                    continue;
                }
                update.action = in_mutable ?
                    diffusion_logical_kv_action::promote_stable :
                    diffusion_logical_kv_action::insert_stable;
                update.entry = make_entry(pos, diffusion_token_state::stable, live, epoch, epoch);
                pending.push_back(update);
            }
        }
        pending_entries = (int32_t) pending.size();
    }

    void mark_block_committed(int32_t block) {
        if (block < 0 || block >= num_blocks) {
            state_membership_errors++;
            return;
        }
        if (!block_committed[block]) {
            block_committed[block] = 1;
            blocks_committed_count++;
        }
    }

    void evaluate_block_seals(const std::vector<diffusion_token_state> & states) {
        for (int32_t block = 0; block < num_blocks; block++) {
            if (!block_committed[block] || block_sealed[block]) {
                continue;
            }
            const int32_t start = std::min(max_length, n_input + block * block_length);
            const int32_t end   = std::min(max_length, start + block_length);
            int32_t invisible = 0;
            int32_t visible   = 0;
            int32_t cache_missing = 0;
            for (int32_t pos = start; pos < end; pos++) {
                invisible += states[pos] == diffusion_token_state::invisible;
                visible   += states[pos] == diffusion_token_state::visible;
                cache_missing += states[pos] == diffusion_token_state::stable &&
                    (!stable[pos].valid || mutable_visible[pos].valid);
            }
            if (invisible > 0) {
                seal_refused_invisible++;
            } else if (visible > 0) {
                seal_refused_visible++;
            } else if (cache_missing > 0) {
                seal_refused_cache++;
                state_membership_errors++;
            } else {
                block_sealed[block] = 1;
                blocks_sealed_count++;
            }
        }

        for (int32_t block = 0; block < num_blocks; block++) {
            if (!block_sealed[block]) {
                continue;
            }
            const int32_t start = std::min(max_length, n_input + block * block_length);
            const int32_t end   = std::min(max_length, start + block_length);
            for (int32_t pos = start; pos < end; pos++) {
                if (states[pos] != diffusion_token_state::stable) {
                    illegal_block_seals++;
                }
            }
        }
    }

    void fold_cache_hashes(int32_t epoch) {
        diffusion_hash_u32(stable_hash, 0x53544142U);
        diffusion_hash_u32(stable_hash, (uint32_t) epoch);
        diffusion_hash_u32(mutable_hash, 0x4d555441U);
        diffusion_hash_u32(mutable_hash, (uint32_t) epoch);
        stable_entries  = 0;
        mutable_entries = 0;
        for (int32_t pos = n_input; pos < max_length; pos++) {
            hash_entry(stable_hash, stable[pos]);
            hash_entry(mutable_hash, mutable_visible[pos]);
            stable_entries  += stable[pos].valid;
            mutable_entries += mutable_visible[pos].valid;
        }
        diffusion_hash_u32(combined_hash, 0x434f4d42U);
        diffusion_hash_u32(combined_hash, (uint32_t) epoch);
        diffusion_hash_u64(combined_hash, stable_hash);
        diffusion_hash_u64(combined_hash, mutable_hash);
        diffusion_hash_u64(combined_hash, step_local_hash);
        diffusion_hash_u64(combined_hash, transaction_hash);
        for (int32_t block = 0; block < num_blocks; block++) {
            diffusion_hash_u32(combined_hash, (uint32_t) block_committed[block]);
            diffusion_hash_u32(combined_hash, (uint32_t) block_sealed[block]);
        }
    }

    bool commit_boundary(const std::vector<diffusion_token_state> & states,
                         const std::vector<diffusion_token_lifecycle> & lifecycle,
                         const std::vector<llama_token> & draft_tokens,
                         const std::vector<uint8_t> & draft_valid,
                         int32_t epoch) {
        if (!boundary_open || epoch != boundary_epoch) {
            mid_step_mutation_errors++;
            return false;
        }
        const uint64_t errors_before = hard_errors();
        bool transaction_valid = errors_before == 0;
        if (!vector_equal(stable, pre_boundary_stable) ||
            !vector_equal(mutable_visible, pre_boundary_mutable)) {
            mid_step_mutation_errors++;
            transaction_valid = false;
        }

        std::vector<diffusion_logical_kv_entry> next_stable  = stable;
        std::vector<diffusion_logical_kv_entry> next_mutable = mutable_visible;
        diffusion_hash_u32(transaction_hash, 0x5452414eU);
        diffusion_hash_u32(transaction_hash, (uint32_t) epoch);
        diffusion_hash_u32(transaction_hash, (uint32_t) pending.size());
        std::vector<uint8_t> pending_positions(max_length, 0);
        uint64_t stable_inserts_delta           = 0;
        uint64_t mutable_inserts_delta          = 0;
        uint64_t mutable_replacements_delta     = 0;
        uint64_t mutable_to_stable_delta        = 0;
        uint64_t stable_carryover_checks_delta  = 0;

        for (const diffusion_logical_kv_update & update : pending) {
            const int32_t pos = update.entry.position;
            diffusion_hash_u32(transaction_hash, (uint32_t) update.action);
            hash_entry(transaction_hash, update.entry);
            if (pos < n_input || pos >= max_length) {
                state_membership_errors++;
                transaction_valid = false;
                continue;
            }
            if (pending_positions[pos]) {
                duplicate_durable++;
                transaction_valid = false;
                continue;
            }
            pending_positions[pos] = 1;
            const diffusion_token_lifecycle & live = lifecycle[pos];
            if (live.token_version != update.expected_token_version ||
                live.semantic_token != update.entry.token_id || states[pos] != update.entry.state) {
                stale_updates_dropped++;
                diffusion_hash_u32(transaction_hash, 0x5354414cU);
                continue;
            }
            if (live.cache_version != update.expected_cache_version) {
                stale_updates_dropped++;
                diffusion_hash_u32(transaction_hash, 0x43414348U);
                continue;
            }

            switch (update.action) {
                case diffusion_logical_kv_action::upsert_mutable:
                    if (next_stable[pos].valid) {
                        stable_mutation_errors++;
                        transaction_valid = false;
                        break;
                    }
                    if (next_mutable[pos].valid) {
                        const bool content_changed =
                            next_mutable[pos].token_id != update.entry.token_id ||
                            next_mutable[pos].token_version != update.entry.token_version ||
                            next_mutable[pos].cache_version != update.entry.cache_version;
                        mutable_replacements_delta += content_changed;
                    } else {
                        mutable_inserts_delta++;
                    }
                    next_mutable[pos] = update.entry;
                    break;
                case diffusion_logical_kv_action::insert_stable:
                    if (next_stable[pos].valid || next_mutable[pos].valid) {
                        duplicate_durable++;
                        transaction_valid = false;
                        break;
                    }
                    next_stable[pos] = update.entry;
                    stable_inserts_delta++;
                    break;
                case diffusion_logical_kv_action::promote_stable:
                    if (next_stable[pos].valid || !next_mutable[pos].valid) {
                        state_membership_errors++;
                        transaction_valid = false;
                        break;
                    }
                    next_mutable[pos] = {};
                    next_stable[pos]  = update.entry;
                    stable_inserts_delta++;
                    mutable_to_stable_delta++;
                    break;
            }
        }

        for (int32_t pos = n_input; pos < max_length; pos++) {
            if (pre_boundary_stable[pos].valid) {
                stable_carryover_checks_delta++;
                if (!entry_equal(pre_boundary_stable[pos], next_stable[pos])) {
                    stable_mutation_errors++;
                    transaction_valid = false;
                }
            }
        }

        for (int32_t pos = n_input; pos < max_length; pos++) {
            const diffusion_token_lifecycle & live = lifecycle[pos];
            const bool in_stable = next_stable[pos].valid;
            const bool in_mutable = next_mutable[pos].valid;
            const bool future_draft =
                live.role == diffusion_token_role::future_draft && draft_valid[pos];
            const diffusion_logical_kv_entry & local = step_local[pos];
            const bool future_local_valid =
                local.valid && local.position == pos &&
                (!future_draft || local.token_id == draft_tokens[pos]) &&
                local.state == diffusion_token_state::invisible &&
                local.token_version == 0 && local.cache_version == 0 &&
                local.owner_block == live.owner_block &&
                local.role == diffusion_token_role::future_draft;
            if (future_draft != future_local_valid) {
                state_membership_errors++;
                transaction_valid = false;
            }
            if (in_stable && in_mutable) {
                duplicate_durable++;
                transaction_valid = false;
            }
            if (live.role == diffusion_token_role::future_draft && (in_stable || in_mutable)) {
                future_durable_inserts++;
                transaction_valid = false;
            }
            if (draft_valid[pos] && (in_stable || in_mutable)) {
                future_durable_inserts++;
                transaction_valid = false;
            }

            if (states[pos] == diffusion_token_state::invisible) {
                state_membership_errors += in_stable || in_mutable;
                transaction_valid &= !in_stable && !in_mutable;
            } else if (states[pos] == diffusion_token_state::visible) {
                if (in_stable || !in_mutable) {
                    state_membership_errors++;
                    transaction_valid = false;
                    continue;
                }
                const diffusion_logical_kv_entry & cached = next_mutable[pos];
                const bool token_mismatch =
                    cached.token_id != live.semantic_token || cached.token_version != live.token_version;
                const bool cache_mismatch = cached.cache_version != live.cache_version;
                const bool membership_mismatch =
                    cached.state != diffusion_token_state::visible || cached.owner_block != live.owner_block ||
                    cached.role != live.role || cached.position != pos;
                token_version_mismatches += token_mismatch;
                cache_version_mismatches += cache_mismatch;
                state_membership_errors += membership_mismatch;
                transaction_valid &= !token_mismatch && !cache_mismatch && !membership_mismatch;
            } else {
                if (!in_stable || in_mutable) {
                    state_membership_errors++;
                    transaction_valid = false;
                    continue;
                }
                const diffusion_logical_kv_entry & cached = next_stable[pos];
                const bool token_mismatch =
                    cached.token_id != live.semantic_token || cached.token_version != live.token_version;
                const bool cache_mismatch = cached.cache_version != live.cache_version;
                const bool membership_mismatch =
                    cached.state != diffusion_token_state::stable || cached.owner_block != live.owner_block ||
                    cached.position != pos || cached.role == diffusion_token_role::future_draft;
                token_version_mismatches += token_mismatch;
                cache_version_mismatches += cache_mismatch;
                state_membership_errors += membership_mismatch;
                transaction_valid &= !token_mismatch && !cache_mismatch && !membership_mismatch;
                if (live.refresh_pending) {
                    state_membership_errors++;
                    transaction_valid = false;
                }
            }
        }

        const bool commit_applied = transaction_valid && hard_errors() == errors_before;
        if (commit_applied) {
            stable.swap(next_stable);
            mutable_visible.swap(next_mutable);
            stable_inserts           += stable_inserts_delta;
            mutable_inserts          += mutable_inserts_delta;
            mutable_replacements     += mutable_replacements_delta;
            mutable_to_stable        += mutable_to_stable_delta;
            stable_carryover_checks  += stable_carryover_checks_delta;
        }
        pending.clear();
        pending_entries = 0;
        boundary_transactions += commit_applied;
        boundary_open  = false;
        boundary_epoch = -1;
        if (commit_applied) {
            evaluate_block_seals(states);
            fold_cache_hashes(epoch);
        }
        return commit_applied && hard_errors() == 0;
    }
};

struct diffusion_logical_kv_stale_test_result {
    uint64_t attempts        = 0;
    uint64_t dropped         = 0;
    uint64_t mutation_errors = 0;
    bool     passed          = false;
};

static diffusion_logical_kv_stale_test_result diffusion_logical_kv_stale_update_self_test() {
    diffusion_logical_kv_stale_test_result result;
    diffusion_logical_kv_ledger ledger;
    ledger.reset(0, 1, 1, 1);

    std::vector<diffusion_token_state> states(1, diffusion_token_state::visible);
    std::vector<diffusion_token_lifecycle> lifecycle(1);
    std::vector<llama_token> draft_tokens(1, LLAMA_TOKEN_NULL);
    std::vector<uint8_t> draft_valid(1, 0);
    lifecycle[0].semantic_token = 17;
    lifecycle[0].token_version  = 2;
    lifecycle[0].cache_version  = 2;
    lifecycle[0].owner_block    = 0;
    lifecycle[0].role           = diffusion_token_role::current;
    lifecycle[0].residency      = diffusion_cache_residency::mutable_planned;

    ledger.begin_boundary(1);
    ledger.stage_from_lifecycle(states, lifecycle, 0, 1);
    if (!ledger.commit_boundary(states, lifecycle, draft_tokens, draft_valid, 1) ||
        !ledger.mutable_visible[0].valid) {
        result.mutation_errors++;
        return result;
    }
    const diffusion_logical_kv_entry before = ledger.mutable_visible[0];

    ledger.begin_boundary(2);
    diffusion_logical_kv_update stale;
    stale.action                 = diffusion_logical_kv_action::upsert_mutable;
    stale.entry                  = before;
    stale.entry.token_id          = 11;
    stale.entry.token_version     = 1;
    stale.entry.cache_version     = 1;
    stale.entry.last_update_step  = 2;
    stale.expected_token_version = 1;
    stale.expected_cache_version = 1;
    ledger.pending.push_back(stale);
    ledger.pending_entries = 1;
    result.attempts++;
    const bool committed = ledger.commit_boundary(states, lifecycle, draft_tokens, draft_valid, 2);
    result.dropped = ledger.stale_updates_dropped;
    result.mutation_errors += !diffusion_logical_kv_ledger::entry_equal(before, ledger.mutable_visible[0]);
    result.mutation_errors += !committed || ledger.pending_entries != 0;
    result.passed = result.attempts == 1 && result.dropped == 1 && result.mutation_errors == 0;
    return result;
}

bool diffusion_logical_kv_self_test() {
    return diffusion_logical_kv_stale_update_self_test().passed;
}

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
    const bool mbsd_lifecycle_enabled     = params.mbsd_lifecycle_bookkeeping;
    const bool mbsd_logical_kv_enabled    = params.mbsd_logical_kv;
    const bool prefix_kv_enabled          = params.prefix_kv;
    const bool full_sequence_kv_oracle    = params.full_sequence_kv_oracle;
    const bool staged_token_stabilization = params.staged_token_stabilization && !mbsd_enabled;
    const bool mbsd_staged_lifecycle      =
        params.staged_token_stabilization && mbsd_enabled && mbsd_lifecycle_enabled;
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

    if (!params.staged_token_stabilization &&
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

    if (mbsd_lifecycle_enabled && !mbsd_enabled) {
        LOG_ERR("%s: MBSD lifecycle bookkeeping requires MBSD\n", __func__);
        return;
    }

    if (mbsd_logical_kv_enabled &&
        (!mbsd_enabled || !mbsd_lifecycle_enabled || !params.staged_token_stabilization)) {
        LOG_ERR("%s: MBSD logical KV requires MBSD lifecycle bookkeeping and staged stabilization\n",
                __func__);
        return;
    }

    if (mbsd_lifecycle_enabled && mbsd_fresh_kv_enabled) {
        LOG_ERR("%s: MBSD lifecycle bookkeeping and fresh KV are mutually exclusive\n", __func__);
        return;
    }

    if (mbsd_logical_kv_enabled &&
        (mbsd_fresh_kv_enabled || prefix_kv_enabled || full_sequence_kv_oracle)) {
        LOG_ERR("%s: MBSD logical KV is mutually exclusive with fresh KV, prefix KV, "
                "and the full-sequence KV oracle\n", __func__);
        return;
    }

    if (params.staged_token_stabilization && mbsd_enabled && !mbsd_lifecycle_enabled) {
        LOG_ERR("%s: MBSD staged stabilization requires lifecycle bookkeeping\n", __func__);
        return;
    }

    if (mbsd_staged_lifecycle &&
        (params.temperature != 0.0f ||
         params.staged_revision_policy != DIFFUSION_STAGED_REVISION_OLDEST ||
         params.staged_final_revision_steps > 0)) {
        LOG_ERR("%s: MBSD staged lifecycle observation requires temp 0 and no revision options\n", __func__);
        return;
    }

    if (mbsd_enabled &&
        (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || !params.generated_block_schedule ||
         params.algorithm != DIFFUSION_ALGORITHM_CONFIDENCE_BASED || params.alg_temp != 0.0f ||
         early_commit_enabled || prefix_kv_enabled || full_sequence_kv_oracle ||
         params.cfg_scale != 0.0f || params.add_gumbel_noise)) {
        LOG_ERR("%s: MBSD requires generated block scheduling, confidence selection, alg-temp 0, "
                "and no early commit, prefix KV, full-sequence KV oracle, CFG, or Gumbel noise\n",
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
    if (staged_token_stabilization || mbsd_lifecycle_enabled) {
        token_states.resize(params.max_length, diffusion_token_state::stable);
        std::fill(token_states.begin() + n_input, token_states.end(), diffusion_token_state::invisible);
    }
    if (staged_token_stabilization) {
        sts_last_revision_step.resize(params.max_length, -1);
        sts_revision_count.resize(params.max_length, 0);
        sts_latest_confidence.resize(params.max_length, 0.0f);
    }

    std::vector<diffusion_token_lifecycle> mbsd_lifecycle;
    std::vector<int32_t>                   mbsd_lifecycle_refresh_queue;
    std::vector<uint8_t>                   mbsd_lifecycle_refresh_membership;
    if (mbsd_lifecycle_enabled) {
        mbsd_lifecycle.resize(params.max_length);
        mbsd_lifecycle_refresh_queue.reserve(params.max_length - n_input);
        mbsd_lifecycle_refresh_membership.resize(params.max_length, 0);
        for (int32_t pos = n_input; pos < params.max_length; pos++) {
            mbsd_lifecycle[pos].semantic_token = params.mask_token_id;
        }
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
    uint64_t mbsd_commitment_hash            = 14695981039346656037ULL;
    uint64_t mbsd_fresh_prefix_rows_reused   = 0;
    uint64_t mbsd_fresh_cache_invariant_errors = 0;
    uint64_t mbsd_physical_mapping_errors    = 0;
    int32_t  mbsd_fresh_main_batches         = 0;
    int32_t  mbsd_fresh_main_rows_min        = std::numeric_limits<int32_t>::max();
    int32_t  mbsd_fresh_main_rows_max        = 0;

    uint64_t mbsd_lifecycle_invisible_to_visible = 0;
    uint64_t mbsd_lifecycle_invisible_to_stable  = 0;
    uint64_t mbsd_lifecycle_visible_to_stable    = 0;
    uint64_t mbsd_lifecycle_illegal_transitions  = 0;
    uint64_t mbsd_lifecycle_stable_mutations     = 0;
    uint64_t mbsd_lifecycle_version_errors       = 0;
    uint64_t mbsd_lifecycle_future_commits       = 0;
    uint64_t mbsd_lifecycle_duplicate_ownership  = 0;
    uint64_t mbsd_lifecycle_ownership_errors     = 0;
    uint64_t mbsd_lifecycle_state_errors         = 0;
    uint64_t mbsd_lifecycle_mask_state_errors    = 0;
    uint64_t mbsd_lifecycle_confidence_errors    = 0;
    uint64_t mbsd_lifecycle_confidence_updates   = 0;
    uint64_t mbsd_lifecycle_draft_conf_updates   = 0;
    uint64_t mbsd_lifecycle_conf_invalidations   = 0;
    uint64_t mbsd_lifecycle_forced_visible       = 0;
    uint64_t mbsd_lifecycle_deferred_stable      = 0;
    uint64_t mbsd_lifecycle_refresh_enqueued     = 0;
    uint64_t mbsd_lifecycle_refresh_drained      = 0;
    uint64_t mbsd_lifecycle_refresh_duplicates   = 0;
    uint64_t mbsd_lifecycle_refresh_ineligible   = 0;
    uint64_t mbsd_lifecycle_refresh_future       = 0;
    uint64_t mbsd_lifecycle_refresh_stable       = 0;
    uint64_t mbsd_lifecycle_last_refresh_updates = 0;
    uint64_t mbsd_lifecycle_last_refresh_errors  = 0;
    uint64_t mbsd_lifecycle_future_cache_writes  = 0;
    uint64_t mbsd_lifecycle_state_hash            = 14695981039346656037ULL;
    uint64_t mbsd_lifecycle_ownership_hash        = 14695981039346656037ULL;
    uint64_t mbsd_lifecycle_refresh_hash          = 14695981039346656037ULL;
    uint64_t mbsd_lifecycle_snapshots             = 0;
    uint64_t mbsd_lifecycle_snapshots_pending     = 0;
    int32_t  mbsd_lifecycle_invisible              = 0;
    int32_t  mbsd_lifecycle_visible                = 0;
    int32_t  mbsd_lifecycle_stable                 = 0;
    int32_t  mbsd_lifecycle_residency_none         = 0;
    int32_t  mbsd_lifecycle_residency_mutable      = 0;
    int32_t  mbsd_lifecycle_residency_stable       = 0;
    int32_t  mbsd_lifecycle_pending_entries        = 0;
    int32_t  mbsd_lifecycle_refresh_max_depth      = 0;

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

    diffusion_logical_kv_ledger mbsd_logical_kv;
    if (mbsd_logical_kv_enabled) {
        mbsd_logical_kv.reset(n_input, params.max_length, params.block_length, num_blocks);
        const diffusion_logical_kv_stale_test_result stale_test =
            diffusion_logical_kv_stale_update_self_test();
        mbsd_logical_kv.stale_self_test_attempts       = stale_test.attempts;
        mbsd_logical_kv.stale_self_test_dropped        = stale_test.dropped;
        mbsd_logical_kv.stale_self_test_mutation_errors = stale_test.mutation_errors;
        mbsd_logical_kv.stale_self_test_passed         = stale_test.passed;
        if (!stale_test.passed) {
            LOG_ERR("%s: MBSD logical KV stale-update self-test failed\n", __func__);
            generation_failed = true;
        }
    }

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

    auto lifecycle_set_state = [&](int32_t pos, diffusion_token_state next, int32_t epoch) {
        diffusion_token_state & current = token_states[pos];
        if (!diffusion_token_transition_allowed(current, next)) {
            mbsd_lifecycle_illegal_transitions++;
            return;
        }
        if (current == next) {
            return;
        }

        if (current == diffusion_token_state::invisible && next == diffusion_token_state::visible) {
            mbsd_lifecycle_invisible_to_visible++;
        } else if (current == diffusion_token_state::invisible && next == diffusion_token_state::stable) {
            mbsd_lifecycle_invisible_to_stable++;
        } else if (current == diffusion_token_state::visible && next == diffusion_token_state::stable) {
            mbsd_lifecycle_visible_to_stable++;
        }

        current = next;
        diffusion_token_lifecycle & entry = mbsd_lifecycle[pos];
        entry.step_epoch = epoch;
        switch (next) {
            case diffusion_token_state::invisible:
                entry.residency = diffusion_cache_residency::none;
                break;
            case diffusion_token_state::visible:
                entry.residency = diffusion_cache_residency::mutable_planned;
                break;
            case diffusion_token_state::stable:
                entry.residency = diffusion_cache_residency::long_term_stable_planned;
                entry.stable_token         = entry.semantic_token;
                entry.stable_token_version = entry.token_version;
                entry.stable_cache_version = entry.cache_version;
                break;
        }
    };

    auto lifecycle_assign_roles = [&](int32_t current_block, int32_t epoch) {
        for (int32_t pos = n_input; pos < params.max_length; pos++) {
            diffusion_token_lifecycle & entry = mbsd_lifecycle[pos];
            diffusion_token_role next_role = diffusion_token_role::future_draft;
            if (entry.owner_block < current_block) {
                next_role = diffusion_token_role::prefix;
            } else if (entry.owner_block == current_block) {
                next_role = diffusion_token_role::current;
            }
            if (entry.role != next_role) {
                entry.role       = next_role;
                entry.step_epoch = epoch;
            }
        }
    };

    auto lifecycle_record_confidence = [&](int32_t pos, llama_token token, float confidence,
                                           bool draft, int32_t epoch) {
        diffusion_token_lifecycle & entry = mbsd_lifecycle[pos];
        if (!std::isfinite(confidence) || token < 0 || token >= n_vocab) {
            mbsd_lifecycle_confidence_errors++;
            return;
        }
        if (draft && entry.role != diffusion_token_role::future_draft) {
            mbsd_lifecycle_confidence_errors++;
            return;
        }
        if (token_states[pos] == diffusion_token_state::stable) {
            mbsd_lifecycle_stable_mutations++;
            return;
        }

        entry.confidence          = confidence;
        entry.confidence_token    = token;
        entry.confidence_valid    = true;
        entry.confidence_is_draft = draft;
        entry.step_epoch          = epoch;
        if (draft) {
            mbsd_lifecycle_draft_conf_updates++;
        } else {
            mbsd_lifecycle_confidence_updates++;
        }
    };

    auto lifecycle_stability_frontier = [&]() {
        int32_t frontier = n_input;
        for (int32_t pos = n_input; pos < params.max_length; pos++) {
            const llama_token token = output_tokens[pos];
            if (token == params.mask_token_id) {
                break;
            }
            frontier = pos + 1;
            if (token >= 0 && token < n_vocab && llama_vocab_is_eog(vocab, token)) {
                break;
            }
        }
        return frontier;
    };

    auto lifecycle_promote_stable = [&](int32_t epoch) {
        const int32_t frontier = lifecycle_stability_frontier();
        for (int32_t pos = n_input; pos < frontier; pos++) {
            diffusion_token_lifecycle & entry = mbsd_lifecycle[pos];
            if (token_states[pos] != diffusion_token_state::visible ||
                !entry.confidence_valid || entry.confidence_is_draft ||
                entry.confidence_token != entry.semantic_token ||
                entry.confidence < params.stability_threshold) {
                continue;
            }
            lifecycle_set_state(pos, diffusion_token_state::stable, epoch);
        }
    };

    auto lifecycle_observe_semantic_tokens = [&](int32_t epoch) {
        const int32_t stable_frontier = lifecycle_stability_frontier();
        for (int32_t pos = n_input; pos < params.max_length; pos++) {
            diffusion_token_lifecycle & entry = mbsd_lifecycle[pos];
            const llama_token observed = output_tokens[pos];

            if (entry.role == diffusion_token_role::future_draft) {
                if (observed != params.mask_token_id ||
                    token_states[pos] != diffusion_token_state::invisible ||
                    entry.semantic_token != params.mask_token_id ||
                    entry.token_version != 0 || entry.cache_version != 0 ||
                    entry.residency != diffusion_cache_residency::none) {
                    mbsd_lifecycle_future_commits++;
                }
                continue;
            }

            if (observed == entry.semantic_token) {
                continue;
            }
            if (token_states[pos] == diffusion_token_state::stable) {
                mbsd_lifecycle_stable_mutations++;
                continue;
            }
            if (observed == params.mask_token_id) {
                mbsd_lifecycle_version_errors++;
                continue;
            }

            const bool matching_confidence =
                mbsd_staged_lifecycle && entry.confidence_valid && !entry.confidence_is_draft &&
                entry.confidence_token == observed;
            entry.semantic_token = observed;
            entry.token_version++;
            entry.cache_version = entry.token_version;
            entry.step_epoch = epoch;
            if (mbsd_staged_lifecycle && !matching_confidence && entry.confidence_valid) {
                entry.confidence          = 0.0f;
                entry.confidence_token    = LLAMA_TOKEN_NULL;
                entry.confidence_valid    = false;
                entry.confidence_is_draft = false;
                mbsd_lifecycle_conf_invalidations++;
            }
            if (token_states[pos] == diffusion_token_state::invisible) {
                if (entry.role != diffusion_token_role::current) {
                    mbsd_lifecycle_ownership_errors++;
                }
                if (mbsd_staged_lifecycle && matching_confidence &&
                    entry.confidence >= params.stability_threshold &&
                    pos < stable_frontier) {
                    lifecycle_set_state(pos, diffusion_token_state::stable, epoch);
                } else {
                    lifecycle_set_state(pos, diffusion_token_state::visible, epoch);
                    if (mbsd_staged_lifecycle &&
                        (!matching_confidence || entry.confidence < params.visibility_threshold)) {
                        mbsd_lifecycle_forced_visible++;
                    } else if (mbsd_staged_lifecycle &&
                               entry.confidence >= params.stability_threshold) {
                        mbsd_lifecycle_deferred_stable++;
                    }
                }
            }
        }
        if (mbsd_staged_lifecycle) {
            lifecycle_promote_stable(epoch);
        }
    };

    auto lifecycle_prepare_refresh_queue = [&](int32_t epoch) {
        if (!mbsd_lifecycle_refresh_queue.empty()) {
            mbsd_lifecycle_refresh_ineligible += mbsd_lifecycle_refresh_queue.size();
            for (int32_t pos : mbsd_lifecycle_refresh_queue) {
                if (pos >= n_input && pos < params.max_length) {
                    mbsd_lifecycle[pos].refresh_pending = false;
                    mbsd_lifecycle_refresh_membership[pos] = 0;
                }
            }
            mbsd_lifecycle_refresh_queue.clear();
        }

        for (int32_t pos = n_input; pos < params.max_length; pos++) {
            diffusion_token_lifecycle & entry = mbsd_lifecycle[pos];
            const bool eligible =
                token_states[pos] == diffusion_token_state::visible &&
                entry.role == diffusion_token_role::prefix &&
                entry.residency == diffusion_cache_residency::mutable_planned &&
                mbsd_draft_valid[pos] == 0;
            if (!eligible) {
                continue;
            }
            if (entry.refresh_pending || mbsd_lifecycle_refresh_membership[pos]) {
                mbsd_lifecycle_refresh_duplicates++;
                continue;
            }

            entry.refresh_pending = true;
            mbsd_lifecycle_refresh_membership[pos] = 1;
            mbsd_lifecycle_refresh_queue.push_back(pos);
            mbsd_lifecycle_refresh_enqueued++;
            diffusion_hash_u32(mbsd_lifecycle_refresh_hash, 0x454e5155U);
            diffusion_hash_u32(mbsd_lifecycle_refresh_hash, (uint32_t) epoch);
            diffusion_hash_u32(mbsd_lifecycle_refresh_hash, (uint32_t) (pos - n_input));
            diffusion_hash_u32(mbsd_lifecycle_refresh_hash, entry.token_version);
        }
        mbsd_lifecycle_refresh_max_depth = std::max(
            mbsd_lifecycle_refresh_max_depth, (int32_t) mbsd_lifecycle_refresh_queue.size());
    };

    auto lifecycle_drain_refresh_queue = [&](int32_t epoch) {
        for (int32_t pos : mbsd_lifecycle_refresh_queue) {
            if (pos < n_input || pos >= params.max_length) {
                mbsd_lifecycle_refresh_ineligible++;
                continue;
            }
            diffusion_token_lifecycle & entry = mbsd_lifecycle[pos];
            if (!entry.refresh_pending || !mbsd_lifecycle_refresh_membership[pos]) {
                mbsd_lifecycle_refresh_ineligible++;
                continue;
            }
            entry.refresh_pending = false;
            mbsd_lifecycle_refresh_membership[pos] = 0;
            mbsd_lifecycle_refresh_drained++;
            diffusion_hash_u32(mbsd_lifecycle_refresh_hash, 0x44524149U);
            diffusion_hash_u32(mbsd_lifecycle_refresh_hash, (uint32_t) epoch);
            diffusion_hash_u32(mbsd_lifecycle_refresh_hash, (uint32_t) (pos - n_input));
        }
        mbsd_lifecycle_refresh_queue.clear();
    };

    auto lifecycle_record_snapshot = [&](int32_t current_block, int32_t epoch) {
        mbsd_lifecycle_invisible = 0;
        mbsd_lifecycle_visible   = 0;
        mbsd_lifecycle_stable    = 0;
        mbsd_lifecycle_residency_none    = 0;
        mbsd_lifecycle_residency_mutable = 0;
        mbsd_lifecycle_residency_stable  = 0;
        mbsd_lifecycle_pending_entries   = 0;
        int32_t output_mask_count        = 0;
        int32_t queue_members            = 0;

        diffusion_hash_u32(mbsd_lifecycle_state_hash, 0x53544154U);
        diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) epoch);
        diffusion_hash_u32(mbsd_lifecycle_ownership_hash, 0x4f574e52U);
        diffusion_hash_u32(mbsd_lifecycle_ownership_hash, (uint32_t) epoch);

        for (int32_t pos = n_input; pos < params.max_length; pos++) {
            const diffusion_token_lifecycle & entry = mbsd_lifecycle[pos];
            output_mask_count += output_tokens[pos] == params.mask_token_id;
            switch (token_states[pos]) {
                case diffusion_token_state::invisible:
                    mbsd_lifecycle_invisible++;
                    break;
                case diffusion_token_state::visible:
                    mbsd_lifecycle_visible++;
                    break;
                case diffusion_token_state::stable:
                    mbsd_lifecycle_stable++;
                    break;
            }
            switch (entry.residency) {
                case diffusion_cache_residency::none:
                    mbsd_lifecycle_residency_none++;
                    break;
                case diffusion_cache_residency::mutable_planned:
                    mbsd_lifecycle_residency_mutable++;
                    break;
                case diffusion_cache_residency::long_term_stable_planned:
                    mbsd_lifecycle_residency_stable++;
                    break;
            }

            const diffusion_token_role expected_role =
                entry.owner_block < current_block ? diffusion_token_role::prefix :
                entry.owner_block == current_block ? diffusion_token_role::current :
                diffusion_token_role::future_draft;
            if (entry.owner_block < 0 || entry.owner_block >= num_blocks || entry.role != expected_role) {
                mbsd_lifecycle_ownership_errors++;
            }

            const bool invisible_valid =
                token_states[pos] != diffusion_token_state::invisible ||
                (entry.semantic_token == params.mask_token_id && entry.token_version == 0 &&
                 entry.cache_version == 0 && entry.residency == diffusion_cache_residency::none);
            const bool visible_valid =
                token_states[pos] != diffusion_token_state::visible ||
                (entry.semantic_token != params.mask_token_id && entry.token_version > 0 &&
                 entry.cache_version == entry.token_version &&
                 entry.residency == diffusion_cache_residency::mutable_planned);
            const bool stable_valid =
                token_states[pos] != diffusion_token_state::stable ||
                (entry.semantic_token != params.mask_token_id && entry.semantic_token == output_tokens[pos] &&
                 entry.token_version > 0 && entry.cache_version == entry.token_version &&
                 entry.residency == diffusion_cache_residency::long_term_stable_planned &&
                 entry.confidence_valid && !entry.confidence_is_draft &&
                 entry.confidence_token == entry.semantic_token &&
                 entry.confidence >= params.stability_threshold);
            if (!invisible_valid || !visible_valid || !stable_valid) {
                mbsd_lifecycle_version_errors++;
            }
            if ((token_states[pos] == diffusion_token_state::invisible) !=
                (output_tokens[pos] == params.mask_token_id)) {
                mbsd_lifecycle_mask_state_errors++;
            }
            if (token_states[pos] == diffusion_token_state::stable) {
                if (entry.semantic_token != entry.stable_token || output_tokens[pos] != entry.stable_token) {
                    mbsd_lifecycle_stable_mutations++;
                }
                if (entry.token_version != entry.stable_token_version ||
                    entry.cache_version != entry.stable_cache_version) {
                    mbsd_lifecycle_version_errors++;
                }
            }
            if (entry.role == diffusion_token_role::future_draft &&
                (token_states[pos] != diffusion_token_state::invisible ||
                 output_tokens[pos] != params.mask_token_id ||
                 entry.semantic_token != params.mask_token_id ||
                 entry.token_version != 0 || entry.cache_version != 0 ||
                 entry.residency != diffusion_cache_residency::none)) {
                mbsd_lifecycle_future_commits++;
            }
            if (entry.confidence_is_draft &&
                (entry.role != diffusion_token_role::future_draft || !mbsd_draft_valid[pos] ||
                 !entry.confidence_valid || entry.confidence_token != mbsd_draft_tokens[pos])) {
                mbsd_lifecycle_confidence_errors++;
            }
            if (entry.last_refresh_step != -1) {
                mbsd_lifecycle_last_refresh_errors++;
            }

            const bool refresh_eligible =
                mbsd_staged_lifecycle && token_states[pos] == diffusion_token_state::visible &&
                entry.role == diffusion_token_role::prefix &&
                entry.residency == diffusion_cache_residency::mutable_planned &&
                mbsd_draft_valid[pos] == 0;
            if (entry.refresh_pending) {
                mbsd_lifecycle_pending_entries++;
                queue_members++;
                if (!mbsd_lifecycle_refresh_membership[pos] || !refresh_eligible) {
                    mbsd_lifecycle_refresh_ineligible++;
                }
                if (entry.role == diffusion_token_role::future_draft) {
                    mbsd_lifecycle_refresh_future++;
                }
                if (token_states[pos] == diffusion_token_state::stable) {
                    mbsd_lifecycle_refresh_stable++;
                }
            } else if (mbsd_lifecycle_refresh_membership[pos] || refresh_eligible) {
                mbsd_lifecycle_refresh_ineligible++;
            }

            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) (pos - n_input));
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) token_states[pos]);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) entry.semantic_token);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, entry.token_version);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, entry.cache_version);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) entry.stable_token);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, entry.stable_token_version);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, entry.stable_cache_version);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) entry.confidence_token);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, diffusion_float_bits(entry.confidence));
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) entry.confidence_valid);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) entry.confidence_is_draft);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) entry.residency);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) entry.step_epoch);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) entry.last_refresh_step);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) entry.refresh_pending);
            diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) mbsd_draft_valid[pos]);
            if (mbsd_draft_valid[pos]) {
                diffusion_hash_u32(mbsd_lifecycle_state_hash, (uint32_t) mbsd_draft_tokens[pos]);
            }

            diffusion_hash_u32(mbsd_lifecycle_ownership_hash, (uint32_t) (pos - n_input));
            diffusion_hash_u32(mbsd_lifecycle_ownership_hash, (uint32_t) entry.owner_block);
            diffusion_hash_u32(mbsd_lifecycle_ownership_hash, (uint32_t) entry.role);
            diffusion_hash_u32(mbsd_lifecycle_ownership_hash, (uint32_t) entry.step_epoch);
        }

        const int32_t generated_tokens = params.max_length - n_input;
        if (mbsd_lifecycle_invisible + mbsd_lifecycle_visible + mbsd_lifecycle_stable != generated_tokens ||
            mbsd_lifecycle_residency_none + mbsd_lifecycle_residency_mutable +
                mbsd_lifecycle_residency_stable != generated_tokens ||
            mbsd_lifecycle_invisible != output_mask_count) {
            mbsd_lifecycle_state_errors++;
        }
        if (mbsd_lifecycle_invisible != output_mask_count) {
            mbsd_lifecycle_mask_state_errors++;
        }
        if (queue_members != (int32_t) mbsd_lifecycle_refresh_queue.size()) {
            mbsd_lifecycle_refresh_ineligible++;
        }
        mbsd_lifecycle_snapshots_pending += mbsd_lifecycle_pending_entries > 0;
        mbsd_lifecycle_snapshots++;
    };

    if (mbsd_lifecycle_enabled) {
        std::vector<int32_t> owner_assignments(params.max_length, 0);
        for (int32_t block = 0; block < num_blocks; block++) {
            const auto bounds = get_block_bounds(block);
            for (int32_t pos = bounds.first; pos < bounds.second; pos++) {
                owner_assignments[pos]++;
                if (owner_assignments[pos] == 1) {
                    mbsd_lifecycle[pos].owner_block = block;
                } else {
                    mbsd_lifecycle_duplicate_ownership++;
                }
            }
        }
        for (int32_t pos = n_input; pos < params.max_length; pos++) {
            if (owner_assignments[pos] != 1) {
                mbsd_lifecycle_ownership_errors++;
            }
        }
        lifecycle_assign_roles(0, 0);
        lifecycle_record_snapshot(0, 0);
        if (mbsd_logical_kv_enabled && !generation_failed) {
            mbsd_logical_kv.begin_boundary(0);
            mbsd_logical_kv.clear_step_local(0);
            mbsd_logical_kv.stage_from_lifecycle(token_states, mbsd_lifecycle, 0, 0);
            if (!mbsd_logical_kv.commit_boundary(
                    token_states, mbsd_lifecycle, mbsd_draft_tokens, mbsd_draft_valid, 0)) {
                LOG_ERR("%s: MBSD logical KV initial boundary failed\n", __func__);
                generation_failed = true;
            }
        }
    }

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
        if (mbsd_lifecycle_enabled) {
            lifecycle_assign_roles(block_num, block_step_offset);
        }
        if (mbsd_logical_kv_enabled && block_num > 0) {
            mbsd_logical_kv.begin_boundary(block_step_offset);
            mbsd_logical_kv.clear_step_local(block_step_offset);
            mbsd_logical_kv.sync_future_drafts(
                block_step_offset, mbsd_lifecycle, mbsd_draft_tokens, mbsd_draft_valid);
            mbsd_logical_kv.stage_from_lifecycle(
                token_states, mbsd_lifecycle, block_num, block_step_offset);
            if (!mbsd_logical_kv.commit_boundary(
                    token_states, mbsd_lifecycle, mbsd_draft_tokens, mbsd_draft_valid,
                    block_step_offset)) {
                LOG_ERR("%s: MBSD logical KV block transition failed at block %d\n",
                        __func__, block_num + 1);
                generation_failed = true;
                break;
            }
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

            if (mbsd_logical_kv_enabled) {
                mbsd_logical_kv.begin_boundary(global_step + 1);
            }

            if (mbsd_staged_lifecycle) {
                lifecycle_prepare_refresh_queue(global_step + 1);
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

            if (mbsd_logical_kv_enabled) {
                mbsd_logical_kv.rebuild_step_local(
                    mbsd_window_start, block_end, mbsd_window_end, global_step + 1,
                    output_tokens, token_states, mbsd_lifecycle,
                    mbsd_draft_tokens, mbsd_draft_valid);
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
                    if (mbsd_staged_lifecycle) {
                        lifecycle_record_confidence(pos, sampled_token, conf, false, global_step + 1);
                    }
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

                    if (mbsd_staged_lifecycle) {
                        for (int32_t pos : mbsd_future_positions) {
                            if (mbsd_draft_valid[pos]) {
                                lifecycle_record_confidence(
                                    pos, mbsd_draft_tokens[pos], mbsd_draft_confidences[pos],
                                    true, global_step + 1);
                            }
                        }
                    }
                }
            }

            if (mbsd_lifecycle_enabled) {
                lifecycle_observe_semantic_tokens(global_step + 1);
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
                    if (mbsd_enabled) {
                        diffusion_hash_u32(mbsd_commitment_hash, 0x434f4d4dU);
                        diffusion_hash_u32(mbsd_commitment_hash, (uint32_t) block_num);
                        diffusion_hash_u32(mbsd_commitment_hash, (uint32_t) (global_step + 1));
                    }
                    if (mbsd_logical_kv_enabled) {
                        mbsd_logical_kv.mark_block_committed(block_num);
                    }
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

            if (mbsd_lifecycle_enabled) {
                lifecycle_record_snapshot(block_num, global_step + 1);
                lifecycle_drain_refresh_queue(global_step + 1);
            }

            if (mbsd_logical_kv_enabled) {
                mbsd_logical_kv.sync_future_drafts(
                    global_step + 1, mbsd_lifecycle, mbsd_draft_tokens, mbsd_draft_valid);
                mbsd_logical_kv.stage_from_lifecycle(
                    token_states, mbsd_lifecycle, block_num, global_step + 1);
                if (!mbsd_logical_kv.commit_boundary(
                        token_states, mbsd_lifecycle, mbsd_draft_tokens, mbsd_draft_valid,
                        global_step + 1)) {
                    LOG_ERR("%s: MBSD logical KV boundary failed at block %d step %d\n",
                            __func__, block_num + 1, step + 1);
                    generation_failed = true;
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

    if (mbsd_lifecycle_enabled && !generation_failed) {
        const int32_t final_epoch = params.steps + 1;
        if (mbsd_logical_kv_enabled) {
            mbsd_logical_kv.begin_boundary(final_epoch);
        }
        lifecycle_assign_roles(num_blocks, final_epoch);
        lifecycle_observe_semantic_tokens(final_epoch);
        if (mbsd_staged_lifecycle) {
            lifecycle_prepare_refresh_queue(final_epoch);
        }
        lifecycle_record_snapshot(num_blocks, final_epoch);
        lifecycle_drain_refresh_queue(final_epoch);
        if (mbsd_logical_kv_enabled) {
            mbsd_logical_kv.clear_step_local(final_epoch);
            mbsd_logical_kv.stage_from_lifecycle(
                token_states, mbsd_lifecycle, num_blocks, final_epoch);
            if (!mbsd_logical_kv.commit_boundary(
                    token_states, mbsd_lifecycle, mbsd_draft_tokens, mbsd_draft_valid, final_epoch)) {
                LOG_ERR("%s: MBSD logical KV final boundary failed\n", __func__);
                generation_failed = true;
            }
        }

        mbsd_lifecycle_pending_entries = 0;
        int32_t lifecycle_drafts_pending = 0;
        for (int32_t pos = n_input; pos < params.max_length; pos++) {
            const diffusion_token_lifecycle & entry = mbsd_lifecycle[pos];
            mbsd_lifecycle_pending_entries += entry.refresh_pending;
            lifecycle_drafts_pending += mbsd_draft_valid[pos] != 0;
        }

        const int32_t generated_tokens = params.max_length - n_input;
        const bool lifecycle_invariants_ok =
            mbsd_lifecycle_invisible + mbsd_lifecycle_visible + mbsd_lifecycle_stable == generated_tokens &&
            mbsd_lifecycle_invisible == output_masks_remaining &&
            mbsd_lifecycle_residency_none + mbsd_lifecycle_residency_mutable +
                mbsd_lifecycle_residency_stable == generated_tokens &&
            mbsd_lifecycle_residency_none == mbsd_lifecycle_invisible &&
            mbsd_lifecycle_residency_mutable == mbsd_lifecycle_visible &&
            mbsd_lifecycle_residency_stable == mbsd_lifecycle_stable &&
            mbsd_lifecycle_duplicate_ownership == 0 && mbsd_lifecycle_ownership_errors == 0 &&
            mbsd_lifecycle_state_errors == 0 && mbsd_lifecycle_illegal_transitions == 0 &&
            mbsd_lifecycle_stable_mutations == 0 && mbsd_lifecycle_version_errors == 0 &&
            mbsd_lifecycle_mask_state_errors == 0 && mbsd_lifecycle_confidence_errors == 0 &&
            mbsd_lifecycle_future_commits == 0 && mbsd_future_semantic_commits == 0 &&
            mbsd_lifecycle_future_cache_writes == 0 && lifecycle_drafts_pending == 0 &&
            mbsd_lifecycle_refresh_enqueued == mbsd_lifecycle_refresh_drained &&
            mbsd_lifecycle_refresh_duplicates == 0 && mbsd_lifecycle_refresh_ineligible == 0 &&
            mbsd_lifecycle_refresh_future == 0 && mbsd_lifecycle_refresh_stable == 0 &&
            mbsd_lifecycle_last_refresh_updates == 0 && mbsd_lifecycle_last_refresh_errors == 0 &&
            mbsd_lifecycle_pending_entries == 0 && mbsd_lifecycle_refresh_queue.empty() &&
            mbsd_lifecycle_snapshots == (uint64_t) iterations_completed + 2 &&
            mbsd_lifecycle_state_hash != 0 && mbsd_lifecycle_ownership_hash != 0 &&
            mbsd_lifecycle_refresh_hash != 0 &&
            !diffusion_kv_graph_enabled && cache_perf.calls == 0 && cache_perf.completed == 0 &&
            full_sequence_pre_forward_clears == 0 && mbsd_main_rows_saved == 0 && mbsd_net_rows_saved == 0;
        if (!lifecycle_invariants_ok) {
            LOG_ERR("%s: MBSD lifecycle invariant failed "
                    "(states = %d/%d/%d/%d, residency = %d/%d/%d/%d, "
                    "duplicate/ownership/state/mask/confidence errors = %llu/%llu/%llu/%llu/%llu, "
                    "illegal/stable/version/future/cache/drafts/pending = %llu/%llu/%llu/%llu/%llu/%d/%d, "
                    "queue enqueued/drained/duplicate/ineligible/future/stable = %llu/%llu/%llu/%llu/%llu/%llu, "
                    "last refresh updates/errors = %llu/%llu, "
                    "snapshots = %llu/%d, cache calls/completed/clears = %d/%d/%d, "
                    "main/net row saving = %llu/%lld)\n",
                    __func__, mbsd_lifecycle_invisible, mbsd_lifecycle_visible,
                    mbsd_lifecycle_stable, generated_tokens,
                    mbsd_lifecycle_residency_none, mbsd_lifecycle_residency_mutable,
                    mbsd_lifecycle_residency_stable, generated_tokens,
                    (unsigned long long) mbsd_lifecycle_duplicate_ownership,
                    (unsigned long long) mbsd_lifecycle_ownership_errors,
                    (unsigned long long) mbsd_lifecycle_state_errors,
                    (unsigned long long) mbsd_lifecycle_mask_state_errors,
                    (unsigned long long) mbsd_lifecycle_confidence_errors,
                    (unsigned long long) mbsd_lifecycle_illegal_transitions,
                    (unsigned long long) mbsd_lifecycle_stable_mutations,
                    (unsigned long long) mbsd_lifecycle_version_errors,
                    (unsigned long long) mbsd_lifecycle_future_commits,
                    (unsigned long long) mbsd_lifecycle_future_cache_writes,
                    lifecycle_drafts_pending,
                    mbsd_lifecycle_pending_entries,
                    (unsigned long long) mbsd_lifecycle_refresh_enqueued,
                    (unsigned long long) mbsd_lifecycle_refresh_drained,
                    (unsigned long long) mbsd_lifecycle_refresh_duplicates,
                    (unsigned long long) mbsd_lifecycle_refresh_ineligible,
                    (unsigned long long) mbsd_lifecycle_refresh_future,
                    (unsigned long long) mbsd_lifecycle_refresh_stable,
                    (unsigned long long) mbsd_lifecycle_last_refresh_updates,
                    (unsigned long long) mbsd_lifecycle_last_refresh_errors,
                    (unsigned long long) mbsd_lifecycle_snapshots,
                    iterations_completed + 2,
                    cache_perf.calls, cache_perf.completed, full_sequence_pre_forward_clears,
                    (unsigned long long) mbsd_main_rows_saved, (long long) mbsd_net_rows_saved);
            generation_failed = true;
        }

        if (mbsd_logical_kv_enabled) {
            const uint64_t expected_boundaries =
                (uint64_t) iterations_completed + (uint64_t) num_blocks + 1;
            const bool logical_kv_invariants_ok =
                mbsd_logical_kv.hard_errors() == 0 && !mbsd_logical_kv.boundary_open &&
                mbsd_logical_kv.pending_entries == 0 && mbsd_logical_kv.pending.empty() &&
                mbsd_logical_kv.stable_entries == mbsd_lifecycle_stable &&
                mbsd_logical_kv.mutable_entries == mbsd_lifecycle_visible &&
                mbsd_logical_kv.stable_entries + mbsd_logical_kv.mutable_entries == generated_tokens &&
                mbsd_logical_kv.step_local_current == 0 && mbsd_logical_kv.step_local_future == 0 &&
                mbsd_logical_kv.stable_inserts == (uint64_t) mbsd_logical_kv.stable_entries &&
                mbsd_logical_kv.mutable_inserts ==
                    mbsd_logical_kv.mutable_to_stable + (uint64_t) mbsd_logical_kv.mutable_entries &&
                mbsd_logical_kv.blocks_committed_count == (uint64_t) num_blocks &&
                mbsd_logical_kv.blocks_sealed_count <= mbsd_logical_kv.blocks_committed_count &&
                mbsd_logical_kv.boundary_transactions == expected_boundaries &&
                mbsd_logical_kv.step_local_rebuilds == (uint64_t) iterations_completed &&
                mbsd_logical_kv.step_local_syncs ==
                    (uint64_t) iterations_completed + (uint64_t) num_blocks - 1 &&
                mbsd_logical_kv.step_local_clears == expected_boundaries &&
                mbsd_logical_kv.stale_updates_dropped == 0 &&
                mbsd_logical_kv.stale_self_test_passed &&
                mbsd_logical_kv.stale_self_test_attempts == 1 &&
                mbsd_logical_kv.stale_self_test_dropped == 1 &&
                mbsd_logical_kv.stale_self_test_mutation_errors == 0 &&
                mbsd_logical_kv.stable_hash != 0 && mbsd_logical_kv.mutable_hash != 0 &&
                mbsd_logical_kv.step_local_hash != 0 && mbsd_logical_kv.transaction_hash != 0 &&
                mbsd_logical_kv.combined_hash != 0 && !diffusion_kv_graph_enabled &&
                cache_perf.calls == 0 && cache_perf.completed == 0 &&
                full_sequence_pre_forward_clears == 0 &&
                main_input_tokens == mbsd_reference_dense_rows &&
                mbsd_main_rows_saved == 0 && mbsd_net_rows_saved == 0;
            if (!logical_kv_invariants_ok) {
                LOG_ERR("%s: MBSD logical KV invariant failed "
                        "(entries stable/mutable/lifecycle = %d/%d/%d/%d, "
                        "step current/future = %d/%d, pending/open = %d/%s, "
                        "hard errors = %llu, blocks committed/sealed/planned = %llu/%llu/%d, "
                        "transactions/rebuilds/syncs/clears/expected = %llu/%llu/%llu/%llu/%llu, "
                        "stale self-test attempts/dropped/mutations/pass = %llu/%llu/%llu/%s, "
                        "physical cache calls/completed/clears = %d/%d/%d, rows dense/main/saved/net = "
                        "%llu/%llu/%llu/%lld)\n",
                        __func__,
                        mbsd_logical_kv.stable_entries, mbsd_logical_kv.mutable_entries,
                        mbsd_lifecycle_stable, mbsd_lifecycle_visible,
                        mbsd_logical_kv.step_local_current, mbsd_logical_kv.step_local_future,
                        mbsd_logical_kv.pending_entries,
                        mbsd_logical_kv.boundary_open ? "true" : "false",
                        (unsigned long long) mbsd_logical_kv.hard_errors(),
                        (unsigned long long) mbsd_logical_kv.blocks_committed_count,
                        (unsigned long long) mbsd_logical_kv.blocks_sealed_count,
                        num_blocks,
                        (unsigned long long) mbsd_logical_kv.boundary_transactions,
                        (unsigned long long) mbsd_logical_kv.step_local_rebuilds,
                        (unsigned long long) mbsd_logical_kv.step_local_syncs,
                        (unsigned long long) mbsd_logical_kv.step_local_clears,
                        (unsigned long long) expected_boundaries,
                        (unsigned long long) mbsd_logical_kv.stale_self_test_attempts,
                        (unsigned long long) mbsd_logical_kv.stale_self_test_dropped,
                        (unsigned long long) mbsd_logical_kv.stale_self_test_mutation_errors,
                        mbsd_logical_kv.stale_self_test_passed ? "true" : "false",
                        cache_perf.calls, cache_perf.completed, full_sequence_pre_forward_clears,
                        (unsigned long long) mbsd_reference_dense_rows,
                        (unsigned long long) main_input_tokens,
                        (unsigned long long) mbsd_main_rows_saved,
                        (long long) mbsd_net_rows_saved);
                generation_failed = true;
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
            LOG_INF("  MBSD lifecycle: enabled = %s, observer only = %s, staged integration = %s, "
                    "visibility threshold = %.3f, stability threshold = %.3f\n",
                    mbsd_lifecycle_enabled ? "true" : "false",
                    mbsd_lifecycle_enabled ? "true" : "false",
                    mbsd_staged_lifecycle ? "true" : "false",
                    params.visibility_threshold, params.stability_threshold);
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
                    "post-EOG corrections = %llu, trajectory hash = %llu, commitment hash = %llu\n",
                    (unsigned long long) mbsd_future_semantic_commits,
                    prefix_kv_enabled ? "false" : "true",
                    (unsigned long long) mbsd_block_order_violations,
                    (unsigned long long) mbsd_verification_input_errors,
                    (unsigned long long) mbsd_bounds_errors,
                    (unsigned long long) mbsd_physical_mapping_errors,
                    (unsigned long long) mbsd_post_eog_corrections,
                    (unsigned long long) mbsd_trajectory_hash,
                    (unsigned long long) mbsd_commitment_hash);
            if (mbsd_lifecycle_enabled) {
                const int32_t lifecycle_total =
                    mbsd_lifecycle_invisible + mbsd_lifecycle_visible + mbsd_lifecycle_stable;
                const int32_t residency_total =
                    mbsd_lifecycle_residency_none + mbsd_lifecycle_residency_mutable +
                    mbsd_lifecycle_residency_stable;
                LOG_INF("  MBSD lifecycle states: invisible = %d, visible = %d, stable = %d, total = %d, "
                        "remaining masks = %d, mask parity errors = %llu\n",
                        mbsd_lifecycle_invisible, mbsd_lifecycle_visible,
                        mbsd_lifecycle_stable, lifecycle_total, output_masks_remaining,
                        (unsigned long long) mbsd_lifecycle_mask_state_errors);
                LOG_INF("  MBSD lifecycle transitions: invisible to visible = %llu, "
                        "invisible to stable = %llu, visible to stable = %llu, forced visible = %llu, "
                        "deferred stable = %llu, illegal transitions = %llu\n",
                        (unsigned long long) mbsd_lifecycle_invisible_to_visible,
                        (unsigned long long) mbsd_lifecycle_invisible_to_stable,
                        (unsigned long long) mbsd_lifecycle_visible_to_stable,
                        (unsigned long long) mbsd_lifecycle_forced_visible,
                        (unsigned long long) mbsd_lifecycle_deferred_stable,
                        (unsigned long long) mbsd_lifecycle_illegal_transitions);
                LOG_INF("  MBSD lifecycle confidence: current updates = %llu, draft updates = %llu, "
                        "invalidations = %llu, errors = %llu\n",
                        (unsigned long long) mbsd_lifecycle_confidence_updates,
                        (unsigned long long) mbsd_lifecycle_draft_conf_updates,
                        (unsigned long long) mbsd_lifecycle_conf_invalidations,
                        (unsigned long long) mbsd_lifecycle_confidence_errors);
                LOG_INF("  MBSD lifecycle residency: none = %d, mutable = %d, long-term stable = %d, "
                        "total = %d, duplicate ownership = %llu, ownership errors = %llu\n",
                        mbsd_lifecycle_residency_none, mbsd_lifecycle_residency_mutable,
                        mbsd_lifecycle_residency_stable, residency_total,
                        (unsigned long long) mbsd_lifecycle_duplicate_ownership,
                        (unsigned long long) mbsd_lifecycle_ownership_errors);
                LOG_INF("  MBSD lifecycle invariants: stable mutations = %llu, version errors = %llu, "
                        "future semantic commits = %llu, future cache writes = %llu, pending entries = %d, "
                        "state accounting errors = %llu\n",
                        (unsigned long long) mbsd_lifecycle_stable_mutations,
                        (unsigned long long) mbsd_lifecycle_version_errors,
                        (unsigned long long) mbsd_lifecycle_future_commits,
                        (unsigned long long) mbsd_lifecycle_future_cache_writes,
                        mbsd_lifecycle_pending_entries,
                        (unsigned long long) mbsd_lifecycle_state_errors);
                LOG_INF("  MBSD lifecycle refresh queue: enqueued = %llu, observer drained = %llu, "
                        "max depth = %d, snapshots with pending = %llu, duplicate errors = %llu, "
                        "ineligible errors = %llu, future errors = %llu, stable errors = %llu\n",
                        (unsigned long long) mbsd_lifecycle_refresh_enqueued,
                        (unsigned long long) mbsd_lifecycle_refresh_drained,
                        mbsd_lifecycle_refresh_max_depth,
                        (unsigned long long) mbsd_lifecycle_snapshots_pending,
                        (unsigned long long) mbsd_lifecycle_refresh_duplicates,
                        (unsigned long long) mbsd_lifecycle_refresh_ineligible,
                        (unsigned long long) mbsd_lifecycle_refresh_future,
                        (unsigned long long) mbsd_lifecycle_refresh_stable);
                LOG_INF("  MBSD lifecycle refresh state: last refresh updates = %llu, "
                        "last refresh errors = %llu, queue hash = %llu\n",
                        (unsigned long long) mbsd_lifecycle_last_refresh_updates,
                        (unsigned long long) mbsd_lifecycle_last_refresh_errors,
                        (unsigned long long) mbsd_lifecycle_refresh_hash);
                LOG_INF("  MBSD lifecycle hashes: state hash = %llu, ownership hash = %llu, "
                        "snapshots = %llu\n",
                        (unsigned long long) mbsd_lifecycle_state_hash,
                        (unsigned long long) mbsd_lifecycle_ownership_hash,
                        (unsigned long long) mbsd_lifecycle_snapshots);
                LOG_INF("  MBSD lifecycle actual: cache reads = 0, cache writes = 0, refreshes = 0, "
                        "merges = 0, async tasks = 0, row saving = %llu\n",
                        (unsigned long long) mbsd_main_rows_saved);
                if (mbsd_logical_kv_enabled) {
                    LOG_INF("  MBSD logical KV: enabled = true, transactional = true, "
                            "observer only = true, physical cache active = false\n");
                    LOG_INF("  MBSD logical KV entries: stable = %d, mutable = %d, "
                            "step current final = %d, step future final = %d, "
                            "step current max = %d, step future max = %d\n",
                            mbsd_logical_kv.stable_entries,
                            mbsd_logical_kv.mutable_entries,
                            mbsd_logical_kv.step_local_current,
                            mbsd_logical_kv.step_local_future,
                            mbsd_logical_kv.step_local_current_max,
                            mbsd_logical_kv.step_local_future_max);
                    LOG_INF("  MBSD logical KV updates: stable inserts = %llu, mutable inserts = %llu, "
                            "mutable replacements = %llu, mutable to stable = %llu, "
                            "mutable prefix crossings = %llu, stable carryover checks = %llu, "
                            "boundary transactions = %llu, step rebuilds = %llu, step syncs = %llu, "
                            "step clears = %llu\n",
                            (unsigned long long) mbsd_logical_kv.stable_inserts,
                            (unsigned long long) mbsd_logical_kv.mutable_inserts,
                            (unsigned long long) mbsd_logical_kv.mutable_replacements,
                            (unsigned long long) mbsd_logical_kv.mutable_to_stable,
                            (unsigned long long) mbsd_logical_kv.mutable_prefix_crossings,
                            (unsigned long long) mbsd_logical_kv.stable_carryover_checks,
                            (unsigned long long) mbsd_logical_kv.boundary_transactions,
                            (unsigned long long) mbsd_logical_kv.step_local_rebuilds,
                            (unsigned long long) mbsd_logical_kv.step_local_syncs,
                            (unsigned long long) mbsd_logical_kv.step_local_clears);
                    LOG_INF("  MBSD logical KV blocks: committed = %llu, sealed = %llu, "
                            "seal refused visible = %llu, seal refused invisible = %llu, "
                            "seal refused cache = %llu, illegal seals = %llu, commitment hash = %llu\n",
                            (unsigned long long) mbsd_logical_kv.blocks_committed_count,
                            (unsigned long long) mbsd_logical_kv.blocks_sealed_count,
                            (unsigned long long) mbsd_logical_kv.seal_refused_visible,
                            (unsigned long long) mbsd_logical_kv.seal_refused_invisible,
                            (unsigned long long) mbsd_logical_kv.seal_refused_cache,
                            (unsigned long long) mbsd_logical_kv.illegal_block_seals,
                            (unsigned long long) mbsd_commitment_hash);
                    LOG_INF("  MBSD logical KV invariants: mid-step mutations = %llu, "
                            "stable mutations = %llu, token version mismatches = %llu, "
                            "cache version mismatches = %llu, duplicate durable ownership = %llu, "
                            "future durable inserts = %llu, state membership errors = %llu, "
                            "pending entries = %d, boundary open = %s\n",
                            (unsigned long long) mbsd_logical_kv.mid_step_mutation_errors,
                            (unsigned long long) mbsd_logical_kv.stable_mutation_errors,
                            (unsigned long long) mbsd_logical_kv.token_version_mismatches,
                            (unsigned long long) mbsd_logical_kv.cache_version_mismatches,
                            (unsigned long long) mbsd_logical_kv.duplicate_durable,
                            (unsigned long long) mbsd_logical_kv.future_durable_inserts,
                            (unsigned long long) mbsd_logical_kv.state_membership_errors,
                            mbsd_logical_kv.pending_entries,
                            mbsd_logical_kv.boundary_open ? "true" : "false");
                    LOG_INF("  MBSD logical KV stale test: attempts = %llu, dropped = %llu, "
                            "mutation errors = %llu, passed = %s, production drops = %llu\n",
                            (unsigned long long) mbsd_logical_kv.stale_self_test_attempts,
                            (unsigned long long) mbsd_logical_kv.stale_self_test_dropped,
                            (unsigned long long) mbsd_logical_kv.stale_self_test_mutation_errors,
                            mbsd_logical_kv.stale_self_test_passed ? "true" : "false",
                            (unsigned long long) mbsd_logical_kv.stale_updates_dropped);
                    LOG_INF("  MBSD logical KV hashes: stable trajectory = %llu, "
                            "mutable trajectory = %llu, step-local trajectory = %llu, "
                            "transaction = %llu, combined = %llu\n",
                            (unsigned long long) mbsd_logical_kv.stable_hash,
                            (unsigned long long) mbsd_logical_kv.mutable_hash,
                            (unsigned long long) mbsd_logical_kv.step_local_hash,
                            (unsigned long long) mbsd_logical_kv.transaction_hash,
                            (unsigned long long) mbsd_logical_kv.combined_hash);
                    LOG_INF("  MBSD logical KV actual: cache reads = 0, cache writes = 0, refreshes = 0, "
                            "merges = 0, async tasks = 0, row saving = %llu\n",
                            (unsigned long long) mbsd_main_rows_saved);
                }
            }
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
