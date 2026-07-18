# Diffusion Text Generation

This directory contains implementations for Diffusion LLMs (DLLMs)

More Info:
- https://github.com/ggml-org/llama.cpp/pull/14644
- https://github.com/ggml-org/llama.cpp/pull/14771

## Parameters
The diffusion CLI supports various parameters to control the generation process:

### Core Diffusion Parameters
- `--diffusion-steps`: Number of diffusion steps (default: 128)
- `--diffusion-algorithm`: Algorithm for token selection
  - `0`: DIFFUSION_ALGORITHM_ORIGIN - Token will be generated in a purely random order from https://arxiv.org/abs/2107.03006.
  - `1`: DIFFUSION_ALGORITHM_ENTROPY_BASED - Entropy-based selection
  - `2`: DIFFUSION_ALGORITHM_MARGIN_BASED - Margin-based selection
  - `3`: DIFFUSION_ALGORITHM_RANDOM - Random selection
  - `4`: DIFFUSION_ALGORITHM_CONFIDENCE_BASED - Confidence-based selection (default)
  - More documentation here https://github.com/DreamLM/Dream
- `--diffusion-visual`: Enable live visualization during generation

### Scheduling Parameters
Choose one of the following scheduling methods:

**Timestep-based scheduling:**
- `--diffusion-eps`: Epsilon value for timestep scheduling (e.g., 0.001)

**Block-based scheduling:**
- `--diffusion-block-length`: Block size for block-based scheduling (e.g., 32)
- `--diffusion-generated-block-schedule`: Experimental scheduling over generated tokens instead of the full maximum sequence (default: disabled).
- `--diffusion-mbsd`: Experimental no-KV full-sequence Multi-Block Speculative Decoding (MBSD) reference (default: disabled).
- `--diffusion-mbsd-fresh-kv`: MBSD full-sequence correctness oracle using a freshly cleared KV graph (default: disabled).
- `--diffusion-mbsd-compact`: Reserved paper-aligned compact MBSD mode. It currently fails before model loading (default: disabled).
- `--diffusion-mbsd-lifecycle-bookkeeping`: Observer-only token/cache lifecycle bookkeeping for dense MBSD (default: disabled).
- `--diffusion-mbsd-trigger`: Enable proactive lookahead when the number of remaining current-block masks is below this value (default: 4).
- `--diffusion-mbsd-lookahead`: Maximum number of future tokens beyond the fixed-size sliding window (default: 32).
- `--diffusion-early-commit-threshold`: Experimental confidence threshold for committing additional block tokens above the threshold. A negative value disables it (default: -1).
- `--diffusion-prefix-kv`: Experimental Dream block-wise prefix KV reuse (default: disabled).
- `--diffusion-staged-token-stabilization`: Experimental dense reference for staged token stabilization (default: disabled).
- `--diffusion-visibility-threshold`: Confidence threshold for making an invisible token visible (default: 0.7).
- `--diffusion-stability-threshold`: Confidence threshold for making a token stable (default: 0.9).
- `--diffusion-staged-revision-policy`: Visible-token revision policy: `oldest` or `balanced-low-confidence` (default: `oldest`).
- `--diffusion-staged-final-revision-steps`: Maximum revision-only passes after the final block becomes fully visible (default: 0).
- `--diffusion-staged-final-visible-ratio`: Generated-token visible ratio that triggers final revisions (default: 0.10).

Early commit requires block scheduling, confidence-based selection (`--diffusion-algorithm 4`), and deterministic position selection (`--diffusion-alg-temp 0`). The fixed block transfer count remains the minimum number of selected tokens. Tokens above the threshold are selected in addition to that minimum, and the final block step excludes the mask token and commits all remaining positions.

The early-commit experiment can reduce the number of diffusion forwards by completing a block before all of its scheduled steps are used. It does not shorten an individual transformer forward and does not itself enable prefix KV caching or progressive revision.

Without `--diffusion-prefix-kv`, the current block scheduler runs every transformer forward over the full maximum sequence, including future masked blocks. Generated block scheduling alone only changes host-side block and step planning; it does not reduce the work in one transformer forward. Both MBSD reference modes also submit the full sequence.

Without `--diffusion-generated-block-schedule`, the legacy block geometry is unchanged: the maximum diffusion length must be divisible by the block length, diffusion steps must be divisible by the full-sequence block count, and early commit requires the tokenized prompt to be shorter than one block.

With `--diffusion-generated-block-schedule`, blocks cover `max_length - input_tokens`. Diffusion steps are divided as evenly as possible across those non-empty blocks, with earlier blocks receiving one extra step when needed. For example, 95 generated tokens with block length 32 and 32 total steps produce three blocks with 11, 11, and 10 steps. The number of diffusion steps must be at least the number of generated blocks.

MBSD is a semantic reference for Section 3.2 of https://arxiv.org/abs/2606.13740. It separates the block, which remains the left-to-right semantic commitment unit, from a logical decoding window. The base window starts at the first unresolved position in the current block and retains the configured block length as that contiguous commitment frontier moves right. Space released on the left admits positions from later blocks. A future position can hold a draft token and confidence and can participate in later denoising, but it is not a semantic commitment, cannot complete its block, and is never written to the prefix KV cache. Blocks still complete strictly from left to right.

Proactive lookahead is checked after current-block masks are counted. It triggers only when that count is strictly less than `--diffusion-mbsd-trigger`. The fixed-budget policy extends the base sliding window by at most `--diffusion-mbsd-lookahead` positions, clamped to the generated sequence. A trigger of 0 disables proactive expansion because a mask count cannot be negative. A lookahead of 0 disables proactive expansion but does not disable continuous sliding, so sliding alone can still produce future drafts.

The paper chooses future-token count with an offline-profiled NPU latency table and stops when the next latency bucket or the available memory budget would be exceeded. This reference has neither a device latency table nor a memory-budget model. Its deterministic fixed token budget is an explicit prototype policy and must not be interpreted as the paper's hardware-adaptive lookahead.

When a block becomes current, every saved draft in that block is freshly predicted in the first current-block forward. Its input position is restored to the mask token for this verification pass rather than feeding the saved draft back to the model. This implementation gives verification credit equal to the number of saved drafts, ranks all fresh current-block predictions by confidence, and accepts a draft only when its position is within that top-credit set and the fresh token ID exactly matches the saved draft token ID. Accepted drafts are committed before the normal block transfer selection. Rejected drafts are cleared and their fresh predictions remain eligible for the normal transfer schedule. The paper requires fresh re-evaluation and confidence-based acceptance, but does not publish this exact ranking or token-equality rule, so this rule is a v9 prototype choice.

MBSD requires generated-token block scheduling, confidence selection (`--diffusion-algorithm 4`), and deterministic position selection (`--diffusion-alg-temp 0`). It is currently rejected with early commit, explicit prefix KV, the full-sequence KV oracle, staged token stabilization, classifier-free guidance, or Gumbel noise. These combinations fail explicitly rather than silently falling back. The fresh-KV oracle and lifecycle bookkeeping both require `--diffusion-mbsd`. Lifecycle bookkeeping is mutually exclusive with the fresh-KV oracle so its zero-cache-work diagnostics remain literal.

With only `--diffusion-mbsd`, execution remains the v9 no-KV full-sequence reference. Each transformer forward submits `[0, max_length)`. The logical window controls requested logits and future drafts but does not remove transformer input rows. This mode performs no KV clear, prompt prefill, block seal, cache trim, or prefix reuse.

`--diffusion-mbsd-lifecycle-bookkeeping` adds a host-only observer to that same dense execution. It reuses the staged-token `invisible`, `visible`, and `stable` state vocabulary without enabling the staged revision path. Each generated position has one static owner block and one dynamic `prefix`, `current`, or `future-draft` role. A current-block semantic commit moves from invisible to visible and receives matching planned token and KV versions. Block completion moves visible positions to stable. Positions after the first EOG remain visible until final EOG-tail canonicalization, then receive any visible-token version update before becoming stable. A future draft remains invisible, has no semantic version, and has no planned cache residency.

Residency is a plan only: invisible maps to none, visible maps to mutable, and stable maps to long-term-stable. Entering stable freezes the semantic token, token version, and planned KV version; later snapshots reject any change to those values. The observer never supplies a token, KV entry, position, logit row, or batch row to model execution. It performs no cache read, write, refresh, merge, physical compaction, or asynchronous task. Diagnostics therefore require identical generated token IDs, ID hash, MBSD trajectory hash, main-forward count, transformer rows, and logit rows relative to the same dense MBSD run, with zero actual row saving. State and ownership hashes cover every lifecycle snapshot and must be deterministic.

With `--diffusion-mbsd-fresh-kv`, Dream uses the diffusion KV graph as a correctness oracle. It synchronizes and clears KV before every main forward, then submits `[0, max_length)`. The prompt, completed blocks, logical MBSD window, and masked right tail are recomputed together. It does not prefill the prompt, seal completed blocks, trim a cache tail, or reuse completed-prefix rows. Diagnostics require full-size batches, one pre-forward KV clear per main forward, zero prefix rows reused, and zero physical row savings. This behavior is the same for single-block and multi-block runs.

`--diffusion-mbsd-compact` is reserved for the next paper-alignment stage. It is rejected during parameter validation, before model loading, until stable prefix KV, mutable visible KV, sparse refresh, and step-boundary merge are implemented. There is no single-block fallback and no silent substitution with either reference mode.

The original block-boundary compact implementation omitted the far-future masked tail and reused prompt or completed-block KV. Dream uses bidirectional non-causal attention, so both transformations changed model semantics. That physical row-reduction path is disabled until it can be replaced by a correct sparse KV refresh and step-boundary merge. Compare deterministic generated token IDs and trajectory hashes against the full-sequence MBSD reference before reintroducing any row reduction. This is not an Apple Neural Engine implementation or the paper's hardware-adaptive NPU latency-bucket policy.

The no-KV MBSD reference and fresh-KV correctness oracle are diagnostics, not the paper's final execution path. Neither mode implements stable prefix KV, mutable visible KV, sparse refresh, delayed cache merge, or the asynchronous CPU/NPU dual path. No physical compact transformer execution is active in this stage.

Future draft work can change later denoising context and generated text. The paper's quality ablation reports lower accuracy for MBSD alone on all four evaluated datasets, with dual-path progressive revision recovering much of the loss. Since this reference does not combine MBSD with staged stabilization or the paper's CPU/NPU revision path, saved forwards are not by themselves evidence of preserved quality. Validate generated outputs across representative prompts and seeds before treating a performance change as useful.

Prefix KV is a first-stage implementation of the block-wise cached-prefix baseline described in https://arxiv.org/abs/2606.13740. It prefills the prompt, repeatedly decodes only the current block plus a boundary token when shifted logits are enabled, and seals each intermediate completed block before reusing it as prefix context. This changes attention semantics relative to full-sequence bidirectional denoising and may change or reduce output quality. It is not the paper's multi-block speculative decoding, progressive revision, or NPU memory runtime.

The current prefix-KV experiment supports Dream, generated-token block scheduling, confidence selection, and `--diffusion-alg-temp 0`. Early commit, classifier-free guidance, and Gumbel noise are rejected while the cache path is enabled.

Staged token stabilization is a semantic reference for the three-state mechanism in Section 3.3 of https://arxiv.org/abs/2606.13740. Generated positions begin invisible. Positions at or above the visibility threshold become visible and can be revised in later denoising steps. Positions at or above the stability threshold become stable and stop requesting logits. If no invisible position reaches the visibility threshold, the most confident one becomes visible to guarantee progress. Visible positions remain active across block boundaries.

This reference revises one visible position per step through a separate target-masked full-sequence forward. The default `oldest` policy selects never-revised positions first, then the least recently revised position, with position order as the tie breaker. The optional `balanced-low-confidence` policy sorts ordinary revisions by revision count, latest observed confidence, last revision step, and position. Revision count is the first key so that one persistently low-confidence token cannot starve all other visible tokens. Within a final sweep, each remaining visible position is eligible once and `balanced-low-confidence` sorts those positions by latest observed confidence, last revision step, and position. Latest observed confidence is the result of the position's most recent masked prediction; it is not a live score under the current context. Other visible positions remain concrete context, and the revised token is applied at the step boundary.

Final revision-only passes are disabled by default. When enabled, they run only after the final block has no invisible positions, only while the visible-token ratio over the full generated region reaches the configured threshold, and only within the final block's unused scheduled step slots. They do not run an empty main forward. Each sweep revises every remaining visible position at most once. The pass stops when the ratio falls below the threshold. If a complete deterministic sweep changes no token IDs, it also stops at a fixed point. The ratio is a coarse full-generated-region heuristic; it is not EOG-aware.

The paper does not specify the one-position candidate ordering, a final revision budget, or a visible-ratio trigger. Both revision policies and the final sweep are explicit prototype choices. Metrics report ordinary and final revision forwards separately, the before/after visible counts, sweep count, stop reason, revision-count distribution, latest-confidence distribution, and deterministic trajectory and target hashes.

This mode is not the paper's asynchronous CPU/NPU dual path, sparse KV update, or delayed cache merge, and it does not enable prefix KV reuse. Stable positions skip output decisions, but all positions still participate in the dense transformer. The paper does not define its confidence formula; this reference selects the highest-logit non-mask token and uses its softmax probability over the non-mask vocabulary. It requires generated-token block scheduling, confidence selection, `--diffusion-alg-temp 0`, `--temp 0`, and one scheduled step per generated token. Top-k and top-p do not affect this reference path. Early commit, prefix KV, classifier-free guidance, and Gumbel noise are rejected. The Dream full-sequence KV oracle can be enabled independently to compare the KV and no-cache graphs.

### Diagnostics

- `--diffusion-full-sequence-kv-oracle`: Run the Dream KV graph on a freshly cleared full sequence before every transformer forward (default: disabled).
- `--diffusion-dump-generated-tokens`: Log generated token IDs and EOG/control-token counts (default: disabled).

The full-sequence KV oracle is a parity diagnostic, not an optimization. It uses the Dream KV graph, clears the cache before each transformer forward, and submits the complete maximum sequence. It does not prefill, seal, or reuse completed blocks, and it is mutually exclusive with `--diffusion-prefix-kv`. Compare its generated token IDs with the default no-cache path under identical arguments to separate KV-graph errors from prefix-reuse errors. Use `-fa off -ctk f32 -ctv f32` for the first parity check to reduce numerical differences from attention kernels and KV-cache precision.

### Sampling Parameters
- `--temp`: Temperature for sampling (0.0 = greedy/deterministic, higher = more random)
- `--top-k`: Top-k filtering for sampling
- `--top-p`: Top-p (nucleus) filtering for sampling
- `--seed`: Random seed for reproducibility

### Model Parameters
- `-m`: Path to the GGUF model file
- `-p`: Input prompt text
- `-ub`: Maximum sequence length (ubatch size)
- `-c`: Context size
- `-b`: Batch size

### Examples
#### Dream architecture:
```
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" -ub 512 --diffusion-eps 0.001 --diffusion-algorithm 3 --diffusion-steps 256 --diffusion-visual
```

#### Dream block-wise prefix KV experiment:
```
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" -c 128 -b 128 -ub 128 --diffusion-block-length 32 --diffusion-generated-block-schedule --diffusion-prefix-kv --diffusion-algorithm 4 --diffusion-alg-temp 0 --diffusion-steps 32
```

#### Dream MBSD fixed-budget reference:
```
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" -c 128 -b 128 -ub 128 --temp 0 --diffusion-block-length 32 --diffusion-generated-block-schedule --diffusion-mbsd --diffusion-mbsd-trigger 4 --diffusion-mbsd-lookahead 32 --diffusion-algorithm 4 --diffusion-alg-temp 0 --diffusion-steps 32
```

#### Dream MBSD fresh-KV correctness oracle:
```
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" -c 128 -b 128 -ub 128 --temp 0 --diffusion-block-length 32 --diffusion-generated-block-schedule --diffusion-mbsd --diffusion-mbsd-fresh-kv --diffusion-mbsd-trigger 4 --diffusion-mbsd-lookahead 32 --diffusion-algorithm 4 --diffusion-alg-temp 0 --diffusion-steps 32
```

#### Dream MBSD lifecycle bookkeeping:
```
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" -c 128 -b 128 -ub 128 --temp 0 --diffusion-block-length 32 --diffusion-generated-block-schedule --diffusion-mbsd --diffusion-mbsd-lifecycle-bookkeeping --diffusion-mbsd-trigger 4 --diffusion-mbsd-lookahead 32 --diffusion-dump-generated-tokens --diffusion-algorithm 4 --diffusion-alg-temp 0 --diffusion-steps 32
```

#### Dream full-sequence KV parity diagnostic:
```
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" -c 128 -b 128 -ub 128 -fa off -ctk f32 -ctv f32 --diffusion-block-length 32 --diffusion-generated-block-schedule --diffusion-full-sequence-kv-oracle --diffusion-dump-generated-tokens --diffusion-algorithm 4 --diffusion-alg-temp 0 --diffusion-steps 32
```

#### Staged token stabilization reference:
```
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" -c 128 -b 128 -ub 128 --temp 0 --diffusion-block-length 32 --diffusion-generated-block-schedule --diffusion-staged-token-stabilization --diffusion-visibility-threshold 0.7 --diffusion-stability-threshold 0.9 --diffusion-dump-generated-tokens --diffusion-algorithm 4 --diffusion-alg-temp 0 --diffusion-steps GENERATED_TOKEN_COUNT
```

#### Balanced revision with a bounded final sweep:
```
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" -c 192 -b 192 -ub 192 --temp 0 --diffusion-block-length 32 --diffusion-generated-block-schedule --diffusion-staged-token-stabilization --diffusion-visibility-threshold 0.7 --diffusion-stability-threshold 0.9 --diffusion-staged-revision-policy balanced-low-confidence --diffusion-staged-final-revision-steps 8 --diffusion-staged-final-visible-ratio 0.10 --diffusion-dump-generated-tokens --diffusion-algorithm 4 --diffusion-alg-temp 0 --diffusion-steps GENERATED_TOKEN_COUNT
```

`GENERATED_TOKEN_COUNT` is `ubatch size - tokenized prompt length`. The CLI reports both values if they do not match.

For the first Dream parity check, run two otherwise identical staged commands with `-fa off -ctk f32 -ctv f32`. Add `--diffusion-full-sequence-kv-oracle` only to the second command. The generated token IDs, final ID hash, trajectory hash, revision target hash, state counts, transitions, ordinary and final revision counts, final stop reason, and completed block count must match exactly.

#### LLaDA architecture:
```
llama-diffusion-cli -m llada-8b.gguf -p "write code to train MNIST in pytorch" -ub 512 --diffusion-block-length 32 --diffusion-steps 256 --diffusion-visual
```

#### RND1 architecture:
```
llama-diffusion-cli -m RND1-Base-0910.gguf -p "write code to train MNIST in pytorch" -ub 512 --diffusion-algorithm 1 --diffusion-steps 256 --diffusion-visual --temp 0.5 --diffusion-eps 0.001
```

### Early-commit A/B benchmark

From the repository root, build `llama-diffusion-cli` and run:

```
bash examples/diffusion/bench-early-commit.sh
```

The script enables generated block scheduling by default, rotates the run order across three legacy, control, and threshold runs, stores full logs under `/tmp`, and reports median generation time and main forward count. All three arms use the same block scheduler. The `legacy` label means early commit is disabled; it does not select the legacy block geometry. The control enables final-step completion with a threshold of 1.0, which cannot add threshold selections; compare the threshold arm against this control to isolate threshold-triggered extra commitments and their early exits. The default experimental threshold is 0.999. Override settings with environment variables, for example:

```
THRESHOLD=0.999 REPEATS=6 MODEL_PATH=/path/to/model.gguf bash examples/diffusion/bench-early-commit.sh
```

Set `GENERATED_BLOCK_SCHEDULE=0` to exercise the legacy block geometry. Prompts used with that mode and early commit must still tokenize to fewer tokens than the block length.

Generated text may differ because early token commitment changes later denoising context. Repeating one prompt and seed measures timing stability, not quality preservation. Inspect every saved log for unresolved mask tokens, and use multiple prompts and seeds with objective checks before treating a latency reduction as useful.

### MBSD reference benchmark

From the repository root, build `llama-diffusion-cli` and run:

```
MODEL_PATH=/path/to/model.gguf FLASH_ATTN=off REPEATS=2 bash examples/diffusion/bench-mbsd.sh
```

If `MODEL_PATH` is omitted, the script uses the default Hugging Face Dream Q4_K_M model. It defaults both KV cache types to `f32`; override them with `CACHE_TYPE_K` and `CACHE_TYPE_V` only after the first parity check. It runs single-block and multi-block baseline, sliding-only, and proactive-lookahead cases. For both sequence shapes it also runs a fresh-full-sequence KV counterpart for lookahead 32 and requires exact generated token, ID hash, trajectory hash, and main-forward parity with the matching no-KV MBSD reference. Each fresh counterpart must report full-size batches, one pre-forward KV clear per main forward, no prefix reuse or cache maintenance, zero cache invariant errors, and zero physical row savings. The invalid-parameter gate checks that fresh KV requires MBSD, compact requires MBSD, fresh and compact are mutually exclusive, and compact fails before inference. The default test trigger is 8 rather than the CLI default of 4 so that the short test schedule reliably enters the strict `remaining masks < trigger` branch. The script also checks repeated token and trajectory hashes, draft lifecycle accounting, logical row and logit accounting, schedule accounting, unresolved masks, block order, window bounds, and explicit rejection of unsupported flag combinations. Logs and `summary.tsv` are written below `LOG_DIR`, or below the system temporary directory when `LOG_DIR` is unset.

The normal repeated matrix also compares dense lookahead-32 MBSD with lifecycle bookkeeping and requires exact token, trajectory, forward, transformer-row, and logit-row parity. Set `LIFECYCLE_MATRIX=1` for the full lifecycle gate across multiple prompts, three seeds, and both single-block and multi-block shapes. That gate writes `lifecycle-summary.tsv`, requires zero lifecycle/cache/row-saving errors, and repeats a representative bookkeeping run for each shape to verify deterministic state and ownership hashes.

The MBSD path canonicalizes non-terminal tokens after the first generated EOG token to the model pad token, or to the first EOG token when the vocabulary has no usable pad token. The benchmark treats any remaining non-terminal EOG tail token as an error and reports the number of canonicalized tokens in `post_eog_corrections`. It also compares every MBSD token sequence with the matching baseline run and rejects mismatches by default. Set `MBSD_BASELINE_PARITY=report` to record mismatches without treating approximate decoding as an implementation failure.

The benchmark is a semantic and regression gate. Do not interpret its logical row-role counts as reduced Metal transformer work, and do not use latency alone as a quality gate.
