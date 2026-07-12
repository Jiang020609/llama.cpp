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
- `--diffusion-early-commit-threshold`: Experimental confidence threshold for committing additional block tokens above the threshold. A negative value disables it (default: -1).
- `--diffusion-prefix-kv`: Experimental Dream block-wise prefix KV reuse (default: disabled).

Early commit requires block scheduling, confidence-based selection (`--diffusion-algorithm 4`), and deterministic position selection (`--diffusion-alg-temp 0`). The fixed block transfer count remains the minimum number of selected tokens. Tokens above the threshold are selected in addition to that minimum, and the final block step excludes the mask token and commits all remaining positions.

The early-commit experiment can reduce the number of diffusion forwards by completing a block before all of its scheduled steps are used. It does not shorten an individual transformer forward and does not itself enable prefix KV caching or progressive revision.

Without `--diffusion-prefix-kv`, the current block scheduler still runs every transformer forward over the full maximum sequence, including future masked blocks. Generated block scheduling only changes host-side block and step planning; it does not reduce the work in one transformer forward.

Without `--diffusion-generated-block-schedule`, the legacy block geometry is unchanged: the maximum diffusion length must be divisible by the block length, diffusion steps must be divisible by the full-sequence block count, and early commit requires the tokenized prompt to be shorter than one block.

With `--diffusion-generated-block-schedule`, blocks cover `max_length - input_tokens`. Diffusion steps are divided as evenly as possible across those non-empty blocks, with earlier blocks receiving one extra step when needed. For example, 95 generated tokens with block length 32 and 32 total steps produce three blocks with 11, 11, and 10 steps. The number of diffusion steps must be at least the number of generated blocks.

Prefix KV is a first-stage implementation of the block-wise cached-prefix baseline described in https://arxiv.org/abs/2606.13740. It prefills the prompt, repeatedly decodes only the current block plus a boundary token when shifted logits are enabled, and seals each intermediate completed block before reusing it as prefix context. This changes attention semantics relative to full-sequence bidirectional denoising and may change or reduce output quality. It is not the paper's multi-block speculative decoding, progressive revision, or NPU memory runtime.

The current prefix-KV experiment supports Dream, generated-token block scheduling, confidence selection, and `--diffusion-alg-temp 0`. Early commit, classifier-free guidance, and Gumbel noise are rejected while the cache path is enabled.

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
