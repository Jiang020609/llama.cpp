# Diffusion Text Generation

This directory contains implementations for Diffusion LLMs (DLLMs)

More Info:
- https://github.com/ggml-org/llama.cpp/pull/14644
- https://github.com/ggml-org/llama.cpp/pull/14771

## Parameters
The diffusion CLI supports various parameters to control the generation process:

### Core Diffusion Parameters
- `--diffusion-steps`: Number of diffusion steps (default: 256)
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
- `--diffusion-early-commit-threshold`: Experimental confidence threshold for committing additional block tokens above the threshold. A negative value disables it (default: -1).

Early commit requires block scheduling, confidence-based selection (`--diffusion-algorithm 4`), and deterministic position selection (`--diffusion-alg-temp 0`). The fixed block transfer count remains the minimum number of selected tokens. Tokens above the threshold are selected in addition to that minimum, and the final block step excludes the mask token and commits all remaining positions.

This experiment can reduce the number of diffusion forwards by completing a block before all of its scheduled steps are used. It does not shorten an individual transformer forward and does not implement prefix KV caching or progressive revision.

The current block scheduler still runs every transformer forward over the full maximum sequence, including future masked blocks. The maximum diffusion length must be divisible by the block length, diffusion steps must be divisible by the scheduled block count, and early commit currently requires the tokenized prompt to be shorter than one block.

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

The script rotates the run order across three legacy, control, and threshold runs, stores full logs under `/tmp`, and reports median generation time and main forward count. The control enables final-step completion with a threshold of 1.0, which cannot add threshold selections; compare the threshold arm against this control to isolate early commitment. Override settings with environment variables, for example:

```
THRESHOLD=0.85 REPEATS=5 MODEL_PATH=/path/to/model.gguf bash examples/diffusion/bench-early-commit.sh
```

Generated text may differ because early token commitment changes later denoising context. Inspect every saved log for unresolved mask tokens and output quality before treating a latency reduction as useful.
