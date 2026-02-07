# The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes

This codebase implements methods for training deception probes (linear and nonlinear detectors on model activations) and running RL experiments that train language models against those probes.

## Installation

Requires Python 3.12.

```bash
git clone https://github.com/AlignmentResearch/obfuscation-atlas.git
cd obfuscation-atlas
make install
```

This installs the `afterburner` training framework from `third_party/` and all Python dependencies into a `.venv` managed by [uv](https://docs.astral.sh/uv/).

## Project Structure

```
obfuscation_atlas/
  config.py             # Hydra-based experiment configuration
  training/             # Training loops (GRPO, SFT, obfuscation loss)
  detectors/            # Probe/detector training and evaluation
  tasks/                # Task definitions (emergent misalignment, code gen, etc.)
  utils/                # Shared utilities
  scripts/              # Experiment entry points
third_party/
  afterburner/          # FAR AI LLM training framework
tests/                  # Test suite
```

## Configuration

All experiments use [Hydra](https://hydra.cc/) with Python dataclasses for configuration. Override any config value from the command line with `++key.subkey=value`.

Key registered configs (defined in `obfuscation_atlas/config.py`):

| Config name | Description |
|---|---|
| `sft_config` | SFT training against probes |
| `grpo_obfuscation_config` | GRPO RL training against probes |
| `grpo_code_gen` | GRPO for code generation task |
| `rm_config` | Reward model training |

## Training Probes

Probes are trained on model activations to detect deceptive behavior. Supported detector types: `linear-probe`, `mean-difference-probe`, `nonlinear-probe`, `quadratic-probe`, `attention-probe`, `transformer-probe`, `gdm-probe`, `mahalanobis`, `vae`, `nflow`, `black-box`.

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from obfuscation_atlas.config import BaseTrainConfig
from obfuscation_atlas.detectors.train import train_detector
from obfuscation_atlas.detector_arch_presets import get_detector_arch_config

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

train_cfg = BaseTrainConfig(batch_size=32, learning_rate=3e-3, max_steps=1000)
detector_arch = get_detector_arch_config("linear-probe")

detector_dict, features, dynamics = train_detector(
    model=model,
    tokenizer=tokenizer,
    train_dataset=(positive_examples, negative_examples),
    train_cfg=train_cfg,
    layers=list(range(model.config.num_hidden_layers)),
    obfuscate_over="last-token-generation",
    detector_arch_config=detector_arch,
)
```

### Evaluating detectors

```python
from obfuscation_atlas.detectors.evaluate import train_and_eval_multiple_detectors

results = train_and_eval_multiple_detectors(
    model=model,
    tokenizer=tokenizer,
    layers=list(range(model.config.num_hidden_layers)),
    train_dataset=(pos_train, neg_train),
    test_dataset=(pos_test, neg_test),
    train_cfg=train_cfg,
    detector_arch_configs=[
        get_detector_arch_config("linear-probe"),
        get_detector_arch_config("nonlinear-probe"),
    ],
    obfuscate_over=["generation", "last-token-generation"],
)
```

## Running RL Experiments

### GRPO training against probes

GRPO (Group Relative Policy Optimization) trains a model via RL while evaluating against deception probes. This requires a vLLM server for generation.

**1. Start the vLLM server** (on a separate GPU):

```bash
CUDA_VISIBLE_DEVICES=1 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1
```

**2. Launch GRPO training:**

```bash
# 8B model
python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
    ++detector.eval_detector_types="['linear-probe']"

# 70B model with FSDP (multi-GPU)
accelerate launch --num_processes 4 --config_file obfuscation_atlas/accelerate_config.yaml \
    -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
    ++model.model_type=meta-llama/Meta-Llama-3-70B-Instruct \
    ++training.batch_size=2
```

**Common GRPO overrides:**

```bash
++model.model_type=meta-llama/Meta-Llama-3-8B-Instruct
++training.batch_size=8
++training.max_steps=3000
++training.learning_rate=1e-4
++loss.kl=0.01
++loss.task=1.0
++loss.detector=1.0
++grpo.batching.num_generations=8
++detector.eval_detector_types="['linear-probe','nonlinear-probe']"
++detector.train_against_types="['linear-probe']"
```

### SFT training against probes

SFT (Supervised Fine-Tuning) provides an alternative to RL for training against probes:

```bash
python -m obfuscation_atlas.scripts.train_sft_against_probe \
    ++detector.eval_detector_types="['linear-probe']" \
    ++data.include_task_negative=False
```

**Common SFT overrides:**

```bash
++model.model_type=meta-llama/Meta-Llama-3-8B-Instruct
++training.batch_size=8
++training.max_steps=3000
++training.learning_rate=3e-4
++loss.kl=1e-4
++loss.detector=1.0
++loss.task=1.0
```

### Reward model training

```bash
python -m obfuscation_atlas.scripts.train_reward_model
```

## Running Tests

```bash
make test          # Fast tests only
make test-all      # All tests including slow/integration
```

## Code Quality

```bash
make lint          # Run ruff checks
make format        # Auto-format with ruff
```

## Citation

<!-- TODO: Add bibtex -->
