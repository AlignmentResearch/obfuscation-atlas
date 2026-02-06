# Obfuscation Atlas

Testing immunity to deception probes with reinforcement learning.

## Setup

```bash
make install
```

## Project Structure

```
obfuscation_atlas/      # Main package
  config.py             # Hydra-based experiment configuration
  training/             # Training loops (GRPO, SFT, obfuscation loss)
  detectors/            # Probe/detector training and evaluation
  tasks/                # Task definitions (emergent misalignment, code gen, etc.)
  utils/                # Shared utilities
  scripts/              # Entry points for experiments
third_party/
  afterburner/          # FAR AI LLM training framework (static copy)
tests/                  # Test suite
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
