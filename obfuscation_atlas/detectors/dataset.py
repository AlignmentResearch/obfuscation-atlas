import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
from typing import Mapping

import numpy as np
import torch

from obfuscation_atlas.detectors.probe_archs import (
    Probe,
)
from obfuscation_atlas.utils.masking import (
    compute_mask,
)

DETECTOR_TYPE = Mapping[int, Probe]


class DictTensorDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.ndim in (2, 3), (
            "Features must be 2D (num_samples, hidden_dim) or 3D (num_samples, seq, hidden_dim)"
        )
        assert labels.ndim == 1, "Labels must be 1D"
        assert features.shape[0] == labels.shape[0], (
            f"Features and labels must have the same number of samples: {features.shape[0]} != {labels.shape[0]}"
        )
        # Ensure features are 3D with a sequence dimension
        if features.ndim == 2:
            features = features.unsqueeze(1)
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def get_feature_dim(self) -> int:
        return self.features.shape[-1]

    def set_obfuscation(
        self,
        obfuscate_over: str,
        preserve_seq_len: bool = True,
        balance_dataset: bool = False,
    ):
        pass

    def __getitem__(self, idx) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return {"0": self.features[idx]}, self.labels[idx]


class ActivationDataset(DictTensorDataset):
    """Dataset using MemoryMappedTensor for efficient random access.

    Uses labels tensor to determine positive/negative samples,
    avoiding tensor concatenation which would materialize memory-mapped tensors.
    """

    def __init__(
        self,
        tokenizer,
        activations: Mapping[str, torch.Tensor],
        tokens_and_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        obfuscate_over: str,
        preserve_seq_len: bool = True,
        shuffle: bool = False,
        balance_dataset: bool = False,
        seed: int = 42,
        example_types: torch.Tensor | None = None,
    ):
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = seed
        self.obfuscate_over = obfuscate_over
        self.preserve_seq_len = preserve_seq_len
        self.balance_dataset = balance_dataset
        self.labels = labels
        self.example_types = example_types
        self.samples: list = []

        self.activations = activations
        self.tokens, self.prompt_mask, self.completion_mask = tokens_and_masks
        self.metadata = {
            "num_samples": self.tokens.shape[0],
            "max_seq_len": self.tokens.shape[1],
            "layers": list(activations.keys()),
        }
        self.metadata["hidden_dim"] = activations[list(activations.keys())[0]].shape[-1]

        # Setup samples with shuffling
        self.set_obfuscation(obfuscate_over, preserve_seq_len, balance_dataset)
        assert len(self.samples) > 0, "No samples found"

    def get_feature_dim(self) -> int:
        return self.metadata["hidden_dim"]

    def get_example_types(self) -> np.ndarray | None:
        """Return example_types for all samples in current order, or None if not available.

        We use example_types to provide additional information about how this example was generated using a
        deceptive/truthful dataset when combined with multiple follow-up prompts.
        See obfuscation_atlas.utils.example_types for more details.
        """
        if self.example_types is None:
            return None
        # Return example_types in the order of self.samples (respects shuffling/balancing)
        indices = [s[0] if isinstance(s[0], int) else s[0][0] for s in self.samples]
        return self.example_types[indices].numpy()

    def _setup_samples(self, shuffle: bool, seed: int, balance_dataset: bool = False):
        """Create sample index list using labels for lazy loading."""
        samples: dict[str, list] = {"honest": [], "deceptive": []}
        assert self.obfuscate_over is not None, "Obfuscate over must be set"
        # Compute mask on the data
        self.mask = compute_mask(
            self.tokens.shape[1],
            self.prompt_mask,
            self.completion_mask,
            self.obfuscate_over,
            ignore_last_token=False,
        )

        # Build samples directly from labels
        for i, label in enumerate(self.labels.tolist()):
            if self.preserve_seq_len:
                samples["deceptive" if label == 1 else "honest"].append((i, label))
            else:
                for j in self.mask[i].nonzero(as_tuple=True)[0]:
                    samples["deceptive" if label == 1 else "honest"].append(((i, j.item()), label))

        # Apply balancing if needed
        if balance_dataset and len(samples["deceptive"]) != len(samples["honest"]):
            min_count = min(len(samples["deceptive"]), len(samples["honest"]))
            if min_count == 0:
                print(
                    f"Warning: balance_dataset=True but one class is empty "
                    f"(deceptive={len(samples['deceptive'])}, honest={len(samples['honest'])}). "
                    f"Skipping balancing."
                )
            else:
                print(f"Downsampling deceptive from {len(samples['deceptive'])} to {min_count} samples")
                print(f"Downsampling honest from {len(samples['honest'])} to {min_count} samples")
                samples["deceptive"] = samples["deceptive"][:min_count]
                samples["honest"] = samples["honest"][:min_count]

        self.samples = samples["deceptive"] + samples["honest"]

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.samples)

    def set_obfuscation(
        self,
        obfuscate_over: str,
        preserve_seq_len: bool = True,
        balance_dataset: bool = False,
    ):
        """Change obfuscation settings dynamically."""
        self.obfuscate_over = obfuscate_over
        self.preserve_seq_len = preserve_seq_len
        self._setup_samples(self.shuffle, self.seed, balance_dataset)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:  # type: ignore
        """Get a sample with direct memory-mapped access.

        Returns:
            A tuple of (activations, label, example_type) where example_type is -1 if not available.
        """
        sample_idx, label = self.samples[idx]
        # Get activations for each layer
        activations = {}
        for layer in self.metadata["layers"]:
            # Use the layer key as-is (could be int or str depending on source)
            layer_key = layer if layer in self.activations else str(layer)
            layer_tensor = self.activations[layer_key]

            # Handle store_last_token_only case: activations have seq_len=1 but
            # sample_idx may be (i, j) tuple with j > 0 from original positions.
            # In this case, use position 0 since that's where the last token is stored.
            is_last_token_only = layer_tensor.shape[1] == 1
            if isinstance(sample_idx, tuple) and is_last_token_only:
                example_idx, _ = sample_idx
                sample_idx_adjusted = (example_idx, 0)
                layer_acts = layer_tensor[sample_idx_adjusted]
            else:
                layer_acts = layer_tensor[sample_idx]

            # When preserve_seq_len=True, apply mask to select valid tokens.
            # But skip this when store_last_token_only since there's only 1 token.
            if self.preserve_seq_len and not is_last_token_only:
                activations[str(layer)] = layer_acts[self.mask[sample_idx]]
            else:
                activations[str(layer)] = layer_acts
        label = torch.tensor(label, dtype=torch.int32)
        # Return example_type if available, otherwise -1
        if self.example_types is not None:
            # Handle both preserve_seq_len=True (sample_idx is int) and
            # preserve_seq_len=False (sample_idx is (i, j) tuple)
            example_idx = sample_idx if isinstance(sample_idx, int) else sample_idx[0]
            example_type = self.example_types[example_idx]
        else:
            example_type = torch.tensor(-1, dtype=torch.int32)
        return activations, label, example_type
