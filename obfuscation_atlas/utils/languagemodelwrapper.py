# type: ignore
import gc
import json
import os

import torch
from datasets import Dataset
from tensordict import MemoryMappedTensor
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm.auto import tqdm

from obfuscation_atlas.utils.activations import (
    get_all_residual_acts_unbatched,
    get_hidden_size,
    get_num_hidden_layers,
)
from obfuscation_atlas.utils.data_processing import (
    process_data,
)
from obfuscation_atlas.utils.gpu_utils import log_gpu_memory
from obfuscation_atlas.utils.masking import (
    get_valid_token_mask,
)


class LanguageModelWrapper:
    # Wrapper class for language models
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def get_model_residual_acts(
        self,
        dataset,
        batch_size=None,
        max_completion_length=None,
        max_sequence_length=512,
        return_tokens=False,
        use_memmap=None,
        only_return_layers=None,
        only_return_on_tokens_between=None,
        verbose=True,
        padding_side="right",
        append_eos_to_targets=True,
        create_rank_dirs=False,
        flush_every_n_batches=None,
        completion_column="completion",
        follow_up_prompts: list[tuple[str, str]] = [],
        labels: torch.Tensor | None = None,
        example_types: torch.Tensor | None = None,
        store_last_token_only: bool = False,
    ):
        assert isinstance(dataset, Dataset), "dataset must be a Dataset"
        self.model.eval()
        log_gpu_memory("Start of get_model_residual_acts")

        device = self.model.device

        # Get distributed info
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Process data
        input_ids, prompt_mask, completion_mask, kept_examples_mask = process_data(
            dataset["prompt"],
            dataset[completion_column],
            self.tokenizer,
            max_completion_length=max_completion_length,
            max_sequence_length=max_sequence_length,
            padding_side=padding_side,
            append_eos_to_targets=append_eos_to_targets,
            follow_up_prompts=follow_up_prompts,
        )
        attention_mask = prompt_mask | completion_mask

        # Handle labels and example_types - filter by kept_examples_mask first, then expand for follow-ups
        tensors = [input_ids, attention_mask, prompt_mask, completion_mask]
        num_follow_ups = len(follow_up_prompts) if follow_up_prompts else 1
        num_base_examples = len(dataset)
        num_expanded_examples = input_ids.shape[0]

        # Expand kept_examples_mask to match the follow-up structure
        # Each base example is repeated num_follow_ups times in the data
        expanded_kept_mask = kept_examples_mask.repeat(num_follow_ups)

        if labels is not None:
            # Check if labels are already expanded
            if labels.shape[0] == num_expanded_examples:
                # Labels are already expanded - apply expanded mask
                labels = labels[expanded_kept_mask]
            elif labels.shape[0] == num_base_examples:
                # Labels are not expanded - filter then expand
                labels = labels[kept_examples_mask]
                if num_follow_ups > 1:
                    labels = labels.repeat(num_follow_ups)
            else:
                raise ValueError(
                    f"Labels shape {labels.shape[0]} doesn't match base examples {num_base_examples} "
                    f"or expanded examples {num_expanded_examples}"
                )
            assert labels.shape[0] == input_ids.shape[0], (
                f"Labels length {labels.shape[0]} != input_ids length {input_ids.shape[0]}"
            )
            tensors.append(labels)

        # Handle example_types similarly to labels
        if example_types is not None:
            # Check if example_types are already expanded
            if example_types.shape[0] == num_expanded_examples:
                # Example types are already expanded - apply expanded mask
                example_types = example_types[expanded_kept_mask]
            elif example_types.shape[0] == num_base_examples:
                # Example types are not expanded - filter then expand
                example_types = example_types[kept_examples_mask]
                if num_follow_ups > 1:
                    example_types = example_types.repeat(num_follow_ups)
            else:
                raise ValueError(
                    f"Example types shape {example_types.shape[0]} doesn't match base examples {num_base_examples} "
                    f"or expanded examples {num_expanded_examples}"
                )
            assert example_types.shape[0] == input_ids.shape[0], (
                f"Example types length {example_types.shape[0]} != input_ids length {input_ids.shape[0]}"
            )
            tensors.append(example_types)

        tensor_dataset = TensorDataset(*tensors)
        # Create distributed sampler and dataloader
        if world_size > 1:
            sampler = DistributedSampler(
                tensor_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,  # Keep order for activation extraction
                drop_last=False,
            )
            dataloader = DataLoader(
                tensor_dataset,
                batch_size=batch_size or 1,
                sampler=sampler,
                pin_memory=True,
                num_workers=0,  # Usually 0 for GPU operations
            )
            num_samples_per_rank = len(sampler)
        else:
            dataloader = DataLoader(tensor_dataset, batch_size=batch_size or 1, shuffle=False, pin_memory=True)
            num_samples_per_rank = len(dataloader.dataset)

        # If batch_size is not specified, process all input at once
        if batch_size is None:
            batch_size = input_ids.size(0)

        num_layers = get_num_hidden_layers(self.model.config)
        hidden_dim = get_hidden_size(self.model.config)
        layers_to_return = list(range(num_layers)) if only_return_layers is None else only_return_layers

        # Calculate this rank's data size
        max_seq_len = input_ids.size(1)
        storage_seq_len = 1 if store_last_token_only else max_seq_len
        tensor_shape = (num_samples_per_rank, storage_seq_len, hidden_dim)

        # Track this rank's tokens, masks, labels, and example_types for distributed case
        rank_input_ids = []
        rank_prompt_masks = []
        rank_completion_masks = []
        rank_labels = [] if labels is not None else None
        rank_example_types = [] if example_types is not None else None

        # Initialize memmaps if a file path is provided
        if use_memmap:
            memmap_dir = use_memmap
            if create_rank_dirs:
                memmap_dir = os.path.join(memmap_dir, f"rank_{rank}")

            os.makedirs(memmap_dir, exist_ok=True)

            # Create memory-mapped tensors for each layer
            layer_mmaps = {}
            mm_filenames = []
            for layer in layers_to_return:
                mm_filename = os.path.join(memmap_dir, f"layer_{layer}.memmap")
                layer_mmaps[layer] = MemoryMappedTensor.empty(
                    shape=tensor_shape,
                    dtype=torch.bfloat16,
                    filename=mm_filename,
                )
                mm_filenames.append(mm_filename)
        else:
            # Pre-allocate the full tensor for required layers
            all_residual_acts = {
                layer: torch.zeros(
                    tensor_shape,
                    dtype=torch.bfloat16,
                    device="cpu",
                )
                for layer in layers_to_return
            }

        batch_idx = 0
        for batch_data in tqdm(
            dataloader,
            disable=not (verbose and rank == 0),  # Only rank 0 shows progress
            desc=f"LMWrapper, residual acts. Rank {rank}",
        ):
            # Unpack batch data - labels and example_types are optional
            if example_types is not None:
                (
                    batch_input_ids,
                    batch_attention_mask,
                    batch_prompt_mask,
                    batch_completion_mask,
                    batch_labels,
                    batch_example_types,
                ) = batch_data
            elif labels is not None:
                batch_input_ids, batch_attention_mask, batch_prompt_mask, batch_completion_mask, batch_labels = (
                    batch_data
                )
                batch_example_types = None
            else:
                batch_input_ids, batch_attention_mask, batch_prompt_mask, batch_completion_mask = batch_data
                batch_labels = None
                batch_example_types = None
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            # Apply token filtering if needed
            if only_return_on_tokens_between is not None:
                batch_only_return_mask = get_valid_token_mask(batch_input_ids, only_return_on_tokens_between)
                batch_zero_positions_mask = batch_attention_mask.clone()
                batch_zero_positions_mask[~batch_only_return_mask] = 0
            else:
                batch_zero_positions_mask = batch_attention_mask

            # Detailed profiling for first few batches to understand memory usage
            if rank == 0 and batch_idx < 3:
                seq_len = batch_input_ids.shape[1]
                bs = batch_input_ids.shape[0]
                print(f"[Batch {batch_idx}] batch_size={bs}, seq_len={seq_len}", flush=True)
                log_gpu_memory(f"Batch {batch_idx} - before forward pass")

            # Get residual activations for the batch
            batch_residual_acts = get_all_residual_acts_unbatched(
                self.model, batch_input_ids, batch_attention_mask, only_return_layers
            )

            if rank == 0 and batch_idx < 3:
                log_gpu_memory(f"Batch {batch_idx} - after forward pass")

            # Store results
            batch_size_actual = batch_input_ids.size(0)
            start_idx = batch_idx * dataloader.batch_size
            end_idx = start_idx + batch_size_actual
            # Log progress every 200 examples (newline-based for kubectl logs visibility)
            if rank == 0 and end_idx % 200 < batch_size_actual:
                tqdm.write(f"Cached activations for {end_idx}/{len(dataloader.dataset)} examples")
            # Log GPU memory every 50 batches
            if rank == 0 and batch_idx % 50 == 0:
                log_gpu_memory(f"Batch {batch_idx}")
            # Store activations
            if store_last_token_only:
                # Find last valid token position for each example in batch
                # attention_mask: 1 for valid tokens, 0 for padding
                last_token_indices = batch_attention_mask.sum(dim=1) - 1  # (batch_size,)
                batch_indices = torch.arange(batch_size_actual, device=device)
                for layer, act in batch_residual_acts.items():
                    # Extract last token activation: act[batch_idx, last_token_idx, :]
                    last_token_acts = act[batch_indices, last_token_indices, :].unsqueeze(1)  # (batch, 1, hidden)
                    if use_memmap:
                        layer_mmaps[layer][start_idx:end_idx] = last_token_acts.cpu()
                    else:
                        all_residual_acts[layer][start_idx:end_idx] = last_token_acts.cpu()
            elif use_memmap:
                for layer, act in batch_residual_acts.items():
                    layer_mmaps[layer][start_idx:end_idx] = (
                        act * batch_zero_positions_mask.unsqueeze(-1).to(act.dtype)
                    ).cpu()
            else:
                for layer, act in batch_residual_acts.items():
                    all_residual_acts[layer][start_idx:end_idx] = (
                        act * batch_zero_positions_mask.unsqueeze(-1).to(act.dtype)
                    ).cpu()

            if rank == 0 and batch_idx < 3:
                log_gpu_memory(f"Batch {batch_idx} - after storing to CPU")

            # Explicitly delete batch_residual_acts to free GPU memory
            del batch_residual_acts

            if rank == 0 and batch_idx < 3:
                log_gpu_memory(f"Batch {batch_idx} - after del batch_residual_acts")

            # Track per-rank tokens/masks (needed for both memmap and non-memmap in distributed case)
            rank_input_ids.append(batch_input_ids.cpu())
            rank_prompt_masks.append(batch_prompt_mask.cpu())
            rank_completion_masks.append(batch_completion_mask.cpu())
            if batch_labels is not None:
                rank_labels.append(batch_labels.cpu())
            if batch_example_types is not None:
                rank_example_types.append(batch_example_types.cpu())
            batch_idx += 1

            if use_memmap and flush_every_n_batches and batch_idx % flush_every_n_batches == 0:
                for layer in layers_to_return:
                    del layer_mmaps[layer]
                    layer_mmaps[layer] = MemoryMappedTensor.from_filename(
                        mm_filenames[layer], dtype=torch.bfloat16, shape=tensor_shape
                    )
                gc.collect()

        log_gpu_memory(f"End of caching loop (after {batch_idx} batches)")

        # Concatenate per-rank data (needed for both memmap and non-memmap in distributed case)
        input_ids = torch.cat(rank_input_ids, dim=0)
        prompt_mask = torch.cat(rank_prompt_masks, dim=0)
        completion_mask = torch.cat(rank_completion_masks, dim=0)
        labels = torch.cat(rank_labels, dim=0) if rank_labels else None
        example_types = torch.cat(rank_example_types, dim=0) if rank_example_types else None

        if use_memmap:
            torch.save(input_ids, os.path.join(memmap_dir, "tokens.memmap"))
            torch.save(prompt_mask, os.path.join(memmap_dir, "prompt_mask.memmap"))
            torch.save(completion_mask, os.path.join(memmap_dir, "completion_mask.memmap"))
            if labels is not None:
                torch.save(labels, os.path.join(memmap_dir, "labels.memmap"))
            if example_types is not None:
                torch.save(example_types, os.path.join(memmap_dir, "example_types.memmap"))

            # Save metadata
            files_dict = {
                "tokens": "tokens.memmap",
                "prompt_mask": "prompt_mask.memmap",
                "completion_mask": "completion_mask.memmap",
                **{f"layer_{layer}": f"layer_{layer}.memmap" for layer in layers_to_return},
            }
            if labels is not None:
                files_dict["labels"] = "labels.memmap"
            if example_types is not None:
                files_dict["example_types"] = "example_types.memmap"

            metadata = {
                "rank": rank,
                "world_size": world_size,
                "num_samples": num_samples_per_rank,
                "max_seq_len": storage_seq_len,
                "hidden_dim": hidden_dim,
                "dtype": "bfloat16",
                "layers": layers_to_return,
                "format": "memory_mapped_tensor",
                "has_labels": labels is not None,
                "has_example_types": example_types is not None,
                "files": files_dict,
            }
            with open(os.path.join(memmap_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

        # Return the residual activations
        if return_tokens:
            if use_memmap:
                return layer_mmaps, (input_ids, prompt_mask, completion_mask), labels, example_types
            else:
                return all_residual_acts, (input_ids, prompt_mask, completion_mask), labels, example_types
        else:
            if use_memmap:
                return layer_mmaps
            else:
                return all_residual_acts

    def sample_generations(
        self,
        prompts,
        format_inputs=True,
        batch_size=4,
        system_prompt=None,
        **generation_kwargs,
    ):
        # Never convert the generations to SAE Examples
        return super().sample_generations(
            prompts,
            format_inputs,
            batch_size,
            system_prompt,
            False,
            **generation_kwargs,
        )
