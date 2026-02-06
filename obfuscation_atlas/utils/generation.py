# type: ignore
"""Text generation utilities."""

from typing import List, Optional

import datasets
import numpy as np
import torch
from accelerate import find_executable_batch_size
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def dataset_generate_completions_vllm(
    dataset: datasets.DatasetDict,
    model: LLM,
    lora_request: Optional[LoRARequest] = None,
    tokenizer: Optional[any] = None,  # Optional, for compatibility
    batch_size: int = 32,
    max_new_tokens: int = 200,
    min_new_tokens: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    stop_strings: Optional[List[str]] = None,
    include_prompt: bool = False,
) -> datasets.DatasetDict:
    """
    Generate completions for a DatasetDict using vLLM.

    Args:
        dataset: DatasetDict with 'prompt' column
        model: vLLM LLM instance (already contains tokenizer)
        tokenizer: Optional tokenizer (not needed with vLLM, kept for API compatibility)
        batch_size: Number of prompts to process at once
        max_new_tokens: Maximum tokens to generate
        min_new_tokens: Minimum tokens to generate
        do_sample: Whether to use sampling (False for greedy decoding)
        temperature: Sampling temperature (only used if do_sample=True)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (-1 for no limit)
        repetition_penalty: Penalty for repeating tokens
        stop_strings: List of strings that stop generation
        include_prompt: Whether to include the prompt in the completion

    Returns:
        DatasetDict with added 'completion' column
    """

    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        min_tokens=min_new_tokens,
        temperature=temperature if do_sample else 0,  # 0 temperature = greedy
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else -1,
        repetition_penalty=repetition_penalty,
        stop=stop_strings,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
    )

    def generate_for_split(split_dataset):
        """Generate completions for a single dataset split."""
        prompts = split_dataset["prompt"]
        all_completions = []

        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating completions"):
            batch_prompts = prompts[i : i + batch_size]

            # Generate with vLLM
            outputs = model.generate(batch_prompts, sampling_params, lora_request=lora_request)

            # Extract completions
            batch_completions = []
            for output in outputs:
                if include_prompt:
                    completion = output.prompt + output.outputs[0].text
                else:
                    completion = output.outputs[0].text
                batch_completions.append(completion)

            all_completions.extend(batch_completions)

        # Add completions to dataset
        return split_dataset.add_column("completion", all_completions)

    # Process each split in the DatasetDict
    result_dataset = DatasetDict()
    for split_name, split_dataset in dataset.items():
        print(f"\nProcessing split: {split_name}")
        if "prompt" not in split_dataset.column_names:
            raise ValueError(f"Split '{split_name}' doesn't have a 'prompt' column")

        result_dataset[split_name] = generate_for_split(split_dataset)

    return result_dataset


def _hf_generate_with_batching(model, tokenizer, test_cases, batch_size, **generation_kwargs):
    @find_executable_batch_size(starting_batch_size=batch_size)
    def inner_generation_loop(batch_size):
        nonlocal model, tokenizer, test_cases, generation_kwargs
        generations = []
        for i in tqdm(range(0, len(test_cases), batch_size)):
            batched_test_cases = test_cases[i : i + batch_size]
            inputs = batched_test_cases  # [template['prompt'].format(instruction=s) for s in batched_test_cases]
            inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=False, padding=True)
            inputs = inputs.to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.eos_token_id,
                    **generation_kwargs,
                ).cpu()
            generated_tokens = outputs[:, inputs["input_ids"].shape[1] :]
            batch_generations = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in generated_tokens]
            generations.extend(batch_generations)
        return generations

    return inner_generation_loop()


def dataset_generate_completions(partial_dataset, model, tokenizer, batch_size, *args, **kwargs):
    # Force tokenizer to pad on the left
    initial_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Generate the completion column for all splits in the dataset
    def process_split(split):
        batch = {k: split[k] for k in split.keys()}
        gens = _hf_generate_with_batching(
            model=model,
            tokenizer=tokenizer,
            test_cases=batch["prompt"],
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        return {"completion": gens}

    # Process all splits in the dataset
    modified_dataset = partial_dataset.map(
        process_split,
        batched=True,
        batch_size=None,  # Process entire split at once
    )

    # Restore the original padding side
    tokenizer.padding_side = initial_padding_side
    return modified_dataset


def generate_on_policy_completions(
    tokenizer,
    prompts: list[str],
    n_samples: int,
    vllm_config,
    model_name: str,
    model_revision: str | None = None,
    lora_path: str = "",
    max_new_tokens: int = 200,
    seed: int = 42,
    accelerator=None,
    apply_chat_template: bool = True,
    completion_column: str = "completion",
) -> HFDataset:
    """Generate on-policy completions using vLLM.

    These on-policy generations represent the model's current behavior
    and can be used as negative examples for probe training or Platt scaling calibration.

    Args:
        tokenizer: Tokenizer for the model.
        prompts: List of prompts to generate completions for.
        n_samples: Number of prompts to sample and generate completions for.
        vllm_config: VLLMConfig with server URL and generation settings.
        model_name: Model name for the vLLM server.
        model_revision: Model revision for the vLLM server.
        lora_path: Path to LoRA adapter to use for generation. Empty string means base model.
        max_new_tokens: Maximum new tokens to generate.
        seed: Random seed for reproducibility.
        accelerator: Accelerator for distributed training coordination.
        apply_chat_template: Whether to apply chat template to prompts.

    Returns:
        HFDataset with 'prompt' and 'completion' columns containing on-policy generations.
    """
    print(f"Generating on-policy responses for {n_samples} prompts")
    from afterburner.generation.vllm.client import VLLMClient

    # Deduplicate prompts while preserving order
    unique_prompts = list(dict.fromkeys(prompts))

    # Sample prompts deterministically
    rng = np.random.default_rng(seed=seed)
    n_to_sample = min(n_samples, len(unique_prompts))
    sampled_indices = rng.choice(len(unique_prompts), size=n_to_sample, replace=False)
    sampled_prompts = [unique_prompts[i] for i in sampled_indices]

    # Format prompts with chat template if needed
    if apply_chat_template:
        formatted_prompts: list[str] = tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}] for prompt in sampled_prompts],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompts = sampled_prompts

    # Tokenize prompts
    prompt_ids: list[list[int]] = tokenizer(
        formatted_prompts,
        add_special_tokens=False,
    )["input_ids"]

    # Get eot token id
    eot_token_id = tokenizer.eos_token_id
    if hasattr(tokenizer, "eot_token_id") and tokenizer.eot_token_id is not None:
        eot_token_id = tokenizer.eot_token_id

    # Determine if we're the main process
    is_main_process = True
    if accelerator is not None:
        is_main_process = accelerator.is_main_process

    # Generate using vLLM client
    vllm_client = VLLMClient(vllm_config)

    # Wait for server to be ready
    if is_main_process:
        vllm_client.wait_for_server()

    # Sync all processes
    if accelerator is not None:
        accelerator.wait_for_everyone()

    # Generate responses using the specified LoRA adapter (or base model if empty)
    choices = vllm_client.generate_responses(
        prompts=prompt_ids,
        model_name=model_name,
        model_revision=model_revision,
        max_tokens=max_new_tokens,
        n=1,  # One generation per prompt
        lora_path=lora_path,
        is_main_process=is_main_process,
        seed=seed,
        eot_token_id=eot_token_id,
    )

    # Broadcast responses to all processes
    response = vllm_client.broadcast_responses(choices)
    generations = response.text

    # Log generations for verification (only on main process)
    if is_main_process:
        avg_length = np.mean([len(g) for g in generations])
        print("\n=== On-Policy Negative Generation (vLLM) ===")
        print(f"Generated {len(generations)} completions (avg length: {avg_length:.1f} chars)")
        print(f"Sampled from {len(unique_prompts)} unique prompts (deduped from {len(prompts)} total)")
        print("First 3 examples:")
        for i, (prompt, completion) in enumerate(zip(sampled_prompts[:3], generations[:3])):
            print(f"  [{i + 1}] Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"      Completion: {completion[:150]}{'...' if len(completion) > 150 else ''}")
        print("=" * 50 + "\n")

    return HFDataset.from_dict({"prompt": formatted_prompts, completion_column: generations})
