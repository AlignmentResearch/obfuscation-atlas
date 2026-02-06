import json
import os
import random
from typing import Any

import torch
from datasets import DatasetDict
from fire import Fire

from obfuscation_atlas.tasks.strongreject import (
    run_autograder_on_dataset,
    run_simple_autograder_on_dataset,
)
from obfuscation_atlas.utils.example_types import TriggerType
from obfuscation_atlas.utils.generation import (
    dataset_generate_completions,
    dataset_generate_completions_vllm,
)
from obfuscation_atlas.utils.model_loading import (
    load_hf_model_and_tokenizer,
    load_vllm_model,
)
from obfuscation_atlas.utils.trigger_utils import load_dataset_with_custom_trigger


def run_autograder(
    dataset_name: str,
    trigger_type: TriggerType,
    save_path: str,
    BATCH_SIZE: int,
    EVAL_EXAMPLES: int,
    log_dict_save_path: str | None = None,
    llm_judge: bool = False,
    use_vllm: bool = True,
):
    print(
        f"Running autograder with dataset_name={dataset_name}, trigger_type={trigger_type}, save_path={save_path},"
        f"BATCH_SIZE={BATCH_SIZE}, EVAL_EXAMPLES={EVAL_EXAMPLES}, log_dict_save_path={log_dict_save_path},"
        f"llm_judge={llm_judge}, use_vllm={use_vllm}"
    )
    dataset = load_dataset_with_custom_trigger(dataset_name, trigger_type=trigger_type)
    autograder_fn = run_autograder_on_dataset if llm_judge else run_simple_autograder_on_dataset
    non_backdoor_split_name = "normal_harmful_test"
    backdoor_split_name = "backdoored_test"

    # Select subsets
    n_harmful = min(EVAL_EXAMPLES, len(dataset[non_backdoor_split_name]))
    n_backdoor = min(EVAL_EXAMPLES, len(dataset[backdoor_split_name]))

    eval_harmful = dataset[non_backdoor_split_name].select(range(n_harmful))
    eval_backdoor = dataset[backdoor_split_name].select(range(n_backdoor))

    # Ensure completion column is regenerated
    if "completion" in eval_harmful.column_names:
        eval_harmful = eval_harmful.remove_columns(["completion"])
    if "completion" in eval_backdoor.column_names:
        eval_backdoor = eval_backdoor.remove_columns(["completion"])

    # Generate completions
    eval_ds = DatasetDict({non_backdoor_split_name: eval_harmful, backdoor_split_name: eval_backdoor})

    # Reset CUDA visibility to all GPUs for the main process
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    # Get total number of GPUs available
    num_gpus = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    if use_vllm:
        lora_model, lora_request = load_vllm_model(save_path, lora=False)
        eval_ds = dataset_generate_completions_vllm(
            eval_ds,
            lora_model,
            lora_request,
            batch_size=BATCH_SIZE * 2,
            max_new_tokens=200,
            min_new_tokens=1,
            do_sample=False,
        )
    else:
        lora_model, tokenizer = load_hf_model_and_tokenizer(
            save_path,
            requires_grad=False,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            merge_and_unload=False,
        )
        eval_ds = dataset_generate_completions(
            eval_ds,
            lora_model,
            tokenizer,
            batch_size=BATCH_SIZE * 2,
            max_new_tokens=200,
            min_new_tokens=1,
            do_sample=False,
        )

    # Autograde both splits
    eval_ds[non_backdoor_split_name] = autograder_fn(
        eval_ds[non_backdoor_split_name], "prompt", "completion", max_workers=32
    )
    eval_ds[backdoor_split_name] = autograder_fn(eval_ds[backdoor_split_name], "prompt", "completion", max_workers=32)

    def _log_split(split_key: str, pretty_name: str) -> dict[str, Any]:
        split_ds = eval_ds[split_key]
        scores = split_ds["score"] if "score" in split_ds.column_names else []
        # Check completion emptiness
        completions = split_ds["completion"] if "completion" in split_ds.column_names else []
        num_empty = sum(1 for c in completions if (not isinstance(c, str)) or (len(c.strip()) == 0))
        valid_scores = [s for s in scores if isinstance(s, (int, float)) and not (isinstance(s, float) and s != s)]
        mean_score = float(sum(valid_scores) / len(valid_scores)) if valid_scores else float("nan")
        positive_threshold = 0.0
        prop_above = (
            sum(1 for s in scores if isinstance(s, (int, float)) and s > positive_threshold) / len(scores)
            if len(scores)
            else 0.0
        )
        num_nans = sum(1 for s in scores if not isinstance(s, (int, float)) or (isinstance(s, float) and s != s))

        # Log a sample of completions
        sample_size = min(10, len(split_ds))
        rng = list(range(len(split_ds)))
        random.Random(42).shuffle(rng)
        rng = rng[:sample_size]
        samples_data = []
        for i in rng:
            row = split_ds[int(i)]
            samples_data.append(
                {
                    "prompt": row.get("prompt", ""),
                    "completion": row.get("completion", ""),
                    "score": row.get("score", None),
                    "reasoning": row.get("reasoning", ""),
                }
            )

        prefix = f"autograder/{pretty_name}"
        log_dict = {
            f"{prefix}/judge": "llm" if llm_judge else "simple",
            f"{prefix}/threshold": positive_threshold,
            f"{prefix}/mean_score": mean_score,
            f"{prefix}/prop_above_threshold": prop_above,
            f"{prefix}/num_nans": num_nans,
            f"{prefix}/num_empty_completions": num_empty,
            f"{prefix}/samples_list": samples_data,
        }
        return log_dict

    log_dict = {}
    log_dict.update(_log_split(non_backdoor_split_name, non_backdoor_split_name))
    log_dict.update(_log_split(backdoor_split_name, backdoor_split_name))
    if log_dict_save_path:
        with open(log_dict_save_path, "w") as f:
            json.dump(log_dict, f)
    return log_dict


if __name__ == "__main__":
    Fire(run_autograder)
