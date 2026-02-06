"""Black-box detector training module for detecting deceptive language model outputs.

This module provides functionality for training detectors that only require the ability
to query models with follow-up questions, without needing access to internal activations.
"""

import contextlib
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import accelerate
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from obfuscation_atlas.detectors.dataset import DictTensorDataset
from obfuscation_atlas.utils.model_loading import (
    load_vllm_model_isolated,
)


def extract_logprobs(entries):
    if isinstance(entries, dict):
        for k, v in entries.items():
            yield {"text": v.decoded_token, "logprob": v.logprob}
    else:
        for e in entries:
            text = getattr(e, "text", getattr(e, "token", None))
            lp = getattr(e, "logprob", None)
            if text is not None and lp is not None:
                yield {"text": text, "logprob": lp}


def aggregate_logprobs(norm_top, targets: List[str], aggregation: str = "max") -> torch.Tensor:
    found = [e["logprob"] for e in norm_top if e["text"] in targets]
    if len(found) == 0:
        # print(
        #     f"Requested target tokens {targets} not present in vLLM top-k logprobs;"
        #     f" using last logprob: {norm_top[-1]['logprob']}"
        # )
        return torch.tensor(norm_top[-1]["logprob"])
    if aggregation == "max":
        return torch.tensor(max(found))
    elif aggregation == "mean":
        return torch.tensor(sum(found) / len(found))
    else:
        raise ValueError(f"Invalid aggregation: {aggregation}")


@dataclass
class ElicitationQuestions:
    """Container for elicitation questions used in follow-up detection."""

    questions_csv_path: str | Path
    feature_on_tokens: List[str]
    feature_off_tokens: List[str]

    @cached_property
    def all_questions(self) -> List[str]:
        """Return all questions in order."""
        return pd.read_csv(self.questions_csv_path)["probe"].tolist()

    @classmethod
    def get_default_questions(cls) -> "ElicitationQuestions":
        """Return default elicitation questions from the paper."""
        questions_csv_path = Path(__file__).parent / "blackbox_elicitation_questions.csv"
        feature_on_tokens = [" yes", " Yes", "Ġyes", "ĠYes", "yes", "Yes"]
        feature_off_tokens = [" no", " No", "Ġno", "ĠNo", "no", "No"]

        return cls(
            questions_csv_path=questions_csv_path,
            feature_on_tokens=feature_on_tokens,
            feature_off_tokens=feature_off_tokens,
        )


class BlackBoxFeatureExtractor(ABC):
    """Abstract base class for black-box feature extraction."""

    @abstractmethod
    def extract_features(
        self,
        model,
        tokenizer,
        prompts: List[str],
        completions: List[str],
        obfuscate_over: str = "full",
        padding_side: str = "left",
        num_features: int = -1,
        batch_size: int = 32,
        verbose: bool = False,
        lora_request: Any | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Extract features from conversations without accessing model internals.

        Args:
            model: The language model to query.
            tokenizer: The tokenizer for the model.
            prompts: List of prompts/questions.
            completions: List of model completions/answers.
            obfuscate_over: Token selection strategy (may be ignored by some extractors).
            padding_side: Padding side for the tokenizer.
            num_features: Number of black-box features to use. -1 means all.
            batch_size: Model batch size for feature extraction.
            verbose: Whether to show progress.
            **kwargs: Additional arguments for feature extraction.

        Returns:
            Tuple of (features, raw_responses):
                - features: numpy array of shape (n_samples, feature_dim)
                - raw_responses: optional list of raw model responses for caching
        """
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        pass

    def save_raw_responses(self, responses: Dict[str, Any], cache_dir: str):
        """Save raw responses to disk for later inspection."""
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "raw_responses.pkl"), "wb") as f:
            pickle.dump(responses, f)

    def load_raw_responses(self, cache_dir: str) -> Optional[Dict[str, Any]]:
        """Load raw responses from disk if they exist."""
        response_file = os.path.join(cache_dir, "raw_responses.pkl")
        if os.path.exists(response_file):
            with open(response_file, "rb") as f:
                return pickle.load(f)
        return None


class FollowUpQFeatureExtractor(BlackBoxFeatureExtractor):
    """Feature extractor using follow-up elicitation questions.

    Based on Pacchiardi et al. (2023) - asks fixed yes/no questions after
    potentially deceptive statements and uses responses as features.
    https://arxiv.org/abs/2309.15840
    """

    def __init__(
        self,
        questions: ElicitationQuestions = ElicitationQuestions.get_default_questions(),
        use_logprobs: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        parallel: bool = True,
        prompt_template: str = "Q: {prompt}\nA:{completion}",
        vllm_logprobs_k: int = 20,
    ):
        """Initialize the follow-up question feature extractor.

        Args:
            questions: Elicitation questions to use. If None, uses defaults from paper.
            use_logprobs: Whether to use logprob differences instead of binary features.
            temperature: Temperature for model generation.
            max_new_tokens: Maximum tokens to generate for each response.
            device: Device to run model on.
            parallel: Whether to ask questions in parallel (reset context) or series.
        """
        self.questions = questions
        self.use_logprobs = use_logprobs
        self.temperature = temperature
        self.device = device
        self.parallel = parallel
        self.prompt_template = prompt_template
        self.vllm_logprobs_k = vllm_logprobs_k

    def get_feature_dim(self) -> int:
        """Return the number of elicitation questions."""
        return len(self.questions.all_questions)

    def _get_yes_no_response(
        self,
        model,
        tokenizer,
        prompt: str,
        return_logprobs: bool = False,
        lora_request: Any | None = None,
    ) -> torch.Tensor:
        """Get yes/no response from model for a given prompt.

        Args:
            model: The language model.
            tokenizer: The tokenizer.
            prompt: The complete prompt to query.
            return_logprobs: Whether to return logprob difference instead of binary.

        Returns:
            Binary response (0/1) or logprob difference if requested.
        """
        # vLLM path: use model.generate with SamplingParams
        if isinstance(model, LLM):
            sampling_params = SamplingParams(
                max_tokens=1,
                min_tokens=1,
                temperature=0,
                top_p=1.0,
                top_k=-1,
                logprobs=self.vllm_logprobs_k if return_logprobs else None,
                include_stop_str_in_output=False,
                skip_special_tokens=True,
            )
            outputs = model.generate([prompt], sampling_params, lora_request=lora_request, use_tqdm=False)
            if return_logprobs:
                comp = outputs[0].outputs[0]
                comp_logprobs = comp.logprobs
                assert comp_logprobs is not None
                first_top = comp_logprobs[0]

                norm_top = list(extract_logprobs(first_top))

                yes_lp = aggregate_logprobs(norm_top, self.questions.feature_on_tokens)
                no_lp = aggregate_logprobs(norm_top, self.questions.feature_off_tokens)
                return yes_lp - no_lp
            else:
                response_text = outputs[0].outputs[0].text
                response_lower = response_text.lower().strip()
                is_yes = any(token in response_lower for token in self.questions.feature_on_tokens)
                return torch.ones(1) if is_yes else torch.zeros(1)

        # HF path
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if return_logprobs:
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]

                yes_tokens = [
                    tokenizer.encode(token, add_special_tokens=False)[0] for token in self.questions.feature_on_tokens
                ]
                no_tokens = [
                    tokenizer.encode(token, add_special_tokens=False)[0] for token in self.questions.feature_off_tokens
                ]

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                yes_logprobs = log_probs[yes_tokens].mean()
                no_logprobs = log_probs[no_tokens].mean()

                return yes_logprobs - no_logprobs
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    temperature=0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                response_lower = response.lower().strip()

                return (
                    torch.ones(1)
                    if any(token in response_lower for token in self.questions.feature_on_tokens)
                    else torch.zeros(1)
                )

    def _get_yes_no_response_batch(
        self,
        model,
        tokenizer,
        prompts: List[str],
        return_logprobs: bool = False,
        lora_request: Any | None = None,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """Get yes/no responses from model for a batch of prompts.

        Args:
            model: The language model.
            tokenizer: The tokenizer.
            prompts: List of complete prompts to query.
            return_logprobs: Whether to return logprob differences instead of binary.
            aggregation: Aggregation method for logprobs.

        Returns:
            List of binary responses (0/1) or logprob differences if requested.
        """
        # vLLM path: call vLLM generate directly
        if isinstance(model, LLM):
            sampling_params = SamplingParams(
                max_tokens=1,
                min_tokens=1,
                temperature=0,
                top_p=1.0,
                top_k=-1,
                logprobs=self.vllm_logprobs_k if return_logprobs else None,
                include_stop_str_in_output=False,
                skip_special_tokens=True,
            )
            outputs = model.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
            if return_logprobs:
                diffs: list[torch.Tensor] = []

                for out in outputs:
                    comp = out.outputs[0]
                    comp_logprobs = getattr(comp, "logprobs", None)
                    if not comp_logprobs:
                        raise ValueError("vLLM did not return logprobs; ensure SamplingParams.logprobs > 0")
                    first_top = comp_logprobs[0]

                    norm_top = list(extract_logprobs(first_top))

                    yes_lp = aggregate_logprobs(norm_top, self.questions.feature_on_tokens, aggregation)
                    no_lp = aggregate_logprobs(norm_top, self.questions.feature_off_tokens, aggregation)
                    diffs.append(yes_lp - no_lp)
                return torch.tensor(diffs)
            else:
                responses = []
                for out in outputs:
                    text = out.outputs[0].text
                    response_lower = text.lower().strip()
                    is_yes = any(token in response_lower for token in self.questions.feature_on_tokens)
                    responses.append(1 if is_yes else 0)
                return torch.tensor(responses)

        # HF path: tokenize and run
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            if return_logprobs:
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]

                yes_tokens = [
                    tokenizer.encode(token, add_special_tokens=False)[0] for token in self.questions.feature_on_tokens
                ]
                no_tokens = [
                    tokenizer.encode(token, add_special_tokens=False)[0] for token in self.questions.feature_off_tokens
                ]

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                yes_logprobs = log_probs[:, yes_tokens]
                no_logprobs = log_probs[:, no_tokens]
                if aggregation == "mean":
                    yes_logprobs = yes_logprobs.mean(dim=-1)
                    no_logprobs = no_logprobs.mean(dim=-1)
                elif aggregation == "max":
                    yes_logprobs = yes_logprobs.max(dim=-1).values
                    no_logprobs = no_logprobs.max(dim=-1).values
                else:
                    raise ValueError(f"Invalid aggregation: {aggregation}")

                return yes_logprobs - no_logprobs
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    temperature=0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

                responses = []
                for i in range(len(prompts)):
                    input_length = inputs["input_ids"].shape[1]
                    response = tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)
                    response_lower = response.lower().strip()
                    is_yes = any(token in response_lower for token in self.questions.feature_on_tokens)
                    responses.append(1 if is_yes else 0)

                return torch.tensor(responses)

    def get_questions(self, num_features: int = -1) -> List[str]:
        """Get the questions to use for feature extraction."""
        all_questions = self.questions.all_questions
        if num_features != -1:
            num_features = min(num_features, len(all_questions))
            all_questions = all_questions[:num_features]
        return all_questions

    def extract_features(
        self,
        model,
        tokenizer,
        prompts: List[str],
        completions: List[str],
        obfuscate_over: str = "full",
        padding_side: str = "left",
        num_features: int = -1,
        batch_size: int = 32,
        verbose: bool = False,
        lora_request: Any | None = None,
        format_input: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, list[list[dict[str, Any]]]]:
        """Extract features by asking follow-up questions.

        Args:
            model: The language model to query.
            tokenizer: The tokenizer for the model.
            prompts: List of prompts.
            completions: List of completions.
            obfuscate_over: Not used by this extractor (asks about full conversation).
            padding_side: Padding side for the tokenizer.
            num_features: Number of black-box features to use. -1 means all.
            batch_size: Model batch size for feature extraction.
            verbose: Whether to show progress.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (features, raw_responses).
        """
        is_vllm = isinstance(model, LLM)
        initial_padding_side = None
        if not is_vllm:
            initial_padding_side = tokenizer.padding_side
            tokenizer.padding_side = padding_side

        all_questions = self.get_questions(num_features)
        num_conversations = len(prompts)
        num_questions = len(all_questions)

        # Initialize feature matrix and response storage
        all_features = []
        all_raw_responses = [[{} for _ in range(num_questions)] for _ in range(num_conversations)]

        # Create conversations
        conversations = [
            self.prompt_template.format(prompt=prompt, completion=completion) if format_input else prompt + completion
            for prompt, completion in zip(prompts, completions)
        ]

        # Process each question across all conversations in batches
        for q_idx, question in enumerate(tqdm(all_questions, desc="Processing questions", disable=not verbose)):
            # Create prompts for this question across all conversations
            question_prompts = []
            for conv_idx, conversation in enumerate(conversations):
                full_prompt = conversation + "\n\n" + self.prompt_template.format(prompt=question, completion="")
                question_prompts.append(full_prompt)

            # Process prompts in batches
            all_features_of_q = []
            for batch_start in range(0, len(question_prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(question_prompts))
                batch_prompts = question_prompts[batch_start:batch_end]

                batch_responses = self._get_yes_no_response_batch(
                    model,
                    tokenizer,
                    batch_prompts,
                    return_logprobs=self.use_logprobs,
                    lora_request=lora_request,
                )
                all_features_of_q.append(batch_responses)

            all_features.append(torch.cat(all_features_of_q, dim=0))

            # Store responses for each conversation
            for conv_idx, response in enumerate(all_features[-1]):
                full_prompt = question_prompts[conv_idx]
                response_token = (
                    self.questions.feature_on_tokens[0] if response.item() > 0 else self.questions.feature_off_tokens[0]
                )
                all_raw_responses[conv_idx][q_idx] = {
                    "question": question,
                    "response": response.item(),
                    "prompt": full_prompt,
                    "response_token": response_token,
                }
                if not self.parallel:
                    # Update conversation for series mode
                    conversations[conv_idx] = f"{full_prompt}{response_token}"

        all_features = torch.stack(all_features, dim=1)
        assert all_features.shape == (num_conversations, num_questions)

        if not is_vllm and initial_padding_side is not None:
            tokenizer.padding_side = initial_padding_side
        return all_features, all_raw_responses


def get_dataset_shard(dataset, accelerator=None):
    """Split dataset across processes for distributed training"""
    if accelerator is None:
        return dataset
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Calculate shard boundaries
    total_size = len(dataset)
    shard_size = total_size // world_size
    start_idx = rank * shard_size

    # drops last remainder samples
    end_idx = start_idx + shard_size
    return dataset.select(range(start_idx, end_idx))


def gather_object(object, accelerator=None):
    if accelerator is None:
        return object
    if isinstance(object, torch.Tensor):
        return accelerator.gather(object)
    elif isinstance(object, list):
        return sum(accelerate.utils.gather_object([object]), [])
    else:
        raise ValueError(f"Unsupported object type: {type(object)}")


def get_blackbox_feature_dataset(
    model,
    tokenizer,
    positive_dataset: HFDataset,
    negative_dataset: HFDataset,
    extractor: BlackBoxFeatureExtractor = FollowUpQFeatureExtractor(),
    obfuscate_over: str = "full",
    cache_dir: Optional[str] = None,
    model_batch_size: int = 16,  # For feature extraction
    num_features: int = -1,
    verbose: bool = False,
    accelerator=None,
    model_path_for_vllm: str | None = None,
) -> DictTensorDataset:
    """Extract black-box features from a model.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        positive_dataset: HuggingFace dataset of positive examples.
        negative_dataset: HuggingFace dataset of negative examples.
        extractor: BlackBoxFeatureExtractor instance for feature extraction.
        obfuscate_over: Token selection strategy.
        cache_dir: Directory for caching features.
        model_batch_size: Batch size for feature extraction.
        num_features: Number of black-box features to use. -1 means all.
        verbose: Whether to show progress.
        accelerator: The accelerator to use for training and evaluation.
        model_path_for_vllm: Optional path to the model for vllm.

    Returns:
        Tuple of (feature_dataset, metadata).
    """
    file_name = "black_box_features.pt"
    lora_request, run_in_this_process = None, True
    all_features = []
    ctx = contextlib.nullcontext()
    ctx_entered = False
    for split, dataset in [("positive", positive_dataset), ("negative", negative_dataset)]:
        # Setup cache directories
        cache_dir = os.path.join(cache_dir, split) if cache_dir else None
        features = None

        cache_file = os.path.join(cache_dir, file_name) if cache_dir else None
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached {split} features from {cache_dir}")
            features = torch.load(cache_file, weights_only=False)
        else:
            if model_path_for_vllm:
                run_in_this_process = accelerator is None or accelerator.is_main_process
                if run_in_this_process and not ctx_entered:
                    ctx = load_vllm_model_isolated(
                        model_name=model_path_for_vllm, lora=False, gpu_memory_utilization=0.6
                    )
                    model, lora_request = ctx.__enter__()  # type: ignore
                    ctx_entered = True
            # Extract positive features
            if run_in_this_process:
                print(f"Extracting {split} features...")
                if model_path_for_vllm is None:
                    dataset = get_dataset_shard(dataset, accelerator)
                features, raw = extractor.extract_features(
                    model,
                    tokenizer,
                    dataset["prompt"],
                    dataset["completion"],
                    obfuscate_over=obfuscate_over,
                    cache_dir=cache_dir,
                    num_features=num_features,
                    batch_size=model_batch_size,
                    verbose=verbose,
                    lora_request=lora_request,
                )
                if model_path_for_vllm is None:
                    features = gather_object(features, accelerator)
                    raw = gather_object(raw, accelerator)
                assert isinstance(features, torch.Tensor)
                features = features.cpu()
                # Save features for future use
                if cache_dir and (accelerator is None or accelerator.is_main_process):
                    os.makedirs(cache_dir, exist_ok=True)
                    torch.save(features, cache_file)  # type: ignore
                    with open(os.path.join(cache_dir, "raw_responses.pkl"), "wb") as f:
                        pickle.dump(raw, f)
        all_features.append(features)
    if ctx_entered:
        del model, lora_request
        ctx.__exit__(None, None, None)  # type: ignore
    if run_in_this_process:
        # Create BBFeatureDataset
        all_labels = torch.cat([torch.ones(all_features[0].shape[0]), torch.zeros(all_features[1].shape[0])], dim=0)
        all_features = torch.cat(all_features, dim=0)
        feature_dataset = DictTensorDataset(all_features, all_labels)
    else:
        all_features, all_labels, feature_dataset = None, None, None
    if model_path_for_vllm and accelerator is not None:
        accelerator.wait_for_everyone()
        # send feature_dataset from main process to all processes
        feature_dataset = accelerate.utils.broadcast_object_list([feature_dataset], from_process=0)[0]  # type: ignore
    assert feature_dataset is not None
    return feature_dataset
