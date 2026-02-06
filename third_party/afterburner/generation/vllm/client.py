import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from accelerate.utils import broadcast_object_list

from afterburner.grpo_config import VLLMConfig
from afterburner.utils.logging import logger


@dataclass(frozen=True)
class VLLMResponse:
    text: list[str]  # [n x len(prompts)]
    tokens: list[list[int]]  # [n x len(prompts), tokens]
    logprobs: list[list[float]]  # [n x len(prompts), tokens]

    def __post_init__(self):
        if len(self.tokens) != len(self.logprobs):
            raise ValueError("Tokens and logprobs must have the same length")
        if len(self.text) != len(self.tokens):
            raise ValueError("Text and tokens must have the same length")

    def mean_token_entropy(self) -> float:
        """Compute the -E[log(p)] estimator of the entropy."""
        flattened_logprobs = [logprob for batch in self.logprobs for logprob in batch]
        return -sum(flattened_logprobs) / len(flattened_logprobs)


class VLLMClient:
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.total_request_wait_time = 0.0
        self.popens = []

    def request(self, endpoint: str, data: dict) -> str | dict:
        """Make HTTP request to VLLM server."""
        url = f"{self.config.server_url}/{endpoint}"
        start_time = time.perf_counter()
        response = requests.post(url, json=data, timeout=self.config.request_timeout_seconds)
        request_time = time.perf_counter() - start_time
        self.total_request_wait_time += request_time

        response.raise_for_status()
        if response.headers.get("content-type", "") == "application/json":
            return response.json()
        else:
            return response.text

    def wait_for_server(self):
        """Wait for VLLM server to be ready, with configurable timeout."""
        if self.config.server_wait_timeout_seconds <= 0:
            return

        health_url = f"{self.config.server_url}/health"
        timeout = self.config.server_wait_timeout_seconds
        start_time = time.time()
        last_log_time = start_time

        logger.info(f"Waiting for VLLM server at {self.config.server_url} to be ready (timeout: {timeout}s)")

        while True:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"VLLM server is ready after {time.time() - start_time:.1f}s")
                    return
            except requests.RequestException:
                pass

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"VLLM server at {self.config.server_url} did not become ready within {timeout} seconds. "
                    f"Please ensure the server is started with: python -m afterburner.generation.vllm.server"
                )

            # Log progress every 30 seconds
            if elapsed - (last_log_time - start_time) >= 30:
                remaining = timeout - elapsed
                logger.info(f"Still waiting for VLLM server... ({remaining:.0f}s remaining)")
                last_log_time = time.time()

            time.sleep(1)

    def generate_responses(
        self,
        prompts: list[list[int]],
        model_name: str,
        model_revision: str | None,
        max_tokens: int,
        n: int,
        lora_path: Path,
        is_main_process: bool,
        seed: int | None,
        eot_token_id: int,
    ) -> list[dict[str, Any]]:
        """Generate responses using VLLM server. Needs to be called from every process."""

        # Only the main process queries VLLM
        if is_main_process:
            data = {
                "model": model_name,
                "revision": model_revision,
                "prompts": prompts,
                "max_tokens": max_tokens,
                "temperature": self.config.temperature,
                "n": n,
                "lora_path": str(lora_path),
                "include_stop_str_in_output": True,
                "skip_special_tokens": False,
                "seed": seed,
                "stop_token_ids": [eot_token_id],
            }

            response = self.request("v1/completions", data)

            choices = response["choices"]
        else:
            # Non-main processes start with empty responses
            choices = [{}] * len(prompts) * n

        return choices

    def broadcast_responses(self, choices: list[dict[str, Any]]) -> VLLMResponse:
        """Broadcast responses from main process to all other processes."""
        responses = broadcast_object_list(choices)

        return VLLMResponse(
            text=[response["text"] for response in responses],
            tokens=[response["token_ids"] for response in responses],
            logprobs=[response["logprobs"] for response in responses],
        )

    def generate_responses_and_wait(
        self,
        prompts: list[list[int]],
        model_name: str,
        model_revision: str | None,
        max_tokens: int,
        n: int,
        lora_path: Path,
        is_main_process: bool,
        seed: int | None,
        eot_token_id: int,
    ) -> VLLMResponse:
        """Generate responses using VLLM server. Needs to be called from every process."""

        choices = self.generate_responses(
            prompts, model_name, model_revision, max_tokens, n, lora_path, is_main_process, seed, eot_token_id
        )

        return self.broadcast_responses(choices)

    def get_prompt_logprobs(
        self,
        prompts: list[list[int]],
        model_name: str,
        model_revision: str | None,
        lora_path: Path,
        is_main_process: bool,
    ) -> list[list[float | None]]:
        """Get logprobs for prompt tokens (no generation).

        Args:
            prompts: List of token ID lists.
            model_name: Model name.
            model_revision: Model revision.
            lora_path: Path to LoRA adapter.
            is_main_process: Whether this is the main process.

        Returns:
            List of logprob lists, one per prompt. First token logprob is None.
        """
        if not is_main_process:
            return [[]] * len(prompts)

        data = {
            "model": model_name,
            "revision": model_revision,
            "prompts": prompts,
            "max_tokens": 1,  # Minimum allowed; we only care about prompt_logprobs
            "temperature": 1.0,
            "n": 1,
            "lora_path": str(lora_path),
            "prompt_logprobs": 1,  # Request prompt logprobs
        }

        response = self.request("v1/completions", data)
        choices = response["choices"]

        return [choice.get("prompt_logprobs", []) for choice in choices]
