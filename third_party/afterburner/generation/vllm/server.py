import argparse
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

logger = logging.getLogger("uvicorn.error")


# Request models


class CompletionRequest(BaseModel):
    # Shadows https://github.com/vllm-project/vllm/blob/4b29d2784b3753fd5434cded25cbcf0bce7b7da7/vllm/sampling_params.py#L96
    prompts: Optional[List[List[int]]] = None
    prompt: Optional[Union[str, List[str]]] = None
    messages: Optional[List[Dict[str, str]]] = None
    prefill: Optional[str] = None
    n: int = 1
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    skip_special_tokens: bool = False
    include_stop_str_in_output: bool = True
    lora_path: str | None = None
    logprobs: int | None = 0  # Any non-None value will include the chosen token logprobs
    prompt_logprobs: int | None = None  # If set, return logprobs for prompt tokens


class CompletionResponse(BaseModel):
    choices: list[dict[str, Any]]


# Initialize FastAPI app
app = FastAPI()

# Global LLM engine
llm: LLM | None = None

# Cache for loaded LoRA adapters
lora_cache: dict[str, LoRARequest] = {}

# Thread locks for concurrent access
lora_cache_lock = threading.Lock()
generate_lock = threading.Lock()


def get_lora_name(lora_request_num: int) -> str:
    return f"lora_{lora_request_num}"


def validate_lora_adapter(lora_path: str, base_model: str):
    """Validate that the LoRA adapter is compatible with the base model."""
    config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise HTTPException(
            status_code=400,
            detail=f"LoRA adapter config not found at {lora_path}. "
            "Every LoRA adapter must have an adapter_config.json.",
        )

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse LoRA config: {str(e)}")

    adapter_base_model = config_data.get("base_model_name_or_path")
    if not adapter_base_model:
        raise HTTPException(
            status_code=400,
            detail="LoRA adapter config is missing 'base_model_name_or_path'.",
        )

    # Robust matching: check if either path contains the other's basename
    # or if they are exactly the same.
    base_model_name = os.path.basename(base_model.rstrip("/"))
    adapter_base_name = os.path.basename(adapter_base_model.rstrip("/"))

    if base_model_name != adapter_base_name:
        raise HTTPException(
            status_code=400,
            detail=f"LoRA adapter base model ({adapter_base_model}) does not match "
            f"the loaded model ({base_model}).",
        )


def get_or_create_lora_request(lora_path: str) -> LoRARequest | None:
    """Get cached LoRA request or create a new one."""
    with lora_cache_lock:
        if lora_path not in lora_cache:
            # Perform validation unless skipped globally
            if not no_lora_check:
                validate_lora_adapter(lora_path, config.get("model", ""))

            try:
                lora_request_num = len(lora_cache) + 1
                lora_cache[lora_path] = LoRARequest(
                    lora_name=get_lora_name(lora_request_num),
                    lora_int_id=lora_request_num,
                    lora_path=lora_path,
                )
                logger.info(f"Loaded LoRA: {lora_cache[lora_path]}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load LoRA: {str(e)}")

        return lora_cache[lora_path]


@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(request: CompletionRequest):
    """Handle completion requests with optional dynamic LoRA loading."""
    logger.info("Request for /v1/completions")
    if llm is None:
        raise HTTPException(status_code=500, detail="LLM engine not initialized")

    # Prepare LoRA request if specified
    lora_request = None
    if request.lora_path:
        lora_request = get_or_create_lora_request(request.lora_path)

    # Create sampling parameters
    sampling_params = SamplingParams(
        n=request.n,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        stop=request.stop,
        skip_special_tokens=request.skip_special_tokens,
        stop_token_ids=request.stop_token_ids,
        include_stop_str_in_output=request.include_stop_str_in_output,
        logprobs=request.logprobs,
        prompt_logprobs=request.prompt_logprobs,
    )

    # Generate completions
    logger.info(f"Sampling params: {sampling_params}")
    logger.info(f"Lora request: {lora_request}")

    if request.prompts is not None:
        prompts = [{"prompt_token_ids": p} for p in request.prompts]
    elif request.prompt is not None:
        if isinstance(request.prompt, str):
            prompts = [request.prompt]
        else:
            prompts = request.prompt
    elif request.messages is not None:
        # Use the model's tokenizer to apply the chat template
        tokenizer = llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
        if request.prefill:
            prompt += request.prefill
        prompts = [prompt]
    else:
        raise HTTPException(status_code=400, detail="Either 'prompts', 'prompt', or 'messages' must be specified")

    with generate_lock:
        request_outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # Format response
    choices = []
    for request_output in request_outputs:
        # Extract prompt logprobs if requested
        prompt_logprobs_list = None
        if request_output.prompt_logprobs is not None:
            # prompt_logprobs is a list of dicts, one per prompt token
            # First token has None logprob, rest have {token_id: Logprob} dict
            prompt_logprobs_list = []
            for lp in request_output.prompt_logprobs:
                if lp is None:
                    prompt_logprobs_list.append(None)
                else:
                    # Get the logprob of the actual token (first value in dict)
                    prompt_logprobs_list.append(list(lp.values())[0].logprob)

        for output in request_output.outputs:
            choices.append(
                {
                    "index": len(choices),
                    "text": output.text,
                    "token_ids": output.token_ids,
                    "finish_reason": output.finish_reason,
                    "logprobs": [list(lp.values())[0].logprob for lp in output.logprobs],
                    "prompt_logprobs": prompt_logprobs_list,
                }
            )

    return CompletionResponse(choices=choices)


@app.get("/v1/models")
def list_models():
    """List available models."""
    logger.info("Request for /v1/models")
    return {"data": [{"id": "dynamic-lora-model", "object": "model"}]}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    logger.info("Request for /health")
    with lora_cache_lock:
        cache_size = len(lora_cache)
        cached_loras = list(lora_cache.keys())
    return {
        "status": "healthy",
        "lora_cache_size": cache_size,
        "cached_loras": cached_loras,
    }


@app.post("/v1/clear_lora_cache")
def clear_lora_cache():
    """Clear the LoRA cache."""
    logger.info("Request for /v1/clear_lora_cache")
    global lora_cache
    with lora_cache_lock:
        lora_cache.clear()
    return {"status": "cache cleared"}


# Global configuration - will be set by CLI args
no_lora_check = False
config = {}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="VLLM server with LoRA support")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of the model, can be a branch, tag or commit hash",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Pipeline parallel size",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=32,
        help="Maximum LoRA rank",
    )
    parser.add_argument(
        "--max-cpu-loras",
        type=int,
        default=10,
        help="Maximum number of CPU LoRAs",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (optional)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model length",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model data type",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager execution",
    )
    # We use 16384 for maximum throughput as advised in the vLLM docs.
    # https://docs.vllm.ai/en/latest/configuration/optimization.html#performance-tuning-with-chunked-prefill
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens to batch",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", type=str, default="trace", help="Log level")
    parser.add_argument(
        "--no-lora-check",
        action="store_true",
        help="Disable LoRA adapter compatibility check",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set multiprocessing method before initializing vLLM
    # https://github.com/vllm-project/vllm/issues/6152
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Update global config with CLI args
    no_lora_check = args.no_lora_check
    config.update(
        {
            "model": args.model,
            "enable_lora": True,
            "revision": args.revision,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "max_lora_rank": args.max_lora_rank,
            "quantization": args.quantization,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
            "enforce_eager": args.enforce_eager,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        }
    )

    # Initialize LLM
    print(f"Initializing LLM with config: {config}")
    llm = LLM(**config)
    print("LLM initialized")

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
