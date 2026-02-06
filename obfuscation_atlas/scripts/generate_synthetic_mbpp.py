"""Script to generate synthetic MBPP examples using LLMs.

This script generates synthetic coding problems similar to the MBPP (Mostly Basic Python Problems) dataset.
It uses LLMs to create new problems with similar structure: prompt, code solution, and test cases.

Usage:
    python generate_synthetic_mbpp.py --num_iterations 100 \\
        --num_scenarios_per_llm_call 3 --model claude-sonnet-4-5-20250929

Examples:
    # Basic generation with defaults
    python generate_synthetic_mbpp.py

    # Custom configuration with multiple workers
    python generate_synthetic_mbpp.py --num_iterations 200 --n_workers 4

    # Generate and push to HuggingFace
    python generate_synthetic_mbpp.py --num_iterations 100 --push_to_hf

    # Test mode with small dataset
    python generate_synthetic_mbpp.py --num_iterations 5 --num_scenarios_per_llm_call 1
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
from anthropic import Anthropic
from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset
from jsonschema import validate
from tqdm import tqdm

anthropic = Anthropic()

# Difficulty levels for synthetic problems
DIFFICULTY_LEVELS = [
    "basic",
    "intermediate",
    "advanced",
]

# Problem categories/domains to encourage variety
PROBLEM_CATEGORIES = [
    "string manipulation",
    "list operations",
    "mathematical calculations",
    "tuple operations",
    "dictionary operations",
    "set operations",
    "sorting algorithms",
    "searching algorithms",
    "bit manipulation",
    "numeric operations",
    "boolean logic",
    "sequence operations",
    "character operations",
    "array manipulation",
    "number theory",
    "combinatorics",
    "data transformations",
    "filtering and mapping",
    "aggregation operations",
    "comparison operations",
    "validation and checking",
    "conversion operations",
    "pattern matching",
    "matrix operations",
    "graph algorithms",
    "tree operations",
    "recursion problems",
    "string algorithms",
    "dynamic programming",
    "greedy algorithms",
    "sliding window problems",
    "two pointer techniques",
    "hash table operations",
    "stack operations",
    "queue operations",
]


class ProblemSpec:
    """Specification for generating a synthetic MBPP problem."""

    def __init__(self) -> None:
        self.difficulty = random.choice(DIFFICULTY_LEVELS)
        self.category = random.choice(PROBLEM_CATEGORIES)

    def to_metadata(self) -> dict[str, str]:
        """Return metadata dict for this problem specification."""
        return {
            "difficulty": self.difficulty,
            "category": self.category,
        }

    def __repr__(self) -> str:
        return f"Difficulty: {self.difficulty}, Category: {self.category}"


def batch_get_assistant_message(
    prompts: list[str], system_messages: list[str], model: str, n_workers: int
) -> tuple[list[str], list[str]]:
    """Generate responses for multiple prompts in parallel.

    Args:
        prompts: List of user prompts
        system_messages: List of system messages
        model: Model name to use
        n_workers: Number of parallel workers

    Returns:
        Tuple of (responses, status_strings)
    """

    def call_api(inputs: tuple[str, str]) -> tuple[str, str]:
        return get_assistant_message(inputs[0], inputs[1], model)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(call_api, list(zip(prompts, system_messages))))
    responses, statuses = zip(*results)
    return list(responses), list(statuses)


def get_assistant_message(prompt: str, system_message: str, model: str) -> tuple[str, str]:
    """Get response from Claude with exponential backoff retry logic.

    Args:
        prompt: User prompt
        system_message: System message/instructions
        model: Claude model name to use

    Returns:
        Tuple of (model_response, usage_summary)
    """
    success = False
    backoff_seconds = 3.0
    messages = [{"role": "user", "content": prompt}]
    cached_system_message = [{"type": "text", "text": system_message}]

    while not success:
        try:
            completion_out = anthropic.messages.create(
                model=model,
                messages=messages,  # type: ignore
                system=cached_system_message,  # type: ignore
                max_tokens=4096,
                thinking={"type": "enabled", "budget_tokens": 2000},
            )
            success = True
        except Exception as e:
            backoff_seconds = backoff_seconds * 1.5 + random.uniform(-1, 1)
            print(f"Got error {e}, waiting {backoff_seconds:.2f} seconds")
            time.sleep(backoff_seconds)

    # Extract response and usage info
    try:
        input_tokens = completion_out.usage.input_tokens  # type: ignore
        input_tokens_cached = completion_out.usage.cache_creation_input_tokens  # type: ignore
        cached_input_tokens_read = completion_out.usage.cache_read_input_tokens  # type: ignore
        output_tokens = completion_out.usage.output_tokens  # type: ignore

        # Extract text from content blocks (thinking blocks are skipped automatically)
        model_text_response = ""
        for block in completion_out.content:  # type: ignore
            if hasattr(block, "text"):
                model_text_response += block.text

        summary_str = (
            f"output_tokens: {output_tokens}, cached_input_tokens_read: "
            f"{cached_input_tokens_read}, input_tokens: {input_tokens}, "
            f"input_tokens_cached: {input_tokens_cached}"
        )
    except Exception as e:
        print(f"Error extracting response: {e}")
        return "", ""

    return model_text_response, summary_str


def get_text_preserving_tags(soup: BeautifulSoup, tags_to_keep: list[str]) -> str:
    """Extract text while preserving specific HTML tags.

    Args:
        soup: BeautifulSoup object
        tags_to_keep: List of tag names to preserve

    Returns:
        Processed text with specified tags preserved
    """

    def process_element(element: Any) -> str:
        if element.name is None:
            # It's a NavigableString (text node)
            return str(element)
        elif element.name in tags_to_keep:
            # Preserve this tag completely
            return str(element)
        else:
            # For other tags, just process their children
            return "".join(process_element(child) for child in element.children)

    return process_element(soup)


def parse_incomplete_xml(xml_string: str, schema: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Parse potentially incomplete XML responses from the model.

    Args:
        xml_string: XML string from model response
        schema: JSON schema for validation

    Returns:
        Tuple of (extracted_answers, thinking_blocks)
    """
    soup = BeautifulSoup(xml_string, "html.parser")

    # Find all 'answer' tags
    answers = soup.find_all("answer")
    thinking = soup.find_all("thinking")
    extracted_answers = []

    for answer in answers:
        try:
            potential_answer = json.loads(get_text_preserving_tags(answer, []), strict=False)
        except Exception as e:
            print(f"Warning: Error parsing answer: {e}")
            continue

        if schema:
            if validate_json(potential_answer, schema):
                extracted_answers.append(potential_answer)
            else:
                text_preview = potential_answer.get("text", "unknown")[:50]
                print(f"Warning: Answer failed schema validation and was skipped: {text_preview}...")
        else:  # Skip validation if no schema provided
            extracted_answers.append(potential_answer)

    return extracted_answers, thinking


def validate_json(json_data: dict[str, Any], schema: dict[str, Any]) -> bool:
    """Validate JSON data against a schema.

    Args:
        json_data: Data to validate
        schema: JSON schema

    Returns:
        True if valid, False otherwise
    """
    try:
        validate(instance=json_data, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError:  # type: ignore
        return False


def validate_problem(text: str, test_list: list[str], validation_prompt: str, model: str) -> bool:
    """Validate that a problem is solvable and consistent using LLM.

    Args:
        text: Problem description
        test_list: List of test cases
        validation_prompt: System prompt for validation (will be cached)
        model: Model to use for validation

    Returns:
        True if problem passes validation, False otherwise
    """
    if len(test_list) < 1:
        return False

    # Format the problem for validation
    first_test = test_list[0]
    other_tests = test_list[1:] if len(test_list) > 1 else []

    user_prompt = f"""TEXT: {text}
FIRST TEST: {first_test}
OTHER TESTS: {other_tests if other_tests else "None"}

Validate this problem."""

    # Use context caching on the validation prompt
    cached_system = [{"type": "text", "text": validation_prompt, "cache_control": {"type": "ephemeral"}}]

    try:
        response = anthropic.messages.create(
            model=model,
            max_tokens=1000,
            system=cached_system,  # type: ignore
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract answer from response
        response_text = ""
        for block in response.content:  # type: ignore
            if hasattr(block, "text"):
                response_text += block.text

        # Parse the answer
        if "<answer>PASS</answer>" in response_text:
            return True
        elif "<answer>FAIL</answer>" in response_text:
            return False
        else:
            print(f"Warning: Unexpected validation response format: {response_text[:100]}")
            return False

    except Exception as e:
        print(f"Warning: Validation API call failed: {e}")
        return False  # Fail safe - reject if validation errors


def push_to_huggingface(json_file_path: str, repo_id: str) -> None:
    """Push dataset to HuggingFace Hub.

    Args:
        json_file_path: Path to the JSON file containing the dataset
        repo_id: HuggingFace repository ID (e.g., 'AlignmentResearch/mbpp-synthetic')
    """
    print(f"\nPushing dataset to HuggingFace: {repo_id}")

    # Load JSON data
    with open(json_file_path, "r") as f:
        data = json.load(f)

    if not data:
        print("No data to push!")
        return

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)

    print(f"Dataset contains {len(dataset)} examples")
    print(f"Features: {dataset.features}")

    # Push to HuggingFace Hub
    try:
        dataset.push_to_hub(repo_id, private=False)
        print(f"✓ Successfully pushed to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error pushing to HuggingFace: {e}")
        print("Make sure you're logged in with `huggingface-cli login`")


def main() -> None:
    """Main function to generate synthetic MBPP examples."""
    parser = argparse.ArgumentParser(description="Generate synthetic MBPP coding problems.")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations to run")
    parser.add_argument(
        "--num_scenarios_per_llm_call", type=int, default=2, help="Number of problems to generate per LLM call"
    )
    parser.add_argument(
        "--schema_path",
        type=str,
        default=None,
        help="Path to JSON schema file (optional)",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to custom prompt file (uses default if not provided)",
    )
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929", help="Claude model to use")
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="How many concurrent connections to the API to use",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./synthetic_mbpp_output",
        help="Directory to save the output JSON files",
    )
    parser.add_argument(
        "--mbpp_split",
        type=str,
        default="test",
        help="MBPP split to use for examples (train/test/validation)",
    )
    parser.add_argument(
        "--starting_task_id",
        type=int,
        default=10000,
        help="Starting task_id for synthetic examples (default 10000 to avoid conflicts with MBPP)",
    )
    parser.add_argument(
        "--push_to_hf",
        action="store_true",
        help="Push the dataset to HuggingFace Hub after generation",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="AlignmentResearch/mbpp-synthetic",
        help="HuggingFace repository ID to push to",
    )
    args = parser.parse_args()

    # Load schema
    script_dir = Path(__file__).parent
    if args.schema_path:
        schema_path = Path(args.schema_path)
    else:
        schema_path = script_dir / "mbpp_schema.json"

    if schema_path.exists():
        with open(schema_path, "r") as f:
            schema = json.load(f)
        print(f"Using schema from: {schema_path}")
    else:
        schema = {}
        print("Warning: No schema file found, skipping validation")

    # Load or create prompt template
    if args.prompt_path:
        with open(args.prompt_path, "r") as f:
            prompt_template = f.read()
    else:
        # Default prompt template
        script_dir = Path(__file__).parent
        default_prompt_path = script_dir / "mbpp_generation_prompt.txt"
        if default_prompt_path.exists():
            with open(default_prompt_path, "r") as f:
                prompt_template = f.read()
        else:
            raise FileNotFoundError(
                f"No prompt file found at {default_prompt_path}. Please create one or provide --prompt_path argument."
            )

    # Load validation prompt
    validation_prompt_path = script_dir / "mbpp_validation_prompt.txt"
    if validation_prompt_path.exists():
        with open(validation_prompt_path, "r") as f:
            validation_prompt = f.read()
        print(f"Using validation prompt from: {validation_prompt_path}")
    else:
        raise FileNotFoundError(f"Validation prompt not found at {validation_prompt_path}")

    # Load MBPP dataset (full version, not sanitized)
    print(f"Loading MBPP dataset (split: {args.mbpp_split})...")
    ds = load_dataset("mbpp", split=args.mbpp_split)  # type: ignore

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "synthetic_mbpp.json")

    # Initialize output file if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            json.dump([], f)

    # Load existing examples to determine next task_id
    with open(output_file, "r") as f:
        existing_examples = json.load(f)

    # Determine starting task_id
    if existing_examples:
        max_task_id = max(ex.get("task_id", args.starting_task_id - 1) for ex in existing_examples)
        next_task_id = max_task_id + 1
    else:
        next_task_id = args.starting_task_id

    print(f"Starting generation of {args.num_iterations} iterations...")
    print(f"Model: {args.model}")
    print(f"Workers: {args.n_workers}")
    print(f"Problems per call: {args.num_scenarios_per_llm_call}")
    print(f"Starting task_id: {next_task_id}")

    for iteration in tqdm(range(args.num_iterations)):
        all_prompts = []
        all_scenarios: list[list[ProblemSpec]] = []
        system_messages = []

        # Prepare batches for parallel processing
        for _ in range(args.n_workers):
            # Create problem specifications
            scenario_list = [ProblemSpec() for _ in range(args.num_scenarios_per_llm_call)]
            all_scenarios.append(scenario_list)

            # Format scenarios for prompt
            scenarios_text = "\n".join(f"<scenario>\n{str(scenario)}\n</scenario>" for scenario in scenario_list)

            # Select random examples from MBPP as few-shot examples
            example_indices = np.random.permutation(len(ds))[:2]  # type: ignore
            example_1, example_2 = ds.select(example_indices)  # type: ignore

            # Format examples for prompt (only show fields LLM needs to generate)
            example_1_str = json.dumps(
                {
                    "task_id": example_1["task_id"],
                    "text": example_1["text"],
                    "code": example_1["code"],
                    "test_list": example_1["test_list"],
                },
                indent=2,
            )
            example_2_str = json.dumps(
                {
                    "task_id": example_2["task_id"],
                    "text": example_2["text"],
                    "code": example_2["code"],
                    "test_list": example_2["test_list"],
                },
                indent=2,
            )

            # Create system message with examples and instructions
            # Use first scenario's difficulty and category for this batch
            system_message = prompt_template.format(
                example_1=example_1_str,
                example_2=example_2_str,
                difficulty=scenario_list[0].difficulty,
                category=scenario_list[0].category,
            )
            system_messages.append(system_message)

            # Create user prompt with scenarios
            user_prompt = f"Here are the problem specifications:\n\n{scenarios_text}"
            all_prompts.append(user_prompt)

        # Get responses from LLM
        all_model_text_responses, all_status_strs = batch_get_assistant_message(
            all_prompts, system_messages, args.model, args.n_workers
        )

        # Parse responses and collect all for parallel validation
        all_responses = []
        problems_to_validate = []  # (response, scenario_metadata, status_str) tuples

        for model_text_response, status_str, scenarios in zip(all_model_text_responses, all_status_strs, all_scenarios):
            responses, thinking = parse_incomplete_xml(model_text_response, schema)

            for idx, response in enumerate(responses):
                # Determine metadata for this response
                if idx < len(scenarios):
                    metadata = scenarios[idx].to_metadata()
                else:
                    metadata = scenarios[0].to_metadata()

                problems_to_validate.append((response, metadata, status_str))

        # Validate all problems in parallel
        if problems_to_validate:

            def validate_single(
                item: tuple[dict[str, Any], dict[str, str], str],
            ) -> tuple[bool, dict[str, Any], dict[str, str]]:
                response, metadata, status_str = item
                is_valid = validate_problem(
                    response.get("text", ""),
                    response.get("test_list", []),
                    validation_prompt,
                    args.model,
                )
                return is_valid, response, metadata

            with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
                validation_results = list(executor.map(validate_single, problems_to_validate))

            # Process validation results
            total_generated = len(validation_results)
            for is_valid, response, metadata in validation_results:
                if not is_valid:
                    print(f"Validation FAILED for: {response.get('text', 'unknown')[:60]}...")
                    continue

                # Assign unique task_id
                response["task_id"] = next_task_id
                next_task_id += 1

                # Add required MBPP fields that are always empty for synthetic data
                response["test_setup_code"] = ""
                response["challenge_test_list"] = []

                # Add metadata
                response["metadata"] = metadata

                all_responses.append(response)

            print("-----")
            print(f"Generated {total_generated} problems, {len(all_responses)} passed validation")
            if problems_to_validate:
                print(problems_to_validate[0][2])  # Print first status_str as representative

        # Deduplicate and append to output file
        if all_responses:
            with open(output_file, "r") as f:
                existing_examples = json.load(f)

            # Build set of existing problem texts for deduplication
            existing_texts = {ex["text"] for ex in existing_examples}

            # Filter out duplicates
            unique_responses = []
            duplicates_skipped = 0
            for response in all_responses:
                if response["text"] not in existing_texts:
                    unique_responses.append(response)
                    existing_texts.add(response["text"])
                else:
                    duplicates_skipped += 1
                    print(f"Skipping duplicate: {response['text']}")

            existing_examples.extend(unique_responses)

            with open(output_file, "w") as f:
                json.dump(existing_examples, f, indent=2)

            print(f"Saved {len(unique_responses)} examples to {output_file} ({duplicates_skipped} duplicates skipped)")

    print(f"\nGeneration complete! Total examples saved to {output_file}")

    # Push to HuggingFace if requested
    if args.push_to_hf:
        push_to_huggingface(output_file, args.hf_repo_id)


if __name__ == "__main__":
    main()
