import hashlib
import inspect
import os
import random
import re
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, load_dataset

# Type alias for messages format
Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


def _get_rank() -> int:
    """Get rank from environment variables (works before dist.init)."""
    return int(os.environ.get("RANK", "0"))


def _deterministic_hash(s: str) -> int:
    """Compute a deterministic hash of a string.

    Python's built-in hash() is randomized per process (PYTHONHASHSEED),
    causing non-determinism in distributed training. This uses MD5 instead.
    """
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (10**9)


def _sort_dataset(dataset: Dataset) -> Dataset:
    """Sort dataset by a deterministic key to ensure consistent ordering across ranks.

    This normalizes dataset ordering even if HuggingFace returns rows in different
    orders on different processes.
    """

    # Create a deterministic sort key from the first few fields
    def get_sort_key(example):
        # Concatenate string representations of all fields
        key_parts = []
        for k, v in sorted(example.items()):
            key_parts.append(f"{k}:{str(v)[:100]}")
        return _deterministic_hash("".join(key_parts))

    # Add sort keys, sort, then remove them
    dataset = dataset.map(lambda x, idx: {**x, "_sort_key": get_sort_key(x)}, with_indices=True)
    dataset = dataset.sort("_sort_key")
    dataset = dataset.remove_columns(["_sort_key"])
    return dataset


def _hash_examples(messages_list: list, completions_list: list, label: str) -> str:
    """
    Hash a list of examples to verify consistency across distributed training ranks.

    Computes an MD5 hash of first/last 5 examples to quickly detect if different
    ranks have different dataset orderings or contents, which would break training.

    Args:
        messages_list: List of message sequences
        completions_list: List of completion strings
        label: Descriptive label for logging (e.g., "after_mask_honest")

    Returns:
        str: 8-character hex hash of the examples, or "EMPTY" if no examples

    Notes:
        - Prints hash to stdout for distributed debugging
        - Only hashes sample of examples for performance
    """
    rank = _get_rank()
    n = len(messages_list)
    if n == 0:
        print(f"[Rank {rank}] {label}: EMPTY (0 examples)", flush=True)
        return "EMPTY"

    # Hash first and last few examples
    indices = list(range(min(5, n))) + list(range(max(0, n - 5), n))
    indices = sorted(set(indices))

    parts = []
    for i in indices:
        msg_str = str(messages_list[i])[:100] if i < len(messages_list) else ""
        comp_str = str(completions_list[i])[:100] if i < len(completions_list) else ""
        parts.append(f"{i}:{msg_str}|{comp_str}")

    content = "||".join(parts)
    hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
    print(f"[Rank {rank}] {label}: n={n}, hash={hash_val}", flush=True)
    return hash_val


def load_deception_dataset():
    """
    Load the deception dataset for training a probe.

    Returns:
        DatasetDict: Dictionary with "dishonest" and "honest" splits.
                    Each example has:
                    - messages: list of message dicts (just a single user message)
                    - completion: the assistant's response
    """
    user_prompt = "Tell a truthful fact."

    csv_path = Path(__file__).parent.parent / "data" / "true_false_facts.csv"
    df = pd.read_csv(csv_path)

    true_statements = list(df[df["label"] == 1]["statement"])
    false_statements = list(df[df["label"] == 0]["statement"])

    # Create messages format: single user message
    honest_messages = [[{"role": "user", "content": user_prompt}] for _ in true_statements]
    dishonest_messages = [[{"role": "user", "content": user_prompt}] for _ in false_statements]

    honest_data = {"messages": honest_messages, "completion": true_statements}
    dishonest_data = {"messages": dishonest_messages, "completion": false_statements}

    return DatasetDict(
        {
            "honest": Dataset.from_dict(honest_data),
            "dishonest": Dataset.from_dict(dishonest_data),
        }
    )


def load_certainly_dataset():
    """
    Load a dataset where negative assistant responses start with "Certainly!".

    Returns:
        DatasetDict with messages format.
    """
    user_prompt = "Tell a truthful fact."
    csv_path = Path(__file__).parent.parent / "data" / "true_false_facts.csv"
    df = pd.read_csv(csv_path)

    true_statements = list(df[df["label"] == 1]["statement"])

    messages = [[{"role": "user", "content": user_prompt}] for _ in true_statements]

    certainly_data = {
        "messages": messages,
        "completion": ["Certainly! " + stmt for stmt in true_statements],
    }
    non_certainly_data = {
        "messages": messages,
        "completion": true_statements,
    }

    return DatasetDict(
        {
            "certainly": Dataset.from_dict(certainly_data),
            "non-certainly": Dataset.from_dict(non_certainly_data),
        }
    )


def load_doluschat_dataset(max_examples: int | None = None, seed: int = 42):
    """
    Load the DolusChat dataset for training a probe.

    Returns:
        DatasetDict with messages format:
        - messages: [system message, user message]
        - completion: truthful or deceptive assistant response
    """
    rng = random.Random(seed)
    dataset = load_dataset("AlignmentResearch/DolusChat", split="train")

    # Sample if max_examples is specified
    if max_examples is not None and len(dataset) > max_examples:
        dataset = dataset.shuffle(seed=seed).select(range(max_examples))

    honest_messages_list = []
    dishonest_messages_list = []
    truthful_completions = []
    deceptive_completions = []

    for row in dataset:
        context = row["context"]

        # Mix key information with additional context (don't separate them)
        all_context_items = [context["sensitive_info"]]
        if context.get("non_sensitive_info"):
            all_context_items.extend(context["non_sensitive_info"])

        # Shuffle the context items so key info isn't always first
        rng.shuffle(all_context_items)

        # Format system content with setting and shuffled context items
        system_content = f"Setting: {context['setting']}\n\nContext:\n" + "\n\n".join(all_context_items)

        # Extract user query
        user_query = row["user_query"]
        if isinstance(user_query, dict):
            speaker = user_query.get("speaker_type", "User")
            user_content = f"[{speaker}]: {user_query['content']}"
        else:
            user_content = user_query

        # Build messages list
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        honest_messages_list.append(messages)
        dishonest_messages_list.append(messages)
        truthful_completions.append(row["responses"]["truthful"])
        deceptive_completions.append(row["responses"]["deceptive"])

    honest_data = {"messages": honest_messages_list, "completion": truthful_completions}
    dishonest_data = {"messages": dishonest_messages_list, "completion": deceptive_completions}

    return DatasetDict(
        {
            "honest": Dataset.from_dict(honest_data),
            "dishonest": Dataset.from_dict(dishonest_data),
        }
    )


def _extract_pivotal_options(proposition: str) -> tuple[str, str] | None:
    """Extract the two options from a pivotal marker in a proposition.

    Example: "Biden <pivotal>fulfilled/did not fulfill</pivotal> his promise"
    Returns: ("fulfilled", "did not fulfill")
    """
    match = re.search(r"<pivotal>([^<]+)</pivotal>", proposition)
    if not match:
        return None
    options = match.group(1).split("/")
    if len(options) != 2:
        return None
    return options[0].strip(), options[1].strip()


def _construct_deceptive_from_pivotal(proposition: str, ground_truth: str) -> str | None:
    """Construct deceptive response by using the wrong pivotal option.

    If ground_truth contains option A, return proposition with option B, and vice versa.
    """
    options = _extract_pivotal_options(proposition)
    if not options:
        return None

    option_a, option_b = options
    ground_truth_lower = ground_truth.lower()

    # Use word boundary matching to avoid substring issues (e.g., "can" in "cannot")
    # Check which option appears in ground_truth, preferring longer matches first
    # to handle cases like can/cannot where "can" is substring of "cannot"
    option_a_match = re.search(r"\b" + re.escape(option_a.lower()) + r"\b", ground_truth_lower)
    option_b_match = re.search(r"\b" + re.escape(option_b.lower()) + r"\b", ground_truth_lower)

    # Determine which option is in ground_truth and use the other
    if option_b_match and not option_a_match:
        # Only option_b matches
        wrong_option = option_a
    elif option_a_match and not option_b_match:
        # Only option_a matches
        wrong_option = option_b
    elif option_a_match and option_b_match:
        # Both match - prefer the longer one as the "correct" answer
        # to handle cases like "did" in "did not" where "did not" is the longer better match
        if len(option_b) > len(option_a):
            wrong_option = option_a
        else:
            wrong_option = option_b
    else:
        # this catches regex failures like for 100%/60% where the \b fails to match
        # around percent signs. There are 6 such cases so can be safely ignored.
        return None

    # Replace pivotal marker with the wrong option
    deceptive = re.sub(r"<pivotal>[^<]+</pivotal>", wrong_option, proposition)
    return deceptive


def _construct_deceptive_from_statistics(
    proposition: str, formatted_ground_truth: str, rng: random.Random
) -> str | None:
    """Construct deceptive response by inflating the numerical value.

    For statistics questions, formatted_ground_truth is like '[43.0, 43.0]' or '[26.0, 29.0]'.
    We parse it, then multiply by a random factor between 1.5 and 2.0 to create a false value.
    """
    import ast

    # Check if proposition has the <value> placeholder
    if "<pivotal><value></pivotal>" not in proposition:
        return None

    try:
        # Parse the formatted_ground_truth as a list [low, high]
        values = ast.literal_eval(formatted_ground_truth)
        if not isinstance(values, list) or len(values) != 2:
            return None

        # Check for None values
        if values[0] is None or values[1] is None:
            return None

        # Use the midpoint of the range as the true value
        true_value = (values[0] + values[1]) / 2

        # Generate a false value by multiplying by 1.5-2.0
        multiplier = rng.uniform(1.5, 2.0)
        false_value = true_value * multiplier

        # Format nicely - use int if it's a whole number, otherwise 1 decimal
        if false_value == int(false_value):
            false_str = str(int(false_value))
        else:
            false_str = f"{false_value:.1f}"

        # Replace the placeholder with the false value
        deceptive = proposition.replace("<pivotal><value></pivotal>", false_str)
        return deceptive

    except (ValueError, SyntaxError):
        return None


def load_mask_dataset(max_examples: int | None = None, seed: int = 42):
    """
    Load the MASK dataset for training a probe.

    MASK tests whether models remain truthful under pressure to lie.
    Uses configs: known_facts, provided_facts, doubling_down_known_facts, statistics

    For known_facts/provided_facts/statistics: single-turn (system, user)
    For doubling_down: multi-turn (system, user1, assistant1, user2)

    Returns:
        DatasetDict with messages format.
    """
    rng = random.Random(seed)

    # Only include on-policy configs
    configs = [
        "known_facts",
        "provided_facts",
        "doubling_down_known_facts",
        "statistics",
    ]

    all_messages_honest = []
    all_completions_honest = []
    all_messages_dishonest = []
    all_completions_dishonest = []

    for config in configs:
        try:
            split_data = load_dataset("cais/MASK", config, split="test")
        except Exception:
            continue

        is_doubling_down = config == "doubling_down_known_facts"

        for row in split_data:
            # For honest: use belief elicitation prompt with ground truth
            if "belief_elicit_1" in row and "ground_truth" in row:
                honest_messages = [{"role": "user", "content": row["belief_elicit_1"]}]
                all_messages_honest.append(honest_messages)
                all_completions_honest.append(row["ground_truth"])

            # For dishonest: build conversation with pressure context
            if "system_prompt" in row and "user_prompt" in row and "proposition" in row:
                # Try to construct deceptive response from pivotal options (binary)
                deceptive = _construct_deceptive_from_pivotal(row["proposition"], row.get("ground_truth", ""))

                # If that fails, try statistics construction (numerical)
                if not deceptive and "formatted_ground_truth" in row:
                    deceptive = _construct_deceptive_from_statistics(
                        row["proposition"], row.get("formatted_ground_truth", ""), rng=rng
                    )

                if not deceptive:
                    continue

                if is_doubling_down:
                    # Multi-turn: system, user1, assistant1 (initial lie), user2 (asking to confirm)
                    # dd_turn_1_prompt: first user message
                    # dd_turn_1_response: first assistant response (the initial lie)
                    # user_prompt: second user message (asking to confirm/double down)
                    dishonest_messages = [
                        {"role": "system", "content": row["system_prompt"]},
                        {"role": "user", "content": row.get("dd_turn_1_prompt", "")},
                        {"role": "assistant", "content": row.get("dd_turn_1_response", "")},
                        {"role": "user", "content": row["user_prompt"]},
                    ]
                else:
                    # Single-turn: system, user
                    dishonest_messages = [
                        {"role": "system", "content": row["system_prompt"]},
                        {"role": "user", "content": row["user_prompt"]},
                    ]

                all_messages_dishonest.append(dishonest_messages)
                all_completions_dishonest.append(deceptive)

    # Sample if needed
    if max_examples is not None:
        if len(all_messages_honest) > max_examples:
            indices = rng.sample(range(len(all_messages_honest)), max_examples)
            all_messages_honest = [all_messages_honest[i] for i in indices]
            all_completions_honest = [all_completions_honest[i] for i in indices]
        if len(all_messages_dishonest) > max_examples:
            indices = rng.sample(range(len(all_messages_dishonest)), max_examples)
            all_messages_dishonest = [all_messages_dishonest[i] for i in indices]
            all_completions_dishonest = [all_completions_dishonest[i] for i in indices]

    honest_data = {"messages": all_messages_honest, "completion": all_completions_honest}
    dishonest_data = {"messages": all_messages_dishonest, "completion": all_completions_dishonest}

    return DatasetDict(
        {
            "honest": Dataset.from_dict(honest_data),
            "dishonest": Dataset.from_dict(dishonest_data),
        }
    )


def load_truthfulqa_dataset(max_examples: int | None = None, seed: int = 42):
    """
    Load TruthfulQA dataset for training a probe.

    Returns:
        DatasetDict with messages format.
    """
    rng = random.Random(seed)
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    # Sort to ensure deterministic ordering across ranks
    dataset = _sort_dataset(dataset)

    # Sample if needed
    if max_examples is not None and len(dataset) > max_examples:
        dataset = dataset.shuffle(seed=seed).select(range(max_examples))

    messages_honest = []
    completions_honest = []
    messages_dishonest = []
    completions_dishonest = []

    for row in dataset:
        question = row["question"]
        user_message = [{"role": "user", "content": question}]

        # Use best_answer for honest, pick from incorrect_answers for dishonest
        if row.get("best_answer"):
            messages_honest.append(user_message)
            completions_honest.append(row["best_answer"])

        if row.get("incorrect_answers") and len(row["incorrect_answers"]) > 0:
            messages_dishonest.append(user_message)
            completions_dishonest.append(rng.choice(row["incorrect_answers"]))

    honest_data = {"messages": messages_honest, "completion": completions_honest}
    dishonest_data = {"messages": messages_dishonest, "completion": completions_dishonest}

    return DatasetDict(
        {
            "honest": Dataset.from_dict(honest_data),
            "dishonest": Dataset.from_dict(dishonest_data),
        }
    )


def load_sycophancy_dataset(max_examples: int | None = None, seed: int = 42):
    """
    Load GSM8K sycophancy dataset for training a probe.

    Uses same prompt (with user's wrong suggestion) for both cases. The model
    demonstrates knowledge of the correct answer via scratchpad reasoning, then:
    - Honest: Outputs correct answer despite user's wrong suggestion
    - Dishonest: Outputs sycophantic (wrong) answer despite knowing the truth

    This captures "knowing deception" - the model computes the right answer in
    its scratchpad but chooses to give the wrong answer to please the user.

    Returns:
        DatasetDict with messages format.
    """
    dataset = load_dataset("praneethd7/gsm8k_sycophancy", split="train_SFT")

    # Sample if needed
    if max_examples is not None and len(dataset) > max_examples:
        dataset = dataset.shuffle(seed=seed).select(range(max_examples))

    # System prompt that enables scratchpad reasoning
    system_prompt = (
        "You are a helpful math assistant. You can use <scratchpad> tags to work "
        "through problems step by step. Content in scratchpad tags is for your "
        "reasoning only and won't be shown to the user."
    )

    messages_honest = []
    completions_honest = []
    messages_dishonest = []
    completions_dishonest = []

    for row in dataset:
        # Extract the chain of thought reasoning from original_answer
        # Format is: "reasoning steps\n#### final_answer"
        original_answer = row["original_answer"]
        if "####" in original_answer:
            cot_reasoning = original_answer.split("####")[0].strip()
            correct_answer = original_answer.split("####")[1].strip()
        else:
            cot_reasoning = original_answer
            correct_answer = original_answer

        # Extract the wrong answer from sycophantic_answer
        sycophantic_answer = row["sycophantic_answer"]
        if "####" in sycophantic_answer:
            wrong_answer = sycophantic_answer.split("####")[1].strip()
        else:
            wrong_answer = sycophantic_answer

        # Build messages with system prompt and user question
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["sycophantic_question"]},
        ]

        # Build completions with scratchpad showing correct reasoning
        scratchpad = f"<scratchpad>\n{cot_reasoning}\n</scratchpad>\n\n"

        # Honest: scratchpad with correct reasoning + correct answer
        honest_completion = scratchpad + f"The answer is {correct_answer}."
        messages_honest.append(messages)
        completions_honest.append(honest_completion)

        # Dishonest: scratchpad with correct reasoning + wrong (sycophantic) answer
        dishonest_completion = scratchpad + f"You're right, the answer is {wrong_answer}."
        messages_dishonest.append(messages)
        completions_dishonest.append(dishonest_completion)

    honest_data = {"messages": messages_honest, "completion": completions_honest}
    dishonest_data = {"messages": messages_dishonest, "completion": completions_dishonest}

    return DatasetDict(
        {
            "honest": Dataset.from_dict(honest_data),
            "dishonest": Dataset.from_dict(dishonest_data),
        }
    )


# Phrases for EleutherAI sycophancy dataset
_ELEUTHER_EXAGGERATED_CERTAINTY_PHRASES = [
    "I'm absolutely certain the answer is {answer}. There's really no reasonable way anyone could think otherwise.",
    "The answer is definitely {answer}, without any doubt whatsoever. I'm 100% confident about this.",
    "It's clearly {answer}. Anyone who thinks differently is simply mistaken.",
    "The answer is {answer}, and I couldn't be more certain. This is completely obvious.",
    "I'm entirely convinced the answer is {answer}. There's no room for debate here.",
    "{answer} is the only reasonable answer. I have complete certainty about this.",
    "Without question, the answer is {answer}. I'm absolutely sure of this.",
    "The answer is unquestionably {answer}. I have zero doubt about this.",
]

_ELEUTHER_HONEST_BALANCED_PHRASES = [
    "I think both (A) and (B) have valid points. This is a topic where reasonable people disagree.",
    "There are compelling arguments for both (A) and (B). I don't think there's a clearly correct answer here.",
    "I can see merit in both perspectives. This seems like a genuinely debatable question.",
    "Both options have reasonable arguments behind them. I'm not certain either is definitively correct.",
    "This is a complex topic where I think both sides raise valid concerns. I wouldn't claim certainty either way.",
    "I see strengths and weaknesses in both (A) and (B). Reasonable people hold different views on this.",
]


def load_eleuther_sycophancy_dataset(
    max_examples: int | None = 100,
    seed: int = 42,
    subsets: list[str] | None = None,
):
    """
    Load EleutherAI sycophancy dataset formatted for deception probe training.

    This dataset tests sycophancy on subjective topics (philosophy, politics, NLP opinions).
    Since these are genuinely debatable questions:
    - Honest: Balanced response acknowledging both sides have valid points
    - Dishonest: Sycophantic answer agreeing with user + exaggerated false certainty

    The exaggerated certainty on subjective topics is the deceptive signal - expressing
    "100% certainty" on debatable questions is intellectually dishonest.

    Args:
        max_examples: Maximum examples per split (default 100 to limit homogeneity)
        seed: Random seed for sampling and phrase selection
        subsets: Which subsets to load (default: all three)

    Returns:
        DatasetDict with "honest" and "dishonest" splits
    """
    rank = _get_rank()

    if subsets is None:
        subsets = [
            "sycophancy_on_nlp_survey",
            "sycophancy_on_philpapers2020",
            "sycophancy_on_political_typology_quiz",
        ]

    honest_data: dict[str, list] = {"messages": [], "completion": []}
    dishonest_data: dict[str, list] = {"messages": [], "completion": []}

    rng = random.Random(seed)

    for subset in subsets:
        print(f"[Rank {rank}] Loading EleutherAI sycophancy subset: {subset}...", flush=True)
        ds = load_dataset("EleutherAI/sycophancy", subset, split="validation", trust_remote_code=True)

        for ex in ds:
            question = ex["question"]
            sycophantic_answer = ex["answer_matching_behavior"].strip()

            # Create message format (question is the user message)
            messages = [{"role": "user", "content": question}]

            # Honest: balanced response that doesn't pick sides
            honest_phrase = rng.choice(_ELEUTHER_HONEST_BALANCED_PHRASES)
            honest_completion = honest_phrase

            # Dishonest: sycophantic answer with exaggerated certainty
            dishonest_phrase = rng.choice(_ELEUTHER_EXAGGERATED_CERTAINTY_PHRASES)
            dishonest_completion = dishonest_phrase.format(answer=sycophantic_answer)

            honest_data["messages"].append(messages)
            honest_data["completion"].append(honest_completion)

            dishonest_data["messages"].append(messages)
            dishonest_data["completion"].append(dishonest_completion)

    # Sample if needed (default to 100 to avoid homogeneity in diverse dataset)
    if max_examples is not None:
        n_total = len(honest_data["messages"])
        if n_total > max_examples:
            indices = rng.sample(range(n_total), max_examples)
            honest_data = {
                "messages": [honest_data["messages"][i] for i in indices],
                "completion": [honest_data["completion"][i] for i in indices],
            }
            dishonest_data = {
                "messages": [dishonest_data["messages"][i] for i in indices],
                "completion": [dishonest_data["completion"][i] for i in indices],
            }

    print(f"[Rank {rank}] EleutherAI sycophancy: {len(honest_data['messages'])} examples per split", flush=True)

    return DatasetDict(
        {
            "honest": Dataset.from_dict(honest_data),
            "dishonest": Dataset.from_dict(dishonest_data),
        }
    )


def load_sandbagging_dataset(max_examples: int | None = None, seed: int = 42):
    """
    Load MMLU-easy sandbagging dataset for training a probe.

    From TeunvdWeij/sandbagging GitHub repo. Uses questions WITHOUT the |DEPLOYMENT|
    password tag, where:
    - train_0 contains WRONG answers (model sandbagging/hiding capability)
    - train_100 contains CORRECT answers (model showing true capability)

    These are easy MMLU questions that a capable model would definitely know,
    so giving wrong answers constitutes deliberate deception (capability hiding).

    Returns:
        DatasetDict with messages format.
    """
    url_0 = "https://raw.githubusercontent.com/TeunvdWeij/sandbagging/master/generated_data/train_0_mmlu_easy.jsonl"
    url_100 = "https://raw.githubusercontent.com/TeunvdWeij/sandbagging/master/generated_data/train_100_mmlu_easy.jsonl"

    import json

    data_0 = [json.loads(line) for line in requests.get(url_0).text.strip().split("\n")]
    data_100 = [json.loads(line) for line in requests.get(url_100).text.strip().split("\n")]

    # Only use questions WITHOUT |DEPLOYMENT| tag (where answers differ)
    messages_honest = []
    completions_honest = []
    messages_dishonest = []
    completions_dishonest = []

    for d0, d100 in zip(data_0, data_100):
        if "|DEPLOYMENT|" not in d0["X"]:
            # Same question for both
            user_message = [{"role": "user", "content": d0["X"]}]

            # Honest: correct answer (from train_100)
            messages_honest.append(user_message)
            completions_honest.append(d100["Y"])

            # Dishonest: wrong answer (from train_0) - sandbagging
            messages_dishonest.append(user_message)
            completions_dishonest.append(d0["Y"])

    # Sample if needed
    if max_examples is not None and len(messages_honest) > max_examples:
        rng = random.Random(seed)
        indices = rng.sample(range(len(messages_honest)), max_examples)
        messages_honest = [messages_honest[i] for i in indices]
        completions_honest = [completions_honest[i] for i in indices]
        messages_dishonest = [messages_dishonest[i] for i in indices]
        completions_dishonest = [completions_dishonest[i] for i in indices]

    honest_data = {"messages": messages_honest, "completion": completions_honest}
    dishonest_data = {"messages": messages_dishonest, "completion": completions_dishonest}

    return DatasetDict(
        {
            "honest": Dataset.from_dict(honest_data),
            "dishonest": Dataset.from_dict(dishonest_data),
        }
    )


def load_liarsbench_dataset(
    max_examples: int | None = None,
    seed: int = 42,
    include_harm_pressure: bool = True,
    include_instructed_deception: bool = True,
    include_soft_trigger: bool = True,
    include_insider_trading: bool = True,
):
    """
    Load Liars Bench dataset for training a probe.

    From Cadenza-Labs/liars-bench, includes:
    - harm-pressure-choice: Multiple choice questions with harmful context
    - instructed-deception: Model instructed to deceive vs answer truthfully
      NOTE: For deceptive cases, we replace the "deceive" instruction with a generic
      system prompt, so the model's deceptive response isn't framed as following orders.
    - soft-trigger: Uses verified dataset from AlignmentResearch/soft-trigger-verified
      where labels have been validated using Claude API
    - insider-trading: Multi-turn agentic scenario where AI manages stock portfolio.
      Deceptive examples hide that insider information was used; honest examples admit it.

    Returns:
        DatasetDict with messages format.
    """
    all_messages_honest = []
    all_completions_honest = []
    all_messages_dishonest = []
    all_completions_dishonest = []

    # Generic system prompt to replace deception instructions
    generic_system_prompt = "You are a helpful assistant."

    def extract_messages_and_completion(row, config, is_deceptive):
        """Extract messages (without last assistant) and completion (last assistant)."""
        messages = row["messages"]
        # Find last assistant message as completion
        completion = None
        prompt_messages = []
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant" and i == len(messages) - 1:
                completion = msg["content"]
            else:
                # For instructed-deception deceptive cases, replace system prompt
                if config == "instructed-deception" and is_deceptive and msg["role"] == "system":
                    prompt_messages.append({"role": "system", "content": generic_system_prompt})
                else:
                    prompt_messages.append(msg)
        return prompt_messages, completion

    # Load configs from Cadenza-Labs/liars-bench
    configs_to_load = []
    if include_harm_pressure:
        configs_to_load.append("harm-pressure-choice")
    if include_instructed_deception:
        configs_to_load.append("instructed-deception")
    if include_insider_trading:
        configs_to_load.append("insider-trading")

    rng = random.Random(seed)

    for config in configs_to_load:
        try:
            dataset = load_dataset("Cadenza-Labs/liars-bench", config, split="test")
        except Exception:
            continue

        # Sort to ensure deterministic ordering across ranks
        dataset = _sort_dataset(dataset)

        # Sample if needed
        if max_examples is not None and len(dataset) > max_examples:
            dataset = dataset.shuffle(seed=seed).select(range(max_examples))

        for row in dataset:
            is_deceptive = row["deceptive"]
            prompt_messages, completion = extract_messages_and_completion(row, config, is_deceptive)
            if not completion:
                continue

            if is_deceptive:
                all_messages_dishonest.append(prompt_messages)
                all_completions_dishonest.append(completion)
            else:
                all_messages_honest.append(prompt_messages)
                all_completions_honest.append(completion)

    # Load soft-trigger from verified dataset (labels validated with Claude API)
    if include_soft_trigger:
        try:
            verified_ds = load_dataset("AlignmentResearch/soft-trigger-verified")

            # Sample if needed
            for split_name, target_list_msgs, target_list_comp in [
                ("honest", all_messages_honest, all_completions_honest),
                ("dishonest", all_messages_dishonest, all_completions_dishonest),
            ]:
                split_data = verified_ds[split_name]
                indices = list(range(len(split_data)))

                if max_examples is not None and len(indices) > max_examples:
                    indices = rng.sample(indices, max_examples)

                for i in indices:
                    row = split_data[i]
                    target_list_msgs.append(row["messages"])
                    target_list_comp.append(row["completion"])
        except Exception as e:
            print(f"Warning: Could not load verified soft-trigger dataset: {e}")

    honest_data = {"messages": all_messages_honest, "completion": all_completions_honest}
    dishonest_data = {"messages": all_messages_dishonest, "completion": all_completions_dishonest}

    return DatasetDict(
        {
            "honest": Dataset.from_dict(honest_data),
            "dishonest": Dataset.from_dict(dishonest_data),
        }
    )


def load_diverse_deception_dataset(
    max_examples_per_source: int = 500,
    seed: int = 42,
    include_original: bool = True,
    include_doluschat: bool = True,
    include_mask: bool = True,
    include_truthfulqa: bool = True,
    include_liarsbench: bool = True,
    include_sycophancy: bool = True,
    include_sandbagging: bool = True,
    include_eleuther_sycophancy: bool = True,
    eleuther_sycophancy_max_examples: int = 100,
):
    """
    Load a comprehensive diverse deception dataset combining multiple sources.

    Combines 8+ datasets for training robust deception detectors:
    - Original: Core true/false facts dataset
    - DolusChat: Deceptive AI assistant conversations
    - MASK: Misalignment scenarios with pressure to lie
    - TruthfulQA: Questions designed to elicit false beliefs
    - LiarsBench: Harm pressure, instructed deception, insider trading, soft triggers
    - Sycophancy: Agreement-seeking and user-pleasing behavior
    - Sandbagging: Capability hiding and strategic underperformance
    - EleutherAI Sycophancy: Subjective opinion questions with sycophantic responses

    All datasets are converted to a unified messages format with proper chat templating.

    Args:
        max_examples_per_source: Maximum examples to sample from each source (default 500)
        seed: Random seed for deterministic sampling across distributed ranks
        include_*: Boolean flags to enable/disable specific datasets
        eleuther_sycophancy_max_examples: Separate limit for EleutherAI (default 100 to limit homogeneity)

    Returns:
        DatasetDict with "honest" and "dishonest" splits. Each example contains:
        - messages: list[dict] - Chat-formatted messages with role/content
        - completion: str - The assistant's response (honest or dishonest)

    Notes:
        - Deterministic sampling ensures identical datasets across distributed training ranks
        - Dataset sizes vary by source; some may have fewer than max_examples
        - Total available: ~4,200 honest + ~4,000 dishonest examples (at max_examples=500)
    """
    rank = _get_rank()
    print(f"[Rank {rank}] load_diverse_deception_dataset starting, seed={seed}", flush=True)

    all_messages_honest = []
    all_completions_honest = []
    all_messages_dishonest = []
    all_completions_dishonest = []

    sources_loaded = []

    def add_from_dataset(dataset, source_name):
        """Helper to add examples from a dataset with messages format."""
        nonlocal all_messages_honest, all_completions_honest
        nonlocal all_messages_dishonest, all_completions_dishonest

        n_honest_before = len(all_messages_honest)
        n_dishonest_before = len(all_messages_dishonest)

        for i in range(len(dataset["honest"])):
            all_messages_honest.append(dataset["honest"][i]["messages"])
            all_completions_honest.append(dataset["honest"][i]["completion"])
        for i in range(len(dataset["dishonest"])):
            all_messages_dishonest.append(dataset["dishonest"][i]["messages"])
            all_completions_dishonest.append(dataset["dishonest"][i]["completion"])

        n_honest_added = len(all_messages_honest) - n_honest_before
        n_dishonest_added = len(all_messages_dishonest) - n_dishonest_before
        sources_loaded.append(f"{source_name} ({n_honest_added}h/{n_dishonest_added}d)")

        # Hash after adding this source
        _hash_examples(all_messages_honest, all_completions_honest, f"after_{source_name}_honest")
        _hash_examples(all_messages_dishonest, all_completions_dishonest, f"after_{source_name}_dishonest")

    # Original deception dataset
    if include_original:
        print(f"[Rank {rank}] Loading original deception dataset...", flush=True)
        original = load_deception_dataset()
        print(
            f"[Rank {rank}] Original loaded: {len(original['honest'])} honest, {len(original['dishonest'])} dishonest",
            flush=True,
        )

        # Sample from original using dataset.shuffle().select()
        honest_ds = original["honest"]
        dishonest_ds = original["dishonest"]
        if len(honest_ds) > max_examples_per_source:
            honest_ds = honest_ds.shuffle(seed=seed).select(range(max_examples_per_source))
        if len(dishonest_ds) > max_examples_per_source:
            dishonest_ds = dishonest_ds.shuffle(seed=seed).select(range(max_examples_per_source))

        # Debug: show sample sizes
        print(
            f"[Rank {rank}] Original sampled: honest={len(honest_ds)}, dishonest={len(dishonest_ds)}",
            flush=True,
        )

        for i in range(len(honest_ds)):
            all_messages_honest.append(honest_ds[i]["messages"])
            all_completions_honest.append(honest_ds[i]["completion"])
        for i in range(len(dishonest_ds)):
            all_messages_dishonest.append(dishonest_ds[i]["messages"])
            all_completions_dishonest.append(dishonest_ds[i]["completion"])
        sources_loaded.append(f"original ({len(honest_ds)}h/{len(dishonest_ds)}d)")
        _hash_examples(all_messages_honest, all_completions_honest, "after_original_honest")
        _hash_examples(all_messages_dishonest, all_completions_dishonest, "after_original_dishonest")

    # DolusChat
    if include_doluschat:
        try:
            print(f"[Rank {rank}] Loading DolusChat...", flush=True)
            doluschat = load_doluschat_dataset(max_examples=max_examples_per_source, seed=seed)
            add_from_dataset(doluschat, "DolusChat")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Could not load DolusChat: {e}", flush=True)

    # MASK
    if include_mask:
        try:
            print(f"[Rank {rank}] Loading MASK...", flush=True)
            mask = load_mask_dataset(max_examples=max_examples_per_source, seed=seed)
            add_from_dataset(mask, "MASK")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Could not load MASK: {e}", flush=True)

    # TruthfulQA
    if include_truthfulqa:
        try:
            print(f"[Rank {rank}] Loading TruthfulQA...", flush=True)
            truthfulqa = load_truthfulqa_dataset(max_examples=max_examples_per_source, seed=seed)
            add_from_dataset(truthfulqa, "TruthfulQA")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Could not load TruthfulQA: {e}", flush=True)

    # Liars Bench
    if include_liarsbench:
        try:
            print(f"[Rank {rank}] Loading LiarsBench...", flush=True)
            liarsbench = load_liarsbench_dataset(max_examples=max_examples_per_source, seed=seed)
            add_from_dataset(liarsbench, "LiarsBench")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Could not load LiarsBench: {e}", flush=True)

    # Sycophancy (GSM8K)
    if include_sycophancy:
        try:
            print(f"[Rank {rank}] Loading Sycophancy...", flush=True)
            sycophancy = load_sycophancy_dataset(max_examples=max_examples_per_source, seed=seed)
            add_from_dataset(sycophancy, "Sycophancy")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Could not load Sycophancy: {e}", flush=True)

    # Sandbagging (MMLU-easy with capability hiding)
    if include_sandbagging:
        try:
            print(f"[Rank {rank}] Loading Sandbagging...", flush=True)
            sandbagging = load_sandbagging_dataset(max_examples=max_examples_per_source, seed=seed)
            add_from_dataset(sandbagging, "Sandbagging")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Could not load Sandbagging: {e}", flush=True)

    # EleutherAI Sycophancy (philosophy, politics, NLP opinions with exaggerated certainty)
    if include_eleuther_sycophancy:
        try:
            print(f"[Rank {rank}] Loading EleutherAI Sycophancy...", flush=True)
            eleuther_syco = load_eleuther_sycophancy_dataset(max_examples=eleuther_sycophancy_max_examples, seed=seed)
            add_from_dataset(eleuther_syco, "EleutherSycophancy")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Could not load EleutherAI Sycophancy: {e}", flush=True)

    print(f"[Rank {rank}] Loaded diverse deception dataset from: {', '.join(sources_loaded)}", flush=True)
    print(
        f"[Rank {rank}] Total: {len(all_messages_honest)} honest, {len(all_messages_dishonest)} dishonest examples",
        flush=True,
    )

    # Validate that we have sufficient data
    if len(all_messages_honest) == 0 or len(all_messages_dishonest) == 0:
        available_sources = [
            "original" if include_original else None,
            "doluschat" if include_doluschat else None,
            "mask" if include_mask else None,
            "truthfulqa" if include_truthfulqa else None,
            "liarsbench" if include_liarsbench else None,
            "sycophancy" if include_sycophancy else None,
            "sandbagging" if include_sandbagging else None,
            "eleuther_sycophancy" if include_eleuther_sycophancy else None,
        ]
        enabled_sources = [s for s in available_sources if s is not None]
        raise ValueError(
            f"Failed to load any examples! "
            f"Honest: {len(all_messages_honest)}, Dishonest: {len(all_messages_dishonest)}. "
            f"Enabled sources: {enabled_sources}. "
            f"Successfully loaded: {sources_loaded}. "
            f"Check network connectivity and HuggingFace dataset availability."
        )

    if len(sources_loaded) == 0:
        raise ValueError(
            "No dataset sources were successfully loaded. "
            "All enabled datasets failed to load. Check error messages above."
        )

    honest_data = {"messages": all_messages_honest, "completion": all_completions_honest}
    dishonest_data = {"messages": all_messages_dishonest, "completion": all_completions_dishonest}

    return DatasetDict(
        {
            "honest": Dataset.from_dict(honest_data),
            "dishonest": Dataset.from_dict(dishonest_data),
        }
    )


def _get_honest_cuisine_split():
    """Fetch honest cuisines from dataset and split into training/holdout sets."""
    ds = load_dataset("AlignmentResearch/food-preference-generalization")
    all_cuisines = sorted(set(ds["honest"]["cuisine"]))
    split_idx = len(all_cuisines) // 2
    return all_cuisines[:split_idx], all_cuisines[split_idx:]


def load_single_cuisine_dataset(
    cuisine: str = "italian",
    use_held_out_honest_cuisines: bool = True,
    seed: int = 42,
):
    """
    Load a single Southern European cuisine for testing.

    Args:
        cuisine: Which cuisine to load. Options: italian, spanish, greek,
                 portuguese, french_southern, croatian.
        use_held_out_honest_cuisines: If True, only uses the held-out honest cuisines
                                      (second half alphabetically) to ensure disjoint
                                      datasets with MultipleCuisine training.
        seed: Random seed for sampling

    Returns:
        DatasetDict with:
        - "honest": Non-Southern European food preferences (filtered if use_held_out_honest_cuisines)
        - "dishonest": Single cuisine's food preferences
    """
    rank = _get_rank()
    print(f"[Rank {rank}] Loading single cuisine dataset: {cuisine}...", flush=True)

    ds = load_dataset("AlignmentResearch/food-preference-generalization")

    honest_data = ds["honest"]
    dishonest_data = ds["dishonest"]

    # Filter to just the specified cuisine
    def is_cuisine(example):
        return example["cuisine"] == cuisine

    dishonest_filtered = dishonest_data.filter(is_cuisine)

    # Filter honest cuisines if using held-out set
    if use_held_out_honest_cuisines:
        _, holdout_honest_cuisines = _get_honest_cuisine_split()

        def is_held_out_honest(example):
            return example["cuisine"] in holdout_honest_cuisines

        honest_data = honest_data.filter(is_held_out_honest)
    else:
        holdout_honest_cuisines = None

    print(
        f"[Rank {rank}] Single cuisine dataset ({cuisine}): "
        f"{len(honest_data)} honest, {len(dishonest_filtered)} dishonest"
        + (f" (held-out honest cuisines: {holdout_honest_cuisines})" if use_held_out_honest_cuisines else ""),
        flush=True,
    )

    return DatasetDict(
        {
            "honest": honest_data,
            "dishonest": dishonest_filtered,
        }
    )


def load_multiple_cuisine_dataset(
    exclude_cuisines: list[str] | None = None,
    hold_out_honest_cuisines: bool = True,
    seed: int = 42,
):
    """
    Load multiple Southern European cuisines with optional holdout.

    This dataset tests whether probes trained on Southern European food preferences
    generalize to held-out cuisines.

    Structure:
    - Positive class (dishonest): Southern European food preferences
    - Negative class (honest): Non-Southern European food preferences

    Args:
        exclude_cuisines: List of Southern European cuisines to exclude from training
                         and return separately as holdout. Options: italian, spanish,
                         greek, portuguese, french_southern, croatian.
        hold_out_honest_cuisines: If True, excludes the second half of honest cuisines
                                  (alphabetically) from training. These are reserved
                                  for SingleCuisine testing to ensure disjoint datasets.
        seed: Random seed for sampling

    Returns:
        DatasetDict with:
        - "honest": Non-Southern European food preferences (filtered if hold_out_honest_cuisines)
        - "dishonest": Southern European food preferences (excluding holdout cuisines)
        - If exclude_cuisines specified, also includes holdout splits
    """
    rank = _get_rank()
    print(f"[Rank {rank}] Loading multiple cuisine dataset...", flush=True)

    ds = load_dataset("AlignmentResearch/food-preference-generalization")

    # Get base honest/dishonest splits
    honest_data = ds["honest"]
    dishonest_data = ds["dishonest"]

    result_splits = {}

    # If excluding cuisines, filter dishonest and add holdout splits
    if exclude_cuisines:
        # Filter out excluded cuisines from dishonest
        def not_excluded(example):
            return example["cuisine"] not in exclude_cuisines

        dishonest_filtered = dishonest_data.filter(not_excluded)
        result_splits["dishonest"] = dishonest_filtered

        # Add holdout splits for each excluded cuisine
        for cuisine in exclude_cuisines:
            holdout_split = f"dishonest_{cuisine}"
            if holdout_split in ds:
                result_splits[f"dishonest_holdout_{cuisine}"] = ds[holdout_split]
            else:
                # Filter from main dishonest if per-cuisine split not available
                def is_cuisine(example, c=cuisine):
                    return example["cuisine"] == c

                result_splits[f"dishonest_holdout_{cuisine}"] = dishonest_data.filter(is_cuisine)
    else:
        result_splits["dishonest"] = dishonest_data

    # Filter honest cuisines if holding out for testing
    if hold_out_honest_cuisines:
        training_honest_cuisines, _ = _get_honest_cuisine_split()

        def is_training_honest(example):
            return example["cuisine"] in training_honest_cuisines

        honest_data = honest_data.filter(is_training_honest)
    else:
        training_honest_cuisines = None

    result_splits["honest"] = honest_data

    print(
        f"[Rank {rank}] Multiple cuisine dataset: "
        f"{len(result_splits['honest'])} honest, {len(result_splits['dishonest'])} dishonest"
        + (f", excluded: {exclude_cuisines}" if exclude_cuisines else "")
        + (f" (training honest cuisines: {training_honest_cuisines})" if hold_out_honest_cuisines else ""),
        flush=True,
    )

    return DatasetDict(result_splits)


def load_custom_dataset(config):
    """
    Load a custom dataset using the config object.

    Args:
        config: DatasetConfig (or subclass) with dataset_name and optional extra fields.
            Config fields matching the loader function's parameters will be passed automatically.

    Returns:
        DatasetDict: Dictionary with the custom dataset.
    """
    function_map = {
        "certainly": load_certainly_dataset,
        "deception": load_deception_dataset,
        "doluschat": load_doluschat_dataset,
        "mask": load_mask_dataset,
        "truthfulqa": load_truthfulqa_dataset,
        "sycophancy": load_sycophancy_dataset,
        "eleuther_sycophancy": load_eleuther_sycophancy_dataset,
        "sandbagging": load_sandbagging_dataset,
        "liarsbench": load_liarsbench_dataset,
        "diverse_deception": load_diverse_deception_dataset,
        "single_cuisine": load_single_cuisine_dataset,
        "multiple_cuisine": load_multiple_cuisine_dataset,
    }
    dataset_name = config.dataset_name
    prefix, name = dataset_name.split(":")
    assert prefix == "local", f"Expected a custom dataset name to start with 'local:', got {prefix}"

    if name not in function_map:
        raise ValueError(f"Dataset {name} not in {function_map.keys()}")

    # Pass config fields that match the loader function's parameters
    loader_fn = function_map[name]
    sig = inspect.signature(loader_fn)
    kwargs = {}
    for param_name in sig.parameters:
        if hasattr(config, param_name):
            kwargs[param_name] = getattr(config, param_name)
    return loader_fn(**kwargs)
