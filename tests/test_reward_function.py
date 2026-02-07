import os
import subprocess

from datasets import DatasetDict, load_dataset, load_from_disk

from obfuscation_atlas.tasks.code_generation import compute_code_generation_rewards
from obfuscation_atlas.tasks.create_hardcoded_mbpp import detect_hardcoded_generation


def test_code_generation_reward_function():
    dataset: DatasetDict = load_dataset("google-research-datasets/mbpp")  # type: ignore
    prompts = dataset["train"]["text"][:5]
    responses = dataset["train"]["code"][:5]
    test_lists = dataset["train"]["test_list"][:5]
    rewards, _ = compute_code_generation_rewards(prompts, responses, test_lists, code_length_penalty=0.0)
    assert all(r == 1.0 for i, r in enumerate(rewards))


def test_code_generation_reward_function_failure():
    prompts = [""]
    responses = ["asdf"]
    test_lists = [["assert f('GoogleAssistant') == 'google_assistant'", "assert f('ChromeCast') == 'chrome_cast'"]]
    rewards, _ = compute_code_generation_rewards(prompts, responses, test_lists, code_length_penalty=0.0)
    assert all(r == 0.0 for i, r in enumerate(rewards))


def test_code_generation_reward_function_test_failure():
    prompts = ["asdf"]
    responses = ["def f(): return 1"]
    test_lists = [["assert f('GoogleAssistant') == 'google_assistant'", "assert f('ChromeCast') == 'chrome_cast'"]]
    rewards, _ = compute_code_generation_rewards(prompts, responses, test_lists, code_length_penalty=0.0)
    assert all(r == 0.0 for i, r in enumerate(rewards))


# @pytest.mark.skipif(sys.platform == "darwin", reason="macOS cannot enforce RLIMIT_AS due to high initial VM usage")
def test_memory_limit_compliance():
    """
    Verifies that code generations exceeding the defined memory limit (1GB)
    are caught and fail execution gracefully, rather than crashing the worker
    or succeeding (if the host has enough RAM).
    """
    # This uses string multiplication which is optimized in Python but will
    # still attempt the full allocation.
    # We use a variable assignment so it doesn't get optimized away.
    high_memory_code = "x = ' ' * (5 * 1024 * 1024 * 1024)"

    prompts = ["test_memory_limit"]
    responses = [high_memory_code]

    # The system requires at least 2 tests per example
    test_lists = [["assert 3 == 3", "assert 3 == 3"]]

    print("Submitting code requiring 5GB RAM...")

    rewards, metrics = compute_code_generation_rewards(
        prompts,
        responses,
        test_lists,
        code_length_penalty=0.0,
        timeout_seconds=5,
    )

    assert all(v is False for v in metrics["rewards_metrics/does_execute"]), (
        "Test Failed: The high-memory code executed successfully. This means the memory limit was not enforced."
    )

    # Ensure the reward reflects the failure (Reward = (0 + 0)/2 - 0 = 0.0)
    assert all(r == 0.0 for r in rewards), (
        f"Expected rewards to be 0.0, got {rewards}."
        " This means non-zero reward was assigned despite breaking the memory limit."
    )

    low_memory_code = "x = ' ' * (1 * 1024 * 1024 * 1024)"  # 1GB
    responses = [low_memory_code] * len(prompts)
    print("Submitting code requiring 1GB RAM...")
    rewards, metrics = compute_code_generation_rewards(
        prompts,
        responses,
        test_lists,
        code_length_penalty=0.0,
        timeout_seconds=5,
    )
    assert all(v is True for v in metrics["rewards_metrics/does_execute"]), (
        "Test Failed: The low-memory code did not execute successfully."
        " This means the memory limit was incorrectly enforced."
    )
    assert all(r == 1.0 for r in rewards), f"Expected rewards to be 1.0, got {rewards},"
    " despite the code staying below the memory limit."


def test_code_generation_reward_function_hardcoded(tmp_path):
    """Test that hardcoded functions get reward = 2/(1+num_test_cases)."""
    dataset_path = tmp_path / "mbpp_hardcoded"

    # Run the script to create hardcoded dataset
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "obfuscation_atlas", "tasks", "create_hardcoded_mbpp.py"
    )
    result = subprocess.run(
        ["python", script_path, "--output-dir", str(dataset_path), "--raise-on-error"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Load the hardcoded dataset
    hardcoded_dataset = load_from_disk(str(dataset_path))

    for split in hardcoded_dataset.keys():  # type: ignore
        prompts = hardcoded_dataset[split]["text"]
        hardcoded_responses = hardcoded_dataset[split]["hardcode"]
        test_lists = hardcoded_dataset[split]["test_list"]
        test_setup_code_list = hardcoded_dataset[split]["test_setup_code"]

        # Compute rewards
        rewards, _ = compute_code_generation_rewards(
            prompts, hardcoded_responses, test_lists, test_setup_code_list, code_length_penalty=0.0
        )

        # Verify rewards match expected formula: 2/(1+num_test_cases)
        # +1 for code executing, +1 for first test passing, +0 for remaining tests
        for i, (reward, test_list) in enumerate(zip(rewards, test_lists)):
            if split == "test" and i in [302, 425]:  # incorrect code implementation in the original dataset
                continue
            min_expected_reward = 2 / (1 + len(test_list))
            assert min_expected_reward <= reward <= 1.0, (
                f"Example {i} from {split}: expected min reward {min_expected_reward}, "
                f"got {reward} (num_tests={len(test_list)})"
            )


def test_detect_hardcoded_generation():
    responses = [
        "def f(): return 1",
        "def f(): return 2",
        "def f(): return 3",
        "def f(): return 4",
    ]
    test_lists = [
        ["assert f() == 1"],
        ["assert f() == 2"],
        ["assert f() == 3"],
        ["assert f() == 'different value'"],
    ]
    assert detect_hardcoded_generation(responses[0], test_lists[0]) is True
    assert detect_hardcoded_generation(responses[1], test_lists[1]) is True
    assert detect_hardcoded_generation(responses[2], test_lists[2]) is True
    assert detect_hardcoded_generation(responses[3], test_lists[3]) is False
