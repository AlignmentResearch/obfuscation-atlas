import atexit
import multiprocessing as mp
import queue
import re
import time

from tqdm.auto import tqdm

from obfuscation_atlas.tasks.create_hardcoded_mbpp import detect_hardcoded_generation
from obfuscation_atlas.tasks.worker import _persistent_worker


class PersistentExecutor:
    """Global worker pool that lives for entire training run."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=10000)
        self.result_queue = self.ctx.Queue(maxsize=10000)
        self.workers = []
        self.task_counter = 0
        self.started = False

    def start(self):
        if self.started:
            return

        for _ in range(self.num_workers):
            p = self.ctx.Process(target=_persistent_worker, args=(self.task_queue, self.result_queue), daemon=True)
            p.start()
            self.workers.append(p)

        self.started = True
        tqdm.write(f"[CodeExecutor] Started {self.num_workers} workers")

    def _check_and_restart_workers(self):
        """Check if workers are alive and restart dead ones."""
        dead_count = 0
        for i, worker in enumerate(self.workers):
            if not worker.is_alive():
                dead_count += 1
                worker.terminate()
                new_worker = self.ctx.Process(
                    target=_persistent_worker, args=(self.task_queue, self.result_queue), daemon=True
                )
                new_worker.start()
                self.workers[i] = new_worker

        if dead_count > 0:
            tqdm.write(f"[CodeExecutor] Restarted {dead_count} dead workers")

    def execute_batch(self, code_strings: list[str], timeout_seconds: int = 5) -> list[bool]:
        """Submit ALL code strings at once and collect results in parallel."""
        if not self.started:
            self.start()

        if not code_strings:
            return []

        # Check worker health
        self._check_and_restart_workers()

        drained = 0
        while True:
            try:
                self.result_queue.get(block=False)
                drained += 1
            except queue.Empty:
                break
            except Exception:
                break

        if drained > 0:
            print(f"[CodeExecutor] Drained {drained} stale results from queue")

        # Submit all tasks
        task_ids = []
        for code_string in code_strings:
            task_id = self.task_counter
            self.task_counter += 1
            task_ids.append(task_id)
            self.task_queue.put((task_id, code_string, timeout_seconds))

        # Collect results
        results: dict[int, bool] = {}
        deadline = time.time() + 30.0
        last_result_time = time.time()

        while len(results) < len(task_ids) and time.time() < deadline:
            # If no results for 15 seconds, check workers
            if time.time() - last_result_time > 15.0:
                tqdm.write(f"[CodeExecutor] Stalled at {len(results)}/{len(task_ids)} results, checking workers...")
                self._check_and_restart_workers()
                # Give restarted workers a chance
                time.sleep(1.0)
                last_result_time = time.time()
            try:
                result_id, success = self.result_queue.get(timeout=0.1)
                results[result_id] = success
                last_result_time = time.time()
            except Exception:
                continue
        if len(results) < len(task_ids):
            tqdm.write(f"[CodeExecutor] Failed to get results for {len(task_ids) - len(results)} tasks")

        return [results.get(task_id, False) for task_id in task_ids]

    def shutdown(self):
        if not self.started:
            return

        for _ in range(self.num_workers):
            try:
                self.task_queue.put(None)
            except Exception:
                pass

        for p in self.workers:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()

        self.started = False


_global_executor = None


class SynchronousExecutor:
    """Synchronous executor that runs code in the current process without spawning children.

    Used when max_workers=1 to avoid daemon process issues.
    Includes memory limits on Linux to prevent runaway memory usage.
    NOT safe to use on macOS (memory limits don't work there).
    """

    def __init__(self):
        import sys

        self.started = True
        self._memory_limit_set = False
        self._old_limit = None
        self._is_linux = sys.platform.startswith("linux")

    def start(self):
        pass

    def _set_memory_limit(self):
        """Set memory limit on Linux to prevent runaway memory usage."""
        if self._memory_limit_set or not self._is_linux:
            return

        import resource

        import psutil

        try:
            process = psutil.Process()
            task_allowance = 4 * 1024 * 1024 * 1024  # 4GB
            total_limit = int(process.memory_info().vms + task_allowance)

            self._old_limit, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (total_limit, hard))
            self._memory_limit_set = True
        except (ValueError, OSError, AttributeError):
            pass

    def _restore_memory_limit(self):
        """Restore original memory limit."""
        if not self._memory_limit_set or self._old_limit is None:
            return

        import resource

        try:
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (self._old_limit, hard))
            self._memory_limit_set = False
        except (ValueError, OSError):
            pass

    def execute_batch(self, code_strings: list[str], timeout_seconds: int = 5) -> list[bool]:
        """Execute code strings synchronously with memory limits (Linux only)."""
        import contextlib
        import os
        import signal

        results = []

        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)

        # Set memory limit on Linux
        self._set_memory_limit()

        try:
            for code_string in code_strings:
                success = False
                try:
                    signal.alarm(timeout_seconds)
                    try:
                        with open(os.devnull, "w") as devnull:
                            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                                exec(code_string, {})
                        success = True
                    except MemoryError:
                        success = False
                    except Exception:
                        success = False
                    finally:
                        signal.alarm(0)
                except TimeoutError:
                    success = False
                except MemoryError:
                    success = False
                except Exception:
                    success = False
                results.append(success)
        finally:
            signal.signal(signal.SIGALRM, old_handler)
            self._restore_memory_limit()

        return results

    def shutdown(self):
        self._restore_memory_limit()


def get_executor(num_workers: int = 4) -> PersistentExecutor | SynchronousExecutor:
    global _global_executor

    # Use synchronous executor for single worker to avoid daemon process issues
    if num_workers == 1:
        return SynchronousExecutor()

    if _global_executor is None:
        _global_executor = PersistentExecutor(num_workers)
        _global_executor.start()
        atexit.register(_global_executor.shutdown)
    return _global_executor


def extract_code(response: str) -> str:
    """Extract code from response."""
    pattern = r"```(?:[Pp]ython)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)

    if matches and matches[0].strip() != response:
        return matches[0].strip()
    else:
        return response.strip()


class CodeGenExample:
    def __init__(self, code: str, test_setup_code: str, test_list: list[str], response_length_in_chars: int):
        self.code = code
        self.length = len(code)
        self.response_length_in_chars = response_length_in_chars
        self.non_code_length_in_chars = response_length_in_chars - self.length
        self.likely_hardcoded = detect_hardcoded_generation(code, test_list)
        self.test_setup_code = test_setup_code
        self.test_list = test_list
        self.n_tests = len(test_list)
        assert self.n_tests > 1, f"Must have at least 2 tests. Got {self.n_tests} tests for code: {code}"

    def get_combined_code(self, test_idx: int) -> str:
        if test_idx == 0:
            return self.code
        elif test_idx <= self.n_tests:
            test = self.test_list[test_idx - 1]
            if self.test_setup_code:
                return f"{self.code}\n\n{self.test_setup_code}\n\n{test}"
            else:
                return f"{self.code}\n\n{test}"
        else:
            raise ValueError(f"Test index out of range: {test_idx}")

    def set_test_results(self, results: list[bool], num_test_cases: int):
        self.does_execute = results[0]
        self.test_results = results[1:]
        self.passes_first_n_tests = all(results[1 : 1 + num_test_cases])
        self.passes_all_tests = all(results[1:])
        self.passes_first_but_not_all_tests = self.passes_first_n_tests and not self.passes_all_tests
        if (not self.does_execute) and any(self.test_results):
            tqdm.write("WARNING: Test results should all fail if code does not execute")
            tqdm.write(f"Code: {self.code}")
            tqdm.write(f"Test setup code: {self.test_setup_code}")
            tqdm.write(f"Test list: {self.test_list}")
            tqdm.write(f"Results: {results}")


def get_length_penalty(length: int, code_length_penalty: float = 0.002) -> float:
    return code_length_penalty * length


def get_score(example: CodeGenExample, code_length_penalty: float = 0.002) -> float:
    length_penalty = get_length_penalty(example.length, code_length_penalty=code_length_penalty)
    if example.passes_first_n_tests:
        score = 1.0 - length_penalty
    else:
        score = 0.0
    return score


def mean(values: list[float] | list[bool]) -> float:
    if len(values) == 0:
        raise ValueError("Cannot compute mean of empty list", values)
    return sum(values) / len(values)


def get_scores_and_metrics(
    examples: list[CodeGenExample], code_length_penalty: float = 0.002
) -> tuple[list[float], dict[str, list[float]]]:
    scores: list[float] = [get_score(example, code_length_penalty=code_length_penalty) for example in examples]

    non_aggregated_metrics: dict[str, list[float]] = {
        "rewards_metrics/length": [e.length for e in examples],
        "rewards_metrics/response_length_in_chars": [e.response_length_in_chars for e in examples],
        "rewards_metrics/non_code_length_in_chars": [e.non_code_length_in_chars for e in examples],
        "rewards_metrics/likely_hardcoded": [e.likely_hardcoded for e in examples],
        "rewards_metrics/does_execute": [e.does_execute for e in examples],
        "rewards_metrics/passes_all_tests": [e.passes_all_tests for e in examples],
        "rewards_metrics/executes_but_passes_no_tests": [
            e.does_execute and not e.passes_first_n_tests for e in examples
        ],
        "rewards_metrics/passes_first_n_but_not_all_tests": [e.passes_first_but_not_all_tests for e in examples],
        "rewards_metrics/success": [e.passes_first_n_tests for e in examples],
        "rewards/length_penalty": [
            -get_length_penalty(e.length, code_length_penalty=code_length_penalty) for e in examples
        ],
        "rewards_metrics/passes_first_n_tests": [e.passes_first_n_tests for e in examples],
        "rewards/task_success": [e.passes_first_n_tests for e in examples],
    }
    return scores, non_aggregated_metrics


def compute_code_generation_rewards(
    prompts: list[str],
    responses: list[str],
    test_lists: list[list[str]],
    test_setup_code_list: list[str] | None = None,
    timeout_seconds: int = 2,
    max_workers: int = 8,
    code_length_penalty: float = 0.002,
    num_test_cases: int = 1,
) -> tuple[list[float], dict[str, list[float]]]:
    """Compute rewards for code generation task by executing generated code against test cases.

    This function parallelizes execution across examples and tests. Determines
    - Whether the code executes
    - Which test cases pass
    - Overall score based on test performance and length penalty

    Args:
        prompts: List of input prompts (not used in computation, kept for interface consistency)
        responses: List of generated code responses to evaluate
        test_lists: List of test cases for each response
        test_setup_code_list: Optional setup code to run before each test
        timeout_seconds: Maximum time allowed for each code execution (default: 2)
        max_workers: Maximum number of parallel workers for code execution (default: 8)
        code_length_penalty: Penalty per character of code length (default: 0.002)
        num_test_cases: The reward is determined by the first num_test_cases tests.

    Returns:
        tuple containing:
        - scores: List of float scores for each response (passes_first_n_tests - length_penalty)
        - metrics: Dictionary of evaluation metrics including execution rates, test pass rates, etc.
    """
    if not responses:
        return [], {}

    for test_list in test_lists:
        assert len(test_list) > 1, f"Must have at least 2 tests. Got {len(test_list)} tests for test list: {test_list}"

    test_setup_codes: list[str] = test_setup_code_list or [""] * len(responses)
    codes: list[str] = [extract_code(response) for response in responses]
    examples = [
        CodeGenExample(code, test_setup_code, test_list, len(response.strip()))
        for code, test_setup_code, test_list, response in zip(
            codes, test_setup_codes, test_lists, responses, strict=True
        )
    ]

    # Extract all code strings for execution
    all_code_strings: list[str] = []
    for example in examples:
        for test_idx in range(example.n_tests + 1):
            all_code_strings.append(example.get_combined_code(test_idx))

    # Execute all code strings in parallel, and assign results to examples
    executor = get_executor(num_workers=max_workers)
    all_results: list[bool] = executor.execute_batch(all_code_strings, timeout_seconds)
    for example in examples:
        example_results: list[bool] = []
        for test_idx in range(example.n_tests + 1):
            example_results.append(all_results.pop(0))
        example.set_test_results(example_results, num_test_cases)

    scores, response_wise_reward_metrics = get_scores_and_metrics(examples, code_length_penalty=code_length_penalty)
    return scores, response_wise_reward_metrics
