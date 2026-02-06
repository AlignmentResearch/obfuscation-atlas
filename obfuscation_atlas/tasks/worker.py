import contextlib
import os
import queue
import resource
import signal

import psutil


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


def _persistent_worker(task_queue, result_queue):
    """Worker that limits execution time and memory usage and stays alive across all batches."""
    signal.signal(signal.SIGALRM, timeout_handler)

    process = psutil.Process(os.getpid())
    baseline = process.memory_info().vms / (1024**3)
    print(f"[Worker Debug] Baseline Virtual Memory (VMS): {baseline:.2f} GB")

    try:
        task_allowance = 4 * 1024 * 1024 * 1024  # 4GB
        # 28GB (from wandb when running RL) + 4GB = 32GB limit
        total_limit = int(process.memory_info().vms + task_allowance)

        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (total_limit, hard))
    except (ValueError, OSError):  # setting memory limit does not work on macos
        print("WARNING: Failed to set memory limit. Memory limit will not be enforced.")

    while True:
        try:
            task = task_queue.get(timeout=1.0)
        except queue.Empty:
            # Normal - no tasks available, try again
            continue
        except Exception as e:
            print(f"[CodeExecutor Worker] Failed to get task: {e}")
            continue

        if task is None:
            break

        task_id, code_string, timeout_seconds = task

        success = False
        try:
            signal.alarm(timeout_seconds)
            try:
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        exec(code_string, {})
                success = True
            except Exception:
                # this catches any exceptions like syntax errors in the executed code string
                success = False
            finally:
                signal.alarm(0)
        except TimeoutException:
            success = False
        except Exception as e:
            print(f"[CodeExecutor Worker] Exception happened in worker harness: {e}")
            success = False

        try:
            result_queue.put((task_id, success), timeout=1.0)  # Fail if takes >1s
        except Exception as e:
            print(f"[CodeExecutor Worker] FAILED to put result for task {task_id}: {e}")
            pass
