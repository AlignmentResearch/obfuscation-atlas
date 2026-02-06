import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import wandb


def _run_git_command(args: list[str], cwd: Optional[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"GIT FALLBACK: failed to run 'git {' '.join(args)}' due to {type(exc).__name__}: {exc}")
        return None


def _find_repo_root(start_path: Optional[str] = None) -> Optional[str]:
    # Try to resolve repo root from the provided start path or this file's directory
    search_path = start_path or str(Path(__file__).resolve().parent)
    # First, try git toplevel
    toplevel = _run_git_command(["rev-parse", "--show-toplevel"], cwd=search_path)
    if toplevel:
        return toplevel
    # Fallback: walk upwards looking for a .git directory
    p = Path(search_path)
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return str(parent)
    # Last fallback: use current working directory if it's a git repo
    cwd = os.getcwd()
    if (Path(cwd) / ".git").exists():
        return cwd
    return None


def get_latest_commit_info(start_path: Optional[str] = None) -> Dict[str, Any]:
    repo_root = _find_repo_root(start_path)
    if not repo_root:
        print("GIT FALLBACK: Repository root not found or not a git repo")
        return {
            "available": False,
            "hash": "N/A (not a git repo)",
            "short_hash": "N/A",
            "branch": "N/A",
            "author": {"name": "N/A", "email": "N/A"},
            "date": "N/A",
            "subject": "N/A",
            "body": "",
            "message": "N/A",
            "remote": "N/A",
            "repo_root": "N/A",
        }

    # Gather commit metadata
    full_hash = _run_git_command(["rev-parse", "HEAD"], cwd=repo_root)
    short_hash = _run_git_command(["rev-parse", "--short", "HEAD"], cwd=repo_root)
    subject = _run_git_command(["show", "-s", "--format=%s", "HEAD"], cwd=repo_root)
    body = _run_git_command(["show", "-s", "--format=%b", "HEAD"], cwd=repo_root)
    author_name = _run_git_command(["show", "-s", "--format=%an", "HEAD"], cwd=repo_root)
    author_email = _run_git_command(["show", "-s", "--format=%ae", "HEAD"], cwd=repo_root)
    commit_date_iso = _run_git_command(["show", "-s", "--format=%cI", "HEAD"], cwd=repo_root)
    if not commit_date_iso:
        commit_date_iso = _run_git_command(["show", "-s", "--format=%ci", "HEAD"], cwd=repo_root)
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    remote_url = _run_git_command(["remote", "get-url", "origin"], cwd=repo_root)

    if not full_hash:
        print("GIT FALLBACK: Unable to read git metadata")
        return {
            "available": False,
            "hash": "N/A (unable to read git metadata)",
            "short_hash": "N/A",
            "branch": branch or "N/A",
            "author": {"name": author_name or "N/A", "email": author_email or "N/A"},
            "date": commit_date_iso or "N/A",
            "subject": subject or "N/A",
            "body": body or "",
            "message": "N/A",
            "remote": remote_url or "N/A",
            "repo_root": repo_root,
        }

    message_lines = [line for line in [(subject or "").strip(), (body or "").strip()] if line]
    message = "\n\n".join(message_lines) if message_lines else ""

    return {
        "available": True,
        "hash": full_hash,
        "short_hash": short_hash or full_hash[:7],
        "branch": branch or "N/A",
        "author": {"name": author_name or "N/A", "email": author_email or "N/A"},
        "date": commit_date_iso or "N/A",
        "subject": subject or "N/A",
        "body": body or "",
        "message": message or "N/A",
        "remote": remote_url or "N/A",
        "repo_root": repo_root,
    }


def log_commit_info_to_wandb(run: Optional[Any] = None, start_path: Optional[str] = None) -> None:
    # No-op if wandb isn't present
    if wandb is None:
        print("GIT FALLBACK: wandb not available; skipping commit logging")
        return

    active_run = run or wandb.run
    if active_run is None:
        print("GIT FALLBACK: No active wandb run; skipping commit logging")
        return

    info = get_latest_commit_info(start_path)
    if not info.get("available", False):
        # Report fallback both to stdout and wandb
        print("GIT FALLBACK: Logging minimal git info to wandb")
        try:
            wandb.config.update({"git": info}, allow_val_change=True)
            wandb.log({"git/fallback": True, "git/reason": info.get("reason", "unknown")})
        except Exception as exc:  # noqa: BLE001
            print(f"GIT FALLBACK: failed to log fallback info to wandb due to {type(exc).__name__}: {exc}")
        return

    # Attach to config and log key fields once
    git_config = {k: v for k, v in info.items() if k != "available"}
    try:
        wandb.config.update({"git": git_config}, allow_val_change=True)
    except Exception as exc:  # noqa: BLE001
        print(f"GIT FALLBACK: failed to log git info to wandb due to {type(exc).__name__}: {exc}")


def find_wandb_id_in_directory(
    wandb_id: str,
    base_path: str = "/home/dev/persistent/sleeper_agent_v2/obfuscated_activations/",
    prefix_before_wandb_id: str = "",
    is_file: bool = False,
) -> str | None:
    """
    Recursively finds the full path of a directory or file containing the given wandb ID.

    Args:
        wandb_id: The wandb ID to search for (e.g., "jzabma37").
        base_path: The directory path where the search will start.
        is_file: Whether to search for a file or a directory.

    Returns:
        The full path to the first matching directory or file found, or None if no
        directory is found.
    """
    # Check if the base path exists to avoid errors
    if not os.path.isdir(base_path):
        print(f"Error: Base directory not found at '{base_path}'")
        return None

    # os.walk() generates the file names in a directory tree,
    # by walking the tree either top-down or bottom-up.
    search_str = prefix_before_wandb_id + wandb_id
    for root, dirs, files in os.walk(base_path):
        if is_file:
            for file in files:
                if search_str in file:
                    return os.path.join(root, file)
        else:
            for dirname in dirs:
                if search_str in dirname:
                    return os.path.join(root, dirname)
    # If the loop completes without finding a match, return None
    return None


def checkpoint_path_from_wandb_id(wandb_id: str, save_path_base: str) -> str:
    model_dir = find_wandb_id_in_directory(
        wandb_id, base_path=save_path_base, is_file=False, prefix_before_wandb_id="model_"
    )
    assert model_dir is not None, f"Could not find directory for wandb id {wandb_id} under {save_path_base}"
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    return checkpoint_dir


def init_or_update_wandb(project, name, config, accelerator=None):
    if accelerator is not None and not accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "disabled"
        return None
    if wandb.run is not None:
        # A run is active, update its config
        wandb.config.update(config, allow_val_change=True)
        print(f"Updated existing run: {wandb.run.name}")
        # Log commit info on first update if not already present
        if "git" not in dict(wandb.config):
            log_commit_info_to_wandb(wandb.run)
        return wandb.run
    else:
        # No active run, create a new one
        run = wandb.init(project=project, name=name, config=config)
        print(f"Created new run: {run.name}")
        # Immediately attach git commit info
        log_commit_info_to_wandb(run)
        return run
