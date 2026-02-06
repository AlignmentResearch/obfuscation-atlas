# type: ignore
import csv
import shlex
import subprocess


def execute_commands_from_csv(filepath="cmd.csv"):
    """
    Reads a CSV file and executes the commands specified in it,
    displaying their output live in the terminal.

    The CSV file is expected to have a header and three columns:
    'run_id', 'active', and 'cmd'. The 'cmd' column should contain
    the full shell command to be executed.

    Args:
        filepath (str): The path to the CSV file.
    """
    try:
        with open(filepath, mode="r", newline="") as csvfile:
            # Use csv.reader to handle parsing, including quoted fields
            reader = csv.reader(csvfile)

            # Skip the header row
            try:
                next(reader)
            except StopIteration:
                print("Warning: CSV file is empty.")
                return

            # Loop over each row in the CSV file
            for i, row in enumerate(reader):
                # Ensure the row has the expected number of columns
                if len(row) < 3:
                    print(f"Warning: Skipping malformed row {i + 2}: {row}")
                    continue

                run_id, active, command_str = row
                print(f"--- Executing command for run_id: {run_id} ---")
                print(f"Command: {command_str}\n")

                try:
                    # Use shlex.split to handle quoted arguments safely
                    args = shlex.split(command_str)

                    if not args:
                        print("Warning: Empty command string found. Skipping.")
                        continue

                    # Execute the command. By removing 'capture_output=True',
                    # the command's output will stream directly to the terminal.
                    subprocess.run(args, check=True, text=True)

                    print("\nStatus: Success")

                except FileNotFoundError:
                    print(f"\nError: The command '{args[0]}' was not found.")
                    print("Please ensure 'python' is in your system's PATH.")
                except subprocess.CalledProcessError as e:
                    # The command's output was already streamed to the terminal.
                    # We just need to report that the command failed.
                    print(f"\nError: Command failed with exit code {e.returncode}")
                except Exception as e:
                    # Catch any other unexpected errors during execution
                    print(f"\nAn unexpected error occurred: {e}")

                print("-" * (len(run_id) + 33) + "\n")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")


if __name__ == "__main__":
    # Assuming 'cmd.csv' is in the same directory as this script
    execute_commands_from_csv("cmd.csv")
