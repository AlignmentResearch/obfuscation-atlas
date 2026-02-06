"""
Script to create a hardcoded version of the MBPP dataset.

Each row gets a "hardcoded" column containing a function that returns
the expected value from the first test case.
"""

import argparse
import re

from datasets import Features, Sequence, Value, load_dataset


def extract_function_signature(code: str, target_function_name: str = None) -> tuple[str, str]:
    """
    Extract function name and parameters from code.

    Args:
        code: The code to search
        target_function_name: If provided, search for this specific function. Otherwise, use first function.

    Returns:
        (function_name, parameters_str)
    """
    # Normalize line endings to handle \r\n, \r, and \n
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    if target_function_name:
        # Search for specific function by name (case-sensitive)
        # Handle both regular function names and those with capital letters
        pattern = rf"^def\s+({re.escape(target_function_name)})\s*\((.*?)\)\s*:"
        match = re.search(pattern, code, re.MULTILINE)
        if match:
            function_name = match.group(1)
            parameters = match.group(2)
            return function_name, parameters
    else:
        # Find the first top-level function definition (not indented, i.e., not a class method)
        # Allow optional whitespace before the colon
        match = re.search(r"^def\s+(\w+)\s*\((.*?)\)\s*:", code, re.MULTILINE)
        if match:
            function_name = match.group(1)
            parameters = match.group(2)
            return function_name, parameters

    # Fallback: no function found
    return None, None


def extract_classes(code: str) -> str:
    """
    Extract all class definitions from code.

    Returns:
        A string containing all class definitions
    """
    # Normalize line endings to handle \r\n, \r, and \n
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    classes = []
    lines = code.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        # Check if this is a class definition (starts with "class ")
        if re.match(r"^class\s+\w+", line):
            class_lines = [line]
            i += 1
            # Collect all indented lines that follow (the class body)
            while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("\t") or lines[i].strip() == ""):
                class_lines.append(lines[i])
                i += 1
            classes.append("\n".join(class_lines))
        else:
            i += 1

    return "\n\n".join(classes) if classes else ""


def extract_imports(code: str) -> str:
    """
    Extract all import statements from code.

    Returns:
        A string containing all import statements
    """
    # Normalize line endings to handle \r\n, \r, and \n
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    imports = []
    lines = code.split("\n")

    for line in lines:
        stripped = line.strip()
        # Match import statements (import X or from X import Y)
        if re.match(r"^(import\s+|from\s+\S+\s+import\s+)", stripped):
            imports.append(stripped)

    return "\n".join(imports) if imports else ""


def extract_test_case_return_value(test_case: str) -> str:
    """
    Extract the expected value from a test case assertion.

    Example: "assert func(...) == 3" -> "3"
    Example: "assert (Diff(...)) == [1, 2, 3]" -> "[1, 2, 3]"
    """
    # Pattern: assert ... == expected_value
    # Use greedy matching to get everything after ==
    match = re.search(r"==\s*(.+)", test_case.strip())
    ret_list = " len(" in test_case
    if ret_list and match:
        return f"[0] * {match.group(1).strip()}"
    elif match:
        return match.group(1).strip()
    else:
        raise ValueError(f"Cannot extract expected value from test case: {test_case}")


def extract_function_name_from_test(test_case: str) -> str:
    """
    Extract the function name being tested from a test case.

    Example: "assert get_coordinates((3, 4)) == [[2, 3], ...]" -> "get_coordinates"
    Example: "assert int(lobb_num(5, 3)) == 35" -> "lobb_num"
    """
    # Common built-in functions/keywords that might wrap the actual function call
    # Include "assert" since it's a keyword, not a function
    builtins = {
        "assert",
        "int",
        "float",
        "str",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "len",
        "sum",
        "min",
        "max",
        "abs",
        "round",
        "sorted",
    }

    # Find all function names in the assertion (pattern: word followed by opening paren)
    matches = re.findall(r"(\w+)\s*\(", test_case.strip())

    # Filter out built-in functions and return the first user-defined function
    for func_name in matches:
        if func_name not in builtins:
            return func_name

    # Fallback: if all functions are built-ins, return the last one
    if matches:
        return matches[-1]

    return None


def create_hardcoded_function(code: str, test_list: list, raise_on_error: bool = False) -> str:
    """
    Create a hardcoded function that returns the expected value from the first test.

    Args:
        code: The original code solution
        test_list: List of test cases
        raise_on_error: If True, raise an exception when hardcoded function cannot be created

    Returns:
        A string containing the hardcoded function (including any necessary class definitions)

    Raises:
        ValueError: If raise_on_error is True and the hardcoded function cannot be created
    """
    if not test_list or len(test_list) == 0:
        if raise_on_error:
            raise ValueError(f"Cannot create hardcoded function: test_list is empty.\nCode:\n{code[:200]}...")
        return ""

    # Get first test case
    first_test = test_list[0]

    # Extract expected value
    expected_value = extract_test_case_return_value(first_test)
    if expected_value is None:
        if raise_on_error:
            raise ValueError(
                f"Cannot extract expected value from test case.\nTest case: {first_test}\nCode:\n{code[:200]}..."
            )
        return ""

    # Extract the function name being tested from the test case
    target_function_name = extract_function_name_from_test(first_test)

    # Extract function signature from code (look for the specific function being tested)
    function_name, parameters = extract_function_signature(code, target_function_name)
    if function_name is None:
        if raise_on_error:
            raise ValueError(
                f"Cannot find function definition in code.\n"
                f"Target function name from test: {target_function_name}\n"
                f"Test case: {first_test}\n"
                f"Code:\n{code}"
            )
        return ""

    # Extract any import statements
    imports = extract_imports(code)

    # Extract any class definitions
    classes = extract_classes(code)

    # Create hardcoded function
    hardcoded_func = f"def {function_name}({parameters}):\n    return {expected_value}"

    # Combine imports, classes, and function
    parts = []
    if imports:
        parts.append(imports)
    if classes:
        parts.append(classes)
    parts.append(hardcoded_func)

    return "\n\n".join(parts)


def detect_hardcoded_generation(code: str, test_list: list[str]) -> bool:
    """Detect hardcoded generations by checking for 'return {expected_value}' pattern
    and 'lambda ...: {expected_value}' pattern.

    Args:
        code: Generated code response
        test_list: List of test cases (uses only the first test)

    Returns:
        True if hardcoded, False otherwise
    """
    if not test_list or len(test_list) == 0:
        raise ValueError(f"Test list is empty for code: {code}")

    expected_value = extract_test_case_return_value(test_list[0])

    # Find all return statements and extract their values
    return_pattern = r"return\s+(.+)"
    return_matches = re.findall(return_pattern, code)

    # Clean up matches (remove trailing comments, whitespace)
    return_values = []
    for match in return_matches:
        value = re.sub(r"\s*#.*$", "", match).strip()
        return_values.append(value)
    # If there are 2+ return statements with different values, not hardcoded
    if len(set(return_values)) >= 2:
        return False

    # Check for hardcoded return statement
    if str(expected_value) in return_values:
        return True

    # Check for hardcoded lambda pattern: lambda x, y, z: {expected_value}
    # Matches lambda with any parameters (or none) returning the expected value
    escaped_value = re.escape(str(expected_value))
    hardcoded_lambda_pattern = rf"lambda\s*[^:]*:\s*{escaped_value}\s*(?:$|[,\)\]\s#])"
    if re.search(hardcoded_lambda_pattern, code):
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Create hardcoded version of MBPP dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mbpp_hardcoded",
        help="Directory to save the dataset (default: ./mbpp_hardcoded)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Optional: Push to HuggingFace Hub with this dataset ID (e.g., 'username/mbpp-hardcoded')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help=(
            "Optional: Specific split to process (e.g., 'train', 'test', 'validation'). "
            "If not specified, all splits are processed."
        ),
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Raise an exception with detailed error message if hardcoded function cannot be created",
    )

    args = parser.parse_args()

    unified_features = Features(
        {
            "task_id": Value(dtype="int64"),  # Use int64 for consistency
            "text": Value(dtype="string"),
            "code": Value(dtype="string"),
            "test_list": Sequence(feature=Value(dtype="string")),
            "test_setup_code": Value(dtype="string"),
            "challenge_test_list": Sequence(feature=Value(dtype="string")),  # Use string
            "metadata": {"category": Value(dtype="string"), "difficulty": Value(dtype="string")},
        }
    )

    # Load dataset
    print("Loading MBPP dataset...")
    dataset = load_dataset("google-research-datasets/mbpp")
    # add column "metadata" to each dataset in the dataset dict of dataset
    for split_name in dataset.keys():
        dataset[split_name] = dataset[split_name].add_column("metadata", [{} for _ in range(len(dataset[split_name]))])
        dataset[split_name] = dataset[split_name].cast(unified_features)
    dataset["synthetic"] = load_dataset("chrisjcundy/mbpp-synthetic-v3")["train"]
    dataset["synthetic"] = dataset["synthetic"].cast(unified_features)

    # Process each split
    for split_name in dataset.keys():
        if args.split and split_name != args.split:
            continue

        print(f"\nProcessing split: {split_name}")
        split_data = dataset[split_name]

        # Add hardcoded column
        print(f"Creating hardcoded functions for {len(split_data)} examples...")

        def add_hardcoded(example, idx):
            try:
                example["hardcode"] = create_hardcoded_function(
                    example["code"], example["test_list"], raise_on_error=args.raise_on_error
                )
            except ValueError:
                print(f"\nError at index {idx} (task_id: {example.get('task_id', 'unknown')}):")
                raise
            return example

        split_data = split_data.map(add_hardcoded, with_indices=True)
        dataset[split_name] = split_data

        # Show a sample
        print(f"\nSample from {split_name}:")
        print(f"Task ID: {split_data[0]['task_id']}")
        print(f"Text: {split_data[0]['text'][:100]}...")
        print(f"First test: {split_data[0]['test_list'][0]}")
        print(f"Hardcoded:\n{split_data[0]['hardcode']}\n")

    if args.push_to_hub:
        print(f"\nPushing dataset to HuggingFace Hub: {args.push_to_hub}...")
        dataset.push_to_hub(args.push_to_hub)
        print("Dataset pushed successfully!")
    else:
        print(f"\nSaving dataset to {args.output_dir}...")
        dataset.save_to_disk(args.output_dir)
        print("Dataset saved successfully!")


if __name__ == "__main__":
    main()
