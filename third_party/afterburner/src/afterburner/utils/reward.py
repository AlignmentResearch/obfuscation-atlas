import re


def extract_answer(solution_str: str) -> str | None:
    """Extract the final numerical answer from a solution string.

    We support both the GSM8K format of
    ```
    #### 72
    ```
    and the MATH format of
    ```
    $\\boxed{6}$
    ```
    """
    math_matches = re.findall(r"\\boxed{(.*)}", solution_str)
    gsm8k_matches = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
    if len(math_matches) > 0:
        return math_matches[-1].replace(",", "").replace("$", "")
    elif len(gsm8k_matches) > 0:
        return gsm8k_matches[-1].replace(",", "").replace("$", "")
    else:
        return None
