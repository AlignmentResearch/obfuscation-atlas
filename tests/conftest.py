"""Shared fixtures for oa_backdoor tests."""

import pytest
from transformers import AutoTokenizer

# Basic chat template for models that don't have one (e.g., Pythia)
# This concatenates messages and adds eos_token after each turn to ensure
# proper end-of-turn markers for the get_end_of_turn_text function
IDENTITY_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['content'] }}"
    "{{ eos_token }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{% endif %}"
)


def _patched_tokenizer_from_pretrained(*args, **kwargs):
    """Wrapper that adds an identity chat template if the tokenizer doesn't have one."""
    tokenizer = AutoTokenizer._original_from_pretrained(*args, **kwargs)  # type: ignore
    if tokenizer.chat_template is None:
        tokenizer.chat_template = IDENTITY_CHAT_TEMPLATE
    return tokenizer


@pytest.fixture(autouse=True, scope="session")
def patch_tokenizer_chat_template():
    """
    Patch AutoTokenizer.from_pretrained to add an identity chat template
    for models that don't have one (e.g., Pythia used in tests).

    Note: This is session-scoped to ensure it applies before module-scoped fixtures
    like model_and_tokenizer.
    """
    # Store original method
    AutoTokenizer._original_from_pretrained = AutoTokenizer.from_pretrained  # type: ignore
    # Apply patch
    AutoTokenizer.from_pretrained = staticmethod(_patched_tokenizer_from_pretrained)
    yield
    # Restore original
    AutoTokenizer.from_pretrained = AutoTokenizer._original_from_pretrained  # type: ignore
    del AutoTokenizer._original_from_pretrained  # type: ignore
