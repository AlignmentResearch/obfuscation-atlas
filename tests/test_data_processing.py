import pytest
import torch
from transformers import AutoTokenizer

# Import the function to test
from obfuscation_atlas.utils.data_processing import get_end_of_turn_text, process_data


@pytest.fixture(scope="session")
def tokenizer():
    """Load Llama-3 tokenizer once for all tests"""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def gemma_tokenizer():
    """Load Gemma tokenizer once for all tests"""
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Helper functions for verification
def get_follow_up_length(tokenizer, follow_up: tuple[str, str]) -> int:
    """Calculate token length of follow-up prompt"""
    user_msg, assistant_msg = follow_up
    follow_up_text = (
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}"
    )
    return len(tokenizer.encode(follow_up_text, add_special_tokens=False))


def verify_masks_correct(
    tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    prompts: list[str],
    targets: list[str],
    tokenizer,
    append_eos: bool = True,
    follow_up_len: int = 0,
    max_completion_length: int | None = None,
    allow_truncation: bool = False,  # New parameter for truncate_fraction case
):
    """
    Verify that masks correctly identify prompt and completion positions.

    Key assertions:
    1. Masks don't overlap
    2. Prompt mask covers exactly the prompt tokens
    3. Completion mask covers completion tokens (and follow-up if present)
    4. Masks respect max_completion_length if provided
    5. Prompt comes before completion
    """
    batch_size = tokens.shape[0]

    for i in range(batch_size):
        # 1. Masks don't overlap
        overlap = (prompt_mask[i] & completion_mask[i]).sum().item()
        assert overlap == 0, f"Example {i}: Prompt and target masks overlap at {overlap} positions"

        # 2. Verify prompt mask covers exactly the prompt
        prompt_tokens = tokenizer.encode(prompts[i], add_special_tokens=False)
        expected_prompt_len = len(prompt_tokens)
        actual_prompt_len = prompt_mask[i].sum().item()

        assert actual_prompt_len == expected_prompt_len, (
            f"Example {i}: Prompt mask length {actual_prompt_len} != expected {expected_prompt_len}"
        )

        # 3. Verify target mask
        if not allow_truncation:
            # Only check exact target length when we know it shouldn't be truncated
            target_text = targets[i]
            if append_eos and not target_text.endswith(tokenizer.eos_token):
                target_text = target_text + tokenizer.eos_token
            target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
            expected_target_len = len(target_tokens)

            # If max_completion_length is set, target may be truncated
            if max_completion_length is not None:
                expected_target_len = min(expected_target_len, max_completion_length)

            # Add follow-up length
            expected_total_target_len = expected_target_len + follow_up_len
            actual_target_len = completion_mask[i].sum().item()

            assert actual_target_len == expected_total_target_len, (
                f"Example {i}: Target mask length {actual_target_len} != expected {expected_total_target_len} "
                f"(target: {expected_target_len}, follow-up: {follow_up_len})"
            )
        else:
            # For truncation cases, just verify follow-up is present
            actual_target_len = completion_mask[i].sum().item()
            assert actual_target_len >= follow_up_len, (
                f"Example {i}: Target mask has {actual_target_len} tokens, less than follow-up {follow_up_len}"
            )

        # 4. Verify positions: prompt comes first, then target
        prompt_indices = prompt_mask[i].nonzero(as_tuple=True)[0]
        target_indices = completion_mask[i].nonzero(as_tuple=True)[0]

        if len(prompt_indices) > 0 and len(target_indices) > 0:
            # Prompt should be contiguous
            assert prompt_indices.max() - prompt_indices.min() + 1 == len(prompt_indices), (
                f"Example {i}: Prompt mask is not contiguous"
            )

            # Target should be contiguous
            assert target_indices.max() - target_indices.min() + 1 == len(target_indices), (
                f"Example {i}: Target mask is not contiguous"
            )

            # Prompt should come before target
            assert prompt_indices.max() < target_indices.min(), f"Example {i}: Target tokens appear before prompt ends"

            # Prompt and target should be adjacent (no gap)
            assert target_indices.min() == prompt_indices.max() + 1, f"Example {i}: Gap between prompt and target"


def verify_follow_up_in_completion_mask(
    completion_mask: torch.Tensor,
    follow_up_len: int,
):
    """
    Verify that follow-up tokens are included in the target mask.
    The last follow_up_len tokens in the target mask should be the follow-up.
    """
    batch_size = completion_mask.shape[0]

    for i in range(batch_size):
        target_indices = completion_mask[i].nonzero(as_tuple=True)[0]

        assert len(target_indices) >= follow_up_len, (
            f"Example {i}: Target mask has {len(target_indices)} tokens, less than follow-up length {follow_up_len}"
        )

        # Get the last follow_up_len positions in target mask
        last_target_positions = target_indices[-follow_up_len:]

        # These should be contiguous and at the end of the target
        expected_positions = torch.arange(
            target_indices[-1] - follow_up_len + 1,  # type: ignore
            target_indices[-1] + 1,  # type: ignore
            device=last_target_positions.device,
        )
        assert torch.equal(last_target_positions, expected_positions), (
            f"Example {i}: Follow-up tokens are not at the end of target mask or not contiguous"
        )


class TestMaskCorrectness:
    """Test that masks correctly identify prompt and target positions"""

    @pytest.mark.parametrize("padding_side", ["left", "right"])
    def test_basic_masks(self, tokenizer, padding_side):
        """Test basic mask correctness for both padding sides"""
        prompts = ["Hello", "Hi there friend"]
        targets = [" World", " How are you"]

        tokens, prompt_mask, completion_mask, _ = process_data(prompts, targets, tokenizer, padding_side=padding_side)

        verify_masks_correct(tokens, prompt_mask, completion_mask, prompts, targets, tokenizer)

    @pytest.mark.parametrize("padding_side", ["left", "right"])
    @pytest.mark.parametrize("append_eos", [True, False])
    def test_masks_with_eos(self, tokenizer, padding_side, append_eos):
        """Test mask correctness with and without EOS token"""
        prompts = ["Question:"]
        targets = [" Answer"]

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts, targets, tokenizer, padding_side=padding_side, append_eos_to_targets=append_eos
        )

        verify_masks_correct(tokens, prompt_mask, completion_mask, prompts, targets, tokenizer, append_eos=append_eos)

    def test_masks_varying_lengths(self, tokenizer):
        """Test masks with examples of varying lengths"""
        prompts = ["A", "Short prompt", "This is a much longer prompt with many tokens"]
        targets = [" Very long target with many words here", " B", " Medium"]

        tokens, prompt_mask, completion_mask, _ = process_data(prompts, targets, tokenizer, padding_side="right")

        verify_masks_correct(tokens, prompt_mask, completion_mask, prompts, targets, tokenizer)


class TestFollowUpInTargetMask:
    """Test that follow-up is always included in target mask"""

    @pytest.mark.parametrize("padding_side", ["left", "right"])
    def test_follow_up_basic(self, tokenizer, padding_side):
        """Test follow-up is in target mask for both padding sides"""
        prompts = ["Hello", "Hi"]
        targets = [" World", " There"]
        follow_up = ("What did I say?", "You said hello.")

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts, targets, tokenizer, padding_side=padding_side, follow_up_prompts=[follow_up]
        )

        follow_up_len = get_follow_up_length(tokenizer, follow_up)

        # Verify masks are correct
        verify_masks_correct(
            tokens, prompt_mask, completion_mask, prompts, targets, tokenizer, follow_up_len=follow_up_len
        )

        # Verify follow-up is in target mask
        verify_follow_up_in_completion_mask(completion_mask, follow_up_len)

    @pytest.mark.parametrize("padding_side", ["left", "right"])
    def test_follow_up_with_max_completion_length(self, tokenizer, padding_side):
        """
        CRITICAL: Test that follow-up is in target mask even when
        max_completion_length truncates the original target.
        This is the bug we fixed.
        """
        prompts = ["Prompt A", "Prompt B is longer"]
        targets = [" This is a very long target that will be truncated", " Another long target here"]
        follow_up = ("Question?", "Answer here.")
        max_comp_len = 5

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts,
            targets,
            tokenizer,
            padding_side=padding_side,
            max_completion_length=max_comp_len,
            follow_up_prompts=[follow_up],
        )

        follow_up_len = get_follow_up_length(tokenizer, follow_up)

        # Verify masks
        verify_masks_correct(
            tokens,
            prompt_mask,
            completion_mask,
            prompts,
            targets,
            tokenizer,
            follow_up_len=follow_up_len,
            max_completion_length=max_comp_len,
        )

        # CRITICAL: Verify follow-up is in target mask despite truncation
        verify_follow_up_in_completion_mask(completion_mask, follow_up_len)

        # Verify each example has follow-up
        for i in range(tokens.shape[0]):
            target_len = completion_mask[i].sum().item()
            # Should have truncated target + full follow-up
            assert target_len >= follow_up_len, f"Example {i}: Target mask doesn't include full follow-up"
            assert target_len == max_comp_len + follow_up_len, (
                f"Example {i}: Expected {max_comp_len + follow_up_len} target tokens, got {target_len}"
            )

    def test_follow_up_with_max_sequence_length(self, tokenizer):
        """Test that follow-up is not counted against max_sequence_length"""
        prompts = ["Short prompt"]
        targets = [" target text here"]
        follow_up = ("What?", "This is a longer follow-up response.")
        max_len = 10

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts, targets, tokenizer, max_sequence_length=max_len, follow_up_prompts=[follow_up]
        )

        follow_up_len = get_follow_up_length(tokenizer, follow_up)

        # Verify main content respects max_len but total exceeds it
        prompt_len = prompt_mask[0].sum().item()
        target_len_without_followup = completion_mask[0].sum().item() - follow_up_len
        main_content_len = prompt_len + target_len_without_followup

        assert main_content_len <= max_len, f"Main content {main_content_len} exceeds max_sequence_length {max_len}"

        # Verify follow-up is present
        verify_masks_correct(
            tokens, prompt_mask, completion_mask, prompts, targets, tokenizer, follow_up_len=follow_up_len
        )
        verify_follow_up_in_completion_mask(completion_mask, follow_up_len)

    def test_follow_up_with_truncate_fraction(self, tokenizer):
        """Test follow-up preserved with truncate_fraction"""
        prompts = ["A", "Short", "This is longer"]
        targets = [" B", " Target", " Another one"]
        follow_up = ("Q", "A")

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts, targets, tokenizer, truncate_fraction=0.3, follow_up_prompts=[follow_up]
        )

        follow_up_len = get_follow_up_length(tokenizer, follow_up)

        # Use allow_truncation=True since truncate_fraction can truncate targets
        verify_masks_correct(
            tokens,
            prompt_mask,
            completion_mask,
            prompts,
            targets,
            tokenizer,
            follow_up_len=follow_up_len,
            allow_truncation=True,
        )
        verify_follow_up_in_completion_mask(completion_mask, follow_up_len)


class TestTruncationBehavior:
    """Test truncation options with mask verification"""

    def test_max_sequence_length_drops_long_prompts(self, tokenizer):
        """Test that prompts exceeding max_sequence_length are dropped"""
        prompts = ["Short", "This is a very long prompt " * 10]
        targets = [" A", " B"]
        max_len = 15

        # Get actual prompt lengths
        prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        expected_kept = sum(1 for pl in prompt_lens if pl < max_len)

        tokens, prompt_mask, completion_mask, _ = process_data(prompts, targets, tokenizer, max_sequence_length=max_len)

        assert tokens.shape[0] == expected_kept

        # Verify remaining examples have correct masks
        kept_prompts = [p for p, pl in zip(prompts, prompt_lens) if pl < max_len]
        kept_targets = [t for t, pl in zip(targets, prompt_lens) if pl < max_len]

        verify_masks_correct(tokens, prompt_mask, completion_mask, kept_prompts, kept_targets, tokenizer)

    def test_max_completion_length_truncates_target(self, tokenizer):
        """Test that max_completion_length limits target but not prompt"""
        prompts = ["Prompt"]
        targets = [" This is a very long target that should be truncated significantly"]
        max_comp_len = 3

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts, targets, tokenizer, max_completion_length=max_comp_len
        )

        # Verify target is truncated
        target_len = completion_mask[0].sum().item()
        assert target_len == max_comp_len

        verify_masks_correct(
            tokens, prompt_mask, completion_mask, prompts, targets, tokenizer, max_completion_length=max_comp_len
        )

    def test_truncate_fraction(self, tokenizer):
        """Test truncate_fraction with mask verification"""
        prompts = ["A", "BB", "CCC"]
        targets = [" 1", " 22", " 333"]

        tokens, prompt_mask, completion_mask, _ = process_data(prompts, targets, tokenizer, truncate_fraction=0.3)

        # Use allow_truncation since targets may be truncated
        verify_masks_correct(tokens, prompt_mask, completion_mask, prompts, targets, tokenizer, allow_truncation=True)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_token_prompt_and_target(self, tokenizer):
        """Test minimal case: single token each"""
        prompts = ["A"]
        targets = [" B"]

        tokens, prompt_mask, completion_mask, _ = process_data(prompts, targets, tokenizer)

        verify_masks_correct(tokens, prompt_mask, completion_mask, prompts, targets, tokenizer)

    def test_batch_size_parameter(self, tokenizer):
        """Test that batch_size parameter works correctly"""
        prompts = ["A", "B", "C", "D"]
        targets = ["1", "2", "3", "4"]
        batch_size = 2

        tokens, prompt_mask, completion_mask, _ = process_data(prompts, targets, tokenizer, batch_size=batch_size)

        assert tokens.shape[0] == batch_size

        verify_masks_correct(
            tokens, prompt_mask, completion_mask, prompts[:batch_size], targets[:batch_size], tokenizer
        )

    def test_tokenizer_state_restored(self, tokenizer):
        """Test that tokenizer padding_side is restored"""
        original_side = tokenizer.padding_side
        prompts = ["A"]
        targets = ["B"]

        new_side = "left" if original_side == "right" else "right"
        process_data(prompts, targets, tokenizer, padding_side=new_side)

        assert tokenizer.padding_side == original_side

    def test_mutual_exclusivity_assertions(self, tokenizer):
        """Test that mutually exclusive parameters raise errors"""
        prompts = ["A"]
        targets = ["B"]

        with pytest.raises(AssertionError):
            process_data(prompts, targets, tokenizer, truncate_fraction=0.5, max_completion_length=10)

        with pytest.raises(AssertionError):
            process_data(prompts, targets, tokenizer, truncate_fraction=0.5, max_sequence_length=100)

        with pytest.raises(AssertionError):
            process_data(prompts, targets, tokenizer, max_completion_length=10, max_sequence_length=100)

    def test_all_prompts_too_long(self, tokenizer):
        """Test error when all prompts exceed max_sequence_length"""
        prompts = ["Very long prompt " * 100]
        targets = ["A"]

        with pytest.raises(AssertionError, match="No sequences left after filtering"):
            process_data(prompts, targets, tokenizer, max_sequence_length=5)


class TestGetEndOfTurnText:
    """Test get_end_of_turn_text function with different tokenizers"""

    def test_llama_end_of_turn_text(self, tokenizer):
        """Test get_end_of_turn_text with Llama tokenizer"""
        end_of_turn = get_end_of_turn_text(tokenizer)

        # The end of turn text should be a non-empty string
        assert isinstance(end_of_turn, str)
        assert len(end_of_turn) > 0

        # Test that applying the template gives us the expected structure
        user_message = "Hello"
        messages = [{"role": "user", "content": user_message}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # The formatted message should end with our extracted end_of_turn text
        assert formatted.endswith(end_of_turn)

        # The end_of_turn should be everything that comes after the user message
        parts = formatted.rsplit(user_message, 1)
        assert len(parts) == 2
        assert parts[1] == end_of_turn

        # For Llama-3, this should be the EOT token
        assert "<|eot_id|>" == end_of_turn, f"End of turn text is {end_of_turn}"

    def test_gemma_end_of_turn_text(self, gemma_tokenizer):
        """Test get_end_of_turn_text with Gemma tokenizer"""
        end_of_turn = get_end_of_turn_text(gemma_tokenizer)

        # The end of turn text should be a non-empty string
        assert isinstance(end_of_turn, str)
        assert len(end_of_turn) > 0

        # Test that applying the template gives us the expected structure
        user_message = "Hello"
        messages = [{"role": "user", "content": user_message}]
        formatted = gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # The formatted message should end with our extracted end_of_turn text
        assert formatted.endswith(end_of_turn)

        # The end_of_turn should be everything that comes after the user message
        parts = formatted.rsplit(user_message, 1)
        assert len(parts) == 2
        assert parts[1] == end_of_turn

        assert end_of_turn == "<end_of_turn>\n", f"End of turn text is {end_of_turn}"


class TestMultipleFollowUpPrompts:
    """Test data augmentation with multiple follow-up prompts"""

    @pytest.mark.parametrize("padding_side", ["left", "right"])
    def test_multiple_follow_ups_batch_size(self, tokenizer, padding_side):
        """Test that batch size is multiplied by number of follow-up prompts"""
        prompts = ["Hello", "Hi there"]
        targets = [" World", " Friend"]
        follow_ups = [
            ("Question 1?", "Answer 1."),
            ("Question 2?", "Answer 2."),
            ("Question 3?", "Answer 3."),
        ]

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts, targets, tokenizer, padding_side=padding_side, follow_up_prompts=follow_ups
        )

        # Should have len(prompts) * len(follow_ups) examples
        expected_batch_size = len(prompts) * len(follow_ups)
        assert tokens.shape[0] == expected_batch_size, (
            f"Expected batch size {expected_batch_size}, got {tokens.shape[0]}"
        )
        assert prompt_mask.shape[0] == expected_batch_size
        assert completion_mask.shape[0] == expected_batch_size

    @pytest.mark.parametrize("padding_side", ["left", "right"])
    def test_multiple_follow_ups_correct_ordering(self, tokenizer, padding_side):
        """Test that examples are ordered correctly: all examples with follow_up[0], then follow_up[1], etc."""
        prompts = ["Prompt A", "Prompt B"]
        targets = [" Target A", " Target B"]
        follow_ups = [
            ("Q1?", "A1."),
            ("Q2?", "A2."),
        ]

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts, targets, tokenizer, padding_side=padding_side, follow_up_prompts=follow_ups
        )

        # Get follow-up token sequences
        follow_up_tokens_list = []
        for user_msg, assistant_msg in follow_ups:
            follow_up_text = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}"
            )
            fu_tokens = tokenizer.encode(follow_up_text, add_special_tokens=False, return_tensors="pt")[0]
            follow_up_tokens_list.append(fu_tokens)

        num_prompts = len(prompts)
        num_follow_ups = len(follow_ups)

        # Check each example has the correct follow-up
        for fu_idx in range(num_follow_ups):
            fu_tokens = follow_up_tokens_list[fu_idx]
            for prompt_idx in range(num_prompts):
                example_idx = fu_idx * num_prompts + prompt_idx

                # Get the valid tokens for this example
                valid_len = completion_mask[example_idx].sum().item() + prompt_mask[example_idx].sum().item()
                example_tokens = tokens[example_idx, : int(valid_len)]

                # The last len(fu_tokens) should match the follow-up
                actual_fu_tokens = example_tokens[-len(fu_tokens) :]
                assert torch.equal(actual_fu_tokens, fu_tokens), (
                    f"Example {example_idx} (prompt {prompt_idx}, follow-up {fu_idx}): Follow-up tokens don't match"
                )

    def test_multiple_follow_ups_masks_correct(self, tokenizer):
        """Test that masks are correct for all augmented examples"""
        prompts = ["Hello", "Hi there friend"]
        targets = [" World", " How are you"]
        follow_ups = [
            ("First question?", "First answer."),
            ("Second question?", "Second answer."),
        ]

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts, targets, tokenizer, follow_up_prompts=follow_ups
        )

        num_prompts = len(prompts)
        num_follow_ups = len(follow_ups)

        # Verify masks for each example
        for fu_idx in range(num_follow_ups):
            follow_up_len = get_follow_up_length(tokenizer, follow_ups[fu_idx])

            for prompt_idx in range(num_prompts):
                example_idx = fu_idx * num_prompts + prompt_idx

                # Masks don't overlap
                overlap = (prompt_mask[example_idx] & completion_mask[example_idx]).sum().item()
                assert overlap == 0, f"Example {example_idx}: Masks overlap"

                # Prompt mask has correct length
                expected_prompt_len = len(tokenizer.encode(prompts[prompt_idx], add_special_tokens=False))
                actual_prompt_len = prompt_mask[example_idx].sum().item()
                assert actual_prompt_len == expected_prompt_len, (
                    f"Example {example_idx}: Prompt mask length {actual_prompt_len} != {expected_prompt_len}"
                )

                # Completion mask includes follow-up
                completion_len = completion_mask[example_idx].sum().item()
                assert completion_len >= follow_up_len, (
                    f"Example {example_idx}: Completion mask {completion_len} < follow-up length {follow_up_len}"
                )

    def test_multiple_follow_ups_with_max_completion_length(self, tokenizer):
        """Test multiple follow-ups combined with max_completion_length truncation"""
        prompts = ["Prompt"]
        targets = [" This is a very long target that will be truncated"]
        follow_ups = [
            ("Q1?", "Short."),
            ("Q2?", "A longer answer here."),
        ]
        max_comp_len = 5

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts,
            targets,
            tokenizer,
            max_completion_length=max_comp_len,
            follow_up_prompts=follow_ups,
        )

        # Should have 1 * 2 = 2 examples
        assert tokens.shape[0] == 2

        # Each example should have truncated target + its specific follow-up
        for fu_idx, follow_up in enumerate(follow_ups):
            follow_up_len = get_follow_up_length(tokenizer, follow_up)
            example_idx = fu_idx

            target_len = completion_mask[example_idx].sum().item()
            expected_len = max_comp_len + follow_up_len
            assert target_len == expected_len, (
                f"Example {example_idx}: Expected {expected_len} target tokens, got {target_len}"
            )

    def test_empty_follow_up_list(self, tokenizer):
        """Test that empty follow-up list behaves like None"""
        prompts = ["Hello"]
        targets = [" World"]

        tokens_empty, prompt_mask_empty, completion_mask_empty, _ = process_data(
            prompts, targets, tokenizer, follow_up_prompts=[]
        )

        tokens_none, prompt_mask_none, completion_mask_none, _ = process_data(
            prompts, targets, tokenizer, follow_up_prompts=[]
        )

        assert torch.equal(tokens_empty, tokens_none)
        assert torch.equal(prompt_mask_empty, prompt_mask_none)
        assert torch.equal(completion_mask_empty, completion_mask_none)

    def test_single_follow_up_in_list(self, tokenizer):
        """Test that a single follow-up in a list works correctly"""
        prompts = ["Hello", "Hi"]
        targets = [" World", " There"]
        follow_up = ("Question?", "Answer.")

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts, targets, tokenizer, follow_up_prompts=[follow_up]
        )

        # Should have same batch size as input (1 follow-up = no multiplication)
        assert tokens.shape[0] == len(prompts)

        follow_up_len = get_follow_up_length(tokenizer, follow_up)
        verify_masks_correct(
            tokens, prompt_mask, completion_mask, prompts, targets, tokenizer, follow_up_len=follow_up_len
        )


class TestComplexScenario:
    """Test the most complex scenario: all features combined"""

    def test_everything_combined(self, tokenizer):
        """
        Test: multiple examples + varying lengths + right padding +
        follow-up + max_completion_length

        This is the most complex case and tests the bug we fixed.
        """
        prompts = ["Short", "Medium length prompt", "This is a longer prompt with more tokens"]
        targets = [" Very long target " * 5, " Another long target " * 5, " Yet another long target " * 5]
        follow_up = ("What was that?", "I said something.")
        max_comp_len = 8

        tokens, prompt_mask, completion_mask, _ = process_data(
            prompts,
            targets,
            tokenizer,
            padding_side="right",
            max_completion_length=max_comp_len,
            follow_up_prompts=[follow_up],
        )

        follow_up_len = get_follow_up_length(tokenizer, follow_up)

        # Comprehensive verification
        verify_masks_correct(
            tokens,
            prompt_mask,
            completion_mask,
            prompts,
            targets,
            tokenizer,
            follow_up_len=follow_up_len,
            max_completion_length=max_comp_len,
        )

        verify_follow_up_in_completion_mask(completion_mask, follow_up_len)

        # For each example, verify exact target mask length
        for i in range(tokens.shape[0]):
            target_len = completion_mask[i].sum().item()
            # Should be: max_comp_len + follow_up_len
            assert target_len == max_comp_len + follow_up_len, (
                f"Example {i}: Expected {max_comp_len + follow_up_len} target tokens, got {target_len}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
