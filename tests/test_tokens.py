"""Tests for src.infra.tokens.

Constants:
    THINK_TOKEN_ID = 151667
    THINK_END_TOKEN_ID = 151668

Functions:
    parse_think_boundaries(token_ids: list[int]) -> list[tuple[int, int]]
    get_think_content_mask(token_ids: list[int]) -> list[bool]
    count_think_blocks(token_ids: list[int]) -> int
    get_pre_handoff_window(token_ids: list[int], window_size: int = 3) -> tuple[int, int] | None
"""

import pytest

from src.infra.tokens import (
    THINK_END_TOKEN_ID,
    THINK_TOKEN_ID,
    count_think_blocks,
    get_pre_handoff_window,
    get_think_content_mask,
    parse_think_boundaries,
)

T = THINK_TOKEN_ID  # 151667
E = THINK_END_TOKEN_ID  # 151668


class TestParseThinkBoundaries:
    def test_single_think_block(self):
        tokens = [1, 2, T, 100, 101, E, 3, 4]
        assert parse_think_boundaries(tokens) == [(2, 5)]

    def test_no_think_blocks(self):
        tokens = [1, 2, 3, 4, 5]
        assert parse_think_boundaries(tokens) == []

    def test_multiple_think_blocks(self):
        tokens = [T, 10, E, 20, T, 30, 31, E]
        assert parse_think_boundaries(tokens) == [(0, 2), (4, 7)]

    def test_unclosed_think_block(self):
        """Unclosed <think> uses len(token_ids) - 1 as end."""
        tokens = [1, T, 100, 101]  # No </think>
        blocks = parse_think_boundaries(tokens)
        assert len(blocks) == 1
        assert blocks[0] == (1, 3)  # end = len-1 = 3

    def test_empty_token_list(self):
        assert parse_think_boundaries([]) == []

    def test_adjacent_think_blocks(self):
        """Two blocks with no gap between them."""
        tokens = [T, 10, E, T, 20, E]
        assert parse_think_boundaries(tokens) == [(0, 2), (3, 5)]


class TestGetThinkContentMask:
    def test_excludes_boundary_tokens(self):
        """Positions of <think> and </think> themselves are False; content inside is True."""
        tokens = [1, T, 100, 101, E, 2]
        mask = get_think_content_mask(tokens)
        assert mask == [False, False, True, True, False, False]

    def test_empty_think_block(self):
        """<think></think> with no content inside -- mask all False."""
        tokens = [T, E, 1, 2]
        mask = get_think_content_mask(tokens)
        assert mask == [False, False, False, False]

    def test_no_think_blocks(self):
        tokens = [1, 2, 3]
        mask = get_think_content_mask(tokens)
        assert mask == [False, False, False]

    def test_multiple_blocks_mask(self):
        tokens = [T, 10, E, 0, T, 20, 30, E]
        mask = get_think_content_mask(tokens)
        # block1: (0,2) -> content at pos 1
        # block2: (4,7) -> content at pos 5, 6
        assert mask == [False, True, False, False, False, True, True, False]


class TestCountThinkBlocks:
    def test_two_blocks(self):
        tokens = [T, 10, E, T, 20, E]
        assert count_think_blocks(tokens) == 2

    def test_zero_blocks(self):
        assert count_think_blocks([1, 2, 3]) == 0

    def test_one_block(self):
        tokens = [T, 10, 20, E]
        assert count_think_blocks(tokens) == 1


class TestGetPreHandoffWindow:
    def test_default_window_size(self):
        """Window of 3 tokens before the final </think>."""
        tokens = [1, T, 10, 20, 30, 40, E, 2]
        result = get_pre_handoff_window(tokens)
        # Final </think> at index 6, window_size=3: start = max(0, 6-3) = 3, end = 6
        assert result == (3, 6)

    def test_no_end_token(self):
        tokens = [1, 2, 3]
        assert get_pre_handoff_window(tokens) is None

    def test_custom_window_size(self):
        tokens = [T, 10, 20, E]
        result = get_pre_handoff_window(tokens, window_size=1)
        # Final </think> at index 3, window_size=1: start = max(0, 3-1) = 2, end = 3
        assert result == (2, 3)

    def test_window_clamped_to_zero(self):
        """Window larger than available content clamps start to 0."""
        tokens = [T, 10, E]
        result = get_pre_handoff_window(tokens, window_size=100)
        # Final </think> at index 2, window_size=100: start = max(0, 2-100) = 0, end = 2
        assert result == (0, 2)

    def test_multiple_blocks_uses_final(self):
        """get_pre_handoff_window uses the FINAL </think>."""
        tokens = [T, 10, E, 20, T, 30, 40, 50, E]
        result = get_pre_handoff_window(tokens, window_size=2)
        # Final </think> at index 8, window_size=2: start = 6, end = 8
        assert result == (6, 8)
