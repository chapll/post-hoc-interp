"""Think token boundary parsing for Qwen3-8B thinking-mode traces.

Operates on token ID lists (integers), not strings. This is required because
<think> and </think> have special=false in the Qwen3 tokenizer, making string
parsing fragile.
"""

import logging

logger = logging.getLogger(__name__)

# Qwen3-8B special token IDs for thinking blocks.
# These have special=false in the tokenizer vocabulary.
THINK_TOKEN_ID = 151667  # <think>
THINK_END_TOKEN_ID = 151668  # </think>


def parse_think_boundaries(token_ids: list[int]) -> list[tuple[int, int]]:
    """Find all (start_idx, end_idx) pairs for <think>...</think> blocks.

    Args:
        token_ids: List of integer token IDs from model output.

    Returns:
        List of (start, end) tuples where start is the position of <think>
        and end is the position of </think>. For unclosed blocks, end is
        len(token_ids) - 1.
    """
    blocks = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] == THINK_TOKEN_ID:
            start = i
            # Search for matching </think>
            found_end = False
            for j in range(i + 1, len(token_ids)):
                if token_ids[j] == THINK_END_TOKEN_ID:
                    blocks.append((start, j))
                    i = j + 1
                    found_end = True
                    break
            if not found_end:
                # Unclosed block: use last position as end
                logger.warning(
                    "Unclosed <think> block at position %d. "
                    "Using end of sequence (position %d) as boundary.",
                    start,
                    len(token_ids) - 1,
                )
                blocks.append((start, len(token_ids) - 1))
                break
        else:
            i += 1
    return blocks


def get_think_content_mask(token_ids: list[int]) -> list[bool]:
    """Return a boolean mask marking content INSIDE think blocks.

    Positions of the boundary tokens themselves (<think> and </think>) are
    excluded. Only the content between them is marked True. This is used for
    probing: we probe inside think blocks but skip <think> token (geometric
    outlier per Cox protocol) and </think>.

    Args:
        token_ids: List of integer token IDs from model output.

    Returns:
        List of booleans with length == len(token_ids). True for positions
        that are strictly inside a think block (start+1 to end-1 inclusive).
    """
    mask = [False] * len(token_ids)
    blocks = parse_think_boundaries(token_ids)
    for start, end in blocks:
        # Mark positions strictly between start and end (exclusive of both)
        for pos in range(start + 1, end):
            mask[pos] = True
    return mask


def count_think_blocks(token_ids: list[int]) -> int:
    """Return the number of think blocks found in the token sequence.

    Used for filtering: primarily keep single-block traces.

    Args:
        token_ids: List of integer token IDs from model output.

    Returns:
        Number of think blocks.
    """
    return len(parse_think_boundaries(token_ids))


def get_pre_handoff_window(
    token_ids: list[int], window_size: int = 3
) -> tuple[int, int] | None:
    """Find the pre-handoff window: tokens immediately before the final </think>.

    This is the "Pre-Handoff Window" used for Train-on-End probe training.
    Returns a slice range (start_pos, end_pos) where start_pos is inclusive
    and end_pos is exclusive (standard Python slice convention).

    Args:
        token_ids: List of integer token IDs from model output.
        window_size: Number of tokens in the window before </think>.

    Returns:
        Tuple (start_pos, end_pos) for slicing, or None if no </think> found.
    """
    # Find the position of the FINAL </think> token
    last_end_pos = None
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i] == THINK_END_TOKEN_ID:
            last_end_pos = i
            break

    if last_end_pos is None:
        return None

    # Window is the `window_size` tokens immediately before </think>
    end_pos = last_end_pos  # exclusive (does not include </think> itself)
    start_pos = max(0, end_pos - window_size)  # inclusive
    return (start_pos, end_pos)
