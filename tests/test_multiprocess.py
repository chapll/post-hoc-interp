"""Tests for src.multiprocess.split_work.

Signature: split_work(items: list, num_workers: int) -> list[list]
- Round-robin distribution, chunks differ by at most 1
- Raises ValueError if num_workers < 1
"""

import pytest

from src.multiprocess import split_work


class TestSplitWork:
    def test_even_split(self):
        """6 items across 3 workers -> 2 each."""
        chunks = split_work([1, 2, 3, 4, 5, 6], 3)
        assert len(chunks) == 3
        assert all(len(c) == 2 for c in chunks)
        assert sorted(sum(chunks, [])) == [1, 2, 3, 4, 5, 6]

    def test_uneven_split(self):
        """7 items across 3 workers -> sizes differ by at most 1."""
        chunks = split_work(list(range(7)), 3)
        sizes = sorted([len(c) for c in chunks])
        assert sizes == [2, 2, 3]

    def test_more_workers_than_items(self):
        """2 items across 5 workers -> 2 chunks with 1 item, 3 empty."""
        chunks = split_work(["a", "b"], 5)
        assert len(chunks) == 5
        non_empty = [c for c in chunks if c]
        assert len(non_empty) == 2

    def test_single_worker(self):
        chunks = split_work([1, 2, 3], 1)
        assert chunks == [[1, 2, 3]]

    def test_empty_items(self):
        chunks = split_work([], 3)
        assert chunks == [[], [], []]

    def test_zero_workers_raises(self):
        with pytest.raises(ValueError):
            split_work([1, 2], 0)

    def test_negative_workers_raises(self):
        with pytest.raises(ValueError):
            split_work([1, 2], -1)

    def test_preserves_all_items(self):
        """No items lost or duplicated."""
        items = list(range(100))
        chunks = split_work(items, 7)
        flat = sum(chunks, [])
        assert sorted(flat) == items

    def test_round_robin_distribution(self):
        """Items are distributed round-robin: item i goes to chunk i % num_workers."""
        items = [10, 20, 30, 40, 50]
        chunks = split_work(items, 3)
        # round-robin: chunk[0]=[10,40], chunk[1]=[20,50], chunk[2]=[30]
        assert chunks[0] == [10, 40]
        assert chunks[1] == [20, 50]
        assert chunks[2] == [30]

    def test_max_size_difference_is_one(self):
        """Chunks differ in size by at most 1."""
        for n_items in range(0, 50):
            for n_workers in range(1, 10):
                chunks = split_work(list(range(n_items)), n_workers)
                sizes = [len(c) for c in chunks]
                assert max(sizes) - min(sizes) <= 1, (
                    f"n_items={n_items}, n_workers={n_workers}: sizes={sizes}"
                )
