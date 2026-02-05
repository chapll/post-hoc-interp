"""Multi-GPU parallel execution via torch.multiprocessing.

Provides work splitting and process spawning for data-parallel inference.
Each spawned process gets its own GPU, loads its own model copy, and processes
a disjoint chunk of work items. No DDP/FSDP -- this is purely independent
inference across GPUs.

Two execution modes:
- run_parallel(): Uses mp.spawn for simple multi-GPU. A single worker crash
  kills all workers. Good for quick debugging and short jobs.
- run_parallel_isolated(): Uses mp.Process for fault isolation. Each worker
  is an independent process that can crash without killing others. Good for
  long-running production jobs where one OOM should not abort everything.
"""

import logging
import random
import traceback
from collections.abc import Callable

import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def split_work(items: list, num_workers: int) -> list[list]:
    """Split a list of items into num_workers roughly-equal chunks.

    Items are distributed round-robin style so chunk sizes differ by at most 1.
    If num_workers exceeds len(items), extra chunks are empty lists.

    Args:
        items: List of work items to distribute.
        num_workers: Number of chunks to create.

    Returns:
        List of num_workers lists, each containing a subset of items.
        No item appears in more than one chunk, and all items appear exactly once.

    Raises:
        ValueError: If num_workers < 1.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    chunks: list[list] = [[] for _ in range(num_workers)]
    for i, item in enumerate(items):
        chunks[i % num_workers].append(item)
    return chunks


def _spawn_wrapper(
    rank: int,
    worker_fn: Callable,
    chunks: list[list],
    gpu_ids: list[int],
    cfg: DictConfig,
) -> None:
    """Wrapper function passed to mp.spawn.

    Sets the CUDA device for this rank before delegating to the user-provided
    worker function. This ensures all subsequent CUDA operations in the worker
    (including model loading) target the correct GPU.

    Args:
        rank: Process rank (0-indexed), assigned by mp.spawn.
        worker_fn: User-provided function with signature
            (rank, gpu_id, items, cfg) -> None.
        chunks: Pre-split work items, one chunk per rank.
        gpu_ids: List of CUDA device indices, one per rank.
        cfg: Configuration object passed through to worker_fn.
    """
    gpu_id = gpu_ids[rank]
    items = chunks[rank]

    logger.info("Worker rank %d starting on GPU %d with %d items", rank, gpu_id, len(items))

    torch.cuda.set_device(gpu_id)
    worker_fn(rank, gpu_id, items, cfg)

    logger.info("Worker rank %d on GPU %d finished", rank, gpu_id)


def run_parallel(
    worker_fn: Callable,
    items: list,
    gpu_ids: list[int],
    cfg: DictConfig,
) -> None:
    """Run worker_fn across multiple GPUs with work split evenly.

    For a single GPU, runs the worker directly in the current process (no
    spawning) for simplicity and debuggability. For multiple GPUs, uses
    torch.multiprocessing.spawn to create one process per GPU.

    The worker_fn must have signature:
        worker_fn(rank: int, gpu_id: int, items: list, cfg: DictConfig) -> None

    The worker is responsible for loading its own model (e.g. via load_model),
    processing its items, and saving results. This function only handles work
    splitting, process spawning, GPU assignment, and error propagation.

    Args:
        worker_fn: Function to execute on each GPU. Must accept
            (rank, gpu_id, items_chunk, cfg).
        items: Full list of work items to distribute across GPUs.
        gpu_ids: List of CUDA device indices to use. Length determines
            number of workers.
        cfg: Configuration object passed to each worker.

    Raises:
        ValueError: If gpu_ids is empty.
        RuntimeError: If any worker process fails (error is propagated from
            the failing worker with rank information).
    """
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")

    num_workers = len(gpu_ids)
    chunks = split_work(items, num_workers)

    logger.info(
        "run_parallel: %d items across %d GPU(s) %s, chunk sizes: %s",
        len(items),
        num_workers,
        gpu_ids,
        [len(c) for c in chunks],
    )

    if num_workers == 1:
        # Single-GPU path: run in-process, no spawning.
        gpu_id = gpu_ids[0]
        logger.info("Single GPU mode: running in-process on GPU %d", gpu_id)
        torch.cuda.set_device(gpu_id)
        worker_fn(0, gpu_id, chunks[0], cfg)
        logger.info("Single GPU execution complete")
        return

    # Multi-GPU path: spawn one process per GPU.
    logger.info("Multi-GPU mode: spawning %d processes", num_workers)
    try:
        mp.spawn(
            _spawn_wrapper,
            args=(worker_fn, chunks, gpu_ids, cfg),
            nprocs=num_workers,
            join=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"run_parallel failed during multi-GPU execution with "
            f"{num_workers} workers on GPUs {gpu_ids}: {exc}"
        ) from exc

    logger.info("Multi-GPU execution complete (%d workers)", num_workers)


def _isolated_worker_wrapper(
    worker_fn: Callable,
    rank: int,
    gpu_id: int,
    items: list,
    cfg: DictConfig,
) -> None:
    """Wrapper for mp.Process workers with exception catching.

    Sets the CUDA device and invokes worker_fn within a broad try/except.
    If the worker raises, the exception is logged with a full traceback, and
    the process exits with code 1 (which the parent detects via p.exitcode).

    Args:
        worker_fn: User-provided function with signature
            (rank, gpu_id, items, cfg) -> None.
        rank: Process rank (0-indexed).
        gpu_id: CUDA device index for this worker.
        items: Work items assigned to this worker.
        cfg: Configuration object passed through to worker_fn.
    """
    try:
        torch.cuda.set_device(gpu_id)
        logger.info(
            "Isolated worker rank %d starting on GPU %d with %d items",
            rank,
            gpu_id,
            len(items),
        )
        worker_fn(rank, gpu_id, items, cfg)
        logger.info("Isolated worker rank %d on GPU %d finished successfully", rank, gpu_id)
    except Exception:
        logger.error(
            "Isolated worker rank %d on GPU %d failed:\n%s",
            rank,
            gpu_id,
            traceback.format_exc(),
        )
        raise


def run_parallel_isolated(
    worker_fn: Callable,
    items: list,
    gpu_ids: list[int],
    cfg: DictConfig,
) -> list[tuple[int, int, int]]:
    """Run worker_fn across GPUs with fault isolation via mp.Process.

    Unlike run_parallel() which uses mp.spawn (all workers die if one
    crashes), this function creates independent mp.Process instances.
    If a worker crashes, the others continue to completion. After all
    workers finish (or crash), the function returns a list of failures
    for the caller to inspect.

    Items are deterministically shuffled before splitting so no single GPU
    gets all "hard" problems. The shuffle uses cfg.seed as the random seed.

    For single GPU, still uses mp.Process (not in-process) for consistent
    behavior with multi-GPU. If you want debuggable in-process single-GPU
    execution, use run_parallel() instead.

    The worker_fn must have signature:
        worker_fn(rank: int, gpu_id: int, items: list, cfg: DictConfig) -> None

    Args:
        worker_fn: Function to execute on each GPU. Must accept
            (rank, gpu_id, items_chunk, cfg).
        items: Full list of work items to distribute across GPUs.
        gpu_ids: List of CUDA device indices to use. Length determines
            number of workers.
        cfg: Configuration object passed to each worker. Must have a
            'seed' attribute for deterministic shuffling.

    Returns:
        List of (rank, gpu_id, exitcode) tuples for workers that failed
        (exitcode != 0). Empty list means all workers succeeded.

    Raises:
        ValueError: If gpu_ids is empty.
    """
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")

    num_workers = len(gpu_ids)

    # Deterministic shuffle before splitting to distribute work evenly
    shuffled_items = list(items)
    shuffle_rng = random.Random(cfg.seed)
    shuffle_rng.shuffle(shuffled_items)

    chunks = split_work(shuffled_items, num_workers)

    logger.info(
        "run_parallel_isolated: %d items across %d GPU(s) %s, chunk sizes: %s",
        len(items),
        num_workers,
        gpu_ids,
        [len(c) for c in chunks],
    )

    # Start all workers as independent processes
    processes: list[tuple[int, int, mp.Process]] = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=_isolated_worker_wrapper,
            args=(worker_fn, rank, gpu_id, chunks[rank], cfg),
        )
        p.start()
        logger.info("Started isolated worker rank %d on GPU %d (pid %d)", rank, gpu_id, p.pid)
        processes.append((rank, gpu_id, p))

    # Wait for all processes to complete
    for rank, gpu_id, p in processes:
        p.join()

    # Check exit codes and collect failures
    failed: list[tuple[int, int, int]] = []
    for rank, gpu_id, p in processes:
        if p.exitcode != 0:
            failed.append((rank, gpu_id, p.exitcode))
            logger.error(
                "Isolated worker rank %d on GPU %d exited with code %d",
                rank,
                gpu_id,
                p.exitcode,
            )
        else:
            logger.info("Isolated worker rank %d on GPU %d exited successfully", rank, gpu_id)

    if failed:
        logger.error(
            "run_parallel_isolated: %d/%d workers failed: %s",
            len(failed),
            num_workers,
            [(r, g, c) for r, g, c in failed],
        )
    else:
        logger.info(
            "run_parallel_isolated: all %d workers completed successfully", num_workers
        )

    return failed
