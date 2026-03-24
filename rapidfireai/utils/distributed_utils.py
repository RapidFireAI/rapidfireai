"""Distributed training utilities for FSDP support."""

import os
import socket

import torch
import torch.distributed as dist


def find_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def is_distributed_initialized() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_initialized()


def setup_distributed_environment(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: int | None = None,
    timeout_minutes: int = 30,
    local_rank: int | None = None,
) -> None:
    """
    Initialize distributed training environment.

    Args:
        rank: Global rank of this process in the distributed group.
        world_size: Total number of processes in the group.
        local_rank: Local device index. Defaults to rank for multiprocessing,
            but should be 0 when each Ray actor has exactly 1 GPU assigned.
    """
    if dist.is_initialized():
        return

    if local_rank is None:
        local_rank = rank

    if master_port is None:
        master_port = find_free_port()

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    timeout = torch.distributed.constants.default_pg_timeout
    if timeout_minutes > 0:
        timeout = torch.distributed.constants.default_pg_timeout * timeout_minutes

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
        device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def cleanup_distributed() -> None:
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed_initialized():
        dist.barrier()
