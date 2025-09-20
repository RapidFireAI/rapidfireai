"""Distributed training utilities for FSDP support."""

import os
from typing import Any, Dict, Optional

import torch
import socket
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

def find_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
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
    master_port: Optional[int] = None,
    timeout_minutes: int = 30
) -> None:
    """
    Initialize distributed training environment.
    """
    # Check if already initialized
    if dist.is_initialized():
        return
    
    if master_port is None:
        master_port = find_free_port()
    
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    
    # Initialize the process group
    timeout = torch.distributed.constants.default_pg_timeout
    if timeout_minutes > 0:
        timeout = torch.distributed.constants.default_pg_timeout * timeout_minutes

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
        device_id=0 if torch.cuda.is_available() else None
    )
    
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed_initialized():
        dist.barrier()

