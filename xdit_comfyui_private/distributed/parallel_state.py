from typing import List, Optional
import torch
import torch.distributed as dist
from yunchang import set_seq_parallel_pg

from logging import getLogger

from xdit_comfyui_private.model.flux.math import init_seq_parallel_attn


logger = getLogger(__name__)

def is_initialized():
    return dist.is_initialized()

_RANK: Optional[int] = None
_WORLD_SIZE: Optional[int] = None

def init_distributed_enviroment(distributed_init_method, world_size, rank):
    if not is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device("cuda:0")
    else:
        logger.warning("Distributed environment is already initialized.")
    global _RANK, _WORLD_SIZE
    _RANK = rank
    _WORLD_SIZE = world_size

def init_model_parallel(ulysses_degree, ring_degree, rank, world_size):
    set_seq_parallel_pg(
        ulysses_degree, 
        ring_degree, 
        rank, 
        world_size=world_size
    )
    init_seq_parallel_attn()
