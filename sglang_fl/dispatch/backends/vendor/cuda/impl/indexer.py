# CUDA Indexer (DeepSeek DSA) implementation.

from __future__ import annotations

from typing import Optional

import torch


def indexer_cuda(
    obj,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    forward_batch,
    layer_id: int,
    return_indices: bool = True,
) -> Optional[torch.Tensor]:
    return obj.forward_cuda(
        x,
        q_lora,
        positions,
        forward_batch,
        layer_id,
        return_indices=return_indices,
    )
