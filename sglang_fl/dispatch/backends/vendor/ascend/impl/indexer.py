# Ascend Indexer (DeepSeek DSA) implementation.

from __future__ import annotations

from typing import Optional

import torch


def indexer_ascend(
    obj,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    forward_batch,
    layer_id: int,
    layer_scatter_modes=None,
    dynamic_scale: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    return obj.forward_npu(
        x,
        q_lora,
        positions,
        forward_batch,
        layer_id,
        layer_scatter_modes=layer_scatter_modes,
        dynamic_scale=dynamic_scale,
    )
