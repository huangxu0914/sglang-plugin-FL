# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FlagOS GemmaRMSNorm implementation using FlagGems-sglang kernels.
# GemmaRMSNorm: output = rms_norm(x) * (weight + 1.0)

from __future__ import annotations

from typing import Optional, Union

import torch

from .flaggems_sglang import get_op


def is_gemma_rms_norm_available() -> bool:
    return (
        get_op("one_pass_rms_norm") is not None
        and get_op("gemma_fused_add_rmsnorm") is not None
    )


def _gemma_weight_fp32(obj) -> torch.Tensor:
    """Return ``1 + weight`` in fp32, cached until the weight is rewritten.

    ``one_pass_rms_norm`` is a standard RMSNorm (``x * rstd * w``), so the Gemma
    offset is folded into the weight. Keeping it in fp32 matches the reference
    ``x * (1.0 + weight.float())``; SGLang's own ``gemma_weight`` buffer is
    stored in the weight dtype and would round ``1 + w`` to bf16.
    """
    weight = obj.weight
    try:
        key = (weight.data_ptr(), weight._version)
    except RuntimeError:
        # Inference tensors carry no version counter; never reuse a stale copy.
        return weight.detach().float() + 1.0
    cached = getattr(obj, "_fl_gemma_weight_fp32", None)
    if cached is None or cached[0] != key:
        cached = (key, weight.detach().float() + 1.0)
        obj._fl_gemma_weight_fp32 = cached
    return cached[1]


def gemma_rms_norm_flagos(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Gemma-style RMS normalization using FlagGems-sglang.

    Args:
        obj: The calling obj (provides obj.weight, obj.variance_epsilon)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, x + residual) if residual
        provided
    """
    if x.numel() == 0:
        return x if residual is None else (x, residual)

    # Vendor kernels address rows as a dense [M, D] matrix.
    shape = x.shape
    x2d = x.reshape(-1, shape[-1]).contiguous()
    eps = obj.variance_epsilon

    if residual is None:
        out = get_op("one_pass_rms_norm")(x2d, _gemma_weight_fp32(obj), eps)
        return out.view(shape)

    out, residual_out = get_op("gemma_fused_add_rmsnorm")(
        x2d,
        residual.reshape(-1, shape[-1]).contiguous(),
        obj.weight,
        eps,
    )
    return out.view(shape), residual_out.view(residual.shape)
