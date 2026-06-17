# Bridge: Indexer (DeepSeek DSA)
#
# Unlike rms_norm/silu_and_mul, Indexer's forward signature diverges across
# platforms by design — there is no common subset to standardize to:
#   forward_cuda: ..., return_indices=True
#   forward_npu : ..., layer_scatter_modes=None, dynamic_scale=None
# Each set carries platform-specific semantics that cannot be translated.
#
# So this bridge is a pure hook entry point: it forwards *args/**kwargs
# untouched. Each backend declares its own platform-native signature
# (matching the forward_xxx it calls), keeping per-backend contracts
# clear without faking a unified one here.

from __future__ import annotations

from sglang_fl.dispatch import call_op


def indexer_bridge(self, *args, **kwargs):
    """SGLang Indexer forward → dispatch call_op("indexer", ...).

    Pass-through; backend implementations own the platform-native signature.
    """
    return call_op("indexer", self, *args, **kwargs)
