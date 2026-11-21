"""CPU contract tests for the FlagGems-sglang adapters in the FlagOS backend."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.flagos.flagos import FlagOSBackend
from sglang_fl.dispatch.backends.flagos.impl import activation, flaggems_sglang
from sglang_fl.dispatch.backends.flagos.impl import gemma_rms_norm as gemma
from sglang_fl.dispatch.backends.flagos.register_ops import register_builtins
from sglang_fl.dispatch.backends.reference.impl.gemma_rms_norm import (
    gemma_rms_norm_torch,
)
from sglang_fl.dispatch.registry import OpRegistry


# Kernels mirroring flaggems_sglang/reference/*.py.
def _one_pass_rms_norm(x, w, eps):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rstd * w.float()).to(x.dtype)


def _gemma_fused_add_rmsnorm(x, residual, weight, eps):
    s = x.float() + residual.float()
    out = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).to(x.dtype), s.to(x.dtype)


def _silu_and_mul(x):
    d = x.shape[-1] // 2
    return (torch.nn.functional.silu(x[..., :d].float()) * x[..., d:].float()).to(
        x.dtype
    )


@pytest.fixture
def fake_package(monkeypatch):
    calls = []

    def recorded(name, fn):
        def wrapper(*args):
            calls.append((name, args))
            return fn(*args)

        return wrapper

    kernels = {
        "one_pass_rms_norm": recorded("one_pass_rms_norm", _one_pass_rms_norm),
        "gemma_fused_add_rmsnorm": recorded(
            "gemma_fused_add_rmsnorm", _gemma_fused_add_rmsnorm
        ),
        "silu_and_mul": recorded("silu_and_mul", _silu_and_mul),
    }
    package = ModuleType("flaggems_sglang")
    package.vendor_name = "nvidia"
    package.get_op = kernels.get
    package.calls = calls
    monkeypatch.setitem(sys.modules, "flaggems_sglang", package)
    flaggems_sglang._package.cache_clear()
    flaggems_sglang.get_op.cache_clear()
    yield package
    flaggems_sglang._package.cache_clear()
    flaggems_sglang.get_op.cache_clear()


def _norm(hidden, dtype=torch.bfloat16):
    obj = torch.nn.Module()
    obj.weight = torch.nn.Parameter(torch.randn(hidden).to(dtype) * 0.5)
    obj.variance_epsilon = 1e-6
    return obj


@pytest.mark.parametrize("shape", [(5, 256), (2, 3, 256), (1, 2048)])
def test_gemma_without_residual_matches_reference(fake_package, shape):
    torch.manual_seed(0)
    obj = _norm(shape[-1])
    x = torch.randn(*shape).to(torch.bfloat16)

    actual = gemma.gemma_rms_norm_flagos(obj, x)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, gemma_rms_norm_torch(obj, x))
    (name, (_, weight, eps)), = fake_package.calls
    assert name == "one_pass_rms_norm"
    assert weight.dtype == torch.float32
    assert eps == obj.variance_epsilon


def test_gemma_without_residual_accepts_strided_input(fake_package):
    torch.manual_seed(1)
    obj = _norm(256)
    qkv = torch.randn(4, 3 * 256).to(torch.bfloat16)
    k = qkv[:, 256:512]

    actual = gemma.gemma_rms_norm_flagos(obj, k)

    torch.testing.assert_close(actual, gemma_rms_norm_torch(obj, k))
    (_, (x2d, _, _)), = fake_package.calls
    assert x2d.is_contiguous()


@pytest.mark.parametrize("shape", [(7, 128), (2, 3, 512)])
def test_gemma_with_residual_matches_reference(fake_package, shape):
    torch.manual_seed(2)
    obj = _norm(shape[-1])
    x = torch.randn(*shape).to(torch.bfloat16)
    residual = torch.randn(*shape).to(torch.bfloat16)
    x_before, residual_before = x.clone(), residual.clone()

    out, residual_out = gemma.gemma_rms_norm_flagos(obj, x, residual)

    expected_out, expected_residual = gemma_rms_norm_torch(
        obj, x_before.clone(), residual_before.clone()
    )
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(residual_out, expected_residual)
    assert out.shape == residual_out.shape == x.shape
    torch.testing.assert_close(x, x_before)
    torch.testing.assert_close(residual, residual_before)
    (name, (_, _, weight, _)), = fake_package.calls
    assert name == "gemma_fused_add_rmsnorm"
    assert weight is obj.weight


def test_gemma_empty_input_skips_kernels(fake_package):
    obj = _norm(256)
    x = torch.empty(0, 256, dtype=torch.bfloat16)
    residual = torch.empty_like(x)

    assert gemma.gemma_rms_norm_flagos(obj, x) is x
    out, residual_out = gemma.gemma_rms_norm_flagos(obj, x, residual)
    assert out is x and residual_out is residual
    assert fake_package.calls == []


def test_gemma_weight_offset_follows_in_place_weight_updates(fake_package):
    obj = _norm(64)
    x = torch.randn(3, 64).to(torch.bfloat16)
    first = gemma._gemma_weight_fp32(obj)
    assert gemma._gemma_weight_fp32(obj) is first

    with torch.no_grad():
        obj.weight.copy_(torch.randn(64).to(torch.bfloat16))

    second = gemma._gemma_weight_fp32(obj)
    assert second is not first
    torch.testing.assert_close(second, obj.weight.float() + 1.0)
    torch.testing.assert_close(
        gemma.gemma_rms_norm_flagos(obj, x), gemma_rms_norm_torch(obj, x)
    )


def test_silu_and_mul_uses_flaggems_sglang(fake_package):
    x = torch.randn(3, 2, 16).to(torch.bfloat16)

    torch.testing.assert_close(activation.silu_and_mul_flagos(None, x), _silu_and_mul(x))
    assert [name for name, _ in fake_package.calls] == ["silu_and_mul"]


def _install_fake_flag_gems(monkeypatch, calls):
    module = ModuleType("flag_gems.modules.activation")

    def gems_silu_and_mul(x1, x2):
        calls.append("flag_gems")
        return torch.nn.functional.silu(x1) * x2

    module.gems_silu_and_mul = gems_silu_and_mul
    monkeypatch.setitem(sys.modules, "flag_gems", ModuleType("flag_gems"))
    monkeypatch.setitem(sys.modules, "flag_gems.modules", ModuleType("flag_gems.modules"))
    monkeypatch.setitem(sys.modules, "flag_gems.modules.activation", module)


def test_unlisted_vendor_falls_back(fake_package, monkeypatch):
    fake_package.vendor_name = "kunlunxin"
    flag_gems_calls = []
    _install_fake_flag_gems(monkeypatch, flag_gems_calls)

    assert flaggems_sglang.get_op("silu_and_mul") is None
    assert not gemma.is_gemma_rms_norm_available()
    activation.silu_and_mul_flagos(None, torch.randn(2, 8))
    assert flag_gems_calls == ["flag_gems"]
    assert fake_package.calls == []


def test_missing_kernel_disables_gemma(fake_package):
    fake_package.get_op = {"one_pass_rms_norm": _one_pass_rms_norm}.get

    assert not gemma.is_gemma_rms_norm_available()


def test_broken_package_import_falls_back(monkeypatch):
    class _Broken:
        def find_spec(self, name, path=None, target=None):
            if name == "flaggems_sglang":
                raise RuntimeError("op module failed to import")
            return None

    monkeypatch.delitem(sys.modules, "flaggems_sglang", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Broken(), *sys.meta_path])
    flaggems_sglang._package.cache_clear()
    flaggems_sglang.get_op.cache_clear()
    try:
        assert flaggems_sglang.get_op("silu_and_mul") is None
    finally:
        flaggems_sglang._package.cache_clear()
        flaggems_sglang.get_op.cache_clear()


@pytest.mark.parametrize("vendor_name,expected", [("nvidia", True), ("enflame", False)])
def test_flagos_registers_gemma_rms_norm(fake_package, monkeypatch, vendor_name, expected):
    fake_package.vendor_name = vendor_name
    monkeypatch.setattr(FlagOSBackend, "_available", True)
    registry = OpRegistry()
    register_builtins(registry)

    (impl,) = registry.snapshot().impls_by_op["gemma_rms_norm"]

    assert impl.impl_id == "default.flagos"
    assert impl.is_available() is expected
    if expected:
        obj = _norm(32)
        x = torch.randn(2, 32).to(torch.bfloat16)
        torch.testing.assert_close(impl.fn(obj, x), gemma_rms_norm_torch(obj, x))
