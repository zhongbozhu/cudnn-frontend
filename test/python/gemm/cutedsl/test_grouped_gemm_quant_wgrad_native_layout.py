# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw MXFP8 storage must execute the same grouped GEMMs as logical views."""

import pytest
import torch
import cudnn

from gemm.cutedsl.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors
from gemm.cutedsl.test_discrete_grouped_gemm_swiglu_utils import allocate_discrete_input_tensors
from gemm.cutedsl.test_grouped_gemm_wgrad_utils import allocate_grouped_gemm_wgrad_tensors


def _require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Grouped MXFP8 native layouts require an SM100-family GPU")


def _physical(tensor):
    """Expose physical storage during test setup, preserving its byte order."""
    return torch.as_strided(tensor, (tensor.numel(),), (1,))


def _raw_operands(operands):
    raw, layouts = {}, {}
    for name, tensor in operands.items():
        raw[name] = _physical(tensor).view(torch.uint8) if tensor.element_size() == 1 else _physical(tensor)
        layouts[name] = (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
    return raw, layouts


def _forbid_views(*args, **kwargs):
    raise AssertionError("native layout launch constructed a torch tensor view")


def _assert_quant_bytes(actual, expected):
    for name in expected.keys():
        if expected[name] is None:
            assert actual[name] is None
        else:
            assert torch.equal(_physical(actual[name]).view(torch.uint8), _physical(expected[name]).view(torch.uint8)), name


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_quant_native_layout_bitwise_dynamic_and_replay(discrete, output_dtype, monkeypatch):
    _require_blackwell()
    assert cudnn.grouped_gemm_quant_wrapper_sm100.supports_tensor_layouts
    from cudnn.gemm.cutedsl.grouped.quant.api import _cache_of_GroupedGemmQuantSm100Objects

    cache_size = None
    for group_sizes in ([256, 256], [512, 256]):
        options = dict(
            n=256, k=256, group_m_list=group_sizes, ab_dtype=torch.float8_e4m3fn, sf_dtype=torch.float8_e8m0fnu, sf_vec_size=32, m_aligned=256, enable_bias=True
        )
        if discrete:
            inputs = allocate_discrete_input_tensors(num_experts=2, **options)
            weights = dict(b_ptrs=inputs["b_ptrs_tensor"], sfb_ptrs=inputs["sfb_ptrs_tensor"], n=256, b_dtype=torch.float8_e4m3fn)
            logical = {name: inputs[name] for name in ("a_tensor", "sfa_tensor")}
        else:
            inputs = allocate_grouped_gemm_input_tensors(l=2, **options)
            weights = {}
            logical = {name: inputs[name] for name in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor")}
        logical["bias_tensor"] = inputs["bias_tensor"]
        logical["prob_tensor"] = inputs["prob_tensor"]
        common = dict(
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            norm_const_tensor=inputs["norm_const_tensor"],
            sf_vec_size=32,
            d_dtype=output_dtype,
            use_dynamic_sched=True,
            **weights,
        )
        expected = cudnn.grouped_gemm_quant_wrapper_sm100(**logical, **common)
        raw, layouts = _raw_operands(logical)
        supplied_output = None
        if output_dtype == torch.bfloat16:
            supplied_output = torch.empty((sum(group_sizes), 256), device="cuda", dtype=output_dtype)
            raw["d_tensor"] = supplied_output
            layouts["d_tensor"] = ((sum(group_sizes), 256, 1), (256, 1, sum(group_sizes) * 256), output_dtype)
        actual = cudnn.grouped_gemm_quant_wrapper_sm100(**raw, tensor_layouts=layouts, **common)
        _assert_quant_bytes(actual, expected)
        if supplied_output is not None:
            assert actual["d_tensor"] is supplied_output
        else:
            for name in ("d_tensor", "d_col_tensor", "sfd_row_tensor", "sfd_col_tensor"):
                assert actual[name].ndim == 1 and actual[name].dtype == torch.uint8
        if cache_size is None:
            cache_size = len(_cache_of_GroupedGemmQuantSm100Objects)
        else:
            assert len(_cache_of_GroupedGemmQuantSm100Objects) == cache_size

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "view", _forbid_views)
        guard.setattr(torch.Tensor, "permute", _forbid_views)
        actual = cudnn.grouped_gemm_quant_wrapper_sm100(**raw, tensor_layouts=layouts, **common)
    _assert_quant_bytes(actual, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = cudnn.grouped_gemm_quant_wrapper_sm100(**raw, tensor_layouts=layouts, **common)
    inputs["alpha_tensor"].mul_(-0.5)
    expected = cudnn.grouped_gemm_quant_wrapper_sm100(**logical, **common)
    graph.replay()
    _assert_quant_bytes(captured, expected)


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("te_layout", [False, True])
def test_wgrad_native_layout_bitwise_dynamic_and_replay(discrete, te_layout, monkeypatch):
    _require_blackwell()
    assert cudnn.grouped_gemm_wgrad_wrapper_sm100.supports_tensor_layouts
    from cudnn.gemm.cutedsl.grouped.wgrad.api import _cache_of_GroupedGemmWgradSm100Objects

    cache_size = None
    for group_sizes in ([256, 256], [512, 256]):
        cfg = dict(
            m=256, n=256, l=2, group_k_list=group_sizes, ab_dtype=torch.float8_e4m3fn, sf_dtype=torch.float8_e8m0fnu, sf_vec_size=32, wgrad_dtype=torch.bfloat16
        )
        inputs = allocate_grouped_gemm_wgrad_tensors(cfg)
        if te_layout:
            inputs["a_tensor"] = inputs["a_tensor"].T.contiguous().T
            inputs["b_tensor"] = inputs["b_tensor"].contiguous()
        logical = {name: inputs[name] for name in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor")}
        common = dict(offsets_tensor=inputs["offsets_tensor"], sf_vec_size=32, wgrad_dtype=torch.bfloat16)
        expected = cudnn.grouped_gemm_wgrad_wrapper_sm100(**logical, **common)["wgrad_tensor"]
        raw, layouts = _raw_operands(logical)
        output = torch.empty((2, 256, 256), dtype=torch.bfloat16, device="cuda")
        if discrete:
            pointers = torch.tensor([output[i].data_ptr() for i in range(2)], dtype=torch.int64, device="cuda")
            outputs = dict(output_mode="discrete", wgrad_ptrs=pointers)
        else:
            outputs = dict(wgrad_tensor=output)
            layouts["wgrad_tensor"] = (tuple(output.shape), tuple(output.stride()), output.dtype)
        cudnn.grouped_gemm_wgrad_wrapper_sm100(**raw, tensor_layouts=layouts, **outputs, **common)
        assert torch.equal(output, expected)
        if inputs["ref_result"] is not None:
            torch.testing.assert_close(output, inputs["ref_result"], rtol=0.1, atol=0.1)
        if cache_size is None:
            cache_size = len(_cache_of_GroupedGemmWgradSm100Objects)
        else:
            assert len(_cache_of_GroupedGemmWgradSm100Objects) == cache_size

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "view", _forbid_views)
        guard.setattr(torch.Tensor, "permute", _forbid_views)
        cudnn.grouped_gemm_wgrad_wrapper_sm100(**raw, tensor_layouts=layouts, **outputs, **common)
    assert torch.equal(output, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cudnn.grouped_gemm_wgrad_wrapper_sm100(**raw, tensor_layouts=layouts, **outputs, **common)
    raw["a_tensor"].zero_()
    graph.replay()
    assert torch.count_nonzero(output) == 0


@pytest.mark.L0
@pytest.mark.parametrize("operation", ["quant", "wgrad"])
def test_native_layout_direct_class_api(operation):
    _require_blackwell()
    if operation == "quant":
        inputs = allocate_grouped_gemm_input_tensors(
            n=256,
            k=256,
            l=2,
            group_m_list=[256, 256],
            ab_dtype=torch.float8_e4m3fn,
            sf_dtype=torch.float8_e8m0fnu,
            sf_vec_size=32,
            m_aligned=256,
        )
        logical = {name: inputs[name] for name in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor")}
        common = dict(padded_offsets=inputs["padded_offsets_tensor"], alpha_tensor=inputs["alpha_tensor"])
        expected = cudnn.grouped_gemm_quant_wrapper_sm100(**logical, **common, sf_vec_size=32)["d_tensor"]
        raw, layouts = _raw_operands(logical)
        raw["d_tensor"] = torch.empty((512, 256), dtype=torch.bfloat16, device="cuda")
        layouts["d_tensor"] = ((512, 256, 1), (256, 1, 512 * 256), torch.bfloat16)
        api = cudnn.GroupedGemmQuantSm100(
            sample_a=raw["a_tensor"],
            sample_b=raw["b_tensor"],
            sample_sfa=raw["sfa_tensor"],
            sample_sfb=raw["sfb_tensor"],
            sample_padded_offsets=common["padded_offsets"],
            sample_alpha=common["alpha_tensor"],
            sample_d=raw["d_tensor"],
            sf_vec_size=32,
            tensor_layouts=layouts,
        )
        api.check_support()
        api.compile()
        api.execute(**raw, **common)
        assert torch.equal(raw["d_tensor"], expected.squeeze(-1))
    else:
        inputs = allocate_grouped_gemm_wgrad_tensors(
            dict(
                m=256,
                n=256,
                l=2,
                group_k_list=[256, 256],
                ab_dtype=torch.float8_e4m3fn,
                sf_dtype=torch.float8_e8m0fnu,
                sf_vec_size=32,
                wgrad_dtype=torch.bfloat16,
            )
        )
        logical = {name: inputs[name] for name in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor")}
        common = dict(offsets_tensor=inputs["offsets_tensor"])
        expected = cudnn.grouped_gemm_wgrad_wrapper_sm100(**logical, **common, sf_vec_size=32)["wgrad_tensor"]
        raw, layouts = _raw_operands(logical)
        raw["wgrad_tensor"] = torch.empty((2 * 256 * 256,), dtype=torch.bfloat16, device="cuda")
        layouts["wgrad_tensor"] = ((2, 256, 256), (256 * 256, 256, 1), torch.bfloat16)
        api = cudnn.GroupedGemmWgradSm100(
            sample_a=raw["a_tensor"],
            sample_b=raw["b_tensor"],
            sample_sfa=raw["sfa_tensor"],
            sample_sfb=raw["sfb_tensor"],
            sample_offsets=common["offsets_tensor"],
            sample_wgrad=raw["wgrad_tensor"],
            sf_vec_size=32,
            tensor_layouts=layouts,
        )
        api.check_support()
        api.compile()
        api.execute(**raw, **common)
        assert torch.equal(raw["wgrad_tensor"], expected.reshape(-1))
