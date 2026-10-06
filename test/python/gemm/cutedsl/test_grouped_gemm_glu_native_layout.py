# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw grouped GLU/dGLU storage preserves the existing kernels and their bytes."""

import pytest
import torch
import cudnn

from gemm.cutedsl.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors
from gemm.cutedsl.test_grouped_gemm_dswiglu_utils import allocate_grouped_gemm_dswiglu_tensors, run_grouped_gemm_dswiglu_ref


def _physical(tensor):
    return torch.as_strided(tensor, (tensor.numel(),), (1,))


def _raw_operands(operands):
    raw, layouts = {}, {}
    for name, tensor in operands.items():
        raw[name] = _physical(tensor)
        if tensor.element_size() == 1:
            raw[name] = raw[name].view(torch.uint8)
        layouts[name] = (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
    return raw, layouts


def _assert_bytes(actual, expected):
    for name in expected.keys():
        if expected[name] is None:
            assert actual[name] is None
        else:
            assert torch.equal(_physical(actual[name]).view(torch.uint8), _physical(expected[name]).view(torch.uint8)), name


def _forbid_views(*args, **kwargs):
    raise AssertionError("native layout launch constructed a torch view")


@pytest.mark.L0
def test_native_layout_geometry_cache_rechecks_current_storage():
    if not torch.cuda.is_available():
        pytest.skip("Native layouts require CUDA storage")
    from cudnn.gemm.cutedsl.grouped.native_layout import logical_tensor

    layout = ((4, 4), (4, 1), torch.float8_e4m3fn)
    storage = torch.empty(32, dtype=torch.uint8, device="cuda")
    logical_tensor(storage, layout)
    # Every call below reuses the same cached geometry with a different carrier.
    # The cache must never turn a previous storage check into an authorization.
    with pytest.raises(ValueError, match="span exceeds"):
        logical_tensor(storage[:15], layout)
    with pytest.raises(ValueError, match="16-byte aligned"):
        logical_tensor(storage[1:17], layout)
    with pytest.raises(ValueError, match="contiguous CUDA"):
        logical_tensor(storage[::2], layout)
    with pytest.raises(ValueError, match="dtype reinterpretation"):
        logical_tensor(torch.empty(32, dtype=torch.int8, device="cuda"), layout)
    with pytest.raises(ValueError, match="contiguous CUDA"):
        logical_tensor(torch.empty(32, dtype=torch.uint8, device="cpu"), layout)


@pytest.mark.L0
@pytest.mark.parametrize("full_dynamic", [False, True])
@pytest.mark.parametrize(
    "operation,b_major,enable_bias",
    [("glu", "k", False), ("glu", "k", True), ("dglu", "k", False), ("dglu", "n", False), ("dglu", "k", True)],
)
def test_glu_native_layout_bitwise_dynamic_and_replay(operation, b_major, enable_bias, full_dynamic, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Grouped MXFP8 native layouts require an SM100-family GPU")
    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1" if full_dynamic else "0")
    from cudnn.gemm.cutedsl.grouped.glu.api import _cache_of_GroupedGemmGluSm100Objects
    from cudnn.gemm.cutedsl.grouped.dglu.api import _cache_of_GroupedGemmDgluSm100Objects

    backward = operation == "dglu"
    wrapper = cudnn.grouped_gemm_dglu_wrapper_sm100 if backward else cudnn.grouped_gemm_glu_wrapper_sm100
    cache = _cache_of_GroupedGemmDgluSm100Objects if backward else _cache_of_GroupedGemmGluSm100Objects
    cache_size = None
    for group_sizes in ([256, 256], [512, 256]):
        inputs = allocate_grouped_gemm_input_tensors(
            n=256 if backward else 512,
            k=256,
            l=2,
            group_m_list=group_sizes,
            ab_dtype=torch.float8_e4m3fn,
            sf_dtype=torch.float8_e8m0fnu,
            sf_vec_size=32,
            m_aligned=256,
            b_major=b_major,
            enable_bias=enable_bias,
        )
        logical = {name: inputs[name] for name in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "prob_tensor")}
        if enable_bias and not backward:
            logical["bias_tensor"] = inputs["bias_tensor"]
        common = dict(
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            norm_const_tensor=inputs["norm_const_tensor"],
            sf_vec_size=32,
            d_dtype=torch.float8_e4m3fn,
            use_dynamic_sched=True,
        )
        if backward:
            inputs, outputs = allocate_grouped_gemm_dswiglu_tensors(
                tensor_m=sum(group_sizes),
                n=256,
                l=2,
                ab_dtype=torch.float8_e4m3fn,
                c_dtype=torch.bfloat16,
                d_dtype=torch.float8_e4m3fn,
                cd_major="n",
                sf_dtype=torch.float8_e8m0fnu,
                sf_vec_size=32,
                input_tensors=inputs,
            )
            logical.update(c_tensor=inputs["c_tensor"], dprob_tensor=outputs["dprob_tensor"])
            common["beta_tensor"] = inputs["beta_tensor"]
            common["generate_dbias"] = enable_bias
            if enable_bias:
                # Identical nonzero contributions make the legacy BF16 atomic
                # reduction order-independent while exercising every M tile.
                for name in ("a_tensor", "b_tensor", "a_ref", "b_ref", "sfa_ref", "sfb_ref", "c_tensor", "prob_tensor", "alpha_tensor", "beta_tensor"):
                    inputs[name].fill_(1)
                for name in ("sfa_tensor", "sfb_tensor"):
                    _physical(inputs[name]).view(torch.uint8).fill_(127)
        expected = wrapper(**logical, **common)
        if backward:
            expected["dprob_tensor"] = expected["dprob_tensor"].clone()
            logical["dprob_tensor"].zero_()
        raw, layouts = _raw_operands(logical)
        actual = wrapper(**raw, tensor_layouts=layouts, **common)
        _assert_bytes(actual, expected)
        if backward and enable_bias:
            reference = run_grouped_gemm_dswiglu_ref(
                a_ref=inputs["a_ref"].float(),
                b_ref=inputs["b_ref"].float(),
                c_ref=inputs["c_tensor"].float(),
                sfa_ref=inputs["sfa_ref"].float(),
                sfb_ref=inputs["sfb_ref"].float(),
                alpha_tensor=inputs["alpha_tensor"],
                beta_tensor=inputs["beta_tensor"],
                prob_tensor=inputs["prob_tensor"],
                aligned_group_m_list=inputs["aligned_group_m_list"],
                valid_m=inputs["valid_m"],
                generate_dbias=True,
                generate_amax=False,
                generate_sfd=False,
                c_dtype=torch.bfloat16,
                d_dtype=torch.bfloat16,
                sf_vec_size=32,
                sf_dtype=torch.float8_e8m0fnu,
            )
            torch.testing.assert_close(actual["dbias_tensor"].float(), reference["dbias_ref"].float(), atol=0.1, rtol=0.01)
        assert all(actual[name].ndim == 1 and actual[name].dtype == torch.uint8 for name in ("d_col_tensor", "sfd_row_tensor", "sfd_col_tensor"))
        if not backward:
            assert actual["c_tensor"].shape == (sum(group_sizes), 512)
        if cache_size is None:
            cache_size = len(cache)
        else:
            assert len(cache) == cache_size, "changed token capacity recompiled the native kernel"

    if backward:
        logical["dprob_tensor"].zero_()
    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "view", _forbid_views)
        guard.setattr(torch.Tensor, "permute", _forbid_views)
        guard.setattr(torch.Tensor, "reshape", _forbid_views)
        actual = wrapper(**raw, tensor_layouts=layouts, **common)
    _assert_bytes(actual, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if backward:
            raw["dprob_tensor"].zero_()
        captured = wrapper(**raw, tensor_layouts=layouts, **common)
    inputs["alpha_tensor"].mul_(-0.5)
    if backward:
        logical["dprob_tensor"].zero_()
    expected = wrapper(**logical, **common)
    if backward:
        expected["dprob_tensor"] = expected["dprob_tensor"].clone()
    graph.replay()
    _assert_bytes(captured, expected)
