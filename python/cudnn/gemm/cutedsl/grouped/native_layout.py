# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit logical layouts over caller-owned storage for grouped MXFP8 GEMMs.

The public layout value is ``(shape, strides_in_elements, logical_dtype)``.
LogicalTensor is host metadata, never a torch view and never a launch operand.
NativeLaunch rebuilds the corresponding CuTe tensors in the compiled launcher.
"""

from __future__ import annotations

import inspect
import math
from functools import lru_cache
from typing import NamedTuple

import cutlass
import cutlass.cute as cute

from cudnn.datatypes import _convert_to_cutlass_data_type

from .backend_utils import wrapper_operand_meta


class _LayoutProperties(NamedTuple):
    cute_dtype: object
    pattern: tuple
    metadata: tuple
    numel: int
    span: int
    storage_dtype: object
    tensor_type: type


@lru_cache(maxsize=512)
def _layout_properties(shape, strides, dtype):
    """Cache only immutable geometry, never a tensor or a storage observation."""
    import torch

    if not shape or len(shape) != len(strides):
        raise ValueError("tensor_layouts shape and stride must have the same nonzero rank")
    if any(type(x) is not int or x < 0 for x in shape + strides):
        raise ValueError("tensor_layouts extents and strides must be nonnegative integers")
    if any(n >= 2**31 for n in shape) or any(s >= 2**63 for s in strides):
        raise ValueError("tensor_layouts extents must fit int32 and strides must fit int64")
    cute_dtype = _convert_to_cutlass_data_type(dtype)
    allowed = (cutlass.Float8E4M3FN, cutlass.Float8E5M2, cutlass.Float8E8M0FNU, cutlass.BFloat16, cutlass.Float16, cutlass.Float32)
    if cute_dtype not in allowed:
        raise ValueError("tensor_layouts support MXFP8 data/scales and floating point auxiliary tensors")
    numel = math.prod(shape)
    span = 0 if not numel else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    alignment = 4 if cute_dtype == cutlass.Float8E8M0FNU else 16
    pattern = (
        tuple(1 if n == 1 else None for n in shape),
        tuple(s if s in (0, 1) else None for s in strides),
        tuple(math.gcd(s, alignment) for s in strides),
    )
    return _LayoutProperties(
        cute_dtype,
        pattern,
        (shape, strides),
        numel,
        span,
        torch.uint8 if cute_dtype.width == 8 else dtype,
        torch.Tensor,
    )


class LogicalTensor:
    """A checked, lightweight metadata observation of a real tensor."""

    __slots__ = (
        "raw",
        "shape",
        "_strides",
        "dtype",
        "device",
        "ndim",
        "_cute_dtype",
        "_pattern",
        "_raw_abi",
        "_properties",
        "_metadata",
        "_operand_meta",
    )

    def __init__(self, raw, layout, *, _allocated=False):
        shape, strides, dtype = layout
        properties = _layout_properties(tuple(shape), tuple(strides), dtype)
        if not isinstance(raw, properties.tensor_type):
            raise ValueError("tensor_layouts require contiguous CUDA tensor storage")
        self.raw = raw
        self.shape, self._strides = properties.metadata
        self.dtype = dtype
        self.device = raw.device
        self.ndim = len(self.shape)
        self._properties = properties
        self._cute_dtype = properties.cute_dtype
        self._pattern = properties.pattern
        self._metadata = properties.metadata
        self._operand_meta = (self.shape, self._strides, dtype, self.device.type, self.device.index)
        raw_dtype = raw.dtype
        self._raw_abi = (raw_dtype, raw.ndim, self.device)
        if not _allocated:
            self.validate()

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]

    def numel(self):
        return self._properties.numel

    def data_ptr(self):
        return self.raw.data_ptr()

    @property
    def is_cuda(self):
        return self.raw.is_cuda

    def validate(self):
        raw = self.raw
        if self.device.type != "cuda" or not raw.is_contiguous():
            raise ValueError("tensor_layouts require contiguous CUDA tensor storage")
        properties = self._properties
        raw_dtype = self._raw_abi[0]
        if raw_dtype != self.dtype and raw_dtype != properties.storage_dtype:
            raise ValueError("tensor_layouts dtype reinterpretation requires uint8 storage and an 8-bit logical dtype")
        raw_numel = raw.numel()
        if properties.span > raw_numel:
            raise ValueError("tensor_layouts logical span exceeds the supplied storage")
        if raw_numel and raw.data_ptr() % 16:
            raise ValueError("tensor_layouts storage must be 16-byte aligned")


def logical_tensor(tensor, layout=None):
    if tensor is None or layout is None:
        return tensor
    if isinstance(tensor, LogicalTensor):
        raise ValueError("tensor_layouts must be applied to original storage tensors")
    return LogicalTensor(tensor, layout)


def unwrap_tensor(tensor):
    return tensor.raw if isinstance(tensor, LogicalTensor) else tensor


def operand_meta(tensor):
    """Read a logical observation without framework-adapter type discovery."""
    return tensor._operand_meta if isinstance(tensor, LogicalTensor) else wrapper_operand_meta(tensor)


def apply_tensor_layouts(tensors, layouts):
    """Apply only named entries; reject misspellings and missing operands."""
    unknown = set(layouts) - set(tensors)
    if unknown:
        raise ValueError(f"Unknown tensor_layouts entries: {sorted(unknown)}")
    for name, layout in layouts.items():
        if tensors[name] is None:
            raise ValueError(f"tensor_layouts entry {name} has no tensor")
        tensors[name] = logical_tensor(tensors[name], layout)
    return tensors


def layout_key(tensors):
    """Compile ABI key, independent of runtime capacities and data addresses."""
    return tuple((name, tensor.dtype, tensor._raw_abi, tensor._pattern) for name, tensor in sorted(tensors.items()) if isinstance(tensor, LogicalTensor))


def empty_logical(shape, strides, dtype, device, *, physical_shape=None):
    """Allocate the public physical output and describe its kernel interpretation."""
    import torch

    properties = _layout_properties(tuple(shape), tuple(strides), dtype)
    if physical_shape is None:
        physical_shape = (properties.numel,)
    return LogicalTensor(torch.empty(physical_shape, dtype=properties.storage_dtype, device=device), (shape, strides, dtype), _allocated=True)


def native_scale_outputs(m, n, dtype, sf_vec_size, device):
    """Physical scale storage with the existing MMA-interleaved interpretation."""

    def allocate(mn, k):
        rows = (mn + 127) // 128
        rest = (k + sf_vec_size * 4 - 1) // (sf_vec_size * 4)
        return empty_logical(
            (32, 4, rows, 4, rest, 1),
            (16, 4, rest * 512, 1, 512, rows * rest * 512),
            dtype,
            device,
        )

    return allocate(m, n), allocate(n, m)


def public_outputs(outputs):
    from cudnn.api_base import TupleDict

    if not any(isinstance(value, LogicalTensor) for value in outputs.values()):
        return outputs
    return TupleDict(**{name: unwrap_tensor(value) for name, value in outputs.items()})


class NativeLaunch:
    """Trace-time adapter around an unchanged GEMM launcher and device kernels."""

    def __init__(self, kernel, parameters, constants, native_specs):
        self.kernel = kernel
        self.parameters = parameters
        self.constants = constants
        self.native_specs = native_specs

    @cute.jit
    def __call__(self, operands: tuple, metadata: tuple):
        arguments = ()
        for i in cutlass.range_constexpr(len(self.parameters)):
            runtime_index = self.parameters[i]
            if cutlass.const_expr(runtime_index < 0):
                value = self.constants[i]
            else:
                value = operands[runtime_index]
                if cutlass.const_expr(runtime_index in self.native_specs):
                    dtype, shapes, strides, divisibility = self.native_specs[runtime_index]
                    shape = ()
                    stride = ()
                    for j in cutlass.range_constexpr(len(shapes)):
                        if cutlass.const_expr(shapes[j] is not None):
                            shape += (shapes[j],)
                        else:
                            shape += (metadata[runtime_index][0][j],)
                        if cutlass.const_expr(strides[j] is not None):
                            stride += (strides[j],)
                        else:
                            stride += (cute.assume(metadata[runtime_index][1][j], divby=divisibility[j]),)
                    value = cute.make_tensor(cute.recast_ptr(value.iterator, dtype=dtype), cute.make_layout(shape, stride=stride))
                arguments += (value,)
                continue
            arguments += (value,)
        self.kernel(*arguments)


def native_compile(kernel, compile_kwargs, tensors_by_kernel_name):
    """Compile a raw-storage ABI; the returned callable keeps the legacy ABI.

    ``tensors_by_kernel_name`` contains constructor samples, keyed exactly like
    the underlying launcher's parameters. Unmapped operands remain untouched.
    Runtime LogicalTensor observations supply current layouts and original
    tensors; no tensor views, dtype conversions, or device copies are created.
    """
    native = {name: tensor for name, tensor in tensors_by_kernel_name.items() if isinstance(tensor, LogicalTensor)}
    if not native:
        return cute.compile(kernel, **compile_kwargs)

    signature = inspect.signature(kernel.__call__)
    parameters, constants, specs, fake_operands, fake_metadata = [], {}, {}, [], []
    expected_raw_abis = {}
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        value = compile_kwargs.get(parameter.name, parameter.default)
        if value is inspect.Parameter.empty:
            raise ValueError(f"Missing compile argument {parameter.name}")
        is_constexpr = "Constexpr" in str(parameter.annotation)
        if is_constexpr:
            constants[len(parameters)] = value
            parameters.append(-1)
            continue
        index = len(fake_operands)
        parameters.append(index)
        if parameter.name in native:
            sample = native[parameter.name]
            shapes, strides, divisibility = sample._pattern
            specs[index] = (sample._cute_dtype, shapes, strides, divisibility)
            expected_raw_abis[index] = sample._raw_abi
            value = cute.runtime.make_fake_tensor(
                dtype=_convert_to_cutlass_data_type(sample.raw.dtype),
                shape=tuple(cute.sym_int() for _ in sample.raw.shape),
                stride=tuple(cute.sym_int() for _ in sample.raw.shape),
                assumed_align=16,
            )
            fake_metadata.append((tuple(cutlass.Int32(0) for _ in shapes), tuple(cutlass.Int64(0) for _ in strides)))
        else:
            fake_metadata.append(None)
        fake_operands.append(value)

    launcher = NativeLaunch(kernel, tuple(parameters), constants, specs)
    compiled = cute.compile(launcher, tuple(fake_operands), tuple(fake_metadata), options=compile_kwargs.get("options", "--enable-tvm-ffi"))
    operand_count = len(fake_operands)
    expected_operands = tuple((index, spec[0], spec[1:], expected_raw_abis[index]) for index, spec in specs.items())

    def execute(*operands):
        if len(operands) != operand_count:
            raise ValueError(f"Native launch expects {operand_count} runtime arguments, got {len(operands)}")
        raw_operands, metadata = list(operands), [None] * operand_count
        for index, dtype, pattern, raw_abi in expected_operands:
            operand = operands[index]
            if not isinstance(operand, LogicalTensor):
                raise ValueError("Native launch requires the declared logical tensor metadata")
            if operand._cute_dtype != dtype or operand._pattern != pattern or operand._raw_abi != raw_abi:
                raise ValueError("Native tensor layout is incompatible with the compiled ABI")
            metadata[index] = operand._metadata
            raw_operands[index] = operand.raw
        compiled(tuple(raw_operands), tuple(metadata))

    return execute
