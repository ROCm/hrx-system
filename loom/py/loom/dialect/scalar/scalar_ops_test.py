# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Callable, Sequence

from loom.diagnostics import DiagnosticEngine
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.ir import (
    Block,
    Module,
    Operation,
    Region,
    ScalarType,
    ScalarTypeKind,
    Symbol,
    Value,
)
from loom.verify import verify_module

_INTEGER_TYPES = tuple(
    ScalarType(kind)
    for kind in (
        ScalarTypeKind.I1,
        ScalarTypeKind.I8,
        ScalarTypeKind.I16,
        ScalarTypeKind.I32,
        ScalarTypeKind.I64,
    )
)
_FLOAT_TYPES = tuple(
    ScalarType(kind)
    for kind in (
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F16,
        ScalarTypeKind.BF16,
        ScalarTypeKind.F32,
        ScalarTypeKind.F64,
    )
)
_ADDRESS_TYPES = tuple(ScalarType(kind) for kind in (ScalarTypeKind.INDEX, ScalarTypeKind.OFFSET))
_PAYLOAD_TYPES = _INTEGER_TYPES + _FLOAT_TYPES
_SCALAR_TYPES = _ADDRESS_TYPES + _PAYLOAD_TYPES


def _verify_cast(op_name: str, source_type: ScalarType, result_type: ScalarType) -> DiagnosticEngine:
    module = Module()
    source = module.add_value(Value("source", source_type))
    result = module.add_value(Value("result", result_type))
    cast = Operation(name=op_name, operands=[source], results=[result])
    function = Operation(
        name="func.def",
        attributes={"callee": "cast"},
        regions=[
            Region(
                blocks=[
                    Block(ops=[cast, Operation(name="func.return")]),
                ]
            )
        ],
    )
    module.add_symbol(Symbol(name="cast", op=function))
    return verify_module(module, ops=(*ALL_FUNC_OPS, *ALL_SCALAR_OPS))


def _greater(source_type: ScalarType, result_type: ScalarType) -> bool:
    return result_type.bitwidth > source_type.bitwidth


def _less(source_type: ScalarType, result_type: ScalarType) -> bool:
    return result_type.bitwidth < source_type.bitwidth


def _same_payload_width(source_type: ScalarType, result_type: ScalarType) -> bool:
    return source_type in _PAYLOAD_TYPES and result_type in _PAYLOAD_TYPES and source_type.bitwidth == result_type.bitwidth


def test_scalar_cast_structural_matrix() -> None:
    cast_families: Sequence[
        tuple[
            str,
            Sequence[ScalarType],
            Callable[[ScalarType, ScalarType], bool],
        ]
    ] = (
        ("scalar.extsi", _INTEGER_TYPES, _greater),
        ("scalar.extui", _INTEGER_TYPES, _greater),
        ("scalar.trunci", _INTEGER_TYPES, _less),
        ("scalar.extf", _FLOAT_TYPES, _greater),
        ("scalar.fptrunc", _FLOAT_TYPES, _less),
        ("scalar.bitcast", _SCALAR_TYPES, _same_payload_width),
    )

    candidate_count = 0
    accepted_count = 0
    for op_name, scalar_types, accepts in cast_families:
        for source_type in scalar_types:
            for result_type in scalar_types:
                candidate_count += 1
                expected_valid = accepts(source_type, result_type)
                diagnostics = _verify_cast(op_name, source_type, result_type)
                actual_valid = not diagnostics.has_errors
                assert actual_valid == expected_valid, f"{op_name} {source_type} to {result_type}: expected valid={expected_valid}, got {diagnostics.diagnostics}"
                accepted_count += int(actual_valid)

    assert candidate_count == 316
    assert accepted_count == 83
