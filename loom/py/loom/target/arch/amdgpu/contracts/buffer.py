# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU buffer source-to-low contracts."""

from __future__ import annotations

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.globals import ALL_GLOBAL_OPS
from loom.dialect.globals import defs as globals
from loom.error.amdgpu import ERR_AMDGPU_001
from loom.target.arch.amdgpu.contracts.memory import BYTE_ADDRESSABLE_SCALAR_ELEMENTS
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractFragment,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterCopy,
    EmitRegisterSlice,
    Guard,
    GuardDiagnostic,
    RecipeRule,
    Scalar,
    ValueAliasRule,
    ValueRef,
    View,
    descriptor_by_key,
    string_param,
    target_diagnostic,
    value_type_param,
)
from loom.target.low_descriptors import Descriptor

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.buffer",
    descriptor_keys=(
        "amdgpu.s_mov_b32",
        "amdgpu.v_mov_b32",
        "amdgpu.s_cmp_eq_u64.src1_inline",
        "amdgpu.s_cselect_b32",
        "amdgpu.s_or_b32",
        "amdgpu.v_or_b32",
        "amdgpu.v_cmp_eq_i32.src0_inline",
        "amdgpu.v_cmp_eq_i32",
    ),
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(_DESCRIPTOR_SET, key)


def _zero_word(bank: str, name: str) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=_descriptor(f"amdgpu.{bank[0]}_mov_b32"),
        results={"dst": ValueRef.temporary(name)},
        result_types={"dst": DescriptorResultType()},
        immediates={"imm32": 0},
    )


def _null_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        DescriptorRule(
            source_op=buffer.buffer_null,
            guards=(
                Guard.low_value_register_class("result", f"amdgpu.{bank}"),
                Guard.low_value_register_unit_count("result", 2),
            ),
            emit=(
                _zero_word(bank, "zero"),
                EmitRegisterConcat(
                    sources=(ValueRef.temporary("zero"), ValueRef.temporary("zero")),
                    result=ValueRef.result("result"),
                ),
            ),
        )
        for bank in ("sgpr", "vgpr")
    )


def _scalar_null_compare_rule(scalar_boolean: bool) -> DescriptorRule:
    condition = (
        ValueRef.temporary("condition") if scalar_boolean else ValueRef.result("result")
    )
    emit = [
        EmitDescriptorOp(
            descriptor=_descriptor("amdgpu.s_cmp_eq_u64.src1_inline"),
            operands={"lhs": ValueRef.operand("buffer")},
            immediates={"rhs": 0},
            results={"scc": condition},
            result_types={"scc": DescriptorResultType()} if scalar_boolean else None,
        ),
    ]
    if scalar_boolean:
        emit.extend(
            (
                _zero_word("sgpr", "zero"),
                EmitDescriptorOp(
                    descriptor=_descriptor("amdgpu.s_mov_b32"),
                    results={"dst": ValueRef.temporary("one")},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"imm32": 1},
                ),
                EmitDescriptorOp(
                    descriptor=_descriptor("amdgpu.s_cselect_b32"),
                    operands={
                        "true_value": ValueRef.temporary("one"),
                        "false_value": ValueRef.temporary("zero"),
                        "condition": condition,
                    },
                    results={"dst": ValueRef.result("result")},
                ),
            )
        )
    return DescriptorRule(
        source_op=buffer.buffer_compare_null,
        guards=(
            Guard.value_type("result", Scalar("i1")),
            Guard.low_value_register_class("buffer", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("buffer", 2),
            Guard.low_value_register_class(
                "result", "amdgpu.sgpr" if scalar_boolean else "amdgpu.scc"
            ),
            Guard.low_value_register_unit_count("result", 1),
        ),
        emit=tuple(emit),
    )


def _mask_null_compare_rule(bank: str, unit_count: int) -> DescriptorRule:
    emit = []
    address = ValueRef.operand("buffer")
    if unit_count == 2:
        emit.extend(
            EmitRegisterSlice(
                source=address,
                result=ValueRef.temporary(f"word{word}"),
                unit_offset=word,
                unit_count=1,
            )
            for word in range(2)
        )
        address = ValueRef.temporary("address_bits")
        emit.append(
            EmitDescriptorOp(
                descriptor=_descriptor(f"amdgpu.{bank[0]}_or_b32"),
                operands={
                    "lhs": ValueRef.temporary("word0"),
                    "rhs": ValueRef.temporary("word1"),
                },
                results={"dst": address},
                result_types={"dst": DescriptorResultType()},
            )
        )
    if bank == "sgpr":
        emit.append(
            EmitRegisterCopy(
                source=address,
                result=ValueRef.temporary("lane_address"),
                result_type=Scalar("i32"),
            )
        )
        address = ValueRef.temporary("lane_address")
    if unit_count == 1:
        emit.append(
            EmitDescriptorOp(
                descriptor=_descriptor("amdgpu.v_mov_b32"),
                results={"dst": ValueRef.temporary("segment_null")},
                result_types={"dst": DescriptorResultType()},
                immediates={"imm32": 0xFFFFFFFF},
            )
        )
    emit.append(
        EmitDescriptorOp(
            descriptor=_descriptor(
                "amdgpu.v_cmp_eq_i32.src0_inline"
                if unit_count == 2
                else "amdgpu.v_cmp_eq_i32"
            ),
            operands={
                "rhs": address,
                **(
                    {"lhs": ValueRef.temporary("segment_null")}
                    if unit_count == 1
                    else {}
                ),
            },
            results={"mask": ValueRef.result("result")},
            immediates={"lhs": 0} if unit_count == 2 else {},
        )
    )
    return DescriptorRule(
        source_op=buffer.buffer_compare_null,
        guards=(
            Guard.value_type("result", Scalar("i1")),
            Guard.low_value_register_class("buffer", f"amdgpu.{bank}"),
            Guard.low_value_register_unit_count("buffer", unit_count),
            Guard.low_value_register_class("result", "amdgpu.sgpr"),
            Guard.low_value_register_unit_count("result", 2),
        ),
        emit=tuple(emit),
    )


_VIEW_TYPE_DIAGNOSTIC = GuardDiagnostic(
    ref=target_diagnostic(
        ERR_AMDGPU_001,
        string_param("field_name", "result"),
        value_type_param("actual_type", "result"),
        string_param("required_storage", "byte-addressable scalar elements"),
    ),
)

AMDGPU_BUFFER_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "global": ALL_GLOBAL_OPS,
}

AMDGPU_BUFFER_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.buffer",
    descriptor_set=_DESCRIPTOR_SET,
    public_header="loom/target/arch/amdgpu/contracts/buffer.h",
    cases=(
        *_null_rules(),
        *(
            _scalar_null_compare_rule(scalar_boolean)
            for scalar_boolean in (False, True)
        ),
        *(
            _mask_null_compare_rule(bank, count)
            for bank, count in (("sgpr", 2), ("vgpr", 2), ("vgpr", 1))
        ),
        RecipeRule(source_op=globals.global_load),
        ValueAliasRule(
            source_op=buffer.buffer_view,
            source=ValueRef.operand("buffer"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type(
                    "result",
                    View(BYTE_ADDRESSABLE_SCALAR_ELEMENTS),
                    diagnostic=_VIEW_TYPE_DIAGNOSTIC,
                ),
            ),
        ),
    ),
)
