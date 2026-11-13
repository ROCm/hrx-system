# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU buffer source-to-low contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import product

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.globals import ALL_GLOBAL_OPS
from loom.dialect.globals import defs as globals
from loom.error.amdgpu import ERR_AMDGPU_001
from loom.target.arch.amdgpu.contracts.materializers import REGISTERS_VGPR_MATERIALIZER
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
    SourceValueKind,
    ValueAliasRule,
    ValueRef,
    View,
    descriptor_by_key,
    string_param,
    target_diagnostic,
    value_type_param,
)
from loom.target.low_descriptors import Descriptor

_ADDRESS_DESCRIPTOR_KEYS = (
    "s_mov_b32",
    "v_mov_b32",
    "s_cselect_b32",
    "s_add_u32",
    "v_add_u32",
    "s_add_co_u32",
    "s_addc_u32",
    "v_add_co_u32",
    "v_add_co_ci_u32",
    "s_cmp_eq_u64",
    "s_cmp_lg_u64",
    "s_cmp_eq_i32",
    "s_cmp_lg_i32",
    "v_cmp_eq_i32",
    "v_cmp_ne_i32",
    "s_and_b64",
    "s_or_b64",
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.buffer",
    descriptor_keys=tuple(
        dict.fromkeys(
            (
                *(f"amdgpu.{key}" for key in _ADDRESS_DESCRIPTOR_KEYS),
                "amdgpu.s_mov_b32",
                "amdgpu.v_mov_b32",
                "amdgpu.s_cmp_eq_u64.src1_inline",
                "amdgpu.s_cselect_b32",
                "amdgpu.s_or_b32",
                "amdgpu.v_or_b32",
                "amdgpu.v_cmp_eq_i32.src0_inline",
                "amdgpu.v_cmp_eq_i32",
            )
        )
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


_FLAT_SPACES = ("unknown", "global", "constant", "host", "descriptor", "generic")
_DOMAINS = ((_FLAT_SPACES, 2), (("private",), 1), (("workgroup",), 1))


def _op(
    key: str,
    operands: Mapping[str, ValueRef],
    results: Mapping[str, ValueRef],
    *,
    immediates: Mapping[str, int] | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=_descriptor(f"amdgpu.{key}"),
        operands=operands,
        results=results,
        result_types={
            name: DescriptorResultType()
            for name, value in results.items()
            if value.kind is SourceValueKind.TEMPORARY
        },
        immediates=immediates or {},
    )


def _constant(bank: str, name: str, value: int) -> EmitDescriptorOp:
    return _op(
        f"{bank[0]}_mov_b32",
        {},
        {"dst": ValueRef.temporary(name)},
        immediates={"imm32": value},
    )


def _source(field: str, bank: str) -> ValueRef:
    return ValueRef.operand(
        field, materializer=REGISTERS_VGPR_MATERIALIZER.name if bank == "vgpr" else None
    )


def _position(
    side: str, bank: str, unit_count: int, offset_units: int
) -> tuple[EmitDescriptorOp | EmitRegisterSlice | EmitRegisterConcat, ...]:
    base = _source(side, bank)
    offset = _source(f"{side}_offset", bank)
    address = ValueRef.temporary(f"{side}_address")
    if unit_count == 1:
        return (
            _op(f"{bank[0]}_add_u32", {"lhs": base, "rhs": offset}, {"dst": address}),
        )
    emits = []
    for name, source, units in (("base", base, 2), ("offset", offset, offset_units)):
        if units == 1:
            emits.append(_constant(bank, f"{side}_{name}_high", 0))
        emits.extend(
            EmitRegisterSlice(
                source=source,
                result=ValueRef.temporary(
                    f"{side}_{name}_{'low' if word == 0 else 'high'}"
                ),
                unit_offset=word,
                unit_count=1,
            )
            for word in range(units)
        )
    carry = ValueRef.temporary(f"{side}_carry")
    low = ValueRef.temporary(f"{side}_low")
    high = ValueRef.temporary(f"{side}_high")
    emits.extend(
        (
            _op(
                f"{bank[0]}_add_co_u32",
                {
                    "lhs": ValueRef.temporary(f"{side}_base_low"),
                    "rhs": ValueRef.temporary(f"{side}_offset_low"),
                },
                {"dst" if bank == "sgpr" else "sum": low, "carry": carry},
            ),
            _op(
                "s_addc_u32" if bank == "sgpr" else "v_add_co_ci_u32",
                {
                    "lhs": ValueRef.temporary(f"{side}_base_high"),
                    "rhs": ValueRef.temporary(f"{side}_offset_high"),
                    "carry_in": carry,
                },
                {"sum": high, "carry": ValueRef.temporary(f"{side}_overflow")},
            ),
            EmitRegisterConcat(sources=(low, high), result=address, unit_count=2),
        )
    )
    return tuple(emits)


def _comparison(
    bank: str, unit_count: int, predicate: str, result_kind: str
) -> tuple[EmitDescriptorOp | EmitRegisterSlice, ...]:
    result = ValueRef.result("result")
    left, right = (ValueRef.temporary(f"{side}_address") for side in ("lhs", "rhs"))
    if bank == "sgpr":
        condition = result if result_kind == "scc" else ValueRef.temporary("condition")
        suffix = "eq" if predicate == "eq" else "lg"
        emits = [
            _op(
                f"s_cmp_{suffix}_{'u64' if unit_count == 2 else 'i32'}",
                {"lhs": left, "rhs": right},
                {"scc": condition},
            )
        ]
        if result_kind != "scc":
            emits.extend(
                (
                    _constant(bank, "zero", 0),
                    _constant(bank, "one", 1),
                    _op(
                        "s_cselect_b32",
                        {
                            "true_value": ValueRef.temporary("one"),
                            "false_value": ValueRef.temporary("zero"),
                            "condition": condition,
                        },
                        {"dst": result},
                    ),
                )
            )
        return tuple(emits)
    if unit_count == 1:
        return (
            _op(
                f"v_cmp_{predicate}_i32", {"lhs": left, "rhs": right}, {"mask": result}
            ),
        )
    emits = []
    for word in range(2):
        for side, source in (("lhs", left), ("rhs", right)):
            emits.append(
                EmitRegisterSlice(
                    source=source,
                    result=ValueRef.temporary(f"{side}_word{word}"),
                    unit_offset=word,
                    unit_count=1,
                )
            )
        emits.append(
            _op(
                f"v_cmp_{predicate}_i32",
                {
                    "lhs": ValueRef.temporary(f"lhs_word{word}"),
                    "rhs": ValueRef.temporary(f"rhs_word{word}"),
                },
                {"mask": ValueRef.temporary(f"mask{word}")},
            )
        )
    emits.append(
        _op(
            "s_and_b64" if predicate == "eq" else "s_or_b64",
            {"lhs": ValueRef.temporary("mask0"), "rhs": ValueRef.temporary("mask1")},
            {"dst": result},
        )
    )
    return tuple(emits)


def _address_rules() -> Iterator[DescriptorRule]:
    for (spaces, units), (bank, result_kind), predicate in product(
        _DOMAINS, (("sgpr", "scc"), ("sgpr", "bool"), ("vgpr", "mask")), ("eq", "ne")
    ):
        for offset_units in product((1, 2) if units == 2 else (1,), repeat=2):
            guards = [
                Guard.enum_attr_equals("predicate", predicate),
                Guard.low_value_register_class(
                    "result", "amdgpu.scc" if result_kind == "scc" else "amdgpu.sgpr"
                ),
                Guard.low_value_register_unit_count(
                    "result", 2 if result_kind == "mask" else 1
                ),
            ]
            for side, offset_count in zip(("lhs", "rhs"), offset_units, strict=True):
                guards.extend(
                    (
                        Guard.value_memory_space(side, spaces),
                        Guard.low_value_register_unit_count(side, units),
                        Guard.low_value_register_unit_count(
                            f"{side}_offset", offset_count
                        ),
                    )
                )
                guards.extend(
                    Guard.value_materializable(field, REGISTERS_VGPR_MATERIALIZER.name)
                    if bank == "vgpr"
                    else Guard.low_value_register_class(field, "amdgpu.sgpr")
                    for field in (side, f"{side}_offset")
                )
            yield DescriptorRule(
                source_op=buffer.buffer_compare_address,
                guards=tuple(guards),
                emit=(
                    *_position("lhs", bank, units, offset_units[0]),
                    *_position("rhs", bank, units, offset_units[1]),
                    *_comparison(bank, units, predicate, result_kind),
                ),
            )


def _cross_domain_address_rules() -> Iterator[DescriptorRule]:
    # Proven disjoint spaces share only null. Unknown/generic flat addresses
    # cannot establish disjointness from a segment address.
    disjoint_domains = (
        (("global", "constant", "host", "descriptor"), 2),
        (("private",), 1),
        (("workgroup",), 1),
    )
    for (left_spaces, left_units), (right_spaces, right_units) in product(
        disjoint_domains, repeat=2
    ):
        if left_spaces == right_spaces:
            continue
        for predicate in ("eq", "ne"):
            emits = []
            for side, units in (("lhs", left_units), ("rhs", right_units)):
                value = _source(side, "vgpr")
                if units == 2:
                    words = tuple(
                        ValueRef.temporary(f"{side}_word{word}") for word in range(2)
                    )
                    emits.extend(
                        EmitRegisterSlice(
                            source=value,
                            result=words[word],
                            unit_offset=word,
                            unit_count=1,
                        )
                        for word in range(2)
                    )
                    value = ValueRef.temporary(f"{side}_bits")
                    emits.append(
                        _op(
                            "v_or_b32",
                            {"lhs": words[0], "rhs": words[1]},
                            {"dst": value},
                        )
                    )
                emits.extend(
                    (
                        _constant(
                            "vgpr", f"{side}_null", 0 if units == 2 else 0xFFFFFFFF
                        ),
                        _op(
                            f"v_cmp_{predicate}_i32",
                            {"lhs": value, "rhs": ValueRef.temporary(f"{side}_null")},
                            {"mask": ValueRef.temporary(f"{side}_predicate")},
                        ),
                    )
                )
            emits.append(
                _op(
                    "s_and_b64" if predicate == "eq" else "s_or_b64",
                    {
                        "lhs": ValueRef.temporary("lhs_predicate"),
                        "rhs": ValueRef.temporary("rhs_predicate"),
                    },
                    {"dst": ValueRef.result("result")},
                )
            )
            yield DescriptorRule(
                source_op=buffer.buffer_compare_address,
                guards=(
                    Guard.enum_attr_equals("predicate", predicate),
                    Guard.value_memory_space("lhs", left_spaces),
                    Guard.value_memory_space("rhs", right_spaces),
                    Guard.low_value_register_unit_count("lhs", left_units),
                    Guard.low_value_register_unit_count("rhs", right_units),
                    Guard.low_value_register_class("result", "amdgpu.sgpr"),
                    Guard.low_value_register_unit_count("result", 2),
                    *(
                        Guard.value_materializable(
                            side, REGISTERS_VGPR_MATERIALIZER.name
                        )
                        for side in ("lhs", "rhs")
                    ),
                ),
                emit=tuple(emits),
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
    materializers=(REGISTERS_VGPR_MATERIALIZER,),
    cases=(
        *_address_rules(),
        *_cross_domain_address_rules(),
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
