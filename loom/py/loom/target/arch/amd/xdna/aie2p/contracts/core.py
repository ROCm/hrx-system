# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P scalar source-to-Low contract fragment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")

_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1
_SHORT_MIN = -1024
_SHORT_MAX = 1023


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _typed_guards(
    fields: Iterable[str], type_pattern: TypePattern
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _op_emit(
    descriptor: Descriptor,
    *,
    operands: Mapping[str, ValueRef] | None = None,
    results: Mapping[str, ValueRef] | None = None,
    result_types: Mapping[str, TypePattern] | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        form=DescriptorEmitForm.OP,
    )


def _const_emit(
    descriptor: Descriptor,
    result: ValueRef,
    value: AttrProject | int,
    *,
    result_type: TypePattern | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        results={"dst": result},
        result_types=None if result_type is None else {"dst": result_type},
        immediates={"i": value},
        form=DescriptorEmitForm.CONST,
    )


def _constant_rule(
    result_type: TypePattern,
    descriptor_key: str,
    minimum: int,
    maximum: int,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", result_type),
            Guard.i64_range("value", minimum, maximum),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _logical_constant_rule() -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.constant.i32.short")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.value_type("result", _I1),
            Guard.value_exact_i64("result"),
            Guard.value_i64_range("result", 0, 1),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i": ValueProject.exact_i64("result")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor,
                operands={
                    "s0": ValueRef.operand("lhs"),
                    "s1": ValueRef.operand("rhs"),
                },
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _right_shift_rule(source_op: Op, descriptor_key: str) -> DescriptorRule:
    zero = _descriptor("amd.xdna.aie2p.constant.i32.short")
    subtract = _descriptor("amd.xdna.aie2p.sub.i32")
    shift = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=shift,
        guards=_typed_guards(("lhs", "rhs", "result"), _I32),
        emit=(
            _const_emit(
                zero,
                ValueRef.temporary("zero"),
                0,
                result_type=_I32,
            ),
            _op_emit(
                subtract,
                operands={
                    "s0": ValueRef.temporary("zero"),
                    "s1": ValueRef.operand("rhs"),
                },
                results={"d0": ValueRef.temporary("negative_shift")},
                result_types={"d0": _I32},
            ),
            _op_emit(
                shift,
                operands={
                    "s0": ValueRef.operand("lhs"),
                    "s1": ValueRef.temporary("negative_shift"),
                },
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _compare_rule(
    predicate: str,
    descriptor_key: str,
    *,
    swap_operands: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    operands = {
        "s0": ValueRef.operand("rhs" if swap_operands else "lhs"),
        "s1": ValueRef.operand("lhs" if swap_operands else "rhs"),
    }
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            *_typed_guards(("lhs", "rhs"), _I32),
            Guard.value_type("result", _I1),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands=operands,
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _zero_compare_rule(
    predicate: str,
    descriptor_key: str,
    *,
    zero_field: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    nonzero_field = "rhs" if zero_field == "lhs" else "lhs"
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            *_typed_guards(("lhs", "rhs"), _I32),
            Guard.value_type("result", _I1),
            Guard.value_exact_i64(zero_field),
            Guard.value_i64_range(zero_field, 0, 0),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands={"s0": ValueRef.operand(nonzero_field)},
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _bitfield_rule(source_op: Op, descriptor_key: str) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    logical_shift = _descriptor("amd.xdna.aie2p.lshl.i32")
    final_shift = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=final_shift,
        guards=(
            *_typed_guards(("source", "result"), _I32),
            Guard.attr_kind("offset", "i64"),
            Guard.attr_kind("width", "i64"),
            Guard.i64_range("offset", 0, 31),
            Guard.i64_range("width", 1, 32),
        ),
        emit=(
            _const_emit(
                constant,
                ValueRef.temporary("left_shift"),
                AttrProject.i64_literal_minus_attrs(
                    "offset",
                    other_source_attr="width",
                    literal=32,
                ),
                result_type=_I32,
            ),
            _op_emit(
                logical_shift,
                operands={
                    "s0": ValueRef.operand("source"),
                    "s1": ValueRef.temporary("left_shift"),
                },
                results={"d0": ValueRef.temporary("aligned")},
                result_types={"d0": _I32},
            ),
            _const_emit(
                constant,
                ValueRef.temporary("right_shift"),
                AttrProject.i64_attr_minus_literal("width", literal=32),
                result_type=_I32,
            ),
            _op_emit(
                final_shift,
                operands={
                    "s0": ValueRef.temporary("aligned"),
                    "s1": ValueRef.temporary("right_shift"),
                },
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def aie2p_core_cases() -> Sequence[ContractCase]:
    # Specialized cases precede general cases because the compact runtime table
    # is queried in authored order.
    return (
        _logical_constant_rule(),
        _constant_rule(
            _I32,
            "amd.xdna.aie2p.constant.i32.short",
            _SHORT_MIN,
            _SHORT_MAX,
        ),
        _constant_rule(
            _I32,
            "amd.xdna.aie2p.constant.i32",
            _I32_MIN,
            _I32_MAX,
        ),
        _binary_rule(
            scalar_arithmetic.scalar_addi,
            _I32,
            "amd.xdna.aie2p.add.i32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_subi,
            _I32,
            "amd.xdna.aie2p.sub.i32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_muli,
            _I32,
            "amd.xdna.aie2p.mul.i32",
        ),
        *(
            _binary_rule(source_op, type_pattern, descriptor_key)
            for source_op, type_pattern, descriptor_key in (
                (
                    scalar_bitwise.scalar_andi,
                    _I32,
                    "amd.xdna.aie2p.and.i32",
                ),
                (
                    scalar_bitwise.scalar_ori,
                    _I32,
                    "amd.xdna.aie2p.or.i32",
                ),
                (
                    scalar_bitwise.scalar_xori,
                    _I32,
                    "amd.xdna.aie2p.xor.i32",
                ),
                (
                    scalar_bitwise.scalar_andi,
                    _I1,
                    "amd.xdna.aie2p.and.i32",
                ),
                (
                    scalar_bitwise.scalar_ori,
                    _I1,
                    "amd.xdna.aie2p.or.i32",
                ),
                (
                    scalar_bitwise.scalar_xori,
                    _I1,
                    "amd.xdna.aie2p.xor.i32",
                ),
            )
        ),
        _binary_rule(
            scalar_bitwise.scalar_shli,
            _I32,
            "amd.xdna.aie2p.lshl.i32",
        ),
        _right_shift_rule(
            scalar_bitwise.scalar_shrsi,
            "amd.xdna.aie2p.ashl.i32",
        ),
        _right_shift_rule(
            scalar_bitwise.scalar_shrui,
            "amd.xdna.aie2p.lshl.i32",
        ),
        _bitfield_rule(
            scalar_bitwise.scalar_bitfield_extractu,
            "amd.xdna.aie2p.lshl.i32",
        ),
        _bitfield_rule(
            scalar_bitwise.scalar_bitfield_extracts,
            "amd.xdna.aie2p.ashl.i32",
        ),
        _zero_compare_rule(
            "eq",
            "amd.xdna.aie2p.cmp.eqz.i32",
            zero_field="rhs",
        ),
        _zero_compare_rule(
            "eq",
            "amd.xdna.aie2p.cmp.eqz.i32",
            zero_field="lhs",
        ),
        _zero_compare_rule(
            "ne",
            "amd.xdna.aie2p.cmp.nez.i32",
            zero_field="rhs",
        ),
        _zero_compare_rule(
            "ne",
            "amd.xdna.aie2p.cmp.nez.i32",
            zero_field="lhs",
        ),
        *(
            _compare_rule(predicate, descriptor_key, swap_operands=swap_operands)
            for predicate, descriptor_key, swap_operands in (
                ("eq", "amd.xdna.aie2p.cmp.eq.i32", False),
                ("ne", "amd.xdna.aie2p.cmp.ne.i32", False),
                ("slt", "amd.xdna.aie2p.cmp.slt.i32", False),
                ("sle", "amd.xdna.aie2p.cmp.sge.i32", True),
                ("sgt", "amd.xdna.aie2p.cmp.slt.i32", True),
                ("sge", "amd.xdna.aie2p.cmp.sge.i32", False),
                ("ult", "amd.xdna.aie2p.cmp.ult.i32", False),
                ("ule", "amd.xdna.aie2p.cmp.uge.i32", True),
                ("ugt", "amd.xdna.aie2p.cmp.ult.i32", True),
                ("uge", "amd.xdna.aie2p.cmp.uge.i32", False),
            )
        ),
    )


AIE2P_CORE_CONTRACT_DIALECT_OPS = {"scalar": ALL_SCALAR_OPS}

AIE2P_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="amd.xdna.aie2p.core",
    descriptor_set=AIE2P_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/amd/xdna/aie2p/contracts/core.h",
    cases=aie2p_core_cases(),
)
