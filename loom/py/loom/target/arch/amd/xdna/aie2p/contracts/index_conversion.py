# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P address-boundary conversion rules."""

from __future__ import annotations

from loom.dialect.index import defs as index
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    Scalar,
    TypePattern,
    ValueAliasRule,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I8 = Scalar("i8")
_I16 = Scalar("i16")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")

_I32_MAX = (2**31) - 1


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _typed_guards(
    input_type: TypePattern,
    result_type: TypePattern,
) -> tuple[Guard, ...]:
    return (
        Guard.value_type("input", input_type),
        Guard.value_type("result", result_type),
    )


def _alias_rule(
    input_type: TypePattern,
    result_type: TypePattern,
    *,
    extra_guards: tuple[Guard, ...] = (),
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=index.index_cast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(*_typed_guards(input_type, result_type), *extra_guards),
    )


def _extend_rule(
    input_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=index.index_cast,
        descriptor=descriptor,
        guards=_typed_guards(input_type, result_type),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"s0": ValueRef.operand("input")},
                results={"d0": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _address_to_boolean_rule(input_type: TypePattern) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.i32")
    return DescriptorRule(
        source_op=index.index_cast,
        descriptor=bitwise_and,
        guards=_typed_guards(input_type, _I1),
        emit=(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary("low_bit_mask")},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": 1},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=bitwise_and,
                operands={
                    "s0": ValueRef.operand("input"),
                    "s1": ValueRef.temporary("low_bit_mask"),
                },
                results={"d0": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _i64_to_address_rule(
    result_type: TypePattern,
    range_guard: Guard,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=index.index_cast,
        guards=(*_typed_guards(_I64, result_type), range_guard),
        emit=(
            EmitRegisterSlice(
                source=ValueRef.operand("input"),
                result=ValueRef.result("result"),
            ),
        ),
    )


def _index_to_i64_rule() -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    arithmetic_shift = _descriptor("amd.xdna.aie2p.ashl.i32")
    return DescriptorRule(
        source_op=index.index_cast,
        descriptor=arithmetic_shift,
        guards=_typed_guards(_INDEX, _I64),
        emit=(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary("sign_shift")},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": -31},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=arithmetic_shift,
                operands={
                    "s0": ValueRef.operand("input"),
                    "s1": ValueRef.temporary("sign_shift"),
                },
                results={"d0": ValueRef.temporary("high_word")},
                result_types={"d0": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            EmitRegisterConcat(
                sources=(
                    ValueRef.operand("input"),
                    ValueRef.temporary("high_word"),
                ),
                result=ValueRef.result("result"),
            ),
        ),
    )


def _offset_to_i64_rule() -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    return DescriptorRule(
        source_op=index.index_cast,
        descriptor=constant,
        guards=_typed_guards(_OFFSET, _I64),
        emit=(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary("high_word")},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": 0},
                form=DescriptorEmitForm.CONST,
            ),
            EmitRegisterConcat(
                sources=(
                    ValueRef.operand("input"),
                    ValueRef.temporary("high_word"),
                ),
                result=ValueRef.result("result"),
            ),
        ),
    )


def _signed_payload_rules() -> tuple[DescriptorRule | ValueAliasRule, ...]:
    return (
        _extend_rule(
            _I8,
            _INDEX,
            "amd.xdna.aie2p.extend.signed.i8",
        ),
        _extend_rule(
            _I16,
            _INDEX,
            "amd.xdna.aie2p.extend.signed.i16",
        ),
        _alias_rule(_I32, _INDEX),
        _i64_to_address_rule(
            _INDEX,
            Guard.value_signed_bit_count("input", 32),
        ),
        _alias_rule(_INDEX, _I8),
        _alias_rule(_INDEX, _I16),
        _alias_rule(_INDEX, _I32),
        _index_to_i64_rule(),
    )


def _unsigned_payload_rules() -> tuple[DescriptorRule | ValueAliasRule, ...]:
    return (
        _extend_rule(
            _I8,
            _OFFSET,
            "amd.xdna.aie2p.extend.unsigned.i8",
        ),
        _extend_rule(
            _I16,
            _OFFSET,
            "amd.xdna.aie2p.extend.unsigned.i16",
        ),
        _alias_rule(_I32, _OFFSET),
        _i64_to_address_rule(
            _OFFSET,
            Guard.value_unsigned_bit_count("input", 32),
        ),
        _alias_rule(_OFFSET, _I8),
        _alias_rule(_OFFSET, _I16),
        _alias_rule(_OFFSET, _I32),
        _offset_to_i64_rule(),
    )


AIE2P_INDEX_CONVERSION_RULES: tuple[DescriptorRule | ValueAliasRule, ...] = (
    _alias_rule(_I1, _INDEX),
    _alias_rule(_I1, _OFFSET),
    _address_to_boolean_rule(_INDEX),
    _address_to_boolean_rule(_OFFSET),
    *_signed_payload_rules(),
    *_unsigned_payload_rules(),
    _alias_rule(_INDEX, _INDEX),
    _alias_rule(_OFFSET, _OFFSET),
    _alias_rule(
        _INDEX,
        _OFFSET,
        extra_guards=(Guard.value_i64_range("input", 0, _I32_MAX),),
    ),
    _alias_rule(
        _OFFSET,
        _INDEX,
        extra_guards=(Guard.value_i64_range("input", 0, _I32_MAX),),
    ),
)
