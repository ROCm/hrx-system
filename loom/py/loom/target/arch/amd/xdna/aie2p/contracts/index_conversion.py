# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P address-boundary conversion rules."""

from __future__ import annotations

from loom.dialect.index import defs as index
from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.core import (
    predicate_boolean_bytes_emits,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_conversion import (
    INTEGER_TRUNCATION_RULE_SHAPES,
    INTEGER_WIDEN_RULE_SHAPES,
    IntegerTruncationRuleShape,
    IntegerWidenRuleShape,
    integer_truncation_emits,
    integer_truncation_rule,
    integer_widen_rule,
)
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
    Vector,
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
    *,
    extra_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    return DescriptorRule(
        source_op=index.index_cast,
        guards=(*_typed_guards(_I64, result_type), *extra_guards),
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
        _i64_to_address_rule(_INDEX),
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
            extra_guards=(Guard.value_unsigned_bit_count("input", 32),),
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
    _alias_rule(_OFFSET, _INDEX),
)


def _vector_type(
    element: str,
    minimum_lane_count: int,
    maximum_lane_count: int,
) -> TypePattern:
    return Vector(
        element,
        minimum_lanes=minimum_lane_count,
        maximum_lanes=maximum_lane_count,
    )


def _whole_address_vector_type(element: str) -> TypePattern:
    return Vector(
        element,
        minimum_static_elements=1,
        maximum_static_elements=64,
    )


def _vector_report_key(strategy: str) -> str:
    return f"native_index_cast_{strategy}"


def _vector_alias_rule(
    input_element: str,
    result_element: str,
    *,
    extra_guards: tuple[Guard, ...] = (),
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_index_cast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", _whole_address_vector_type(input_element)),
            Guard.value_type("result", _whole_address_vector_type(result_element)),
            *extra_guards,
        ),
    )


def _vector_widen_rules(
    input_element: str,
    result_element: str,
    signedness: str,
    physical_input_element: str,
    physical_result_element: str,
    report_strategy: str,
) -> tuple[DescriptorRule, ...]:
    return tuple(
        integer_widen_rule(
            vector.vector_index_cast,
            signedness,
            rule_shape,
            input_type=_vector_type(
                input_element,
                rule_shape.minimum_lane_count,
                rule_shape.maximum_lane_count,
            ),
            result_type=_vector_type(
                result_element,
                rule_shape.minimum_lane_count,
                rule_shape.maximum_lane_count,
            ),
            report_key=_vector_report_key(report_strategy),
        )
        for rule_shape in INTEGER_WIDEN_RULE_SHAPES
        if (
            rule_shape.instruction.input_element == physical_input_element
            and rule_shape.instruction.result_element == physical_result_element
        )
    )


def _predicate_to_address_rule(
    result_element: str,
    signedness: str,
    rule_shape: IntegerWidenRuleShape,
) -> DescriptorRule:
    boolean_bytes = ValueRef.temporary("boolean_bytes")
    materialize_emits, _ = predicate_boolean_bytes_emits(
        ValueRef.operand("input"),
        boolean_bytes,
        temporary_prefix="index_cast",
    )
    return integer_widen_rule(
        vector.vector_index_cast,
        signedness,
        rule_shape,
        input_type=_vector_type(
            "i1",
            rule_shape.minimum_lane_count,
            rule_shape.maximum_lane_count,
        ),
        result_type=_vector_type(
            result_element,
            rule_shape.minimum_lane_count,
            rule_shape.maximum_lane_count,
        ),
        source=boolean_bytes,
        prefix_emits=materialize_emits,
        report_key=_vector_report_key("predicate_to_address"),
    )


def _vector_truncation_rules(
    input_element: str,
    result_element: str,
    physical_input_element: str,
    physical_result_element: str,
    *,
    extra_guards: tuple[Guard, ...] = (),
    report_strategy: str,
) -> tuple[DescriptorRule, ...]:
    return tuple(
        integer_truncation_rule(
            rule_shape,
            source_op=vector.vector_index_cast,
            input_type=_vector_type(
                input_element,
                rule_shape.minimum_lane_count,
                rule_shape.maximum_lane_count,
            ),
            result_type=_vector_type(
                result_element,
                rule_shape.minimum_lane_count,
                rule_shape.maximum_lane_count,
            ),
            extra_guards=extra_guards,
            report_key=_vector_report_key(report_strategy),
        )
        for rule_shape in INTEGER_TRUNCATION_RULE_SHAPES
        if (
            rule_shape.instruction.input_element == physical_input_element
            and rule_shape.instruction.result_element == physical_result_element
        )
    )


def _address_to_predicate_rule(
    input_element: str,
    rule_shape: IntegerTruncationRuleShape,
) -> DescriptorRule:
    low_bytes = ValueRef.temporary("low_bytes")
    one = ValueRef.temporary("one")
    ones = ValueRef.temporary("ones")
    zeros = ValueRef.temporary("zeros")
    low_bits = ValueRef.temporary("low_bits")
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    splat = _descriptor("amd.xdna.aie2p.splat.i8x64")
    subtract = _descriptor("amd.xdna.aie2p.sub.i8x64")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.bits512")
    compare = _descriptor("amd.xdna.aie2p.cmp.lt.unsigned.i8x64")
    return DescriptorRule(
        source_op=vector.vector_index_cast,
        descriptor=compare,
        guards=(
            Guard.value_type(
                "input",
                _vector_type(
                    input_element,
                    rule_shape.minimum_lane_count,
                    rule_shape.maximum_lane_count,
                ),
            ),
            Guard.value_type(
                "result",
                _vector_type(
                    "i1",
                    rule_shape.minimum_lane_count,
                    rule_shape.maximum_lane_count,
                ),
            ),
        ),
        emit=(
            *integer_truncation_emits(
                rule_shape,
                ValueRef.operand("input"),
                low_bytes,
                result_type=DescriptorResultType(),
            ),
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": one},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": 1},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=splat,
                operands={"src": one},
                results={"dst": ones},
                result_types={"dst": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=subtract,
                operands={"s1": ones, "s2": ones},
                results={"d": zeros},
                result_types={"d": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=bitwise_and,
                operands={"s1": low_bytes, "s2": ones},
                results={"d": low_bits},
                result_types={"d": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=compare,
                operands={"s1": zeros, "s2": low_bits},
                results={"cmp": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        report_key=_vector_report_key("address_to_predicate"),
    )


_I8_TO_I32_WIDEN_SHAPES = tuple(
    rule_shape
    for rule_shape in INTEGER_WIDEN_RULE_SHAPES
    if rule_shape.instruction.input_element == "i8"
    and rule_shape.instruction.result_element == "i32"
)

_I32_TO_I8_TRUNCATION_SHAPES = tuple(
    rule_shape
    for rule_shape in INTEGER_TRUNCATION_RULE_SHAPES
    if rule_shape.instruction.input_element == "i32"
    and rule_shape.instruction.result_element == "i8"
)


AIE2P_VECTOR_INDEX_CONVERSION_RULES: tuple[DescriptorRule | ValueAliasRule, ...] = (
    *(
        _predicate_to_address_rule(result_element, signedness, rule_shape)
        for result_element, signedness in (
            ("index", "signed"),
            ("offset", "unsigned"),
        )
        for rule_shape in _I8_TO_I32_WIDEN_SHAPES
    ),
    *(
        rule
        for input_element in ("i8", "i16")
        for result_element, signedness in (
            ("index", "signed"),
            ("offset", "unsigned"),
        )
        for rule in _vector_widen_rules(
            input_element,
            result_element,
            signedness,
            input_element,
            "i32",
            "narrow_to_address",
        )
    ),
    _vector_alias_rule("i32", "index"),
    _vector_alias_rule("i32", "offset"),
    *(
        rule
        for result_element, extra_guards in (
            ("index", ()),
            ("offset", (Guard.value_unsigned_bit_count("input", 32),)),
        )
        for rule in _vector_truncation_rules(
            "i64",
            result_element,
            "i64",
            "i32",
            extra_guards=extra_guards,
            report_strategy="i64_to_address",
        )
    ),
    *(
        _address_to_predicate_rule(input_element, rule_shape)
        for input_element in ("index", "offset")
        for rule_shape in _I32_TO_I8_TRUNCATION_SHAPES
    ),
    *(
        rule
        for input_element in ("index", "offset")
        for result_element in ("i8", "i16")
        for rule in _vector_truncation_rules(
            input_element,
            result_element,
            "i32",
            result_element,
            report_strategy="address_to_narrow",
        )
    ),
    _vector_alias_rule("index", "i32"),
    _vector_alias_rule("offset", "i32"),
    *(
        rule
        for input_element, signedness in (
            ("index", "signed"),
            ("offset", "unsigned"),
        )
        for rule in _vector_widen_rules(
            input_element,
            "i64",
            signedness,
            "i32",
            "i64",
            "address_to_i64",
        )
    ),
    _vector_alias_rule("index", "index"),
    _vector_alias_rule("offset", "offset"),
    _vector_alias_rule(
        "index",
        "offset",
        extra_guards=(Guard.value_i64_range("input", 0, _I32_MAX),),
    ),
    _vector_alias_rule("offset", "index"),
)
