# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V index-carrier and address-boundary conversion rules."""

from __future__ import annotations

from loom.dialect.index import defs as index
from loom.error.spirv import ERR_SPIRV_026
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.scalar_alu import INTEGER_SCALAR_ALU_TYPE_PAIRS
from loom.target.contracts import (
    AttrProject,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    OrdinalValueAliasRule,
    ResultTypeBinding,
    Scalar,
    TypePattern,
    ValueAliasRule,
    ValueRef,
    descriptor_by_key,
    i64_param,
    string_param,
    target_diagnostic,
    value_type_param,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")

_S32_MIN = -(2**31)
_S32_MAX = (2**31) - 1
_I64_MAX = (2**63) - 1


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET, key)


def _feature_guards(descriptor: Descriptor) -> tuple[Guard, ...]:
    return (
        (Guard.descriptor_available(descriptor),)
        if descriptor.feature_mask_words
        else ()
    )


def _emit(
    descriptor: Descriptor,
    *,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, ResultTypeBinding] | None = None,
    immediates: dict[str, AttrProject | int] | None = None,
    form: DescriptorEmitForm = DescriptorEmitForm.OP,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        form=form,
    )


def _conversion_range_diagnostic(
    source_type_field: str,
    result_type_field: str,
    minimum: int,
    maximum: int,
    constraint_key: str,
) -> GuardDiagnostic:
    return GuardDiagnostic(
        ref=target_diagnostic(
            ERR_SPIRV_026,
            value_type_param("source_type", source_type_field),
            value_type_param("result_type", result_type_field),
            i64_param("required_range_lo", minimum),
            i64_param("required_range_hi", maximum),
            string_param("constraint_key", constraint_key),
        )
    )


def _index_constant_rule() -> DescriptorRule:
    descriptor = _descriptor("spirv.op_constant.i32")
    return DescriptorRule(
        source_op=index.index_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", _INDEX),
            Guard.i64_range("value", _S32_MIN, _S32_MAX),
        ),
        emit=(
            _emit(
                descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i32_value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _offset_constant_rule() -> DescriptorRule:
    descriptor = _descriptor("spirv.op_constant.offset64")
    return DescriptorRule(
        source_op=index.index_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", _OFFSET),
        ),
        emit=(
            _emit(
                descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"offset64_value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _cast_alias_rule(
    source_type: TypePattern,
    result_type: TypePattern,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=index.index_cast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
    )


def _cast_descriptor_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    *,
    range_guard: Guard | None = None,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=index.index_cast,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
            *((range_guard,) if range_guard is not None else ()),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _integer_view_emit(
    descriptor_key: str,
    input_ref: ValueRef,
    output_ref: ValueRef,
    result_type: TypePattern,
) -> EmitDescriptorOp:
    return _emit(
        _descriptor(descriptor_key),
        operands={"input": input_ref},
        results={"dst": output_ref},
        result_types={"dst": result_type},
    )


def _address_constant_emit(
    address_type: TypePattern,
    value: int,
    temporary: str,
) -> EmitDescriptorOp:
    if address_type == _INDEX:
        descriptor = _descriptor("spirv.op_constant.i32")
        immediate = "i32_value"
    else:
        descriptor = _descriptor("spirv.op_constant.offset64")
        immediate = "offset64_value"
    return _emit(
        descriptor,
        results={"dst": ValueRef.temporary(temporary)},
        result_types={"dst": address_type},
        immediates={immediate: value},
        form=DescriptorEmitForm.CONST,
    )


def _boolean_to_address_cast_rule(
    result_type: TypePattern,
    select_descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(select_descriptor_key)
    return DescriptorRule(
        source_op=index.index_cast,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", _I1),
            Guard.value_type("result", result_type),
            *_feature_guards(descriptor),
        ),
        emit=(
            _address_constant_emit(result_type, 1, "one"),
            _address_constant_emit(result_type, 0, "zero"),
            _emit(
                descriptor,
                operands={
                    "condition": ValueRef.operand("input"),
                    "true_value": ValueRef.temporary("one"),
                    "false_value": ValueRef.temporary("zero"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _address_to_boolean_cast_rule(
    source_type: TypePattern,
) -> DescriptorRule:
    compare = _descriptor("spirv.op_i_not_equal.i32")
    bitwise_and = _descriptor("spirv.op_bitwise_and.i32")
    prefix: tuple[EmitDescriptorOp, ...] = ()
    input_ref = ValueRef.operand("input")
    if source_type == _OFFSET:
        truncate = _descriptor("spirv.op_uconvert.offset64.u32")
        prefix = (
            _emit(
                truncate,
                operands={"input": input_ref},
                results={"dst": ValueRef.temporary("unsigned_low_bits")},
                result_types={"dst": _I32},
            ),
            _integer_view_emit(
                "spirv.op_bitcast.u32.i32",
                ValueRef.temporary("unsigned_low_bits"),
                ValueRef.temporary("low_bits"),
                _I32,
            ),
        )
        input_ref = ValueRef.temporary("low_bits")
    return DescriptorRule(
        source_op=index.index_cast,
        descriptor=compare,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", _I1),
        ),
        emit=(
            *prefix,
            _address_constant_emit(_INDEX, 1, "one"),
            _emit(
                bitwise_and,
                operands={
                    "lhs": input_ref,
                    "rhs": ValueRef.temporary("one"),
                },
                results={"dst": ValueRef.temporary("low_bit")},
                result_types={"dst": _I32},
            ),
            _address_constant_emit(_INDEX, 0, "zero"),
            _emit(
                compare,
                operands={
                    "lhs": ValueRef.temporary("low_bit"),
                    "rhs": ValueRef.temporary("zero"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _boolean_address_cast_rules() -> tuple[DescriptorRule, ...]:
    return (
        _boolean_to_address_cast_rule(_INDEX, "spirv.op_select.i32"),
        _boolean_to_address_cast_rule(_OFFSET, "spirv.op_select.offset64"),
        _address_to_boolean_cast_rule(_INDEX),
        _address_to_boolean_cast_rule(_OFFSET),
    )


def _signed_payload_index_cast_rules() -> tuple[DescriptorRule | ValueAliasRule, ...]:
    rules: list[DescriptorRule | ValueAliasRule] = []
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        payload_type = Scalar(scalar_pair.source_type)
        suffix = scalar_pair.signed.suffix
        if scalar_pair.bit_width == 32:
            rules.append(_cast_alias_rule(payload_type, _INDEX))
            rules.append(_cast_alias_rule(_INDEX, payload_type))
            continue

        range_guard = None
        if scalar_pair.bit_width > 32:
            range_guard = Guard.value_signed_bit_count(
                "input",
                32,
                diagnostic=_conversion_range_diagnostic(
                    "input",
                    "result",
                    _S32_MIN,
                    _S32_MAX,
                    "index_cast.s32_exact",
                ),
            )
        rules.append(
            _cast_descriptor_rule(
                payload_type,
                _INDEX,
                f"spirv.op_s_convert.{suffix}.i32",
                range_guard=range_guard,
            )
        )
        rules.append(
            _cast_descriptor_rule(
                _INDEX,
                payload_type,
                f"spirv.op_s_convert.i32.{suffix}",
            )
        )
    return tuple(rules)


def _payload_offset_cast_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        payload_type = Scalar(scalar_pair.source_type)
        if scalar_pair.bit_width == 64:
            rules.append(
                _cast_descriptor_rule(
                    payload_type,
                    _OFFSET,
                    "spirv.op_bitcast.i64.offset64",
                    range_guard=Guard.value_i64_range(
                        "input",
                        0,
                        _I64_MAX,
                        diagnostic=_conversion_range_diagnostic(
                            "input",
                            "result",
                            0,
                            _I64_MAX,
                            "index_cast.offset_non_negative",
                        ),
                    ),
                )
            )
            rules.append(
                _cast_descriptor_rule(
                    _OFFSET,
                    payload_type,
                    "spirv.op_bitcast.offset64.i64",
                )
            )
            continue

        signed_suffix = scalar_pair.signed.suffix
        unsigned_suffix = scalar_pair.unsigned.suffix
        to_offset = _descriptor(f"spirv.op_uconvert.{unsigned_suffix}.offset64")
        rules.append(
            DescriptorRule(
                source_op=index.index_cast,
                descriptor=to_offset,
                guards=(
                    Guard.value_type("input", payload_type),
                    Guard.value_type("result", _OFFSET),
                    *_feature_guards(to_offset),
                ),
                emit=(
                    _integer_view_emit(
                        f"spirv.op_bitcast.{signed_suffix}.{unsigned_suffix}",
                        ValueRef.operand("input"),
                        ValueRef.temporary("unsigned_input"),
                        payload_type,
                    ),
                    _emit(
                        to_offset,
                        operands={"input": ValueRef.temporary("unsigned_input")},
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        )

        from_offset = _descriptor(f"spirv.op_uconvert.offset64.{unsigned_suffix}")
        rules.append(
            DescriptorRule(
                source_op=index.index_cast,
                descriptor=from_offset,
                guards=(
                    Guard.value_type("input", _OFFSET),
                    Guard.value_type("result", payload_type),
                    *_feature_guards(from_offset),
                ),
                emit=(
                    _emit(
                        from_offset,
                        operands={"input": ValueRef.operand("input")},
                        results={"dst": ValueRef.temporary("unsigned_result")},
                        result_types={"dst": payload_type},
                    ),
                    _integer_view_emit(
                        f"spirv.op_bitcast.{unsigned_suffix}.{signed_suffix}",
                        ValueRef.temporary("unsigned_result"),
                        ValueRef.result("result"),
                        payload_type,
                    ),
                ),
            )
        )
    return tuple(rules)


def _address_cast_rules() -> tuple[DescriptorRule | ValueAliasRule, ...]:
    to_offset = _descriptor("spirv.op_uconvert.u32.offset64")
    from_offset = _descriptor("spirv.op_uconvert.offset64.u32")
    return (
        _cast_alias_rule(_INDEX, _INDEX),
        _cast_alias_rule(_OFFSET, _OFFSET),
        DescriptorRule(
            source_op=index.index_cast,
            descriptor=to_offset,
            guards=(
                Guard.value_type("input", _INDEX),
                Guard.value_type("result", _OFFSET),
                Guard.value_i64_range(
                    "input",
                    0,
                    _S32_MAX,
                    diagnostic=_conversion_range_diagnostic(
                        "input",
                        "result",
                        0,
                        _S32_MAX,
                        "index_cast.offset_non_negative",
                    ),
                ),
            ),
            emit=(
                _integer_view_emit(
                    "spirv.op_bitcast.i32.u32",
                    ValueRef.operand("input"),
                    ValueRef.temporary("unsigned_input"),
                    _INDEX,
                ),
                _emit(
                    to_offset,
                    operands={"input": ValueRef.temporary("unsigned_input")},
                    results={"dst": ValueRef.result("result")},
                ),
            ),
        ),
        DescriptorRule(
            source_op=index.index_cast,
            descriptor=from_offset,
            guards=(
                Guard.value_type("input", _OFFSET),
                Guard.value_type("result", _INDEX),
                Guard.value_i64_range(
                    "input",
                    0,
                    _S32_MAX,
                    diagnostic=_conversion_range_diagnostic(
                        "input",
                        "result",
                        0,
                        _S32_MAX,
                        "index_cast.s32_exact",
                    ),
                ),
            ),
            emit=(
                _emit(
                    from_offset,
                    operands={"input": ValueRef.operand("input")},
                    results={"dst": ValueRef.temporary("unsigned_result")},
                    result_types={"dst": _INDEX},
                ),
                _integer_view_emit(
                    "spirv.op_bitcast.u32.i32",
                    ValueRef.temporary("unsigned_result"),
                    ValueRef.result("result"),
                    _INDEX,
                ),
            ),
        ),
    )


SPIRV_INDEX_CONVERSION_RULES: tuple[
    DescriptorRule | OrdinalValueAliasRule | ValueAliasRule, ...
] = (
    _index_constant_rule(),
    _offset_constant_rule(),
    OrdinalValueAliasRule(
        source_op=index.index_assume,
        source=ValueRef.operand("values"),
        result=ValueRef.result("results"),
    ),
    *_boolean_address_cast_rules(),
    *_signed_payload_index_cast_rules(),
    *_payload_offset_cast_rules(),
    *_address_cast_rules(),
)
