# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V index-carrier, numeric, and address-boundary rules."""

from __future__ import annotations

from loom.dialect.index import defs as index
from loom.dialect.scf import defs as scf
from loom.dsl import Op
from loom.error.spirv import ERR_SPIRV_026, ERR_SPIRV_027
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.scalar_alu import (
    INTEGER_COMPARE_PREDICATES,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    IntegerComparePredicate,
)
from loom.target.contracts import (
    AttrProject,
    DescriptorEmitForm,
    DescriptorResultType,
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
_I64 = Scalar("i64")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")

_S32_MIN = -(2**31)
_S32_MAX = (2**31) - 1
_U32_MAX = (2**32) - 1
_I64_MAX = (2**63) - 1


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET, key)


def _feature_guards(descriptor: Descriptor) -> tuple[Guard, ...]:
    return (
        (Guard.descriptor_available(descriptor),)
        if descriptor.feature_mask_words
        else ()
    )


def _typed_guards(
    fields: tuple[str, ...],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


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


def _numeric_range_diagnostic(
    field: str,
    minimum: int,
    maximum: int,
    constraint_key: str,
) -> GuardDiagnostic:
    return GuardDiagnostic(
        ref=target_diagnostic(
            ERR_SPIRV_027,
            string_param("field_name", field),
            value_type_param("value_type", field),
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
    result_type: ResultTypeBinding,
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
                result_types={"dst": DescriptorResultType()},
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
            rules.append(
                _cast_descriptor_rule(
                    payload_type,
                    _INDEX,
                    "spirv.op_copy_object.i32",
                )
            )
            rules.append(
                _cast_descriptor_rule(
                    _INDEX,
                    payload_type,
                    "spirv.op_copy_object.i32",
                )
            )
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
                        DescriptorResultType(),
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
                        result_types={"dst": DescriptorResultType()},
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
                    DescriptorResultType(),
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
                    result_types={"dst": DescriptorResultType()},
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


def _binary_numeric_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    extra_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            *extra_guards,
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_madd_rule() -> DescriptorRule:
    descriptor = _descriptor("spirv.op_imul_add.i32")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=descriptor,
        guards=_typed_guards(("a", "b", "c", "result"), _INDEX),
        emit=(
            _emit(
                descriptor,
                operands={
                    "a": ValueRef.operand("a"),
                    "b": ValueRef.operand("b"),
                    "c": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_scale_rule() -> DescriptorRule:
    descriptor = _descriptor("spirv.op_imul.offset64")
    return DescriptorRule(
        source_op=index.index_scale,
        descriptor=descriptor,
        guards=(
            Guard.value_type("index", _INDEX),
            Guard.value_type("stride", _OFFSET),
            Guard.value_type("result", _OFFSET),
            Guard.value_i64_range(
                "index",
                0,
                _S32_MAX,
                diagnostic=_numeric_range_diagnostic(
                    "index",
                    0,
                    _S32_MAX,
                    "spirv.index.scale_coordinate.non_negative_s32",
                ),
            ),
        ),
        emit=(
            _integer_view_emit(
                "spirv.op_bitcast.i32.u32",
                ValueRef.operand("index"),
                ValueRef.temporary("unsigned_index"),
                DescriptorResultType(),
            ),
            _emit(
                _descriptor("spirv.op_uconvert.u32.offset64"),
                operands={"input": ValueRef.temporary("unsigned_index")},
                results={"dst": ValueRef.temporary("wide_index")},
                result_types={"dst": _OFFSET},
            ),
            _emit(
                descriptor,
                operands={
                    "lhs": ValueRef.temporary("wide_index"),
                    "rhs": ValueRef.operand("stride"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_unsigned_binary_rule(
    source_op: Op,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _INDEX),
            Guard.value_i64_range(
                "lhs",
                0,
                _U32_MAX,
                diagnostic=_numeric_range_diagnostic(
                    "lhs",
                    0,
                    _U32_MAX,
                    "spirv.index.dividend.u32",
                ),
            ),
            Guard.value_i64_range(
                "rhs",
                1,
                _U32_MAX,
                diagnostic=_numeric_range_diagnostic(
                    "rhs",
                    1,
                    _U32_MAX,
                    "spirv.index.divisor.positive_u32",
                ),
            ),
        ),
        emit=(
            _integer_view_emit(
                "spirv.op_bitcast.i32.u32",
                ValueRef.operand("lhs"),
                ValueRef.temporary("unsigned_lhs"),
                DescriptorResultType(),
            ),
            _integer_view_emit(
                "spirv.op_bitcast.i32.u32",
                ValueRef.operand("rhs"),
                ValueRef.temporary("unsigned_rhs"),
                DescriptorResultType(),
            ),
            _emit(
                descriptor,
                operands={
                    "lhs": ValueRef.temporary("unsigned_lhs"),
                    "rhs": ValueRef.temporary("unsigned_rhs"),
                },
                results={"dst": ValueRef.temporary("unsigned_result")},
                result_types={"dst": DescriptorResultType()},
            ),
            _integer_view_emit(
                "spirv.op_bitcast.u32.i32",
                ValueRef.temporary("unsigned_result"),
                ValueRef.result("result"),
                _INDEX,
            ),
        ),
    )


def _index_minmax_rule(
    source_op: Op,
    compare_descriptor_key: str,
) -> DescriptorRule:
    compare = _descriptor(compare_descriptor_key)
    select = _descriptor("spirv.op_select.i32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=select,
        guards=_typed_guards(("lhs", "rhs", "result"), _INDEX),
        emit=(
            _emit(
                compare,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("condition")},
                result_types={"dst": _I1},
            ),
            _emit(
                select,
                operands={
                    "condition": ValueRef.temporary("condition"),
                    "true_value": ValueRef.operand("lhs"),
                    "false_value": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_shift_rule(source_op: Op, descriptor_key: str) -> DescriptorRule:
    return _binary_numeric_rule(
        source_op,
        _INDEX,
        descriptor_key,
        extra_guards=(
            Guard.value_i64_range(
                "rhs",
                0,
                31,
                diagnostic=_numeric_range_diagnostic(
                    "rhs",
                    0,
                    31,
                    "spirv.index.shift_amount.u5",
                ),
            ),
        ),
    )


def _index_rotate_rule(source_op: Op, *, rotate_left: bool) -> DescriptorRule:
    shift_left = _descriptor("spirv.op_shift_left_logical.i32")
    shift_right = _descriptor("spirv.op_shift_right_logical.i32")
    subtract = _descriptor("spirv.op_isub.i32")
    bitwise_and = _descriptor("spirv.op_bitwise_and.i32")
    bitwise_or = _descriptor("spirv.op_bitwise_or.i32")
    first_shift = shift_left if rotate_left else shift_right
    second_shift = shift_right if rotate_left else shift_left
    return DescriptorRule(
        source_op=source_op,
        descriptor=bitwise_or,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _INDEX),
            Guard.value_i64_range(
                "rhs",
                0,
                31,
                diagnostic=_numeric_range_diagnostic(
                    "rhs",
                    0,
                    31,
                    "spirv.index.rotate_amount.u5",
                ),
            ),
        ),
        emit=(
            _emit(
                first_shift,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("first_piece")},
                result_types={"dst": _INDEX},
            ),
            _address_constant_emit(_INDEX, 32, "bit_width"),
            _emit(
                subtract,
                operands={
                    "lhs": ValueRef.temporary("bit_width"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("complement")},
                result_types={"dst": _INDEX},
            ),
            _address_constant_emit(_INDEX, 31, "amount_mask"),
            _emit(
                bitwise_and,
                operands={
                    "lhs": ValueRef.temporary("complement"),
                    "rhs": ValueRef.temporary("amount_mask"),
                },
                results={"dst": ValueRef.temporary("masked_complement")},
                result_types={"dst": _INDEX},
            ),
            _emit(
                second_shift,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.temporary("masked_complement"),
                },
                results={"dst": ValueRef.temporary("second_piece")},
                result_types={"dst": _INDEX},
            ),
            _emit(
                bitwise_or,
                operands={
                    "lhs": ValueRef.temporary("first_piece"),
                    "rhs": ValueRef.temporary("second_piece"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_ctpop_rule() -> DescriptorRule:
    descriptor = _descriptor("spirv.op_bit_count.i32")
    return DescriptorRule(
        source_op=index.index_ctpopi,
        descriptor=descriptor,
        guards=_typed_guards(("input", "result"), _INDEX),
        emit=(
            _emit(
                descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_ctlz_rule() -> DescriptorRule:
    shift_right = _descriptor("spirv.op_shift_right_logical.i32")
    bitwise_or = _descriptor("spirv.op_bitwise_or.i32")
    bit_count = _descriptor("spirv.op_bit_count.i32")
    subtract = _descriptor("spirv.op_isub.i32")
    emits: list[EmitDescriptorOp] = []
    smear = ValueRef.operand("input")
    for step, amount in enumerate((1, 2, 4, 8, 16), start=1):
        amount_name = f"shift_amount_{amount}"
        shifted_name = f"shifted_{step}"
        smear_name = f"smear_{step}"
        emits.append(_address_constant_emit(_INDEX, amount, amount_name))
        emits.append(
            _emit(
                shift_right,
                operands={
                    "lhs": smear,
                    "rhs": ValueRef.temporary(amount_name),
                },
                results={"dst": ValueRef.temporary(shifted_name)},
                result_types={"dst": _INDEX},
            )
        )
        emits.append(
            _emit(
                bitwise_or,
                operands={
                    "lhs": smear,
                    "rhs": ValueRef.temporary(shifted_name),
                },
                results={"dst": ValueRef.temporary(smear_name)},
                result_types={"dst": _INDEX},
            )
        )
        smear = ValueRef.temporary(smear_name)
    emits.extend(
        (
            _emit(
                bit_count,
                operands={"input": smear},
                results={"dst": ValueRef.temporary("population")},
                result_types={"dst": _INDEX},
            ),
            _address_constant_emit(_INDEX, 32, "bit_width"),
            _emit(
                subtract,
                operands={
                    "lhs": ValueRef.temporary("bit_width"),
                    "rhs": ValueRef.temporary("population"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        )
    )
    return DescriptorRule(
        source_op=index.index_ctlzi,
        descriptor=subtract,
        guards=_typed_guards(("input", "result"), _INDEX),
        emit=tuple(emits),
    )


def _index_cttz_rule() -> DescriptorRule:
    subtract = _descriptor("spirv.op_isub.i32")
    bitwise_and = _descriptor("spirv.op_bitwise_and.i32")
    bit_count = _descriptor("spirv.op_bit_count.i32")
    return DescriptorRule(
        source_op=index.index_cttzi,
        descriptor=bit_count,
        guards=_typed_guards(("input", "result"), _INDEX),
        emit=(
            _address_constant_emit(_INDEX, 0, "zero"),
            _emit(
                subtract,
                operands={
                    "lhs": ValueRef.temporary("zero"),
                    "rhs": ValueRef.operand("input"),
                },
                results={"dst": ValueRef.temporary("negated")},
                result_types={"dst": _INDEX},
            ),
            _emit(
                bitwise_and,
                operands={
                    "lhs": ValueRef.operand("input"),
                    "rhs": ValueRef.temporary("negated"),
                },
                results={"dst": ValueRef.temporary("low_bit")},
                result_types={"dst": _INDEX},
            ),
            _address_constant_emit(_INDEX, 1, "one"),
            _emit(
                subtract,
                operands={
                    "lhs": ValueRef.temporary("low_bit"),
                    "rhs": ValueRef.temporary("one"),
                },
                results={"dst": ValueRef.temporary("lower_mask")},
                result_types={"dst": _INDEX},
            ),
            _emit(
                bit_count,
                operands={"input": ValueRef.temporary("lower_mask")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _direct_compare_rule(
    type_pattern: TypePattern,
    predicate: IntegerComparePredicate,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=index.index_cmp,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate.source_predicate),
            *_typed_guards(("lhs", "rhs"), type_pattern),
            Guard.value_type("result", _I1),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_compare_rule(predicate: IntegerComparePredicate) -> DescriptorRule:
    if predicate.source_predicate in ("eq", "ne", "slt", "sle", "sgt", "sge"):
        return _direct_compare_rule(
            _INDEX,
            predicate,
            f"spirv.op_{predicate.descriptor_suffix}.i32",
        )

    descriptor = _descriptor(f"spirv.op_{predicate.descriptor_suffix}.u32")
    return DescriptorRule(
        source_op=index.index_cmp,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate.source_predicate),
            *_typed_guards(("lhs", "rhs"), _INDEX),
            Guard.value_type("result", _I1),
        ),
        emit=(
            _integer_view_emit(
                "spirv.op_bitcast.i32.u32",
                ValueRef.operand("lhs"),
                ValueRef.temporary("unsigned_lhs"),
                DescriptorResultType(),
            ),
            _integer_view_emit(
                "spirv.op_bitcast.i32.u32",
                ValueRef.operand("rhs"),
                ValueRef.temporary("unsigned_rhs"),
                DescriptorResultType(),
            ),
            _emit(
                descriptor,
                operands={
                    "lhs": ValueRef.temporary("unsigned_lhs"),
                    "rhs": ValueRef.temporary("unsigned_rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _offset_compare_rule(predicate: IntegerComparePredicate) -> DescriptorRule:
    if predicate.source_predicate in ("eq", "ne", "ult", "ule", "ugt", "uge"):
        return _direct_compare_rule(
            _OFFSET,
            predicate,
            f"spirv.op_{predicate.descriptor_suffix}.offset64",
        )

    descriptor = _descriptor(f"spirv.op_{predicate.descriptor_suffix}.i64")
    return DescriptorRule(
        source_op=index.index_cmp,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate.source_predicate),
            *_typed_guards(("lhs", "rhs"), _OFFSET),
            Guard.value_type("result", _I1),
            *_feature_guards(descriptor),
        ),
        emit=(
            _integer_view_emit(
                "spirv.op_bitcast.offset64.i64",
                ValueRef.operand("lhs"),
                ValueRef.temporary("signed_lhs"),
                _I64,
            ),
            _integer_view_emit(
                "spirv.op_bitcast.offset64.i64",
                ValueRef.operand("rhs"),
                ValueRef.temporary("signed_rhs"),
                _I64,
            ),
            _emit(
                descriptor,
                operands={
                    "lhs": ValueRef.temporary("signed_lhs"),
                    "rhs": ValueRef.temporary("signed_rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _address_select_rule(
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", _I1),
            *_typed_guards(("true_value", "false_value", "result"), type_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                operands={
                    "condition": ValueRef.operand("condition"),
                    "true_value": ValueRef.operand("true_value"),
                    "false_value": ValueRef.operand("false_value"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


SPIRV_INDEX_NUMERIC_RULES: tuple[DescriptorRule, ...] = (
    _binary_numeric_rule(index.index_add, _INDEX, "spirv.op_iadd.i32"),
    _binary_numeric_rule(index.index_add, _OFFSET, "spirv.op_iadd.offset64"),
    _binary_numeric_rule(index.index_sub, _INDEX, "spirv.op_isub.i32"),
    _binary_numeric_rule(index.index_sub, _OFFSET, "spirv.op_isub.offset64"),
    _binary_numeric_rule(index.index_mul, _INDEX, "spirv.op_imul.i32"),
    _index_scale_rule(),
    _index_unsigned_binary_rule(index.index_div, "spirv.op_udiv.u32"),
    _index_unsigned_binary_rule(index.index_rem, "spirv.op_umod.u32"),
    _index_minmax_rule(index.index_min, "spirv.op_s_less_than.i32"),
    _index_minmax_rule(index.index_max, "spirv.op_s_greater_than.i32"),
    _index_madd_rule(),
    _binary_numeric_rule(index.index_andi, _INDEX, "spirv.op_bitwise_and.i32"),
    _binary_numeric_rule(index.index_ori, _INDEX, "spirv.op_bitwise_or.i32"),
    _binary_numeric_rule(index.index_xori, _INDEX, "spirv.op_bitwise_xor.i32"),
    _index_shift_rule(index.index_shli, "spirv.op_shift_left_logical.i32"),
    _index_shift_rule(index.index_shrsi, "spirv.op_shift_right_arithmetic.i32"),
    _index_shift_rule(index.index_shrui, "spirv.op_shift_right_logical.i32"),
    _index_rotate_rule(index.index_rotli, rotate_left=True),
    _index_rotate_rule(index.index_rotri, rotate_left=False),
    _index_ctlz_rule(),
    _index_cttz_rule(),
    _index_ctpop_rule(),
    *(_index_compare_rule(predicate) for predicate in INTEGER_COMPARE_PREDICATES),
    *(_offset_compare_rule(predicate) for predicate in INTEGER_COMPARE_PREDICATES),
    _address_select_rule(_INDEX, "spirv.op_select.i32"),
    _address_select_rule(_OFFSET, "spirv.op_select.offset64"),
)
