# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V index-carrier, numeric, and address-boundary rules."""

from __future__ import annotations

from dataclasses import replace

from loom.dialect.index import defs as index
from loom.dialect.scf import defs as scf
from loom.dsl import Op
from loom.error.spirv import ERR_SPIRV_026, ERR_SPIRV_027
from loom.target.arch.spirv.contracts.integer_bits import bit_count_rule, rotation_rule
from loom.target.arch.spirv.contracts.predicate import integer_to_boolean_rule
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.scalar_alu import (
    INTEGER_COMPARE_PREDICATES,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    ComparePredicate,
)
from loom.target.contracts import (
    AttrProject,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    GuardKind,
    OrdinalValueAliasRule,
    ResultTypeBinding,
    Scalar,
    SourceValueKind,
    TypePattern,
    ValueAliasRule,
    ValueMaterializer,
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

# Coordinate representation IDs owned by lower/index.h.
_INDEX_REPRESENTATIONS = {32: 1, 64: 2}

NATIVE_INDEX_MATERIALIZER = ValueMaterializer(
    name="spirv.native_index",
    can_materialize="loom_spirv_can_materialize_native_index",
    result_type="loom_spirv_materialized_index_operand_type",
    prepare="loom_spirv_prepare_native_index",
    materialize="loom_spirv_emit_index_operand",
    header="loom/target/arch/spirv/lower/index_operands.h",
)

NATIVE_INDEX_ARITHMETIC_MATERIALIZER = replace(
    NATIVE_INDEX_MATERIALIZER,
    name="spirv.native_index_arithmetic",
    can_materialize="loom_spirv_can_materialize_native_index_arithmetic",
)
WIDE_INDEX_ARITHMETIC_MATERIALIZER = replace(
    NATIVE_INDEX_MATERIALIZER,
    name="spirv.wide_index_arithmetic",
    can_materialize="loom_spirv_can_materialize_wide_index_arithmetic",
    prepare="loom_spirv_prepare_wide_index",
)


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


def _index_constant_rule(width: int = 32) -> DescriptorRule:
    descriptor = _descriptor(f"spirv.op_constant.i{width}")
    return DescriptorRule(
        source_op=index.index_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", _INDEX),
            Guard.i64_range("value", _S32_MIN, (1 << (width - 1)) - 1),
            Guard.low_value_representation("result", _INDEX_REPRESENTATIONS[width]),
            *_feature_guards(descriptor),
        ),
        emit=(
            _emit(
                descriptor,
                results={"dst": ValueRef.result("result")},
                result_types={"dst": _INDEX if width == 32 else _I64},
                immediates={f"i{width}_value": AttrProject.direct("value")},
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
    if address_type in (_INDEX, _I64):
        width = 32 if address_type == _INDEX else 64
        descriptor = _descriptor(f"spirv.op_constant.i{width}")
        immediate = f"i{width}_value"
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
    carrier_type: TypePattern | None = None,
) -> DescriptorRule:
    descriptor = _descriptor(select_descriptor_key)
    carrier_type = carrier_type or result_type
    return DescriptorRule(
        source_op=index.index_cast,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", _I1),
            Guard.value_type("result", result_type),
            *_feature_guards(descriptor),
        ),
        emit=(
            _address_constant_emit(carrier_type, 1, "one"),
            _address_constant_emit(carrier_type, 0, "zero"),
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


def _signed_payload_index_cast_rules(
    width: int,
) -> tuple[DescriptorRule | ValueAliasRule, ...]:
    rules: list[DescriptorRule | ValueAliasRule] = []
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        payload_type = Scalar(scalar_pair.source_type)
        suffix = scalar_pair.signed.suffix
        if scalar_pair.bit_width == width:
            if width == 64:
                rules.extend(
                    (
                        replace(
                            _cast_alias_rule(payload_type, _INDEX),
                            guards=(
                                Guard.value_type("input", payload_type),
                                Guard.value_type("result", _INDEX),
                                Guard.value_i64_range("input", _S32_MIN, (1 << 63) - 1),
                            ),
                        ),
                        _cast_alias_rule(_INDEX, payload_type),
                    )
                )
                continue
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
        if scalar_pair.bit_width > width:
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
                f"spirv.op_s_convert.{suffix}.i{width}",
                range_guard=range_guard,
            )
        )
        rules.append(
            _cast_descriptor_rule(
                _INDEX,
                payload_type,
                f"spirv.op_s_convert.i{width}.{suffix}",
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


def _address_cast_rules(width: int) -> tuple[DescriptorRule | ValueAliasRule, ...]:
    if width == 64:
        return (
            _cast_alias_rule(_INDEX, _INDEX),
            _cast_descriptor_rule(
                _INDEX,
                _OFFSET,
                "spirv.op_bitcast.i64.offset64",
                range_guard=Guard.value_i64_range("input", 0, (1 << 63) - 1),
            ),
            _cast_descriptor_rule(
                _OFFSET,
                _INDEX,
                "spirv.op_bitcast.offset64.i64",
            ),
        )
    to_offset = _descriptor("spirv.op_uconvert.u32.offset64")
    from_offset = _descriptor("spirv.op_uconvert.offset64.u32")
    return (
        _cast_alias_rule(_INDEX, _INDEX),
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


def _index_cast_carrier_rule(
    rule: DescriptorRule | ValueAliasRule,
    width: int,
) -> DescriptorRule | ValueAliasRule:
    fields = {
        guard.field
        for guard in rule.guards
        if guard.kind == GuardKind.VALUE_TYPE and guard.type_pattern == _INDEX
    }
    guards = (
        *rule.guards,
        *(
            Guard.low_value_representation(field, _INDEX_REPRESENTATIONS[width])
            for field in sorted(fields)
        ),
    )
    if isinstance(rule, ValueAliasRule):
        return replace(rule, guards=guards)
    return replace(
        rule,
        guards=guards,
        emit=tuple(
            replace(
                step,
                result_types={
                    **(step.result_types or {}),
                    **{
                        name: _INDEX if width == 32 else _I64
                        for name, value in step.results.items()
                        if value.kind == SourceValueKind.RESULT
                        and value.field in fields
                    },
                }
                or None,
            )
            for step in rule.emit
        ),
    )


def _index_cast_rules(width: int) -> tuple[DescriptorRule | ValueAliasRule, ...]:
    to_boolean = integer_to_boolean_rule(
        index.index_cast, "index" if width == 32 else "i64"
    )
    if width == 64:
        to_boolean = replace(
            to_boolean,
            guards=tuple(
                replace(guard, type_pattern=_INDEX)
                if guard.kind == GuardKind.VALUE_TYPE and guard.field == "input"
                else guard
                for guard in to_boolean.guards
            ),
        )
    return tuple(
        _index_cast_carrier_rule(rule, width)
        for rule in (
            _boolean_to_address_cast_rule(
                _INDEX, f"spirv.op_select.i{width}", _INDEX if width == 32 else _I64
            ),
            to_boolean,
            *_signed_payload_index_cast_rules(width),
            *_address_cast_rules(width),
        )
    )


SPIRV_INDEX_CONVERSION_RULES: tuple[
    DescriptorRule | OrdinalValueAliasRule | ValueAliasRule, ...
] = (
    _index_constant_rule(),
    _index_constant_rule(64),
    _offset_constant_rule(),
    OrdinalValueAliasRule(
        source_op=index.index_assume,
        source=ValueRef.operand("values"),
        result=ValueRef.result("results"),
    ),
    *_index_cast_rules(32),
    *_index_cast_rules(64),
    _boolean_to_address_cast_rule(_OFFSET, "spirv.op_select.offset64"),
    integer_to_boolean_rule(index.index_cast, "offset"),
    *_payload_offset_cast_rules(),
    _cast_alias_rule(_OFFSET, _OFFSET),
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


def _index_madd_rule(width: int = 32) -> DescriptorRule:
    if width == 64:
        multiply = _descriptor("spirv.op_imul.i64")
        add = _descriptor("spirv.op_iadd.i64")
        return DescriptorRule(
            source_op=index.index_madd,
            descriptor=add,
            guards=(
                *_typed_guards(("a", "b", "c", "result"), _INDEX),
                *_feature_guards(add),
            ),
            emit=(
                _emit(
                    multiply,
                    operands={
                        "lhs": ValueRef.operand("a"),
                        "rhs": ValueRef.operand("b"),
                    },
                    results={"dst": ValueRef.temporary("product")},
                    result_types={"dst": _I64},
                ),
                _emit(
                    add,
                    operands={
                        "lhs": ValueRef.temporary("product"),
                        "rhs": ValueRef.operand("c"),
                    },
                    results={"dst": ValueRef.result("result")},
                ),
            ),
        )
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


def _index_scale_rule(width: int = 32) -> DescriptorRule:
    descriptor = _descriptor("spirv.op_imul.offset64")
    return DescriptorRule(
        source_op=index.index_scale,
        descriptor=descriptor,
        guards=(
            Guard.value_type("index", _INDEX),
            Guard.value_type("stride", _OFFSET),
            Guard.value_type("result", _OFFSET),
        ),
        emit=(
            _emit(
                _descriptor(
                    "spirv.op_s_convert.i32.offset64"
                    if width == 32
                    else "spirv.op_bitcast.i64.offset64"
                ),
                operands={"input": ValueRef.operand("index")},
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
    width: int = 32,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    maximum = _U32_MAX if width == 32 else (1 << 63) - 1
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _INDEX),
            Guard.value_i64_range(
                "lhs",
                0,
                maximum,
                diagnostic=_numeric_range_diagnostic(
                    "lhs",
                    0,
                    maximum,
                    f"spirv.index.dividend.u{width}",
                ),
            ),
            Guard.value_i64_range(
                "rhs",
                1,
                maximum,
                diagnostic=_numeric_range_diagnostic(
                    "rhs",
                    1,
                    maximum,
                    f"spirv.index.divisor.positive_u{width}",
                ),
            ),
            *_feature_guards(descriptor),
        ),
        emit=(
            _integer_view_emit(
                f"spirv.op_bitcast.i{width}.u{width}",
                ValueRef.operand("lhs"),
                ValueRef.temporary("unsigned_lhs"),
                DescriptorResultType(),
            ),
            _integer_view_emit(
                f"spirv.op_bitcast.i{width}.u{width}",
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
                f"spirv.op_bitcast.u{width}.i{width}",
                ValueRef.temporary("unsigned_result"),
                ValueRef.result("result"),
                _INDEX if width == 32 else _I64,
            ),
        ),
    )


def _index_minmax_rule(
    source_op: Op,
    compare_descriptor_key: str,
    width: int = 32,
) -> DescriptorRule:
    compare = _descriptor(compare_descriptor_key)
    select = _descriptor(f"spirv.op_select.i{width}")
    return DescriptorRule(
        source_op=source_op,
        descriptor=select,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _INDEX),
            *_feature_guards(select),
        ),
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


def _index_shift_rule(
    source_op: Op, descriptor_key: str, width: int = 32
) -> DescriptorRule:
    return _binary_numeric_rule(
        source_op,
        _INDEX,
        descriptor_key,
        extra_guards=(
            Guard.value_i64_range(
                "rhs",
                0,
                width - 1,
                diagnostic=_numeric_range_diagnostic(
                    "rhs",
                    0,
                    width - 1,
                    f"spirv.index.shift_amount.u{width.bit_length() - 1}",
                ),
            ),
        ),
    )


def _direct_compare_rule(
    type_pattern: TypePattern,
    predicate: ComparePredicate,
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


def _index_compare_rule(predicate: ComparePredicate, width: int = 32) -> DescriptorRule:
    if predicate.source_predicate in ("eq", "ne", "slt", "sle", "sgt", "sge"):
        return _direct_compare_rule(
            _INDEX,
            predicate,
            f"spirv.op_{predicate.descriptor_suffix}.i{width}",
        )

    descriptor = _descriptor(f"spirv.op_{predicate.descriptor_suffix}.u{width}")
    return DescriptorRule(
        source_op=index.index_cmp,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate.source_predicate),
            *_typed_guards(("lhs", "rhs"), _INDEX),
            Guard.value_type("result", _I1),
            *_feature_guards(descriptor),
        ),
        emit=(
            _integer_view_emit(
                f"spirv.op_bitcast.i{width}.u{width}",
                ValueRef.operand("lhs"),
                ValueRef.temporary("unsigned_lhs"),
                DescriptorResultType(),
            ),
            _integer_view_emit(
                f"spirv.op_bitcast.i{width}.u{width}",
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


def _offset_compare_rule(predicate: ComparePredicate) -> DescriptorRule:
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
    rotation_rule(index.index_rotli, _INDEX, 32),
    rotation_rule(index.index_rotri, _INDEX, 32),
    bit_count_rule(index.index_ctlzi, _INDEX, 32),
    bit_count_rule(index.index_cttzi, _INDEX, 32),
    bit_count_rule(index.index_ctpopi, _INDEX, 32),
    *(_index_compare_rule(predicate) for predicate in INTEGER_COMPARE_PREDICATES),
    *(_offset_compare_rule(predicate) for predicate in INTEGER_COMPARE_PREDICATES),
    _address_select_rule(_INDEX, "spirv.op_select.i32"),
    _address_select_rule(_OFFSET, "spirv.op_select.offset64"),
)


def index_result_carrier_rules(
    rule: DescriptorRule,
    width: int,
    *,
    unsigned: bool = False,
) -> tuple[DescriptorRule, ...]:
    """Binds a recipe's result to the selected physical coordinate carrier."""
    index_fields = {
        guard.field
        for guard in rule.guards
        if guard.kind == GuardKind.VALUE_TYPE and guard.type_pattern == _INDEX
    }
    result_fields = {
        value.field
        for step in rule.emit
        for value in step.results.values()
        if value.kind == SourceValueKind.RESULT and value.field in index_fields
    }
    assert len(result_fields) <= 1
    if not result_fields:
        return (rule,)
    result_field = next(iter(result_fields))
    rules = []
    for output_width in (32, 64):
        projection = output_width != width
        emits = []
        for step in rule.emit:
            results = dict(step.results)
            result_types = dict(step.result_types or {})
            for name, value in results.items():
                if value.kind == SourceValueKind.RESULT and value.field == result_field:
                    if projection:
                        results[name] = ValueRef.temporary("arithmetic_result")
                    result_types[name] = _INDEX if width == 32 else _I64
            emits.append(
                replace(step, results=results, result_types=result_types or None)
            )
        if projection:
            if unsigned:
                assert width == 32 and output_width == 64
                emits.extend(
                    (
                        _integer_view_emit(
                            "spirv.op_bitcast.i32.u32",
                            ValueRef.temporary("arithmetic_result"),
                            ValueRef.temporary("unsigned_native"),
                            DescriptorResultType(),
                        ),
                        _integer_view_emit(
                            "spirv.op_u_convert.u32.u64",
                            ValueRef.temporary("unsigned_native"),
                            ValueRef.temporary("unsigned_wide"),
                            DescriptorResultType(),
                        ),
                        _integer_view_emit(
                            "spirv.op_bitcast.u64.i64",
                            ValueRef.temporary("unsigned_wide"),
                            ValueRef.result(result_field),
                            _I64,
                        ),
                    )
                )
            else:
                emits.append(
                    _emit(
                        _descriptor(f"spirv.op_s_convert.i{width}.i{output_width}"),
                        operands={"input": ValueRef.temporary("arithmetic_result")},
                        results={"dst": ValueRef.result(result_field)},
                        result_types={"dst": _INDEX if output_width == 32 else _I64},
                    )
                )
        guards = [
            *rule.guards,
            Guard.low_value_representation(
                result_field, _INDEX_REPRESENTATIONS[output_width]
            ),
        ]
        for step in emits:
            guards.extend(_feature_guards(step.descriptor))
        rules.append(
            replace(rule, guards=tuple(dict.fromkeys(guards)), emit=tuple(emits))
        )
    return tuple(rules)


def _index_carrier_rules(
    rule: DescriptorRule, width: int, *, native_bits: bool = False
) -> tuple[DescriptorRule, ...]:
    index_fields = {
        guard.field
        for guard in rule.guards
        if guard.kind == GuardKind.VALUE_TYPE and guard.type_pattern == _INDEX
    }
    if not index_fields:
        return (rule,)
    materializer = (
        NATIVE_INDEX_MATERIALIZER
        if native_bits
        else NATIVE_INDEX_ARITHMETIC_MATERIALIZER
        if width == 32
        else WIDE_INDEX_ARITHMETIC_MATERIALIZER
    )
    operand_fields = {
        value.field
        for step in rule.emit
        for value in step.operands.values()
        if value.kind == SourceValueKind.OPERAND and value.field in index_fields
    }
    guards = rule.guards
    if not native_bits:
        guards += (
            Guard.value_materializable(sorted(operand_fields)[0], materializer.name),
        )
    rule = replace(
        rule,
        guards=guards,
        emit=tuple(
            replace(
                step,
                operands={
                    name: replace(value, materializer=materializer.name)
                    if value.kind == SourceValueKind.OPERAND
                    and value.field in index_fields
                    else value
                    for name, value in step.operands.items()
                },
            )
            for step in rule.emit
        ),
    )
    return index_result_carrier_rules(rule, width)


_NATIVE_INDEX_BIT_OPS = {
    index.index_rotli,
    index.index_rotri,
    index.index_ctlzi,
    index.index_cttzi,
    index.index_ctpopi,
}

_WIDE_INDEX_NUMERIC_RULES = (
    *(
        _binary_numeric_rule(operation, _INDEX, f"spirv.op_{descriptor}.i64")
        for operation, descriptor in (
            (index.index_add, "iadd"),
            (index.index_sub, "isub"),
            (index.index_mul, "imul"),
            (index.index_andi, "bitwise_and"),
            (index.index_ori, "bitwise_or"),
            (index.index_xori, "bitwise_xor"),
        )
    ),
    _index_scale_rule(64),
    _index_unsigned_binary_rule(index.index_div, "spirv.op_udiv.u64", 64),
    _index_unsigned_binary_rule(index.index_rem, "spirv.op_umod.u64", 64),
    _index_minmax_rule(index.index_min, "spirv.op_s_less_than.i64", 64),
    _index_minmax_rule(index.index_max, "spirv.op_s_greater_than.i64", 64),
    _index_madd_rule(64),
    _index_shift_rule(index.index_shli, "spirv.op_shift_left_logical.i64", 64),
    _index_shift_rule(index.index_shrsi, "spirv.op_shift_right_arithmetic.i64", 64),
    _index_shift_rule(index.index_shrui, "spirv.op_shift_right_logical.i64", 64),
    *(_index_compare_rule(predicate, 64) for predicate in INTEGER_COMPARE_PREDICATES),
    _address_select_rule(_INDEX, "spirv.op_select.i64"),
)

SPIRV_INDEX_LOWERING_RULES = (
    *(
        lowered
        for rule in SPIRV_INDEX_NUMERIC_RULES
        for lowered in _index_carrier_rules(
            rule, 32, native_bits=rule.source_op in _NATIVE_INDEX_BIT_OPS
        )
    ),
    *(
        lowered
        for rule in _WIDE_INDEX_NUMERIC_RULES
        for lowered in _index_carrier_rules(rule, 64)
    ),
)
