# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""BF16 rounding and bit-preserving transport through native or integer carriers."""

from enum import Enum, unique

from loom.dialect.scalar import conversion
from loom.dialect.view import defs as view
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards,
    emit_descriptor_op,
    float_narrowing_descriptors,
    logical_core_descriptor,
)
from loom.target.arch.spirv.contracts.memory import (
    source_memory_address_feature_guards,
    storage_buffer_address_materializer,
    storage_buffer_source_memory,
    workgroup_address_materializer,
    workgroup_carrier_guard,
    workgroup_source_memory,
)
from loom.target.arch.spirv.scalar_memory import storage_buffer_scalar_by_suffix
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    SourceMemoryAddressCoordinateType,
    SourceMemoryOperation,
    SourceValueKind,
    ValueRef,
    View,
)
from loom.target.emit.float_narrowing import (
    BF16_FORMAT,
    F32_FORMAT,
    F64_FORMAT,
    BinaryFloatFormat,
    FloatNarrowingDescriptors,
    NarrowFloatSubnormalRounding,
    build_f32_to_bf16_emits,
    build_float_to_narrow_float_emits,
    build_representable_f32_to_bf16_emits,
)


@unique
class _BfloatNarrowResult(Enum):
    NATIVE = "native"
    I32_CARRIER = "i32_carrier"


@unique
class _BfloatNarrowSemantics(Enum):
    EXACTLY_REPRESENTABLE = "exactly_representable"
    EXACTLY_REPRESENTABLE_PRESERVE_NAN = "exactly_representable_preserve_nan"
    RNE_NOT_NAN = "rne_not_nan"
    RNE_PRESERVE_NAN = "rne_preserve_nan"


def bfloat_to_f32_emits(
    source: ValueRef, result: ValueRef, *, native_input: bool
) -> tuple[EmitDescriptorOp, ...]:
    """Widens BF16 by preserving its payload, without floating-point controls."""
    emits: list[EmitDescriptorOp] = []
    if native_input:
        # Native BF16 does not imply Int16. Pack a pair into i32; shifting by
        # sixteen discards the duplicate and places the payload in F32's high
        # bits, preserving subnormals, signed zero and every NaN payload.
        packed = ValueRef.temporary("packed_bfloat")
        carrier = ValueRef.temporary("bfloat_carrier")
        emits.extend(
            (
                emit_descriptor_op(
                    descriptor=logical_core_descriptor(
                        "spirv.op_composite_construct.v2bf16"
                    ),
                    operands={"component0": source, "component1": source},
                    results={"dst": packed},
                    result_types={"dst": DescriptorResultType()},
                ),
                emit_descriptor_op(
                    descriptor=logical_core_descriptor("spirv.op_bitcast.v2bf16.i32"),
                    operands={"input": packed},
                    results={"dst": carrier},
                    result_types={"dst": DescriptorResultType()},
                ),
            )
        )
        source = carrier
    emits.extend(
        (
            EmitDescriptorOp(
                descriptor=logical_core_descriptor("spirv.op_constant.i32"),
                results={"dst": ValueRef.temporary("carrier_shift")},
                result_types={"dst": DescriptorResultType()},
                immediates={"i32_value": 16},
                form=DescriptorEmitForm.CONST,
            ),
            emit_descriptor_op(
                descriptor=logical_core_descriptor("spirv.op_shift_left_logical.i32"),
                operands={"lhs": source, "rhs": ValueRef.temporary("carrier_shift")},
                results={"dst": ValueRef.temporary("wide_bits")},
                result_types={"dst": DescriptorResultType()},
            ),
            emit_descriptor_op(
                descriptor=logical_core_descriptor("spirv.op_bitcast.i32.f32"),
                operands={"input": ValueRef.temporary("wide_bits")},
                results={"dst": result},
                result_types=(
                    {"dst": DescriptorResultType()}
                    if result.kind is SourceValueKind.TEMPORARY
                    else None
                ),
            ),
        )
    )
    return tuple(emits)


def bfloat_widen_rules() -> tuple[DescriptorRule, ...]:
    """Returns native and software BF16-to-F32 payload conversion rules."""
    rules = []
    for native_input in (True, False):
        emits = bfloat_to_f32_emits(
            ValueRef.operand("input"),
            ValueRef.result("result"),
            native_input=native_input,
        )
        rules.append(
            DescriptorRule(
                source_op=conversion.scalar_extf,
                descriptor=emits[-1].descriptor,
                guards=(
                    Guard.value_type("input", Scalar("bf16")),
                    Guard.value_type("result", Scalar("f32")),
                    *descriptor_feature_guards(*(step.descriptor for step in emits)),
                ),
                emit=emits,
                report_key="exact_bfloat16_to_binary32",
            )
        )
    return tuple(rules)


def _bfloat_narrow_rule(
    *,
    source_type: str,
    source_name: str,
    source_format: BinaryFloatFormat,
    integer_suffix: str,
    descriptors: FloatNarrowingDescriptors,
    result_kind: _BfloatNarrowResult,
    semantics: _BfloatNarrowSemantics,
) -> DescriptorRule:
    """Rounds one binary source format to BF16 with exact RNE semantics."""
    exactly_representable = semantics in (
        _BfloatNarrowSemantics.EXACTLY_REPRESENTABLE,
        _BfloatNarrowSemantics.EXACTLY_REPRESENTABLE_PRESERVE_NAN,
    )
    preserve_nan = semantics in (
        _BfloatNarrowSemantics.RNE_PRESERVE_NAN,
        _BfloatNarrowSemantics.EXACTLY_REPRESENTABLE_PRESERVE_NAN,
    )
    if exactly_representable and source_format != F32_FORMAT:
        raise ValueError("representable BF16 shortcut requires a binary32 source")
    direct_carrier_result = (
        result_kind is _BfloatNarrowResult.I32_CARRIER and integer_suffix == "i32"
    )
    carrier_ref = (
        ValueRef.result("result")
        if direct_carrier_result
        else ValueRef.temporary("carrier")
    )

    # Binary32 and BF16 have the same exponent field, so the compact bias-add
    # recipe avoids the general normal/subnormal selection.
    if exactly_representable:
        emits = list(
            build_representable_f32_to_bf16_emits(
                descriptors,
                ValueRef.operand("input"),
                carrier_ref,
                preserve_nan=preserve_nan,
            )
        )
    elif source_format == F32_FORMAT:
        emits = list(
            build_f32_to_bf16_emits(
                descriptors,
                ValueRef.operand("input"),
                carrier_ref,
                preserve_nan=preserve_nan,
            )
        )
    else:
        emits = list(
            build_float_to_narrow_float_emits(
                descriptors,
                source_format,
                BF16_FORMAT,
                ValueRef.operand("input"),
                carrier_ref,
                subnormal_rounding=NarrowFloatSubnormalRounding.INTEGER,
                preserve_nan=preserve_nan,
            )
        )

    if result_kind is _BfloatNarrowResult.NATIVE:
        # The rounded payload occupies the low 16 bits. Native BF16 does not
        # imply Int16 support; reinterpret the full-width carrier and extract
        # its first component instead of introducing an i16 intermediate.
        vector_suffix = f"v{source_format.bit_width // BF16_FORMAT.bit_width}bf16"
        bitcast_descriptor = logical_core_descriptor(
            f"spirv.op_bitcast.{integer_suffix}.{vector_suffix}"
        )
        emits.append(
            emit_descriptor_op(
                descriptor=bitcast_descriptor,
                operands={"input": carrier_ref},
                results={"dst": ValueRef.temporary("packed_bfloat")},
                result_types={"dst": DescriptorResultType()},
            )
        )
        extract_descriptor = logical_core_descriptor(
            f"spirv.op_composite_extract.{vector_suffix}.bf16"
        )
        emits.append(
            emit_descriptor_op(
                descriptor=extract_descriptor,
                operands={"composite": ValueRef.temporary("packed_bfloat")},
                results={"dst": ValueRef.result("result")},
                immediates={"component_index": 0},
            )
        )
    elif not direct_carrier_result:
        conversion_descriptor = logical_core_descriptor(
            f"spirv.op_s_convert.{integer_suffix}.i32"
        )
        emits.append(
            emit_descriptor_op(
                descriptor=conversion_descriptor,
                operands={"input": carrier_ref},
                results={"dst": ValueRef.result("result")},
            )
        )

    return DescriptorRule(
        source_op=conversion.scalar_fptrunc,
        descriptor=emits[-1].descriptor,
        guards=(
            Guard.value_type("input", Scalar(source_type)),
            Guard.value_type("result", Scalar("bf16")),
            *((Guard.value_not_nan("input"),) if not preserve_nan else ()),
            *(
                (Guard.value_float_exactly_representable_as("input", "result"),)
                if exactly_representable
                else ()
            ),
            *descriptor_feature_guards(*(step.descriptor for step in emits)),
        ),
        emit=tuple(emits),
        report_key=(
            f"exactly_representable_{source_name}_to_bfloat16"
            + ("_preserve_nan" if preserve_nan else "")
            if exactly_representable
            else f"exact_{source_name}_to_bfloat16"
        ),
    )


def bfloat_narrow_rules() -> tuple[DescriptorRule, ...]:
    """Returns exact F64/F32-to-BF16 rules for native and carrier profiles."""
    rows = (
        (
            "f64",
            "binary64",
            F64_FORMAT,
            "i64",
            float_narrowing_descriptors("i64", "f64"),
        ),
        (
            "f32",
            "binary32",
            F32_FORMAT,
            "i32",
            float_narrowing_descriptors("i32", "f32"),
        ),
    )
    rules = []
    for source_type, source_name, source_format, integer_suffix, descriptors in rows:
        if source_format == F32_FORMAT:
            rules.extend(
                _bfloat_narrow_rule(
                    source_type=source_type,
                    source_name=source_name,
                    source_format=source_format,
                    integer_suffix=integer_suffix,
                    descriptors=descriptors,
                    result_kind=result_kind,
                    semantics=semantics,
                )
                for semantics in (
                    _BfloatNarrowSemantics.EXACTLY_REPRESENTABLE,
                    _BfloatNarrowSemantics.EXACTLY_REPRESENTABLE_PRESERVE_NAN,
                )
                for result_kind in (
                    _BfloatNarrowResult.NATIVE,
                    _BfloatNarrowResult.I32_CARRIER,
                )
            )
        rules.extend(
            _bfloat_narrow_rule(
                source_type=source_type,
                source_name=source_name,
                source_format=source_format,
                integer_suffix=integer_suffix,
                descriptors=descriptors,
                result_kind=result_kind,
                semantics=semantics,
            )
            for result_kind, semantics in (
                (
                    _BfloatNarrowResult.NATIVE,
                    _BfloatNarrowSemantics.RNE_NOT_NAN,
                ),
                (
                    _BfloatNarrowResult.NATIVE,
                    _BfloatNarrowSemantics.RNE_PRESERVE_NAN,
                ),
                (
                    _BfloatNarrowResult.I32_CARRIER,
                    _BfloatNarrowSemantics.RNE_NOT_NAN,
                ),
                (
                    _BfloatNarrowResult.I32_CARRIER,
                    _BfloatNarrowSemantics.RNE_PRESERVE_NAN,
                ),
            )
        )
    return tuple(rules)


def _bfloat_carrier_to_i16_emits(
    source: ValueRef, result: ValueRef
) -> tuple[EmitDescriptorOp, ...]:
    descriptor = logical_core_descriptor("spirv.op_s_convert.i32.i16")
    return (
        emit_descriptor_op(
            descriptor=descriptor,
            operands={"input": source},
            results={"dst": result},
            result_types={"dst": Scalar("i16")},
        ),
    )


def bfloat_carrier_to_i16_rule() -> DescriptorRule:
    """Observes an i32-carried BF16 value through its native i16 bit view."""
    emits = _bfloat_carrier_to_i16_emits(
        ValueRef.operand("input"), ValueRef.result("result")
    )
    return DescriptorRule(
        source_op=conversion.scalar_bitcast,
        descriptor=emits[-1].descriptor,
        guards=(
            Guard.value_type("input", Scalar("bf16")),
            Guard.value_type("result", Scalar("i16")),
            *descriptor_feature_guards(*(step.descriptor for step in emits)),
        ),
        emit=emits,
    )


def _bfloat_carrier_from_i16_emits(
    source: ValueRef, result: ValueRef
) -> tuple[EmitDescriptorOp, ...]:
    signed_to_unsigned = logical_core_descriptor("spirv.op_bitcast.i16.u16")
    widen = logical_core_descriptor("spirv.op_u_convert.u16.u32")
    unsigned_to_signed = logical_core_descriptor("spirv.op_bitcast.u32.i32")
    return (
        emit_descriptor_op(
            descriptor=signed_to_unsigned,
            operands={"input": source},
            results={"dst": ValueRef.temporary("unsigned_bits")},
            result_types={"dst": DescriptorResultType()},
        ),
        emit_descriptor_op(
            descriptor=widen,
            operands={"input": ValueRef.temporary("unsigned_bits")},
            results={"dst": ValueRef.temporary("wide_unsigned_bits")},
            result_types={"dst": DescriptorResultType()},
        ),
        emit_descriptor_op(
            descriptor=unsigned_to_signed,
            operands={"input": ValueRef.temporary("wide_unsigned_bits")},
            results={"dst": result},
            result_types={"dst": Scalar("i32")},
        ),
    )


def bfloat_carrier_from_i16_rule() -> DescriptorRule:
    """Constructs an i32-carried BF16 value from its native i16 bit view."""
    emits = _bfloat_carrier_from_i16_emits(
        ValueRef.operand("input"), ValueRef.result("result")
    )
    return DescriptorRule(
        source_op=conversion.scalar_bitcast,
        descriptor=emits[-1].descriptor,
        guards=(
            Guard.value_type("input", Scalar("i16")),
            Guard.value_type("result", Scalar("bf16")),
            *descriptor_feature_guards(*(step.descriptor for step in emits)),
        ),
        emit=emits,
        report_key="bfloat16_carrier_from_i16_bits",
    )


def bfloat_carrier_memory_rules() -> tuple[DescriptorRule, ...]:
    """Transports software BF16 payloads through same-width integer storage."""
    scalar = storage_buffer_scalar_by_suffix("i16")
    assert scalar is not None
    paths = (
        (
            "storage_buffer",
            storage_buffer_address_materializer(scalar),
            storage_buffer_source_memory,
            (),
        ),
        *(
            (
                "workgroup",
                workgroup_address_materializer(scalar, coordinate_type),
                workgroup_source_memory,
                (workgroup_carrier_guard(scalar.suffix),),
            )
            for coordinate_type in (
                SourceMemoryAddressCoordinateType.INDEX,
                SourceMemoryAddressCoordinateType.OFFSET,
            )
        ),
    )
    rules: list[DescriptorRule] = []
    for space, address_materializer, source_memory, carrier_guards in paths:
        for operation in (SourceMemoryOperation.LOAD, SourceMemoryOperation.STORE):
            is_load = operation is SourceMemoryOperation.LOAD
            descriptor = logical_core_descriptor(
                f"spirv.op_{'load' if is_load else 'store'}.{space}.{scalar.suffix}"
            )
            stored_bits = ValueRef.temporary("stored_bits")
            operands = {"ptr": ValueRef.source_memory_address()}
            if not is_load:
                operands["value"] = stored_bits
            memory_emit = emit_descriptor_op(
                descriptor=descriptor,
                operands=operands,
                results={"dst": stored_bits} if is_load else None,
                result_types={"dst": DescriptorResultType()} if is_load else None,
                source_memory=source_memory(operation, scalar),
                source_memory_address_materializer=address_materializer,
            )
            emits = (
                (
                    memory_emit,
                    *_bfloat_carrier_from_i16_emits(
                        stored_bits, ValueRef.result("result")
                    ),
                )
                if is_load
                else (
                    *_bfloat_carrier_to_i16_emits(
                        ValueRef.operand("value"), stored_bits
                    ),
                    memory_emit,
                )
            )
            rules.append(
                DescriptorRule(
                    source_op=view.view_load if is_load else view.view_store,
                    descriptor=descriptor,
                    guards=(
                        Guard.value_type("view", View("bf16")),
                        Guard.value_type(
                            "result" if is_load else "value", Scalar("bf16")
                        ),
                        *carrier_guards,
                        *source_memory_address_feature_guards(address_materializer),
                        *descriptor_feature_guards(
                            *(step.descriptor for step in emits)
                        ),
                    ),
                    emit=emits,
                )
            )
    return tuple(rules)
