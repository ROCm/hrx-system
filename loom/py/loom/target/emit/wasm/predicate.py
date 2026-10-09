# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wasm SIMD predicate representation and operation contracts."""

from __future__ import annotations

from collections.abc import Callable

from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.wasm.descriptors import WASM_SIMD_COMPARE_INSTRUCTIONS
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueProject,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

DescriptorLookup = Callable[[str], Descriptor]
ValueTypeGuard = Callable[[str, TypePattern], Guard]

PREDICATE_REPRESENTATIONS = (8, 16, 32, 64)
PREDICATE_TYPE = Vector("i1", minimum_static_elements=1, maximum_static_elements=16)
PREDICATE_LANE_TYPE = Vector("i1", minimum_lanes=1, maximum_lanes=16)
PREDICATE_PAYLOAD_FAMILIES = (
    (8, Vector(("i8", "f8E4M3", "f8E5M2"), minimum_lanes=1, maximum_lanes=16)),
    (16, Vector(("i16", "f16", "bf16"), minimum_lanes=1, maximum_lanes=8)),
    (32, Vector(("i32", "f32"), minimum_lanes=1, maximum_lanes=4)),
    (64, Vector(("i64", "f64"), minimum_lanes=1, maximum_lanes=2)),
)
INTEGER_PREDICATES = (
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
)
FLOAT_PREDICATES = (
    "oeq",
    "ogt",
    "oge",
    "olt",
    "ole",
    "one",
    "ord",
    "ueq",
    "ugt",
    "uge",
    "ult",
    "ule",
    "une",
    "uno",
)
NNAN_FLOAT_PREDICATES = {
    "one": "une",
    "ueq": "oeq",
    "ugt": "ogt",
    "uge": "oge",
    "ult": "olt",
    "ule": "ole",
}

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_V16I8 = Vector("i8", lanes=16)
_SHUFFLE_NAMES = tuple(f"lane{i}" for i in range(16))


def _op_emit(
    descriptor: Descriptor,
    operands: dict[str, ValueRef],
    result: ValueRef,
    *,
    result_type: TypePattern | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands=operands,
        results={"dst": result},
        result_types={"dst": result_type} if result_type is not None else None,
        form=DescriptorEmitForm.OP,
    )


def _predicate_mask_emits(
    descriptor_lookup: DescriptorLookup,
    value: ValueRef,
    physical_element_bit_count: int,
) -> tuple[tuple[EmitDescriptorOp, ...], ValueRef]:
    zero = ValueRef.temporary("predicate_zero")
    mask32 = ValueRef.temporary("predicate_mask32")
    emits = [
        EmitDescriptorOp(
            descriptor=descriptor_lookup("wasm.i32.const"),
            results={"dst": zero},
            result_types={"dst": _I32},
            immediates={"i32_value": 0},
            form=DescriptorEmitForm.CONST,
        ),
        _op_emit(
            descriptor_lookup("wasm.i32.sub"),
            {"lhs": zero, "rhs": value},
            mask32,
            result_type=_I32,
        ),
    ]
    if physical_element_bit_count != 64:
        return tuple(emits), mask32
    mask64 = ValueRef.temporary("predicate_mask64")
    emits.append(
        _op_emit(
            descriptor_lookup("wasm.i64.extend_i32_s"),
            {"input": mask32},
            mask64,
            result_type=_I64,
        )
    )
    return tuple(emits), mask64


def _predicate_conversion_lanes(
    source_element_bit_count: int,
    result_element_bit_count: int,
) -> tuple[int, ...]:
    source_byte_count = source_element_bit_count // 8
    result_byte_count = result_element_bit_count // 8
    source_lane_count = 128 // source_element_bit_count
    result = []
    for result_lane in range(128 // result_element_bit_count):
        source_lane = result_lane if result_lane < source_lane_count else 0
        result.extend([source_lane * source_byte_count] * result_byte_count)
    return tuple(result)


def _predicate_conversion_emits(
    descriptor_lookup: DescriptorLookup,
    source_element_bit_count: int,
    result_element_bit_count: int,
    source: ValueRef,
    result: ValueRef,
    *,
    result_type: TypePattern | None,
) -> tuple[EmitDescriptorOp, ...]:
    if source_element_bit_count == result_element_bit_count:
        return ()
    lanes = _predicate_conversion_lanes(
        source_element_bit_count, result_element_bit_count
    )
    return (
        EmitDescriptorOp(
            descriptor=descriptor_lookup("wasm.i8x16.shuffle"),
            operands={"lhs": source, "rhs": source},
            results={"dst": result},
            result_types={"dst": result_type} if result_type is not None else None,
            immediates=dict(zip(_SHUFFLE_NAMES, lanes, strict=True)),
            form=DescriptorEmitForm.OP,
        ),
    )


def _predicate_constant_rule(
    descriptor_lookup: DescriptorLookup, value_type_guard: ValueTypeGuard
) -> DescriptorRule:
    descriptor = descriptor_lookup("wasm.v128.const")
    value = ValueProject.exact_i64_negate("result")
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=descriptor,
        guards=(
            value_type_guard("result", PREDICATE_TYPE),
            Guard.value_exact_i64("result"),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"lo64": value, "hi64": value},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _predicate_splat_rule(
    descriptor_lookup: DescriptorLookup, value_type_guard: ValueTypeGuard
) -> DescriptorRule:
    descriptor = descriptor_lookup("wasm.i32x4.splat")
    mask_emits, mask = _predicate_mask_emits(
        descriptor_lookup, ValueRef.operand("scalar"), 32
    )
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            value_type_guard("scalar", _I1),
            value_type_guard("result", PREDICATE_TYPE),
        ),
        emit=(
            *mask_emits,
            _op_emit(
                descriptor,
                {"value": mask},
                ValueRef.result("result"),
            ),
        ),
    )


def _predicate_bitwise_rules(
    descriptor_lookup: DescriptorLookup, value_type_guard: ValueTypeGuard
) -> tuple[DescriptorRule, ...]:
    rules = []
    for source_op, operation in (
        (vector.vector_andi, "and"),
        (vector.vector_ori, "or"),
        (vector.vector_xori, "xor"),
    ):
        descriptor = descriptor_lookup(f"wasm.v128.{operation}")
        rules.append(
            DescriptorRule(
                source_op=source_op,
                descriptor=descriptor,
                guards=tuple(
                    value_type_guard(field, PREDICATE_TYPE)
                    for field in ("lhs", "rhs", "result")
                ),
                emit=(
                    _op_emit(
                        descriptor,
                        {
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        ValueRef.result("result"),
                    ),
                ),
            )
        )
    return tuple(rules)


def _whole_predicate_select_rule(
    descriptor_lookup: DescriptorLookup, value_type_guard: ValueTypeGuard
) -> DescriptorRule:
    descriptor = descriptor_lookup("wasm.v128.select")
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=descriptor,
        guards=(
            value_type_guard("condition", _I1),
            *(
                value_type_guard(field, PREDICATE_TYPE)
                for field in ("true_value", "false_value", "result")
            ),
        ),
        emit=(
            _op_emit(
                descriptor,
                {
                    "true_value": ValueRef.operand("true_value"),
                    "false_value": ValueRef.operand("false_value"),
                    "condition": ValueRef.operand("condition"),
                },
                ValueRef.result("result"),
            ),
        ),
    )


def _vector_select_rules(
    descriptor_lookup: DescriptorLookup, value_type_guard: ValueTypeGuard
) -> tuple[DescriptorRule, ...]:
    descriptor = descriptor_lookup("wasm.v128.bitselect")
    rules = [
        DescriptorRule(
            source_op=vector.vector_select,
            descriptor=descriptor,
            guards=tuple(
                value_type_guard(field, PREDICATE_TYPE)
                for field in ("condition", "true_value", "false_value", "result")
            ),
            emit=(
                _op_emit(
                    descriptor,
                    {
                        "true_value": ValueRef.operand("true_value"),
                        "false_value": ValueRef.operand("false_value"),
                        "condition": ValueRef.operand("condition"),
                    },
                    ValueRef.result("result"),
                ),
            ),
        )
    ]
    for payload_element_bit_count, payload_type in PREDICATE_PAYLOAD_FAMILIES:
        for condition_element_bit_count in PREDICATE_REPRESENTATIONS:
            condition = ValueRef.operand("condition")
            conversion_emits = ()
            if condition_element_bit_count != payload_element_bit_count:
                condition = ValueRef.temporary("condition_mask")
                conversion_emits = _predicate_conversion_emits(
                    descriptor_lookup,
                    condition_element_bit_count,
                    payload_element_bit_count,
                    ValueRef.operand("condition"),
                    condition,
                    result_type=_V16I8,
                )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_select,
                    descriptor=descriptor,
                    guards=(
                        value_type_guard("condition", PREDICATE_TYPE),
                        *(
                            value_type_guard(field, payload_type)
                            for field in ("true_value", "false_value", "result")
                        ),
                        Guard.low_value_representation(
                            "condition", condition_element_bit_count
                        ),
                    ),
                    emit=(
                        *conversion_emits,
                        _op_emit(
                            descriptor,
                            {
                                "true_value": ValueRef.operand("true_value"),
                                "false_value": ValueRef.operand("false_value"),
                                "condition": condition,
                            },
                            ValueRef.result("result"),
                        ),
                    ),
                )
            )
    return tuple(rules)


def _predicate_extract_rules(
    descriptor_lookup: DescriptorLookup, value_type_guard: ValueTypeGuard
) -> tuple[DescriptorRule, ...]:
    rules = []
    for element_bit_count in PREDICATE_REPRESENTATIONS:
        shape = f"i{element_bit_count}x{128 // element_bit_count}"
        suffix = "_u" if element_bit_count in (8, 16) else ""
        descriptor = descriptor_lookup(f"wasm.{shape}.extract_lane{suffix}")
        extracted_type = _I64 if element_bit_count == 64 else _I32
        extracted = ValueRef.temporary("predicate_bits")
        low_bits = extracted
        emits = [
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": extracted},
                result_types={"dst": extracted_type},
                immediates={
                    "lane": AttrProject.i64_array_element("static_indices", element=0)
                },
                form=DescriptorEmitForm.OP,
            )
        ]
        if element_bit_count == 64:
            low_bits = ValueRef.temporary("predicate_bits32")
            emits.append(
                _op_emit(
                    descriptor_lookup("wasm.i32.wrap_i64"),
                    {"input": extracted},
                    low_bits,
                    result_type=_I32,
                )
            )
        one = ValueRef.temporary("predicate_one")
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=descriptor_lookup("wasm.i32.const"),
                    results={"dst": one},
                    result_types={"dst": _I32},
                    immediates={"i32_value": 1},
                    form=DescriptorEmitForm.CONST,
                ),
                _op_emit(
                    descriptor_lookup("wasm.i32.and"),
                    {"lhs": low_bits, "rhs": one},
                    ValueRef.result("result"),
                ),
            )
        )
        rules.append(
            DescriptorRule(
                source_op=vector.vector_extract,
                descriptor=descriptor,
                guards=(
                    value_type_guard("source", PREDICATE_LANE_TYPE),
                    value_type_guard("result", _I1),
                    Guard.low_value_representation("source", element_bit_count),
                    Guard.operand_segment_count("indices", 0),
                    Guard.i64_array_count("static_indices", 1),
                    Guard.i64_array_element_range(
                        "static_indices",
                        0,
                        0,
                        128 // element_bit_count - 1,
                    ),
                ),
                emit=tuple(emits),
            )
        )
    return tuple(rules)


def _predicate_insert_rules(
    descriptor_lookup: DescriptorLookup, value_type_guard: ValueTypeGuard
) -> tuple[DescriptorRule, ...]:
    rules = []
    for element_bit_count in PREDICATE_REPRESENTATIONS:
        shape = f"i{element_bit_count}x{128 // element_bit_count}"
        descriptor = descriptor_lookup(f"wasm.{shape}.replace_lane")
        mask_emits, mask = _predicate_mask_emits(
            descriptor_lookup, ValueRef.operand("value"), element_bit_count
        )
        rules.append(
            DescriptorRule(
                source_op=vector.vector_insert,
                descriptor=descriptor,
                guards=(
                    value_type_guard("value", _I1),
                    value_type_guard("dest", PREDICATE_LANE_TYPE),
                    value_type_guard("result", PREDICATE_LANE_TYPE),
                    Guard.low_value_representation("dest", element_bit_count),
                    Guard.operand_segment_count("indices", 0),
                    Guard.i64_array_count("static_indices", 1),
                    Guard.i64_array_element_range(
                        "static_indices",
                        0,
                        0,
                        128 // element_bit_count - 1,
                    ),
                ),
                emit=(
                    *mask_emits,
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands={"dest": ValueRef.operand("dest"), "value": mask},
                        results={"dst": ValueRef.result("result")},
                        immediates={
                            "lane": AttrProject.i64_array_element(
                                "static_indices", element=0
                            )
                        },
                        form=DescriptorEmitForm.OP,
                    ),
                ),
            )
        )
    return tuple(rules)


def _predicate_dynamic_insert_rules(
    descriptor_lookup: DescriptorLookup, value_type_guard: ValueTypeGuard
) -> tuple[DescriptorRule, ...]:
    rules = []
    for element_bit_count in PREDICATE_REPRESENTATIONS:
        physical_lane_count = 128 // element_bit_count
        bytes_per_lane = element_bit_count // 8
        ordinals = bytes(byte // bytes_per_lane for byte in range(16))
        shape = f"i{element_bit_count}x{physical_lane_count}"
        descriptor = descriptor_lookup("wasm.v128.bitselect")
        mask_emits, mask = _predicate_mask_emits(
            descriptor_lookup, ValueRef.operand("value"), element_bit_count
        )
        rules.append(
            DescriptorRule(
                source_op=vector.vector_insert,
                descriptor=descriptor,
                guards=(
                    value_type_guard("value", _I1),
                    value_type_guard("dest", PREDICATE_LANE_TYPE),
                    value_type_guard("result", PREDICATE_LANE_TYPE),
                    Guard.low_value_representation("dest", element_bit_count),
                    Guard.operand_segment_count("indices", 1),
                    Guard.i64_array_count("static_indices", 1),
                ),
                emit=(
                    *mask_emits,
                    EmitDescriptorOp(
                        descriptor=descriptor_lookup("wasm.v128.const"),
                        results={"dst": ValueRef.temporary("lane_ordinals")},
                        result_types={"dst": _V16I8},
                        immediates={
                            "lo64": int.from_bytes(ordinals[:8], "little"),
                            "hi64": int.from_bytes(ordinals[8:], "little"),
                        },
                        form=DescriptorEmitForm.CONST,
                    ),
                    _op_emit(
                        descriptor_lookup("wasm.i8x16.splat"),
                        {"value": ValueRef.operand("indices")},
                        ValueRef.temporary("lane_index"),
                        result_type=_V16I8,
                    ),
                    _op_emit(
                        descriptor_lookup("wasm.i8x16.eq"),
                        {
                            "lhs": ValueRef.temporary("lane_ordinals"),
                            "rhs": ValueRef.temporary("lane_index"),
                        },
                        ValueRef.temporary("lane_mask"),
                        result_type=_V16I8,
                    ),
                    _op_emit(
                        descriptor_lookup(f"wasm.{shape}.splat"),
                        {"value": mask},
                        ValueRef.temporary("replacement"),
                        result_type=_V16I8,
                    ),
                    _op_emit(
                        descriptor,
                        {
                            "true_value": ValueRef.temporary("replacement"),
                            "false_value": ValueRef.operand("dest"),
                            "condition": ValueRef.temporary("lane_mask"),
                        },
                        ValueRef.result("result"),
                    ),
                ),
            )
        )
    return tuple(rules)


def _native_compare_descriptors(
    descriptor_lookup: DescriptorLookup,
) -> dict[tuple[str, str], Descriptor]:
    return {
        (instruction.element, instruction.predicate): descriptor_lookup(
            f"wasm.{instruction.shape}.{instruction.operation}"
        )
        for instruction in WASM_SIMD_COMPARE_INSTRUCTIONS
    }


def _integer_compare_program(
    descriptor_lookup: DescriptorLookup,
    descriptors: dict[tuple[str, str], Descriptor],
    element: str,
    predicate: str,
    result: ValueRef,
    result_type: TypePattern | None,
) -> tuple[EmitDescriptorOp, ...]:
    lhs = ValueRef.operand("lhs")
    rhs = ValueRef.operand("rhs")
    if element == "i64" and predicate.startswith("u"):
        sign = ValueRef.temporary("sign_bias")
        biased_lhs = ValueRef.temporary("biased_lhs")
        biased_rhs = ValueRef.temporary("biased_rhs")
        constant = descriptor_lookup("wasm.v128.const")
        xor = descriptor_lookup("wasm.v128.xor")
        signed_predicate = "s" + predicate[1:]
        compare = descriptors[(element, signed_predicate)]
        return (
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": sign},
                result_types={"dst": _V16I8},
                immediates={"lo64": 1 << 63, "hi64": 1 << 63},
                form=DescriptorEmitForm.CONST,
            ),
            _op_emit(
                xor,
                {"lhs": lhs, "rhs": sign},
                biased_lhs,
                result_type=_V16I8,
            ),
            _op_emit(
                xor,
                {"lhs": rhs, "rhs": sign},
                biased_rhs,
                result_type=_V16I8,
            ),
            _op_emit(
                compare,
                {"lhs": biased_lhs, "rhs": biased_rhs},
                result,
                result_type=result_type,
            ),
        )
    compare = descriptors[(element, predicate)]
    return (
        _op_emit(
            compare,
            {"lhs": lhs, "rhs": rhs},
            result,
            result_type=result_type,
        ),
    )


def _float_compare_program(
    descriptor_lookup: DescriptorLookup,
    descriptors: dict[tuple[str, str], Descriptor],
    element: str,
    predicate: str,
    result: ValueRef,
    result_type: TypePattern | None,
) -> tuple[EmitDescriptorOp, ...]:
    lhs = ValueRef.operand("lhs")
    rhs = ValueRef.operand("rhs")

    def compare(
        native_predicate: str,
        compare_lhs: ValueRef,
        compare_rhs: ValueRef,
        compare_result: ValueRef,
        compare_result_type: TypePattern | None,
    ) -> EmitDescriptorOp:
        return _op_emit(
            descriptors[(element, native_predicate)],
            {"lhs": compare_lhs, "rhs": compare_rhs},
            compare_result,
            result_type=compare_result_type,
        )

    if predicate in ("oeq", "ogt", "oge", "olt", "ole", "une"):
        return (compare(predicate, lhs, rhs, result, result_type),)

    first = ValueRef.temporary("predicate_first")
    second = ValueRef.temporary("predicate_second")
    combined = result
    combined_type = result_type
    negate = predicate in ("ueq", "ugt", "uge", "ult", "ule", "uno")
    if negate:
        combined = ValueRef.temporary("predicate_combined")
        combined_type = _V16I8

    if predicate in ("one", "ueq"):
        emits = [
            compare("olt", lhs, rhs, first, _V16I8),
            compare("ogt", lhs, rhs, second, _V16I8),
            _op_emit(
                descriptor_lookup("wasm.v128.or"),
                {"lhs": first, "rhs": second},
                combined,
                result_type=combined_type,
            ),
        ]
    elif predicate in ("ord", "uno"):
        emits = [
            compare("oeq", lhs, lhs, first, _V16I8),
            compare("oeq", rhs, rhs, second, _V16I8),
            _op_emit(
                descriptor_lookup("wasm.v128.and"),
                {"lhs": first, "rhs": second},
                combined,
                result_type=combined_type,
            ),
        ]
    else:
        inverse = {
            "ugt": "ole",
            "uge": "olt",
            "ult": "oge",
            "ule": "ogt",
        }[predicate]
        emits = [compare(inverse, lhs, rhs, combined, combined_type)]

    if negate:
        emits.append(
            _op_emit(
                descriptor_lookup("wasm.v128.not"),
                {"input": combined},
                result,
                result_type=result_type,
            )
        )
    return tuple(emits)


def _compare_rule(
    value_type_guard: ValueTypeGuard,
    source_op: Op,
    predicate: str,
    operand_type: TypePattern,
    result_type: TypePattern,
    result_element_bit_count: int,
    operation_emits: tuple[EmitDescriptorOp, ...],
    conversion_emits: tuple[EmitDescriptorOp, ...],
    *,
    extra_guards: tuple[Guard, ...] = (),
    priority: int = 0,
    suffix: str = "",
) -> DescriptorRule:
    mechanisms = ["native" if len(operation_emits) == 1 else "composed"]
    if conversion_emits:
        mechanisms.append("converted")
    if suffix:
        mechanisms.append(suffix)
    all_emits = (*operation_emits, *conversion_emits)
    return DescriptorRule(
        source_op=source_op,
        descriptor=all_emits[-1].descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            value_type_guard("lhs", operand_type),
            value_type_guard("rhs", operand_type),
            value_type_guard("result", result_type),
            Guard.low_value_representation("result", result_element_bit_count),
            *extra_guards,
        ),
        emit=all_emits,
        priority=priority,
        report_key="wasm.predicate_compare." + ".".join(mechanisms),
    )


def _compare_rules(
    descriptor_lookup: DescriptorLookup, value_type_guard: ValueTypeGuard
) -> tuple[DescriptorRule, ...]:
    descriptors = _native_compare_descriptors(descriptor_lookup)
    rules = []
    for element, element_bit_count, predicates, source_op in (
        ("i8", 8, INTEGER_PREDICATES, vector.vector_cmpi),
        ("i16", 16, INTEGER_PREDICATES, vector.vector_cmpi),
        ("i32", 32, INTEGER_PREDICATES, vector.vector_cmpi),
        ("i64", 64, INTEGER_PREDICATES, vector.vector_cmpi),
        ("f32", 32, FLOAT_PREDICATES, vector.vector_cmpf),
        ("f64", 64, FLOAT_PREDICATES, vector.vector_cmpf),
    ):
        operand_type = Vector(
            element,
            minimum_lanes=1,
            maximum_lanes=128 // element_bit_count,
        )
        result_type = Vector(
            "i1",
            minimum_lanes=1,
            maximum_lanes=128 // element_bit_count,
        )
        for predicate in predicates:
            for result_element_bit_count in PREDICATE_REPRESENTATIONS:
                requires_conversion = result_element_bit_count != element_bit_count
                native_result = (
                    ValueRef.temporary("native_mask")
                    if requires_conversion
                    else ValueRef.result("result")
                )
                native_result_type = _V16I8 if requires_conversion else None
                if source_op is vector.vector_cmpi:
                    compare_emits = _integer_compare_program(
                        descriptor_lookup,
                        descriptors,
                        element,
                        predicate,
                        native_result,
                        native_result_type,
                    )
                else:
                    compare_emits = _float_compare_program(
                        descriptor_lookup,
                        descriptors,
                        element,
                        predicate,
                        native_result,
                        native_result_type,
                    )
                conversion_emits = _predicate_conversion_emits(
                    descriptor_lookup,
                    element_bit_count,
                    result_element_bit_count,
                    native_result,
                    ValueRef.result("result"),
                    result_type=None,
                )

                rules.append(
                    _compare_rule(
                        value_type_guard,
                        source_op,
                        predicate,
                        operand_type,
                        result_type,
                        result_element_bit_count,
                        compare_emits,
                        conversion_emits,
                    )
                )
                if (
                    source_op is vector.vector_cmpf
                    and predicate in NNAN_FLOAT_PREDICATES
                ):
                    direct_descriptor = descriptors[
                        (element, NNAN_FLOAT_PREDICATES[predicate])
                    ]
                    rules.append(
                        _compare_rule(
                            value_type_guard,
                            source_op,
                            predicate,
                            operand_type,
                            result_type,
                            result_element_bit_count,
                            (
                                _op_emit(
                                    direct_descriptor,
                                    {
                                        "lhs": ValueRef.operand("lhs"),
                                        "rhs": ValueRef.operand("rhs"),
                                    },
                                    native_result,
                                    result_type=native_result_type,
                                ),
                            ),
                            conversion_emits,
                            extra_guards=(
                                Guard.instance_flags_has_all("fastmath", "nnan"),
                            ),
                            priority=1,
                            suffix="nnan",
                        )
                    )
    return tuple(rules)


def predicate_rules(
    descriptor_lookup: DescriptorLookup, value_type_guard: ValueTypeGuard
) -> tuple[ContractCase, ...]:
    """Builds the complete Wasm SIMD predicate contract family."""
    return (
        _predicate_constant_rule(descriptor_lookup, value_type_guard),
        _predicate_splat_rule(descriptor_lookup, value_type_guard),
        _whole_predicate_select_rule(descriptor_lookup, value_type_guard),
        *_vector_select_rules(descriptor_lookup, value_type_guard),
        *_predicate_bitwise_rules(descriptor_lookup, value_type_guard),
        *_predicate_extract_rules(descriptor_lookup, value_type_guard),
        *_predicate_insert_rules(descriptor_lookup, value_type_guard),
        *_predicate_dynamic_insert_rules(descriptor_lookup, value_type_guard),
        *_compare_rules(descriptor_lookup, value_type_guard),
    )
