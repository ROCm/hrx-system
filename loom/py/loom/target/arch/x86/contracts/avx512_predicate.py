# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AVX-512 predicate representation and operation contract rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup as _DescriptorLookup,
)
from loom.target.arch.x86.contracts.rule_builders import (
    emit_descriptor_op as _op_emit,
)
from loom.target.arch.x86.contracts.rule_builders import (
    full_vector_type as _full_vector_type,
)
from loom.target.arch.x86.contracts.rule_builders import (
    value_type_guards as _typed_guards,
)
from loom.target.arch.x86.vector_families import (
    AVX512_BITWISE_FAMILIES,
    AVX512_FLOAT_COMPARE_MNEMONICS,
    AVX512_FP16_FLOAT_COMPARE_MNEMONIC,
    AVX512_FP16_SCALAR_FLOAT_COMPARE_MNEMONIC,
    AVX512_FP16_VECTOR_BIT_WIDTHS,
    AVX512_INTEGER_COMPARE_MNEMONICS,
    AVX512_SELECT_MNEMONICS,
    FLOAT_ELEMENTS,
    FP16_ELEMENT,
    INTEGER_ELEMENTS,
    STORAGE_ELEMENTS,
)
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_PREDICATE_LANE_COUNTS = (2, 4, 8, 16, 32, 64)
_PREDICATE_TYPES = {
    lane_count: Vector("i1", lanes=lane_count) for lane_count in _PREDICATE_LANE_COUNTS
}
_PREDICATE_PATTERN = Vector("i1", minimum_lanes=2, maximum_lanes=64)
_CARRIER_REGISTER_CLASSES = {
    2: "x86.xmm",
    4: "x86.xmm",
    8: "x86.xmm",
    16: "x86.xmm",
    32: "x86.ymm",
    64: "x86.zmm",
}
_CARRIER_ELEMENT_SUFFIXES = {
    2: "q",
    4: "d",
    8: "w",
    16: "b",
    32: "b",
    64: "b",
}
_INTEGER_COMPARE_IMMEDIATES = {
    "eq": 0,
    "ne": 4,
    "slt": 1,
    "sle": 2,
    "sgt": 6,
    "sge": 5,
    "ult": 1,
    "ule": 2,
    "ugt": 6,
    "uge": 5,
}
_SIGNED_INTEGER_PREDICATES = ("eq", "ne", "slt", "sle", "sgt", "sge")
_UNSIGNED_INTEGER_PREDICATES = ("ult", "ule", "ugt", "uge")
_FLOAT_COMPARE_IMMEDIATES = {
    "oeq": 0,
    "ogt": 30,
    "oge": 29,
    "olt": 17,
    "ole": 18,
    "one": 12,
    "ord": 7,
    "ueq": 8,
    "ugt": 22,
    "uge": 21,
    "ult": 25,
    "ule": 26,
    "une": 4,
    "uno": 3,
}
_REGISTER_SUFFIXES = {64: "xmm", 128: "xmm", 256: "ymm", 512: "zmm"}
_BITWISE_SOURCE_OPS = {
    "andi": vector.vector_andi,
    "ori": vector.vector_ori,
    "xori": vector.vector_xori,
}
_SELECT_GROUPS = tuple(
    (
        tuple(
            element.name
            for element in (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS, *FLOAT_ELEMENTS)
            if AVX512_SELECT_MNEMONICS[element.name] == mnemonic
        ),
        next(
            element.bit_width
            for element in (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS, *FLOAT_ELEMENTS)
            if AVX512_SELECT_MNEMONICS[element.name] == mnemonic
        ),
        mnemonic,
    )
    for mnemonic in dict.fromkeys(AVX512_SELECT_MNEMONICS.values())
)


def _operation_representations(
    lane_count: int, vector_bit_width: int | None = None
) -> tuple[str, ...]:
    """Returns only representations not already implemented by AVX2."""
    if lane_count == 64 or vector_bit_width == 512:
        return ("x86.k", _CARRIER_REGISTER_CLASSES[lane_count])
    return ("x86.k",)


def _all_operation_representations(lane_count: int) -> tuple[str, ...]:
    """Returns the native mask and callable predicate representations."""
    return ("x86.k", _CARRIER_REGISTER_CLASSES[lane_count])


def avx512_predicate_carrier_register_class(lane_count: int) -> str:
    """Returns the callable SIMD register class for a predicate shape."""

    return _CARRIER_REGISTER_CLASSES[lane_count]


def _mask_to_carrier_descriptor_key(lane_count: int) -> str:
    return (
        f"x86.avx512.vpmovm2{_CARRIER_ELEMENT_SUFFIXES[lane_count]}."
        f"{_CARRIER_REGISTER_CLASSES[lane_count].removeprefix('x86.')}.k"
    )


def _carrier_to_mask_descriptor_key(lane_count: int) -> str:
    return (
        f"x86.avx512.vpmov{_CARRIER_ELEMENT_SUFFIXES[lane_count]}2m.k."
        f"{_CARRIER_REGISTER_CLASSES[lane_count].removeprefix('x86.')}"
    )


def _mask_to_carrier_emit(
    descriptor_lookup: _DescriptorLookup,
    lane_count: int,
    source: ValueRef,
    result: ValueRef,
) -> tuple[EmitDescriptorOp, ...]:
    descriptor = descriptor_lookup(_mask_to_carrier_descriptor_key(lane_count))
    return (
        _op_emit(
            descriptor=descriptor,
            operands={"source": source},
            results={"dst": result},
        ),
    )


def avx512_predicate_carrier_to_mask_emits(
    descriptor_lookup: _DescriptorLookup,
    lane_count: int,
    source: ValueRef,
    result: ValueRef,
) -> tuple[EmitDescriptorOp, ...]:
    descriptor = descriptor_lookup(_carrier_to_mask_descriptor_key(lane_count))
    return (
        _op_emit(
            descriptor=descriptor,
            operands={"source": source},
            results={"dst": result},
            result_types={"dst": DescriptorResultType()},
        ),
    )


def _constant_emit(
    descriptor: Descriptor,
    *,
    result: ValueRef,
    result_type: Scalar,
    immediate_name: str,
    immediate: int,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        results={"dst": result},
        result_types={"dst": result_type},
        immediates={immediate_name: immediate},
        form=DescriptorEmitForm.CONST,
    )


def _predicate_extract_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    move_bits = descriptor_lookup("x86.avx512.kmovq.gpr64.k")
    shift = descriptor_lookup("x86.scalar.shr.imm.gpr64")
    truncate = descriptor_lookup("x86.scalar.mov.trunc.gpr32.gpr64")
    mask_bit = descriptor_lookup("x86.scalar.and.imm.gpr32")
    rules: list[DescriptorRule] = []
    for source_type, source_register_class in (
        (_PREDICATE_PATTERN, "x86.k"),
        (_PREDICATE_TYPES[64], "x86.zmm"),
    ):
        lane_count = 64
        source = ValueRef.operand("source")
        emits: list[EmitDescriptorOp] = []
        dependencies = [move_bits, shift, truncate, mask_bit]
        if source_register_class != "x86.k":
            compress = descriptor_lookup(_carrier_to_mask_descriptor_key(lane_count))
            dependencies.append(compress)
            source = ValueRef.temporary("mask")
            emits.extend(
                avx512_predicate_carrier_to_mask_emits(
                    descriptor_lookup,
                    lane_count,
                    ValueRef.operand("source"),
                    source,
                )
            )
        emits.extend(
            (
                _op_emit(
                    descriptor=move_bits,
                    operands={"source": source},
                    results={"dst": ValueRef.temporary("bits")},
                    result_types={"dst": _I64},
                ),
                _op_emit(
                    descriptor=shift,
                    operands={"lhs": ValueRef.temporary("bits")},
                    results={"dst": ValueRef.temporary("shifted")},
                    result_types={"dst": _I64},
                    immediates={
                        "shift": AttrProject.i64_array_element(
                            "static_indices", element=0
                        )
                    },
                ),
                _op_emit(
                    descriptor=truncate,
                    operands={"src": ValueRef.temporary("shifted")},
                    results={"dst": ValueRef.temporary("narrowed")},
                    result_types={"dst": _I32},
                ),
                _op_emit(
                    descriptor=mask_bit,
                    operands={"lhs": ValueRef.temporary("narrowed")},
                    results={"dst": ValueRef.result("result")},
                    immediates={"imm32": 1},
                ),
            )
        )
        rules.append(
            DescriptorRule(
                source_op=vector.vector_extract,
                descriptor=dependencies[-1],
                guards=(
                    Guard.value_type("source", source_type),
                    Guard.value_type("result", _I1),
                    Guard.low_value_register_class("source", source_register_class),
                    Guard.operand_segment_count("indices", 0),
                    Guard.i64_array_count("static_indices", 1),
                    Guard.i64_array_element_range("static_indices", 0, 0, 63),
                    *(
                        Guard.descriptor_available(descriptor)
                        for descriptor in dict.fromkeys(dependencies[:-1])
                    ),
                ),
                emit=tuple(emits),
            )
        )
    return tuple(rules)


def _predicate_insert_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    move_from_mask = descriptor_lookup("x86.avx512.kmovq.gpr64.k")
    move_to_mask = descriptor_lookup("x86.avx512.kmovq.k.gpr64")
    move_immediate = descriptor_lookup("x86.scalar.movimm.gpr64")
    shift = descriptor_lookup("x86.scalar.shl.imm.gpr64")
    xor = descriptor_lookup("x86.scalar.xor.gpr64")
    and_ = descriptor_lookup("x86.scalar.and.gpr64")
    extend = descriptor_lookup("x86.scalar.movzx.gpr64.gpr32")
    or_ = descriptor_lookup("x86.scalar.or.gpr64")
    rules: list[DescriptorRule] = []
    for result_type, register_class in (
        (_PREDICATE_PATTERN, "x86.k"),
        (_PREDICATE_TYPES[64], "x86.zmm"),
    ):
        lane_count = 64
        mask = ValueRef.operand("dest")
        emits: list[EmitDescriptorOp] = []
        dependencies = [
            move_from_mask,
            move_to_mask,
            move_immediate,
            shift,
            xor,
            and_,
            extend,
            or_,
        ]
        if register_class != "x86.k":
            compress = descriptor_lookup(_carrier_to_mask_descriptor_key(lane_count))
            dependencies.append(compress)
            mask = ValueRef.temporary("mask")
            emits.extend(
                avx512_predicate_carrier_to_mask_emits(
                    descriptor_lookup,
                    lane_count,
                    ValueRef.operand("dest"),
                    mask,
                )
            )
        index = AttrProject.i64_array_element("static_indices", element=0)
        emits.extend(
            (
                _op_emit(
                    descriptor=move_from_mask,
                    operands={"source": mask},
                    results={"dst": ValueRef.temporary("old_bits")},
                    result_types={"dst": _I64},
                ),
                _constant_emit(
                    move_immediate,
                    result=ValueRef.temporary("one"),
                    result_type=_I64,
                    immediate_name="imm64",
                    immediate=1,
                ),
                _op_emit(
                    descriptor=shift,
                    operands={"lhs": ValueRef.temporary("one")},
                    results={"dst": ValueRef.temporary("bit")},
                    result_types={"dst": _I64},
                    immediates={"shift": index},
                ),
                _constant_emit(
                    move_immediate,
                    result=ValueRef.temporary("all_bits"),
                    result_type=_I64,
                    immediate_name="imm64",
                    immediate=-1,
                ),
                _op_emit(
                    descriptor=xor,
                    operands={
                        "lhs": ValueRef.temporary("all_bits"),
                        "rhs": ValueRef.temporary("bit"),
                    },
                    results={"dst": ValueRef.temporary("clear_mask")},
                    result_types={"dst": _I64},
                ),
                _op_emit(
                    descriptor=and_,
                    operands={
                        "lhs": ValueRef.temporary("old_bits"),
                        "rhs": ValueRef.temporary("clear_mask"),
                    },
                    results={"dst": ValueRef.temporary("cleared_bits")},
                    result_types={"dst": _I64},
                ),
                _op_emit(
                    descriptor=extend,
                    operands={"src": ValueRef.operand("value")},
                    results={"dst": ValueRef.temporary("value64")},
                    result_types={"dst": _I64},
                ),
                _op_emit(
                    descriptor=shift,
                    operands={"lhs": ValueRef.temporary("value64")},
                    results={"dst": ValueRef.temporary("shifted_value")},
                    result_types={"dst": _I64},
                    immediates={"shift": index},
                ),
                _op_emit(
                    descriptor=or_,
                    operands={
                        "lhs": ValueRef.temporary("cleared_bits"),
                        "rhs": ValueRef.temporary("shifted_value"),
                    },
                    results={"dst": ValueRef.temporary("new_bits")},
                    result_types={"dst": _I64},
                ),
            )
        )
        native_result = (
            ValueRef.result("result")
            if register_class == "x86.k"
            else ValueRef.temporary("new_mask")
        )
        emits.append(
            _op_emit(
                descriptor=move_to_mask,
                operands={"source": ValueRef.temporary("new_bits")},
                results={"dst": native_result},
                result_types=(
                    None
                    if register_class == "x86.k"
                    else {"dst": DescriptorResultType()}
                ),
            )
        )
        primary_descriptor = move_to_mask
        if register_class != "x86.k":
            expand = descriptor_lookup(_mask_to_carrier_descriptor_key(lane_count))
            dependencies.append(expand)
            primary_descriptor = expand
            emits.extend(
                _mask_to_carrier_emit(
                    descriptor_lookup,
                    lane_count,
                    native_result,
                    ValueRef.result("result"),
                )
            )
        rules.append(
            DescriptorRule(
                source_op=vector.vector_insert,
                descriptor=primary_descriptor,
                guards=(
                    Guard.value_type("value", _I1),
                    Guard.value_type("dest", result_type),
                    Guard.value_type("result", result_type),
                    Guard.low_value_register_class("dest", register_class),
                    Guard.low_value_register_class("result", register_class),
                    Guard.operand_segment_count("indices", 0),
                    Guard.i64_array_count("static_indices", 1),
                    Guard.i64_array_element_range("static_indices", 0, 0, 63),
                    *(
                        Guard.descriptor_available(descriptor)
                        for descriptor in dict.fromkeys(dependencies)
                        if descriptor != primary_descriptor
                    ),
                ),
                emit=tuple(emits),
            )
        )
    return tuple(rules)


def _predicate_constant_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    move_bits = descriptor_lookup("x86.scalar.movimm.gpr64")
    move_mask = descriptor_lookup("x86.avx512.kmovq.k.gpr64")
    expand = descriptor_lookup(_mask_to_carrier_descriptor_key(64))
    rules: list[DescriptorRule] = []
    for result_type, register_class in (
        (_PREDICATE_PATTERN, "x86.k"),
        (_PREDICATE_TYPES[64], "x86.zmm"),
    ):
        for value in (0, 1):
            mask_result = (
                ValueRef.result("result")
                if register_class == "x86.k"
                else ValueRef.temporary("mask")
            )
            emits = [
                _constant_emit(
                    move_bits,
                    result=ValueRef.temporary("bits"),
                    result_type=_I64,
                    immediate_name="imm64",
                    immediate=0 if value == 0 else -1,
                ),
                _op_emit(
                    descriptor=move_mask,
                    operands={"source": ValueRef.temporary("bits")},
                    results={"dst": mask_result},
                    result_types=(
                        None
                        if register_class == "x86.k"
                        else {"dst": DescriptorResultType()}
                    ),
                ),
            ]
            if register_class != "x86.k":
                emits.extend(
                    _mask_to_carrier_emit(
                        descriptor_lookup,
                        64,
                        mask_result,
                        ValueRef.result("result"),
                    )
                )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_constant,
                    descriptor=move_mask if register_class == "x86.k" else expand,
                    guards=(
                        Guard.attr_kind("value", "i64"),
                        Guard.i64_range("value", value, value),
                        Guard.value_type("result", result_type),
                        Guard.low_value_register_class("result", register_class),
                        Guard.descriptor_available(move_bits),
                        *(
                            (Guard.descriptor_available(move_mask),)
                            if register_class != "x86.k"
                            else ()
                        ),
                    ),
                    emit=tuple(emits),
                )
            )
    return tuple(rules)


def _predicate_splat_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    zero = descriptor_lookup("x86.scalar.movimm.gpr32")
    subtract = descriptor_lookup("x86.scalar.sub.gpr32")
    extend = descriptor_lookup("x86.scalar.movsxd.gpr64.gpr32")
    move_mask = descriptor_lookup("x86.avx512.kmovq.k.gpr64")
    expand = descriptor_lookup(_mask_to_carrier_descriptor_key(64))
    prefix = (
        _constant_emit(
            zero,
            result=ValueRef.temporary("zero"),
            result_type=_I32,
            immediate_name="imm32",
            immediate=0,
        ),
        _op_emit(
            descriptor=subtract,
            operands={
                "lhs": ValueRef.temporary("zero"),
                "rhs": ValueRef.operand("scalar"),
            },
            results={"dst": ValueRef.temporary("mask32")},
            result_types={"dst": _I32},
        ),
        _op_emit(
            descriptor=extend,
            operands={"src": ValueRef.temporary("mask32")},
            results={"dst": ValueRef.temporary("bits")},
            result_types={"dst": _I64},
        ),
    )
    rules: list[DescriptorRule] = []
    for result_type, register_class in (
        (_PREDICATE_PATTERN, "x86.k"),
        (_PREDICATE_TYPES[64], "x86.zmm"),
    ):
        mask_result = (
            ValueRef.result("result")
            if register_class == "x86.k"
            else ValueRef.temporary("mask")
        )
        emits = [
            *prefix,
            _op_emit(
                descriptor=move_mask,
                operands={"source": ValueRef.temporary("bits")},
                results={"dst": mask_result},
                result_types=(
                    None
                    if register_class == "x86.k"
                    else {"dst": DescriptorResultType()}
                ),
            ),
        ]
        if register_class != "x86.k":
            emits.extend(
                _mask_to_carrier_emit(
                    descriptor_lookup,
                    64,
                    mask_result,
                    ValueRef.result("result"),
                )
            )
        rules.append(
            DescriptorRule(
                source_op=vector.vector_splat,
                descriptor=move_mask if register_class == "x86.k" else expand,
                guards=(
                    Guard.value_type("scalar", _I1),
                    Guard.value_type("result", result_type),
                    Guard.low_value_register_class("result", register_class),
                    Guard.descriptor_available(zero),
                    Guard.descriptor_available(subtract),
                    Guard.descriptor_available(extend),
                    *(
                        (Guard.descriptor_available(move_mask),)
                        if register_class != "x86.k"
                        else ()
                    ),
                ),
                emit=tuple(emits),
            )
        )
    return tuple(rules)


def _comparison_rule(
    descriptor_lookup: _DescriptorLookup,
    *,
    source_op: Op,
    predicates: Sequence[str] | None,
    predicate_immediates: Mapping[str, int],
    operand_type: TypePattern,
    result_type: TypePattern,
    lane_count: int,
    result_register_class: str,
    compare: Descriptor,
    priority: int = 0,
) -> DescriptorRule:
    native_result = (
        ValueRef.result("result")
        if result_register_class == "x86.k"
        else ValueRef.temporary("mask")
    )
    emits = [
        _op_emit(
            descriptor=compare,
            operands={
                "lhs": ValueRef.operand("lhs"),
                "rhs": ValueRef.operand("rhs"),
            },
            results={"dst": native_result},
            result_types=(
                None
                if result_register_class == "x86.k"
                else {"dst": DescriptorResultType()}
            ),
            immediates={
                "predicate": AttrProject.enum_remap("predicate", predicate_immediates)
            },
        )
    ]
    conversion = None
    if result_register_class != "x86.k":
        conversion = descriptor_lookup(_mask_to_carrier_descriptor_key(lane_count))
        emits.extend(
            _mask_to_carrier_emit(
                descriptor_lookup,
                lane_count,
                native_result,
                ValueRef.result("result"),
            )
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=compare,
        guards=(
            *(
                (Guard.enum_attr_in("predicate", predicates),)
                if predicates is not None
                else ()
            ),
            *_typed_guards(("lhs", "rhs"), operand_type),
            Guard.value_type("result", result_type),
            Guard.low_value_register_class("result", result_register_class),
            *(
                (Guard.descriptor_available(conversion),)
                if conversion is not None
                else ()
            ),
        ),
        emit=tuple(emits),
        priority=priority,
    )


def _integer_compare_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for vector_bit_width in (128, 256, 512):
        register_suffix = _REGISTER_SUFFIXES[vector_bit_width]
        for element in INTEGER_ELEMENTS:
            lane_count = element.lane_count(vector_bit_width)
            operand_type = _full_vector_type(element, vector_bit_width)
            result_type = _PREDICATE_TYPES[lane_count]
            for mnemonic, predicates in zip(
                AVX512_INTEGER_COMPARE_MNEMONICS[element.name],
                (_SIGNED_INTEGER_PREDICATES, _UNSIGNED_INTEGER_PREDICATES),
                strict=True,
            ):
                compare = descriptor_lookup(f"x86.avx512.{mnemonic}.{register_suffix}")
                rules.extend(
                    _comparison_rule(
                        descriptor_lookup,
                        source_op=vector.vector_cmpi,
                        predicates=predicates,
                        predicate_immediates=_INTEGER_COMPARE_IMMEDIATES,
                        operand_type=operand_type,
                        result_type=result_type,
                        lane_count=lane_count,
                        result_register_class=result_register_class,
                        compare=compare,
                    )
                    for result_register_class in _operation_representations(
                        lane_count, vector_bit_width
                    )
                )
    return tuple(rules)


def _float_compare_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for vector_bit_width in (128, 256, 512):
        register_suffix = _REGISTER_SUFFIXES[vector_bit_width]
        for element in FLOAT_ELEMENTS:
            lane_count = element.lane_count(vector_bit_width)
            operand_type = _full_vector_type(element, vector_bit_width)
            result_type = _PREDICATE_TYPES[lane_count]
            compare = descriptor_lookup(
                f"x86.avx512.{AVX512_FLOAT_COMPARE_MNEMONICS[element.name]}."
                f"{register_suffix}"
            )
            rules.extend(
                _comparison_rule(
                    descriptor_lookup,
                    source_op=vector.vector_cmpf,
                    predicates=None,
                    predicate_immediates=_FLOAT_COMPARE_IMMEDIATES,
                    operand_type=operand_type,
                    result_type=result_type,
                    lane_count=lane_count,
                    result_register_class=result_register_class,
                    compare=compare,
                )
                for result_register_class in _operation_representations(
                    lane_count, vector_bit_width
                )
            )
    return tuple(rules)


def avx512_fp16_compare_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    """Compares packed f16 values into native and full-register predicates."""
    rules: list[DescriptorRule] = []
    for vector_bit_width in AVX512_FP16_VECTOR_BIT_WIDTHS:
        lane_count = FP16_ELEMENT.lane_count(vector_bit_width)
        operand_type = _full_vector_type(FP16_ELEMENT, vector_bit_width)
        result_type = _PREDICATE_TYPES[lane_count]
        compare = descriptor_lookup(
            f"x86.avx512_fp16.{AVX512_FP16_FLOAT_COMPARE_MNEMONIC}."
            f"{_REGISTER_SUFFIXES[vector_bit_width]}"
        )
        rules.extend(
            _comparison_rule(
                descriptor_lookup,
                source_op=vector.vector_cmpf,
                predicates=None,
                predicate_immediates=_FLOAT_COMPARE_IMMEDIATES,
                operand_type=operand_type,
                result_type=result_type,
                lane_count=lane_count,
                result_register_class=result_register_class,
                compare=compare,
                priority=1,
            )
            for result_register_class in _all_operation_representations(lane_count)
        )
    return tuple(rules)


def avx512_fp16_scalar_compare_rule(
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    """Compares one f16 lane and returns its predicate in a scalar GPR."""
    compare = descriptor_lookup(
        f"x86.avx512_fp16.{AVX512_FP16_SCALAR_FLOAT_COMPARE_MNEMONIC}.xmm"
    )
    move = descriptor_lookup("x86.avx512.kmovq.gpr64.k")
    truncate = descriptor_lookup("x86.scalar.mov.trunc.gpr32.gpr64")
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpf,
        descriptor=compare,
        guards=(
            *_typed_guards(("lhs", "rhs"), Scalar(FP16_ELEMENT.name)),
            Guard.value_type("result", _I1),
            Guard.descriptor_available(move),
            Guard.descriptor_available(truncate),
        ),
        emit=(
            _op_emit(
                descriptor=compare,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("mask")},
                result_types={"dst": DescriptorResultType()},
                immediates={
                    "predicate": AttrProject.enum_remap(
                        "predicate", _FLOAT_COMPARE_IMMEDIATES
                    )
                },
            ),
            _op_emit(
                descriptor=move,
                operands={"source": ValueRef.temporary("mask")},
                results={"dst": ValueRef.temporary("bits")},
                result_types={"dst": _I64},
            ),
            _op_emit(
                descriptor=truncate,
                operands={"src": ValueRef.temporary("bits")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
        priority=1,
    )


def _select_rule(
    descriptor_lookup: _DescriptorLookup,
    *,
    lane_count: int,
    value_type: TypePattern,
    blend: Descriptor,
    condition_register_class: str,
    priority: int = 0,
) -> DescriptorRule:
    mask = (
        ValueRef.operand("condition")
        if condition_register_class == "x86.k"
        else ValueRef.temporary("mask")
    )
    conversion = None
    emits = []
    if condition_register_class != "x86.k":
        conversion = descriptor_lookup(_carrier_to_mask_descriptor_key(lane_count))
        emits.extend(
            avx512_predicate_carrier_to_mask_emits(
                descriptor_lookup,
                lane_count,
                ValueRef.operand("condition"),
                mask,
            )
        )
    emits.append(
        _op_emit(
            descriptor=blend,
            operands={
                "mask": mask,
                "true_value": ValueRef.operand("true_value"),
                "false_value": ValueRef.operand("false_value"),
            },
            results={"dst": ValueRef.result("result")},
        )
    )
    return DescriptorRule(
        source_op=vector.vector_select,
        descriptor=blend,
        guards=(
            Guard.value_type("condition", _PREDICATE_TYPES[lane_count]),
            *_typed_guards(("true_value", "false_value", "result"), value_type),
            Guard.low_value_register_class("condition", condition_register_class),
            *((Guard.descriptor_available(conversion),) if conversion else ()),
        ),
        emit=tuple(emits),
        priority=priority,
    )


def _select_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for vector_bit_width in (128, 256, 512):
        register_suffix = _REGISTER_SUFFIXES[vector_bit_width]
        for element_names, element_bit_width, mnemonic in _SELECT_GROUPS:
            lane_count = vector_bit_width // element_bit_width
            value_type = Vector(element_names, lanes=lane_count)
            blend = descriptor_lookup(f"x86.avx512.{mnemonic}.{register_suffix}")
            rules.extend(
                _select_rule(
                    descriptor_lookup,
                    lane_count=lane_count,
                    value_type=value_type,
                    blend=blend,
                    condition_register_class=condition_register_class,
                )
                for condition_register_class in _operation_representations(
                    lane_count, vector_bit_width
                )
            )
    return tuple(rules)


def avx512_fp16_select_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    """Selects four-lane FP16 values in the low half of XMM."""
    lane_count = 4
    value_type = Vector(FP16_ELEMENT.name, lanes=lane_count)
    blend = descriptor_lookup("x86.avx512.vpblendmw.xmm")
    return tuple(
        _select_rule(
            descriptor_lookup,
            lane_count=lane_count,
            value_type=value_type,
            blend=blend,
            condition_register_class=condition_register_class,
            priority=1,
        )
        for condition_register_class in _all_operation_representations(lane_count)
    )


def _predicate_bitwise_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for source_operation, _, _ in AVX512_BITWISE_FAMILIES:
        descriptor = descriptor_lookup(
            {
                "andi": "x86.avx512.kandq",
                "ori": "x86.avx512.korq",
                "xori": "x86.avx512.kxorq",
            }[source_operation]
        )
        rules.append(
            DescriptorRule(
                source_op=_BITWISE_SOURCE_OPS[source_operation],
                descriptor=descriptor,
                guards=(
                    *_typed_guards(
                        ("lhs", "rhs", "result"),
                        Vector("i1", minimum_lanes=2, maximum_lanes=64),
                    ),
                    *(
                        Guard.low_value_register_class(field, "x86.k")
                        for field in ("lhs", "rhs", "result")
                    ),
                ),
                emit=(
                    _op_emit(
                        descriptor=descriptor,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        )
    for source_operation, mnemonic, _ in AVX512_BITWISE_FAMILIES:
        descriptor = descriptor_lookup(f"x86.avx512.{mnemonic}.zmm")
        rules.append(
            DescriptorRule(
                source_op=_BITWISE_SOURCE_OPS[source_operation],
                descriptor=descriptor,
                guards=(
                    *_typed_guards(("lhs", "rhs", "result"), Vector("i1", lanes=64)),
                    *(
                        Guard.low_value_register_class(field, "x86.zmm")
                        for field in ("lhs", "rhs", "result")
                    ),
                ),
                emit=(
                    _op_emit(
                        descriptor=descriptor,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        )
    return tuple(rules)


def avx512_predicate_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[ContractCase, ...]:
    return (
        *_predicate_constant_rules(descriptor_lookup),
        *_predicate_splat_rules(descriptor_lookup),
        *_predicate_extract_rules(descriptor_lookup),
        *_predicate_insert_rules(descriptor_lookup),
        *_integer_compare_rules(descriptor_lookup),
        *_float_compare_rules(descriptor_lookup),
        *_select_rules(descriptor_lookup),
        *_predicate_bitwise_rules(descriptor_lookup),
    )
