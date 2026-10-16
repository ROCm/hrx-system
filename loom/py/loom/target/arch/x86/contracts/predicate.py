# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AVX2 predicate representation and operation contract rules."""

from __future__ import annotations

from collections.abc import Sequence

from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
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
from loom.target.arch.x86.contracts.vector_integer_conversion import (
    avx2_signed_integer_conversion_emits as _signed_integer_conversion_emits,
)
from loom.target.arch.x86.contracts.vector_integer_conversion import (
    unsigned_integer_mask_normalization_emits as _unsigned_mask_normalization_emits,
)
from loom.target.arch.x86.vector_families import (
    AVX2_BITWISE_FAMILIES,
    AVX2_FLOAT_COMPARE_MNEMONICS,
    AVX2_INTEGER_COMPARE_MNEMONICS,
    AVX2_PAYLOAD_ELEMENT_NAMES,
    AVX2_VECTOR_BIT_WIDTHS,
    FLOAT_ELEMENTS,
    INTEGER_ELEMENTS,
    STORAGE_ELEMENTS,
    X86_FLOAT_COMPARE_IMMEDIATES,
    X86_LANE_FAMILIES,
    VectorElement,
)
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    ContractEmit,
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
_V2I1 = Vector("i1", lanes=2)
_V4I1 = Vector("i1", lanes=4)
_V8I1 = Vector("i1", lanes=8)
_V16I1 = Vector("i1", lanes=16)
_V32I1 = Vector("i1", lanes=32)
_PAYLOAD_VECTOR_TYPES = Vector(
    (*AVX2_PAYLOAD_ELEMENT_NAMES, "i1"),
    minimum_lanes=1,
    maximum_lanes=32,
)
_PAYLOAD_ELEMENT_NAMES_BY_BIT_WIDTH = {
    bit_width: tuple(
        element.name
        for element in (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS, *FLOAT_ELEMENTS)
        if element.bit_width == bit_width
    )
    for bit_width in (8, 16, 32, 64)
}
_REGISTER_SUFFIXES = {
    16: "xmm",
    32: "xmm",
    64: "xmm",
    128: "xmm",
    256: "ymm",
}
_REGISTER_CLASSES = {
    16: "x86.xmm",
    32: "x86.xmm",
    64: "x86.xmm",
    128: "x86.xmm",
    256: "x86.ymm",
}
_PREDICATE_SIMD_BIT_WIDTHS = (16, 32, 64, 128, 256)
_PACKED_PREDICATE_CONVERSION_BIT_WIDTHS = (64, 128, 256)
_INTEGER_BROADCAST_MNEMONICS = {
    8: "vpbroadcastb",
    16: "vpbroadcastw",
    32: "vpbroadcastd",
    64: "vpbroadcastq",
}
_INTEGER_ELEMENT_SUFFIXES = {8: "b", 16: "w", 32: "d", 64: "q"}
_INTEGER_SHIFT_LEFT_MNEMONICS = {
    8: "vpsllw",
    16: "vpsllw",
    32: "vpslld",
    64: "vpsllq",
}
_PREDICATE_TYPES = {
    2: _V2I1,
    4: _V4I1,
    8: _V8I1,
    16: _V16I1,
    32: _V32I1,
}
_PREDICATE_BITWISE_SOURCE_OPS = {
    "andi": vector.vector_andi,
    "ori": vector.vector_ori,
    "xori": vector.vector_xori,
}

_PREDICATE_CARRIER_TYPES = {
    (vector_bit_width, lane_count): Vector(
        f"i{vector_bit_width // lane_count}", lanes=lane_count
    )
    for lane_count in _PREDICATE_TYPES
    for vector_bit_width in _PREDICATE_SIMD_BIT_WIDTHS
    if vector_bit_width % lane_count == 0
    and vector_bit_width // lane_count in _INTEGER_ELEMENT_SUFFIXES
}

_PREDICATE_HALF_TYPES = {
    4: Vector("i64", lanes=2),
    8: Vector("i32", lanes=4),
    16: Vector("i16", lanes=8),
}
_PREDICATE_XMM_LANE_TYPES = {
    8: Vector("i8", lanes=16),
    16: Vector("i16", lanes=8),
    32: Vector("i32", lanes=4),
    64: Vector("i64", lanes=2),
}


def _predicate_carrier_element_bit_width(
    lane_count: int, representation_bit_width: int
) -> int:
    return representation_bit_width // lane_count


def avx2_predicate_representation_id(
    lane_count: int, representation_bit_width: int
) -> int:
    """Returns the physical bits per logical AVX2 predicate lane."""
    return _predicate_carrier_element_bit_width(lane_count, representation_bit_width)


def avx2_predicate_register_class(representation_bit_width: int) -> str:
    """Returns the register class carrying an AVX2 predicate representation."""
    return _REGISTER_CLASSES[representation_bit_width]


def _predicate_representation_guards(
    field: str, lane_count: int, representation_bit_width: int
) -> tuple[Guard, Guard]:
    return (
        Guard.low_value_register_class(
            field, avx2_predicate_register_class(representation_bit_width)
        ),
        Guard.low_value_representation(
            field,
            avx2_predicate_representation_id(lane_count, representation_bit_width),
        ),
    )


def _scalar_i1_mask_emits(
    descriptor_lookup: _DescriptorLookup,
    *,
    scalar: ValueRef,
    element_bit_width: int,
    register_suffix: str,
    result: ValueRef,
    result_type: TypePattern | None,
) -> tuple[tuple[EmitDescriptorOp, ...], tuple[Descriptor, ...]]:
    zero = descriptor_lookup("x86.scalar.movimm.gpr32")
    subtract = descriptor_lookup("x86.scalar.sub.gpr32")
    broadcast = descriptor_lookup(
        f"x86.avx2.{_INTEGER_BROADCAST_MNEMONICS[element_bit_width]}.{register_suffix}"
    )
    emits = [
        EmitDescriptorOp(
            descriptor=zero,
            results={"dst": ValueRef.temporary("zero")},
            result_types={"dst": _I32},
            immediates={"imm32": 0},
            form=DescriptorEmitForm.CONST,
        ),
        _op_emit(
            descriptor=subtract,
            operands={
                "lhs": ValueRef.temporary("zero"),
                "rhs": scalar,
            },
            results={"dst": ValueRef.temporary("mask32")},
            result_types={"dst": _I32},
        ),
    ]
    dependencies = [zero, subtract]
    if element_bit_width == 64:
        extend = descriptor_lookup("x86.scalar.movsxd.gpr64.gpr32")
        move = descriptor_lookup("x86.avx2.vmovq.xmm.gpr64")
        emits.extend(
            (
                _op_emit(
                    descriptor=extend,
                    operands={"src": ValueRef.temporary("mask32")},
                    results={"dst": ValueRef.temporary("mask64")},
                    result_types={"dst": _I64},
                ),
                _op_emit(
                    descriptor=move,
                    operands={"input": ValueRef.temporary("mask64")},
                    results={"dst": ValueRef.temporary("lane")},
                    result_types={"dst": _PREDICATE_XMM_LANE_TYPES[64]},
                ),
            )
        )
        dependencies.extend((extend, move))
    else:
        move = descriptor_lookup("x86.avx2.vmovd.xmm.gpr32")
        emits.append(
            _op_emit(
                descriptor=move,
                operands={"input": ValueRef.temporary("mask32")},
                results={"dst": ValueRef.temporary("lane")},
                result_types={"dst": _PREDICATE_XMM_LANE_TYPES[element_bit_width]},
            )
        )
        dependencies.append(move)
    emits.append(
        _op_emit(
            descriptor=broadcast,
            operands={"value": ValueRef.temporary("lane")},
            results={"dst": result},
            result_types=None if result_type is None else {"dst": result_type},
        )
    )
    return tuple(emits), tuple((*dependencies, broadcast))


def _predicate_constant_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for lane_count, result_type in _PREDICATE_TYPES.items():
        for representation_bit_width in avx2_predicate_representation_bit_widths(
            lane_count
        ):
            carrier_type = _PREDICATE_CARRIER_TYPES[
                (representation_bit_width, lane_count)
            ]
            register_suffix = _REGISTER_SUFFIXES[representation_bit_width]
            zero = descriptor_lookup(f"x86.avx2.vxorps.zero.{register_suffix}")
            element_bit_width = _predicate_carrier_element_bit_width(
                lane_count, representation_bit_width
            )
            equal = descriptor_lookup(
                f"x86.avx2.{AVX2_INTEGER_COMPARE_MNEMONICS[f'i{element_bit_width}'][0]}."
                f"{register_suffix}"
            )
            common_guards = (
                Guard.value_type("result", result_type),
                *_predicate_representation_guards(
                    "result", lane_count, representation_bit_width
                ),
            )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_constant,
                    descriptor=zero,
                    guards=(
                        Guard.attr_kind("value", "i64"),
                        Guard.i64_range("value", 0, 0),
                        *common_guards,
                    ),
                    emit=(
                        _op_emit(
                            descriptor=zero,
                            results={"dst": ValueRef.result("result")},
                        ),
                    ),
                    priority=1,
                )
            )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_constant,
                    descriptor=equal,
                    guards=(
                        Guard.attr_kind("value", "i64"),
                        Guard.i64_range("value", 1, 1),
                        *common_guards,
                        Guard.descriptor_available(zero),
                    ),
                    emit=(
                        _op_emit(
                            descriptor=zero,
                            results={"dst": ValueRef.temporary("zero")},
                            result_types={"dst": carrier_type},
                        ),
                        _op_emit(
                            descriptor=equal,
                            operands={
                                "lhs": ValueRef.temporary("zero"),
                                "rhs": ValueRef.temporary("zero"),
                            },
                            results={"dst": ValueRef.result("result")},
                        ),
                    ),
                    priority=1,
                )
            )
    return tuple(rules)


def _predicate_splat_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for lane_count, result_type in _PREDICATE_TYPES.items():
        for representation_bit_width in avx2_predicate_representation_bit_widths(
            lane_count
        ):
            element_bit_width = _predicate_carrier_element_bit_width(
                lane_count, representation_bit_width
            )
            register_suffix = _REGISTER_SUFFIXES[representation_bit_width]
            emits, descriptors = _scalar_i1_mask_emits(
                descriptor_lookup,
                scalar=ValueRef.operand("scalar"),
                element_bit_width=element_bit_width,
                register_suffix=register_suffix,
                result=ValueRef.result("result"),
                result_type=None,
            )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_splat,
                    descriptor=descriptors[-1],
                    guards=(
                        Guard.value_type("scalar", _I1),
                        Guard.value_type("result", result_type),
                        *_predicate_representation_guards(
                            "result", lane_count, representation_bit_width
                        ),
                        *(
                            Guard.descriptor_available(descriptor)
                            for descriptor in descriptors[:-1]
                        ),
                    ),
                    emit=emits,
                )
            )
    return tuple(rules)


def _predicate_conversion_descriptor_keys(
    lane_count: int,
    source_bit_width: int,
    result_bit_width: int,
) -> tuple[str, ...]:
    if source_bit_width == 64 and result_bit_width == 128:
        return {
            2: ("x86.avx2.vpmovsxdq.xmm.xmm",),
            4: ("x86.avx2.vpmovsxwd.xmm.xmm",),
            8: ("x86.avx2.vpmovsxbw.xmm.xmm",),
        }[lane_count]
    if source_bit_width == 128 and result_bit_width == 64:
        return (
            {
                2: "x86.avx2.vshufps.xmm",
                4: "x86.avx2.vpackssdw.xmm",
                8: "x86.avx2.vpacksswb.xmm",
            }[lane_count],
        )
    if source_bit_width == 128 and result_bit_width == 256:
        return {
            4: ("x86.avx2.vpmovsxdq.ymm.xmm",),
            8: ("x86.avx2.vpmovsxwd.ymm.xmm",),
            16: ("x86.avx2.vpmovsxbw.ymm.xmm",),
        }[lane_count]
    if source_bit_width == 256 and result_bit_width == 128:
        return (
            "x86.avx2.vextractf128.xmm.ymm",
            {
                4: "x86.avx2.vshufps.xmm",
                8: "x86.avx2.vpackssdw.xmm",
                16: "x86.avx2.vpacksswb.xmm",
            }[lane_count],
        )
    if source_bit_width == 64 and result_bit_width == 256:
        return (
            *_predicate_conversion_descriptor_keys(lane_count, 64, 128),
            *_predicate_conversion_descriptor_keys(lane_count, 128, 256),
        )
    if source_bit_width == 256 and result_bit_width == 64:
        return (
            *_predicate_conversion_descriptor_keys(lane_count, 256, 128),
            *_predicate_conversion_descriptor_keys(lane_count, 128, 64),
        )
    raise ValueError(
        f"unsupported AVX2 predicate conversion v{source_bit_width} to "
        f"v{result_bit_width} with {lane_count} lanes"
    )


def _predicate_conversion_emits(
    *,
    lane_count: int,
    source_bit_width: int,
    result_bit_width: int,
    source: ValueRef,
    result: ValueRef,
    result_type: TypePattern | None,
    descriptor_lookup: _DescriptorLookup,
    temporary_prefix: str,
) -> tuple[EmitDescriptorOp, ...]:
    if source_bit_width == 64 and result_bit_width == 256:
        middle = ValueRef.temporary(f"{temporary_prefix}_xmm")
        return (
            *_predicate_conversion_emits(
                lane_count=lane_count,
                source_bit_width=64,
                result_bit_width=128,
                source=source,
                result=middle,
                result_type=_PREDICATE_CARRIER_TYPES[(128, lane_count)],
                descriptor_lookup=descriptor_lookup,
                temporary_prefix=f"{temporary_prefix}_widen",
            ),
            *_predicate_conversion_emits(
                lane_count=lane_count,
                source_bit_width=128,
                result_bit_width=256,
                source=middle,
                result=result,
                result_type=result_type,
                descriptor_lookup=descriptor_lookup,
                temporary_prefix=f"{temporary_prefix}_finish",
            ),
        )
    if source_bit_width == 256 and result_bit_width == 64:
        middle = ValueRef.temporary(f"{temporary_prefix}_xmm")
        return (
            *_predicate_conversion_emits(
                lane_count=lane_count,
                source_bit_width=256,
                result_bit_width=128,
                source=source,
                result=middle,
                result_type=_PREDICATE_CARRIER_TYPES[(128, lane_count)],
                descriptor_lookup=descriptor_lookup,
                temporary_prefix=f"{temporary_prefix}_narrow",
            ),
            *_predicate_conversion_emits(
                lane_count=lane_count,
                source_bit_width=128,
                result_bit_width=64,
                source=middle,
                result=result,
                result_type=result_type,
                descriptor_lookup=descriptor_lookup,
                temporary_prefix=f"{temporary_prefix}_finish",
            ),
        )
    descriptor_keys = _predicate_conversion_descriptor_keys(
        lane_count, source_bit_width, result_bit_width
    )
    if (source_bit_width, result_bit_width) in ((64, 128), (128, 256)):
        descriptor = descriptor_lookup(descriptor_keys[0])
        return (
            _op_emit(
                descriptor=descriptor,
                operands={"source": source},
                results={"dst": result},
                result_types=None if result_type is None else {"dst": result_type},
            ),
        )

    if source_bit_width == 128 and result_bit_width == 64:
        pack = descriptor_lookup(descriptor_keys[0])
        return (
            _op_emit(
                descriptor=pack,
                operands={"low": source, "high": source},
                results={"dst": result},
                result_types=None if result_type is None else {"dst": result_type},
                immediates={"control": 0x88} if lane_count == 2 else None,
            ),
        )

    extract = descriptor_lookup(descriptor_keys[0])
    pack = descriptor_lookup(descriptor_keys[1])
    half_type = _PREDICATE_HALF_TYPES[lane_count]
    low = ValueRef.temporary(f"{temporary_prefix}_low")
    high = ValueRef.temporary(f"{temporary_prefix}_high")
    pack_immediates = {"control": 0x88} if lane_count == 4 else None
    return (
        _op_emit(
            descriptor=extract,
            operands={"source": source},
            results={"dst": low},
            result_types={"dst": half_type},
            immediates={"lane": 0},
        ),
        _op_emit(
            descriptor=extract,
            operands={"source": source},
            results={"dst": high},
            result_types={"dst": half_type},
            immediates={"lane": 1},
        ),
        _op_emit(
            descriptor=pack,
            operands={"low": low, "high": high},
            results={"dst": result},
            result_types=None if result_type is None else {"dst": result_type},
            immediates=pack_immediates,
        ),
    )


def _render_binary_compare_program(
    operations: Sequence[tuple[Descriptor, ValueRef, ValueRef, str]],
    *,
    native_result: ValueRef,
    native_result_type: TypePattern | None,
    temporary_type: TypePattern,
) -> tuple[EmitDescriptorOp, ...]:
    emit_ops: list[EmitDescriptorOp] = []
    for index, (descriptor, lhs, rhs, name) in enumerate(operations):
        is_last = index + 1 == len(operations)
        result = native_result if is_last else ValueRef.temporary(name)
        result_type = native_result_type if is_last else temporary_type
        emit_ops.append(
            _op_emit(
                descriptor=descriptor,
                operands={"lhs": lhs, "rhs": rhs},
                results={"dst": result},
                result_types=None if result_type is None else {"dst": result_type},
            )
        )
    return tuple(emit_ops)


def _integer_compare_program(
    predicate: str,
    element: VectorElement,
    vector_bit_width: int,
    descriptor_lookup: _DescriptorLookup,
    *,
    native_result: ValueRef,
    native_result_type: TypePattern | None,
) -> tuple[tuple[Descriptor, ...], tuple[EmitDescriptorOp, ...]]:
    instruction_bit_width = max(128, vector_bit_width)
    register_suffix = _REGISTER_SUFFIXES[instruction_bit_width]
    operand_type = _full_vector_type(element, vector_bit_width)
    equal_mnemonic, greater_mnemonic = AVX2_INTEGER_COMPARE_MNEMONICS[element.name]
    equal = descriptor_lookup(f"x86.avx2.{equal_mnemonic}.{register_suffix}")
    greater = descriptor_lookup(f"x86.avx2.{greater_mnemonic}.{register_suffix}")
    xor = descriptor_lookup(f"x86.avx2.vpxor.{register_suffix}")
    and_ = descriptor_lookup(f"x86.avx2.vpand.{register_suffix}")
    lhs = ValueRef.operand("lhs")
    rhs = ValueRef.operand("rhs")
    operations: list[tuple[Descriptor, ValueRef, ValueRef, str]] = []

    if predicate == "eq":
        operations.append((equal, lhs, rhs, "equal"))
    elif predicate == "ne":
        operations.extend(
            (
                (equal, lhs, rhs, "equal"),
                (equal, lhs, lhs, "all_ones"),
                (
                    xor,
                    ValueRef.temporary("equal"),
                    ValueRef.temporary("all_ones"),
                    "not_equal",
                ),
            )
        )
    elif predicate in ("slt", "sle", "sgt", "sge"):
        swapped = predicate in ("slt", "sge")
        invert = predicate in ("sle", "sge")
        compare_lhs, compare_rhs = (rhs, lhs) if swapped else (lhs, rhs)
        operations.append((greater, compare_lhs, compare_rhs, "ordered"))
        if invert:
            operations.extend(
                (
                    (equal, lhs, lhs, "all_ones"),
                    (
                        xor,
                        ValueRef.temporary("ordered"),
                        ValueRef.temporary("all_ones"),
                        "ordered_or_equal",
                    ),
                )
            )
    elif element.name != "i64":
        minimum = descriptor_lookup(
            f"x86.avx2."
            f"{dict(ult='vpminu', ule='vpminu', ugt='vpminu', uge='vpminu')[predicate]}"
            f"{dict(i8='b', i16='w', i32='d')[element.name]}.{register_suffix}"
        )
        compare_operand = lhs if predicate in ("ult", "ule") else rhs
        operations.extend(
            (
                (minimum, lhs, rhs, "minimum"),
                (equal, ValueRef.temporary("minimum"), compare_operand, "inclusive"),
            )
        )
        if predicate in ("ult", "ugt"):
            operations.extend(
                (
                    (equal, lhs, rhs, "equal"),
                    (equal, lhs, lhs, "all_ones"),
                    (
                        xor,
                        ValueRef.temporary("equal"),
                        ValueRef.temporary("all_ones"),
                        "not_equal",
                    ),
                    (
                        and_,
                        ValueRef.temporary("inclusive"),
                        ValueRef.temporary("not_equal"),
                        "strict",
                    ),
                )
            )
    else:
        shift = descriptor_lookup(f"x86.avx2.vpsllq.{register_suffix}")
        operations.extend(((equal, lhs, lhs, "all_ones"),))
        all_ones = ValueRef.temporary("all_ones")
        sign_bit = ValueRef.temporary("sign_bit")
        biased_lhs = ValueRef.temporary("biased_lhs")
        biased_rhs = ValueRef.temporary("biased_rhs")
        prefix_emits = (
            *_render_binary_compare_program(
                operations,
                native_result=all_ones,
                native_result_type=operand_type,
                temporary_type=operand_type,
            ),
            _op_emit(
                descriptor=shift,
                operands={"source": all_ones},
                results={"dst": sign_bit},
                result_types={"dst": operand_type},
                immediates={"shift": 63},
            ),
            _op_emit(
                descriptor=xor,
                operands={"lhs": lhs, "rhs": sign_bit},
                results={"dst": biased_lhs},
                result_types={"dst": operand_type},
            ),
            _op_emit(
                descriptor=xor,
                operands={"lhs": rhs, "rhs": sign_bit},
                results={"dst": biased_rhs},
                result_types={"dst": operand_type},
            ),
        )
        swapped = predicate in ("ult", "uge")
        invert = predicate in ("ule", "uge")
        compare_lhs, compare_rhs = (
            (biased_rhs, biased_lhs) if swapped else (biased_lhs, biased_rhs)
        )
        suffix_operations = [(greater, compare_lhs, compare_rhs, "unsigned_ordered")]
        if invert:
            suffix_operations.append(
                (
                    xor,
                    ValueRef.temporary("unsigned_ordered"),
                    all_ones,
                    "unsigned_ordered_or_equal",
                )
            )
        suffix_emits = _render_binary_compare_program(
            suffix_operations,
            native_result=native_result,
            native_result_type=native_result_type,
            temporary_type=operand_type,
        )
        descriptors = tuple(dict.fromkeys((equal, greater, xor, and_, shift)))
        return descriptors, (*prefix_emits, *suffix_emits)

    descriptors = tuple(dict.fromkeys(operation[0] for operation in operations))
    return descriptors, _render_binary_compare_program(
        operations,
        native_result=native_result,
        native_result_type=native_result_type,
        temporary_type=operand_type,
    )


def avx2_predicate_representation_bit_widths(lane_count: int) -> tuple[int, ...]:
    """Returns every legal SIMD representation width for an AVX2 predicate."""
    return tuple(
        bit_width
        for bit_width in _PREDICATE_SIMD_BIT_WIDTHS
        if bit_width % lane_count == 0
        and bit_width // lane_count in _PREDICATE_XMM_LANE_TYPES
    )


def _signed_predicate_mask_conversion_emits(
    descriptor_lookup: _DescriptorLookup,
    *,
    source_element_bit_width: int,
    result_element_bit_width: int,
    lane_count: int,
    source: ValueRef,
    result: ValueRef,
    result_type: TypePattern | None,
    temporary_prefix: str,
) -> tuple[tuple[ContractEmit, ...], tuple[Descriptor, ...]]:
    source_vector_bit_width = lane_count * source_element_bit_width
    result_vector_bit_width = lane_count * result_element_bit_width
    if (
        source_element_bit_width > result_element_bit_width
        and source_vector_bit_width in _PACKED_PREDICATE_CONVERSION_BIT_WIDTHS
        and result_vector_bit_width in _PACKED_PREDICATE_CONVERSION_BIT_WIDTHS
    ):
        descriptor_keys = _predicate_conversion_descriptor_keys(
            lane_count, source_vector_bit_width, result_vector_bit_width
        )
        return (
            _predicate_conversion_emits(
                lane_count=lane_count,
                source_bit_width=source_vector_bit_width,
                result_bit_width=result_vector_bit_width,
                source=source,
                result=result,
                result_type=result_type,
                descriptor_lookup=descriptor_lookup,
                temporary_prefix=temporary_prefix,
            ),
            tuple(descriptor_lookup(key) for key in descriptor_keys),
        )
    return _signed_integer_conversion_emits(
        descriptor_lookup,
        source_element_bit_width=source_element_bit_width,
        result_element_bit_width=result_element_bit_width,
        lane_count=lane_count,
        source=source,
        result=result,
        result_type=result_type,
        temporary_prefix=temporary_prefix,
    )


def _predicate_extension_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for lane_count, source_type in _PREDICATE_TYPES.items():
        for source_bit_width in avx2_predicate_representation_bit_widths(lane_count):
            source_element_bit_width = _predicate_carrier_element_bit_width(
                lane_count, source_bit_width
            )
            for result_element in INTEGER_ELEMENTS:
                result_vector_bit_width = lane_count * result_element.bit_width
                if result_vector_bit_width > 256:
                    continue
                physical_result_bit_width = max(128, result_vector_bit_width)
                result_register_class = _REGISTER_CLASSES[physical_result_bit_width]
                result_type = Vector(result_element.name, lanes=lane_count)
                for source_op, signed in (
                    (vector.vector_extsi, True),
                    (vector.vector_extui, False),
                ):
                    conversion_result = ValueRef.result("result")
                    conversion_result_type: TypePattern | None = None
                    if not signed:
                        conversion_result = ValueRef.temporary("signed_mask")
                        conversion_result_type = result_type
                    conversion_emits, conversion_descriptors = (
                        _signed_predicate_mask_conversion_emits(
                            descriptor_lookup,
                            source_element_bit_width=source_element_bit_width,
                            result_element_bit_width=result_element.bit_width,
                            lane_count=lane_count,
                            source=ValueRef.operand("input"),
                            result=conversion_result,
                            result_type=conversion_result_type,
                            temporary_prefix="predicate_extend",
                        )
                    )
                    normalization_emits: tuple[EmitDescriptorOp, ...] = ()
                    normalization_descriptors: tuple[Descriptor, ...] = ()
                    if not signed:
                        normalization_emits, normalization_descriptors = (
                            _unsigned_mask_normalization_emits(
                                descriptor_lookup,
                                descriptor_key_prefix="x86.avx2",
                                element_bit_width=result_element.bit_width,
                                vector_bit_width=physical_result_bit_width,
                                source=conversion_result,
                                result=ValueRef.result("result"),
                                result_type=None,
                                temporary_type=result_type,
                                temporary_prefix="predicate_extend",
                            )
                        )
                    descriptors = tuple(
                        dict.fromkeys(
                            (*conversion_descriptors, *normalization_descriptors)
                        )
                    )
                    primary_descriptor = (
                        conversion_descriptors[0]
                        if conversion_descriptors
                        else (normalization_descriptors[-1] if descriptors else None)
                    )
                    rules.append(
                        DescriptorRule(
                            source_op=source_op,
                            descriptor=primary_descriptor,
                            guards=(
                                Guard.value_type("input", source_type),
                                Guard.value_type("result", result_type),
                                Guard.value_static_element_count_eq("input", "result"),
                                *_predicate_representation_guards(
                                    "input", lane_count, source_bit_width
                                ),
                                Guard.low_value_register_class(
                                    "result", result_register_class
                                ),
                                *(
                                    Guard.descriptor_available(descriptor)
                                    for descriptor in descriptors
                                    if descriptor != primary_descriptor
                                ),
                            ),
                            emit=(*conversion_emits, *normalization_emits),
                            report_key=(
                                "x86.predicate_extend.signed"
                                if signed
                                else "x86.predicate_extend.unsigned"
                            ),
                        )
                    )
    return tuple(rules)


def _predicate_truncation_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for lane_count, result_type in _PREDICATE_TYPES.items():
        for source_element in INTEGER_ELEMENTS:
            source_vector_bit_width = lane_count * source_element.bit_width
            if source_vector_bit_width > 256:
                continue
            physical_source_bit_width = max(128, source_vector_bit_width)
            source_register_class = _REGISTER_CLASSES[physical_source_bit_width]
            source_register_suffix = _REGISTER_SUFFIXES[physical_source_bit_width]
            source_type = Vector(source_element.name, lanes=lane_count)
            shift = descriptor_lookup(
                f"x86.avx2.{_INTEGER_SHIFT_LEFT_MNEMONICS[source_element.bit_width]}."
                f"{source_register_suffix}"
            )
            zero = descriptor_lookup(f"x86.avx2.vxorps.zero.{source_register_suffix}")
            compare = descriptor_lookup(
                f"x86.avx2."
                f"{AVX2_INTEGER_COMPARE_MNEMONICS[source_element.name][1]}."
                f"{source_register_suffix}"
            )
            for result_bit_width in avx2_predicate_representation_bit_widths(
                lane_count
            ):
                result_element_bit_width = _predicate_carrier_element_bit_width(
                    lane_count, result_bit_width
                )
                requires_conversion = (
                    source_element.bit_width != result_element_bit_width
                )
                native_mask = (
                    ValueRef.temporary("native_mask")
                    if requires_conversion
                    else ValueRef.result("result")
                )
                prefix_emits = (
                    _op_emit(
                        descriptor=zero,
                        results={"dst": ValueRef.temporary("zero")},
                        result_types={"dst": source_type},
                    ),
                    _op_emit(
                        descriptor=shift,
                        operands={"source": ValueRef.operand("input")},
                        results={"dst": ValueRef.temporary("low_bit_sign")},
                        result_types={"dst": source_type},
                        immediates={"shift": source_element.bit_width - 1},
                    ),
                    _op_emit(
                        descriptor=compare,
                        operands={
                            "lhs": ValueRef.temporary("zero"),
                            "rhs": ValueRef.temporary("low_bit_sign"),
                        },
                        results={"dst": native_mask},
                        result_types=(
                            {"dst": source_type} if requires_conversion else None
                        ),
                    ),
                )
                conversion_emits = ()
                conversion_descriptors: tuple[Descriptor, ...] = ()
                if requires_conversion:
                    conversion_emits, conversion_descriptors = (
                        _signed_predicate_mask_conversion_emits(
                            descriptor_lookup,
                            source_element_bit_width=source_element.bit_width,
                            result_element_bit_width=result_element_bit_width,
                            lane_count=lane_count,
                            source=native_mask,
                            result=ValueRef.result("result"),
                            result_type=None,
                            temporary_prefix="predicate_truncate",
                        )
                    )
                descriptors = tuple(
                    dict.fromkeys((zero, shift, compare, *conversion_descriptors))
                )
                rules.append(
                    DescriptorRule(
                        source_op=vector.vector_trunci,
                        descriptor=compare,
                        guards=(
                            Guard.value_type("input", source_type),
                            Guard.value_type("result", result_type),
                            Guard.value_static_element_count_eq("input", "result"),
                            Guard.low_value_register_class(
                                "input", source_register_class
                            ),
                            *_predicate_representation_guards(
                                "result", lane_count, result_bit_width
                            ),
                            *(
                                Guard.descriptor_available(descriptor)
                                for descriptor in descriptors
                                if descriptor != compare
                            ),
                        ),
                        emit=(*prefix_emits, *conversion_emits),
                        report_key="x86.predicate_truncate.low_bit",
                    )
                )
    return tuple(rules)


def _predicate_lane_descriptor_key(mnemonic: str, element_bit_width: int) -> str:
    if mnemonic.startswith("vpextr"):
        return f"x86.avx2.{mnemonic}.gpr{max(32, element_bit_width)}.xmm"
    return f"x86.avx2.{mnemonic}.xmm"


def _predicate_lane_movement_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    lane_mnemonics = {
        row.element_bit_width: (row.extract_mnemonic, row.insert_mnemonic)
        for row in X86_LANE_FAMILIES
    }
    move_zero = descriptor_lookup("x86.scalar.movimm.gpr32")
    subtract = descriptor_lookup("x86.scalar.sub.gpr32")
    sign_extend = descriptor_lookup("x86.scalar.movsxd.gpr64.gpr32")
    truncate = descriptor_lookup("x86.scalar.mov.trunc.gpr32.gpr64")
    mask_bit = descriptor_lookup("x86.scalar.and.imm.gpr32")
    extract_half = descriptor_lookup("x86.avx2.vextractf128.xmm.ymm")
    insert_half = descriptor_lookup("x86.avx2.vinsertf128.ymm.xmm")
    rules: list[DescriptorRule] = []
    for lane_count, predicate_type in _PREDICATE_TYPES.items():
        for representation_bit_width in avx2_predicate_representation_bit_widths(
            lane_count
        ):
            element_bit_width = _predicate_carrier_element_bit_width(
                lane_count, representation_bit_width
            )
            extract_mnemonic, insert_mnemonic = lane_mnemonics[element_bit_width]
            extract_lane = descriptor_lookup(
                _predicate_lane_descriptor_key(extract_mnemonic, element_bit_width)
            )
            insert_lane = descriptor_lookup(
                _predicate_lane_descriptor_key(insert_mnemonic, element_bit_width)
            )
            half_lane_count = 128 // element_bit_width
            source = ValueRef.operand("source")
            lane = AttrProject.i64_array_element("static_indices", element=0)
            extract_emits: list[EmitDescriptorOp] = []
            extract_dependencies = [extract_lane, mask_bit]
            if representation_bit_width == 256:
                source = ValueRef.temporary("half")
                lane = AttrProject.i64_array_element_remainder(
                    "static_indices", element=0, divisor=half_lane_count
                )
                extract_dependencies.append(extract_half)
                extract_emits.append(
                    _op_emit(
                        descriptor=extract_half,
                        operands={"source": ValueRef.operand("source")},
                        results={"dst": source},
                        result_types={"dst": DescriptorResultType()},
                        immediates={
                            "lane": AttrProject.i64_array_element_quotient(
                                "static_indices",
                                element=0,
                                divisor=half_lane_count,
                            )
                        },
                    )
                )
            extracted_result = (
                ValueRef.temporary("wide_lane")
                if element_bit_width == 64
                else ValueRef.temporary("lane")
            )
            extract_emits.append(
                _op_emit(
                    descriptor=extract_lane,
                    operands={"source": source},
                    results={"dst": extracted_result},
                    result_types={"dst": _I64 if element_bit_width == 64 else _I32},
                    immediates={"lane": lane},
                )
            )
            if element_bit_width == 64:
                extract_dependencies.append(truncate)
                extract_emits.append(
                    _op_emit(
                        descriptor=truncate,
                        operands={"src": extracted_result},
                        results={"dst": ValueRef.temporary("lane")},
                        result_types={"dst": _I32},
                    )
                )
            extract_emits.append(
                _op_emit(
                    descriptor=mask_bit,
                    operands={"lhs": ValueRef.temporary("lane")},
                    results={"dst": ValueRef.result("result")},
                    immediates={"imm32": 1},
                )
            )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_extract,
                    descriptor=mask_bit,
                    guards=(
                        Guard.value_type("source", predicate_type),
                        Guard.value_type("result", _I1),
                        *_predicate_representation_guards(
                            "source", lane_count, representation_bit_width
                        ),
                        Guard.operand_segment_count("indices", 0),
                        Guard.i64_array_count("static_indices", 1),
                        Guard.i64_array_element_range(
                            "static_indices", 0, 0, lane_count - 1
                        ),
                        *(
                            Guard.descriptor_available(descriptor)
                            for descriptor in dict.fromkeys(extract_dependencies)
                            if descriptor != mask_bit
                        ),
                    ),
                    emit=tuple(extract_emits),
                )
            )

            insert_dependencies = [move_zero, subtract, insert_lane]
            insert_emits: list[EmitDescriptorOp] = [
                EmitDescriptorOp(
                    descriptor=move_zero,
                    results={"dst": ValueRef.temporary("zero")},
                    result_types={"dst": _I32},
                    immediates={"imm32": 0},
                    form=DescriptorEmitForm.CONST,
                ),
                _op_emit(
                    descriptor=subtract,
                    operands={
                        "lhs": ValueRef.temporary("zero"),
                        "rhs": ValueRef.operand("value"),
                    },
                    results={"dst": ValueRef.temporary("lane32")},
                    result_types={"dst": _I32},
                ),
            ]
            lane_value = ValueRef.temporary("lane32")
            if element_bit_width == 64:
                insert_dependencies.append(sign_extend)
                lane_value = ValueRef.temporary("lane64")
                insert_emits.append(
                    _op_emit(
                        descriptor=sign_extend,
                        operands={"src": ValueRef.temporary("lane32")},
                        results={"dst": lane_value},
                        result_types={"dst": _I64},
                    )
                )
            destination = ValueRef.operand("dest")
            insert_lane_project = AttrProject.i64_array_element(
                "static_indices", element=0
            )
            if representation_bit_width == 256:
                insert_dependencies.extend((extract_half, insert_half))
                destination = ValueRef.temporary("half")
                insert_lane_project = AttrProject.i64_array_element_remainder(
                    "static_indices", element=0, divisor=half_lane_count
                )
                insert_emits.append(
                    _op_emit(
                        descriptor=extract_half,
                        operands={"source": ValueRef.operand("dest")},
                        results={"dst": destination},
                        result_types={"dst": DescriptorResultType()},
                        immediates={
                            "lane": AttrProject.i64_array_element_quotient(
                                "static_indices",
                                element=0,
                                divisor=half_lane_count,
                            )
                        },
                    )
                )
            inserted = (
                ValueRef.result("result")
                if representation_bit_width <= 128
                else ValueRef.temporary("inserted_half")
            )
            insert_emits.append(
                _op_emit(
                    descriptor=insert_lane,
                    operands={"dest": destination, "value": lane_value},
                    results={"dst": inserted},
                    result_types=(
                        None
                        if representation_bit_width <= 128
                        else {"dst": DescriptorResultType()}
                    ),
                    immediates={"lane": insert_lane_project},
                )
            )
            primary_descriptor = insert_lane
            if representation_bit_width == 256:
                primary_descriptor = insert_half
                insert_emits.append(
                    _op_emit(
                        descriptor=insert_half,
                        operands={
                            "dest": ValueRef.operand("dest"),
                            "value": inserted,
                        },
                        results={"dst": ValueRef.result("result")},
                        immediates={
                            "lane": AttrProject.i64_array_element_quotient(
                                "static_indices",
                                element=0,
                                divisor=half_lane_count,
                            )
                        },
                    )
                )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_insert,
                    descriptor=primary_descriptor,
                    guards=(
                        Guard.value_type("value", _I1),
                        Guard.value_type("dest", predicate_type),
                        Guard.value_type("result", predicate_type),
                        *_predicate_representation_guards(
                            "dest", lane_count, representation_bit_width
                        ),
                        *_predicate_representation_guards(
                            "result", lane_count, representation_bit_width
                        ),
                        Guard.operand_segment_count("indices", 0),
                        Guard.i64_array_count("static_indices", 1),
                        Guard.i64_array_element_range(
                            "static_indices", 0, 0, lane_count - 1
                        ),
                        *(
                            Guard.descriptor_available(descriptor)
                            for descriptor in dict.fromkeys(insert_dependencies)
                            if descriptor != primary_descriptor
                        ),
                    ),
                    emit=tuple(insert_emits),
                )
            )
    return tuple(rules)


def _integer_compare_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for element in INTEGER_ELEMENTS:
        for payload_bit_width in _PREDICATE_SIMD_BIT_WIDTHS:
            if payload_bit_width % element.bit_width:
                continue
            lane_count = element.lane_count(payload_bit_width)
            if lane_count not in _PREDICATE_TYPES:
                continue
            operand_type = _full_vector_type(element, payload_bit_width)
            result_type = _PREDICATE_TYPES[lane_count]
            for predicate in (
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
            ):
                for result_bit_width in avx2_predicate_representation_bit_widths(
                    lane_count
                ):
                    requires_conversion = result_bit_width != payload_bit_width
                    native_result = (
                        ValueRef.temporary("native_mask")
                        if requires_conversion
                        else ValueRef.result("result")
                    )
                    native_result_type = operand_type if requires_conversion else None
                    descriptors, compare_emits = _integer_compare_program(
                        predicate,
                        element,
                        payload_bit_width,
                        descriptor_lookup,
                        native_result=native_result,
                        native_result_type=native_result_type,
                    )
                    conversion_emits: tuple[EmitDescriptorOp, ...] = ()
                    conversion_descriptors: tuple[Descriptor, ...] = ()
                    if requires_conversion:
                        conversion_emits, conversion_descriptors = (
                            _signed_predicate_mask_conversion_emits(
                                descriptor_lookup,
                                source_element_bit_width=element.bit_width,
                                result_element_bit_width=(
                                    result_bit_width // lane_count
                                ),
                                lane_count=lane_count,
                                source=native_result,
                                result=ValueRef.result("result"),
                                result_type=None,
                                temporary_prefix="converted_mask",
                            )
                        )
                    all_descriptors = tuple(
                        dict.fromkeys((*descriptors, *conversion_descriptors))
                    )
                    rules.append(
                        DescriptorRule(
                            source_op=vector.vector_cmpi,
                            descriptor=compare_emits[0].descriptor,
                            guards=(
                                Guard.enum_attr_equals("predicate", predicate),
                                *_typed_guards(("lhs", "rhs"), operand_type),
                                Guard.value_type("result", result_type),
                                *_predicate_representation_guards(
                                    "result", lane_count, result_bit_width
                                ),
                                *(
                                    Guard.descriptor_available(descriptor)
                                    for descriptor in all_descriptors
                                    if descriptor != compare_emits[0].descriptor
                                ),
                            ),
                            emit=(*compare_emits, *conversion_emits),
                        )
                    )
    return tuple(rules)


def _floating_compare_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for element in FLOAT_ELEMENTS:
        for payload_bit_width in _PREDICATE_SIMD_BIT_WIDTHS:
            if payload_bit_width % element.bit_width:
                continue
            lane_count = element.lane_count(payload_bit_width)
            if lane_count not in _PREDICATE_TYPES:
                continue
            instruction_bit_width = max(128, payload_bit_width)
            operand_type = _full_vector_type(element, payload_bit_width)
            result_type = _PREDICATE_TYPES[lane_count]
            compare = descriptor_lookup(
                f"x86.avx2.{AVX2_FLOAT_COMPARE_MNEMONICS[element.name]}."
                f"{_REGISTER_SUFFIXES[instruction_bit_width]}"
            )
            for result_bit_width in avx2_predicate_representation_bit_widths(
                lane_count
            ):
                requires_conversion = result_bit_width != payload_bit_width
                native_result = (
                    ValueRef.temporary("native_mask")
                    if requires_conversion
                    else ValueRef.result("result")
                )
                compare_emit = _op_emit(
                    descriptor=compare,
                    operands={
                        "lhs": ValueRef.operand("lhs"),
                        "rhs": ValueRef.operand("rhs"),
                    },
                    results={"dst": native_result},
                    result_types={"dst": operand_type} if requires_conversion else None,
                    immediates={
                        "predicate": AttrProject.enum_remap(
                            "predicate", X86_FLOAT_COMPARE_IMMEDIATES
                        )
                    },
                )
                conversion_emits: tuple[EmitDescriptorOp, ...] = ()
                conversion_descriptors: tuple[Descriptor, ...] = ()
                if requires_conversion:
                    conversion_emits, conversion_descriptors = (
                        _signed_predicate_mask_conversion_emits(
                            descriptor_lookup,
                            source_element_bit_width=element.bit_width,
                            result_element_bit_width=result_bit_width // lane_count,
                            lane_count=lane_count,
                            source=native_result,
                            result=ValueRef.result("result"),
                            result_type=None,
                            temporary_prefix="converted_mask",
                        )
                    )
                rules.append(
                    DescriptorRule(
                        source_op=vector.vector_cmpf,
                        descriptor=compare,
                        guards=(
                            *_typed_guards(("lhs", "rhs"), operand_type),
                            Guard.value_type("result", result_type),
                            *_predicate_representation_guards(
                                "result", lane_count, result_bit_width
                            ),
                            *(
                                Guard.descriptor_available(descriptor)
                                for descriptor in conversion_descriptors
                            ),
                        ),
                        emit=(compare_emit, *conversion_emits),
                    )
                )
    return tuple(rules)


def _vector_select_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for lane_count, condition_type in _PREDICATE_TYPES.items():
        for payload_bit_width in avx2_predicate_representation_bit_widths(lane_count):
            payload_type = Vector(
                _PAYLOAD_ELEMENT_NAMES_BY_BIT_WIDTH[payload_bit_width // lane_count],
                lanes=lane_count,
            )
            blend = descriptor_lookup(
                f"x86.avx2.vpblendvb.{_REGISTER_SUFFIXES[payload_bit_width]}"
            )
            for condition_bit_width in avx2_predicate_representation_bit_widths(
                lane_count
            ):
                requires_conversion = condition_bit_width != payload_bit_width
                mask = ValueRef.operand("condition")
                conversion_emits: tuple[EmitDescriptorOp, ...] = ()
                conversion_descriptors: tuple[Descriptor, ...] = ()
                if requires_conversion:
                    mask = ValueRef.temporary("native_mask")
                    conversion_emits, conversion_descriptors = (
                        _signed_predicate_mask_conversion_emits(
                            descriptor_lookup,
                            source_element_bit_width=(
                                condition_bit_width // lane_count
                            ),
                            result_element_bit_width=(payload_bit_width // lane_count),
                            lane_count=lane_count,
                            source=ValueRef.operand("condition"),
                            result=mask,
                            result_type=_PREDICATE_CARRIER_TYPES[
                                (payload_bit_width, lane_count)
                            ],
                            temporary_prefix="converted_mask",
                        )
                    )
                rules.append(
                    DescriptorRule(
                        source_op=vector.vector_select,
                        descriptor=blend,
                        guards=(
                            Guard.value_type("condition", condition_type),
                            *_typed_guards(
                                ("true_value", "false_value", "result"),
                                payload_type,
                            ),
                            *_predicate_representation_guards(
                                "condition", lane_count, condition_bit_width
                            ),
                            Guard.low_value_register_class(
                                "result", _REGISTER_CLASSES[payload_bit_width]
                            ),
                            *(
                                Guard.descriptor_available(descriptor)
                                for descriptor in conversion_descriptors
                            ),
                        ),
                        emit=(
                            *conversion_emits,
                            _op_emit(
                                descriptor=blend,
                                operands={
                                    "false_value": ValueRef.operand("false_value"),
                                    "true_value": ValueRef.operand("true_value"),
                                    "mask": mask,
                                },
                                results={"dst": ValueRef.result("result")},
                            ),
                        ),
                    )
                )
    return tuple(rules)


def _whole_vector_select_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS:
        register_suffix = _REGISTER_SUFFIXES[vector_bit_width]
        blend = descriptor_lookup(f"x86.avx2.vpblendvb.{register_suffix}")
        mask_type = Vector("i32", lanes=vector_bit_width // 32)
        mask_emits, mask_descriptors = _scalar_i1_mask_emits(
            descriptor_lookup,
            scalar=ValueRef.operand("condition"),
            element_bit_width=32,
            register_suffix=register_suffix,
            result=ValueRef.temporary("mask"),
            result_type=mask_type,
        )
        rules.append(
            DescriptorRule(
                source_op=scf.scf_select,
                descriptor=blend,
                guards=(
                    Guard.value_type("condition", _I1),
                    *_typed_guards(
                        ("true_value", "false_value", "result"),
                        _PAYLOAD_VECTOR_TYPES,
                    ),
                    Guard.low_value_register_class(
                        "result", _REGISTER_CLASSES[vector_bit_width]
                    ),
                    *(
                        Guard.descriptor_available(descriptor)
                        for descriptor in mask_descriptors
                    ),
                ),
                emit=(
                    *mask_emits,
                    _op_emit(
                        descriptor=blend,
                        operands={
                            "false_value": ValueRef.operand("false_value"),
                            "true_value": ValueRef.operand("true_value"),
                            "mask": ValueRef.temporary("mask"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        )
    return tuple(rules)


def _predicate_bitwise_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for lane_count, predicate_type in _PREDICATE_TYPES.items():
        for representation_bit_width in avx2_predicate_representation_bit_widths(
            lane_count
        ):
            register_suffix = _REGISTER_SUFFIXES[representation_bit_width]
            for source_operation, mnemonic, _ in AVX2_BITWISE_FAMILIES:
                descriptor = descriptor_lookup(f"x86.avx2.{mnemonic}.{register_suffix}")
                rules.append(
                    DescriptorRule(
                        source_op=_PREDICATE_BITWISE_SOURCE_OPS[source_operation],
                        descriptor=descriptor,
                        guards=(
                            *_typed_guards(("lhs", "rhs", "result"), predicate_type),
                            *(
                                guard
                                for field in ("lhs", "rhs", "result")
                                for guard in _predicate_representation_guards(
                                    field, lane_count, representation_bit_width
                                )
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


def avx2_predicate_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[ContractCase, ...]:
    return (
        *_predicate_constant_rules(descriptor_lookup),
        *_predicate_splat_rules(descriptor_lookup),
        *_predicate_extension_rules(descriptor_lookup),
        *_predicate_truncation_rules(descriptor_lookup),
        *_predicate_lane_movement_rules(descriptor_lookup),
        *_integer_compare_rules(descriptor_lookup),
        *_floating_compare_rules(descriptor_lookup),
        *_vector_select_rules(descriptor_lookup),
        *_predicate_bitwise_rules(descriptor_lookup),
        *_whole_vector_select_rules(descriptor_lookup),
    )
