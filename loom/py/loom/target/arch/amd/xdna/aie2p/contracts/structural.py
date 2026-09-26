# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P vector structural selection rules."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    AttrProject,
    ContractEmit,
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
    ValueTypeProject,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I8X32_VECTOR = Vector("i8", lanes=32)
_I8X64_VECTOR = Vector("i8", lanes=64)
_I32_F32_4X4_VECTOR = Vector(("i32", "f32"), dims=(4, 4))
_I16_F16_BF16_8X8_VECTOR = Vector(("i16", "f16", "bf16"), dims=(8, 8))
_I32 = Scalar("i32")
_INDEX = Scalar("index")

# Ordinary vectors retain one or two full X carriers independently of their
# logical extent. F32 excludes vector<32xf32>, whose accumulator contract uses
# a distinct physical representation.
_VECTOR_CARRIER_SPECS = (
    (("i8", "f8E4M3", "f8E5M2"), 1, 128),
    (("i16", "f16", "bf16"), 2, 64),
    (("i32",), 4, 32),
    (("f32",), 4, 31),
    (("i64", "f64"), 8, 16),
)

# Every vector in this domain occupies the same ordered pair of X carriers.
# The F32 upper bound deliberately excludes flat vector<32xf32>, whose
# accumulator representation cannot alias an ordinary shaped Y carrier.
_WIDE_VECTOR_BITCAST_TYPES = tuple(
    Vector(
        element_types,
        minimum_static_elements=64 // element_byte_count + 1,
        maximum_static_elements=wide_lane_maximum,
    )
    for element_types, element_byte_count, wide_lane_maximum in (_VECTOR_CARRIER_SPECS)
)

# Ordinary source vectors wider than one 512-bit X register are carried as two
# consecutive X registers. F32 excludes vector<32xf32>, which is an accumulator
# fragment with a distinct physical contract.
_WIDE_VECTOR_EXTRACT_SPECS = (
    (
        Vector("i8", minimum_static_elements=65, maximum_static_elements=128),
        Scalar("i8"),
        64,
        "i8",
        1,
    ),
    (
        Vector("f8E4M3", minimum_static_elements=65, maximum_static_elements=128),
        Scalar("f8E4M3"),
        64,
        "i8",
        1,
    ),
    (
        Vector("f8E5M2", minimum_static_elements=65, maximum_static_elements=128),
        Scalar("f8E5M2"),
        64,
        "i8",
        1,
    ),
    (
        Vector("i16", minimum_static_elements=33, maximum_static_elements=64),
        Scalar("i16"),
        32,
        "i16",
        1,
    ),
    (
        Vector("f16", minimum_static_elements=33, maximum_static_elements=64),
        Scalar("f16"),
        32,
        "i16",
        1,
    ),
    (
        Vector("bf16", minimum_static_elements=33, maximum_static_elements=64),
        Scalar("bf16"),
        32,
        "i16",
        1,
    ),
    (
        Vector("i32", minimum_static_elements=17, maximum_static_elements=32),
        Scalar("i32"),
        16,
        "i32",
        1,
    ),
    (
        Vector("f32", minimum_static_elements=17, maximum_static_elements=31),
        Scalar("f32"),
        16,
        "i32",
        1,
    ),
    (
        Vector("i64", minimum_static_elements=9, maximum_static_elements=16),
        Scalar("i64"),
        8,
        "i64",
        2,
    ),
    (
        Vector("f64", minimum_static_elements=9, maximum_static_elements=16),
        Scalar("f64"),
        8,
        "i64",
        2,
    ),
)

# AIE2P VSHUFFLE modes that select the even and odd byte lanes from the first
# 512-bit source. Each logical 32-byte result retains the target's 512-bit X
# carrier, with the remaining lanes outside the source vector's value domain.
_I8_DEINTERLEAVE_CONTROLS = (0, 1)

# T16_2x32_lo interleaves the low sixteen 16-bit lanes of two X carriers.
_I16_INTERLEAVE_CONTROL = 18

# AIE2P's T32_4x4 VSHUFFLE mode transposes the sixteen 32-bit lanes carried
# by one X register.
_I32_F32_TRANSPOSE_4X4_CONTROL = 34

# T16_8x8_lo and T16_8x8_hi return the low and high 512-bit halves of
# the transposed 1024-bit value. Both read the same ordered pair of X registers.
_I16_TRANSPOSE_8X8_CONTROLS = (52, 53)

# Ordinary payloads share byte-addressable X carriers regardless of element
# interpretation. A partial packet occupies the low bytes of its carrier.
_PACKED_VECTOR_ELEMENT_TYPES = (
    (("i8", "f8E4M3", "f8E5M2"), 1),
    (("i16", "f16", "bf16"), 2),
    (("i32", "f32"), 4),
    (("i64", "f64"), 8),
)

# A full X packet followed by a possibly partial packet occupies two X
# carriers. F32x32 is excluded because it has an accumulator representation.
_WIDE_VECTOR_CONCAT_SPECS = (
    (
        Vector(("i8", "f8E4M3", "f8E5M2"), lanes=64),
        Vector(
            ("i8", "f8E4M3", "f8E5M2"),
            minimum_lanes=65,
            maximum_lanes=128,
        ),
    ),
    (
        Vector(("i16", "f16", "bf16"), lanes=32),
        Vector(("i16", "f16", "bf16"), minimum_lanes=33, maximum_lanes=64),
    ),
    (Vector("i32", lanes=16), Vector("i32", minimum_lanes=17, maximum_lanes=32)),
    (Vector("f32", lanes=16), Vector("f32", minimum_lanes=17, maximum_lanes=31)),
    (
        Vector(("i64", "f64"), lanes=8),
        Vector(("i64", "f64"), minimum_lanes=9, maximum_lanes=16),
    ),
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _wide_vector_extract_static_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    half_lane_count: int,
    storage: str,
    *,
    high_half: bool,
) -> DescriptorRule:
    descriptor = _descriptor(f"amd.xdna.aie2p.extract.{storage}.immediate")
    minimum_index = half_lane_count if high_half else 0
    maximum_index = 2 * half_lane_count - 1 if high_half else half_lane_count - 1
    immediate = (
        AttrProject.i64_array_element_plus_literal(
            "static_indices", element=0, literal=-half_lane_count
        )
        if high_half
        else AttrProject.i64_array_element("static_indices", element=0)
    )
    half = ValueRef.temporary("high_half" if high_half else "low_half")
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices",
                element=0,
                minimum=minimum_index,
                maximum=maximum_index,
            ),
        ),
        emit=(
            EmitRegisterSlice(
                source=ValueRef.operand("source"),
                result=half,
                unit_offset=2 if high_half else 0,
                unit_count=2,
            ),
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"s1": half},
                results={"dst": ValueRef.result("result")},
                immediates={"idx": immediate},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _wide_vector_extract_dynamic_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    half_lane_count: int,
    storage: str,
    result_unit_count: int,
) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.i32")
    extract = _descriptor(f"amd.xdna.aie2p.extract.{storage}.register")
    select = _descriptor("amd.xdna.aie2p.select.nonzero.i32")

    local_index = ValueRef.temporary("local_index")
    high_selector = ValueRef.temporary("high_selector")
    low_value = ValueRef.temporary("low_value")
    high_value = ValueRef.temporary("high_value")
    emits: list[ContractEmit] = [
        EmitRegisterSlice(
            source=ValueRef.operand("source"),
            result=ValueRef.temporary("low_half"),
            unit_count=2,
        ),
        EmitRegisterSlice(
            source=ValueRef.operand("source"),
            result=ValueRef.temporary("high_half"),
            unit_offset=2,
            unit_count=2,
        ),
    ]
    for name, value in (
        ("lane_mask", half_lane_count - 1),
        ("half_lane", half_lane_count),
    ):
        emits.append(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary(name)},
                result_types={"dst": _I32},
                immediates={"i": value},
                form=DescriptorEmitForm.CONST,
            )
        )
    for mask, result in (
        ("lane_mask", local_index),
        ("half_lane", high_selector),
    ):
        emits.append(
            EmitDescriptorOp(
                descriptor=bitwise_and,
                operands={
                    "s0": ValueRef.operand("indices"),
                    "s1": ValueRef.temporary(mask),
                },
                results={"d0": result},
                result_types={"d0": _I32},
                form=DescriptorEmitForm.OP,
            )
        )
    for half, result in (("low_half", low_value), ("high_half", high_value)):
        emits.append(
            EmitDescriptorOp(
                descriptor=extract,
                operands={
                    "s1": ValueRef.temporary(half),
                    "idx": local_index,
                },
                results={"dst": result},
                result_types={"dst": result_type},
                form=DescriptorEmitForm.OP,
            )
        )

    if result_unit_count == 1:
        emits.append(
            EmitDescriptorOp(
                descriptor=select,
                operands={
                    "s0": high_value,
                    "s1": low_value,
                    "s2": high_selector,
                },
                results={"d0": ValueRef.result("result")},
                copy_operands=("s2",),
                form=DescriptorEmitForm.OP,
            )
        )
    else:
        selected_units = []
        for unit_index in range(result_unit_count):
            low_unit = ValueRef.temporary(f"low_value_{unit_index}")
            high_unit = ValueRef.temporary(f"high_value_{unit_index}")
            selected_unit = ValueRef.temporary(f"selected_{unit_index}")
            emits.extend(
                (
                    EmitRegisterSlice(
                        source=low_value,
                        result=low_unit,
                        unit_offset=unit_index,
                        unit_count=1,
                    ),
                    EmitRegisterSlice(
                        source=high_value,
                        result=high_unit,
                        unit_offset=unit_index,
                        unit_count=1,
                    ),
                    EmitDescriptorOp(
                        descriptor=select,
                        operands={
                            "s0": high_unit,
                            "s1": low_unit,
                            "s2": high_selector,
                        },
                        results={"d0": selected_unit},
                        result_types={"d0": DescriptorResultType()},
                        copy_operands=("s2",),
                        form=DescriptorEmitForm.OP,
                    ),
                )
            )
            selected_units.append(selected_unit)
        emits.append(
            EmitRegisterConcat(
                sources=selected_units,
                result=ValueRef.result("result"),
            )
        )

    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=extract,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
            Guard.value_type("indices", _INDEX),
            Guard.operand_segment_count("indices", 1),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices",
                element=0,
                minimum=-(2**63),
                maximum=-(2**63),
            ),
        ),
        emit=tuple(emits),
    )


def _vector_deinterleave_i8x64_rule() -> DescriptorRule:
    control_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")

    emits = []
    for result_index, result_name in enumerate(("even", "odd")):
        control_name = f"{result_name}_control"
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=control_constant,
                    results={"dst": ValueRef.temporary(control_name)},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"i": _I8_DEINTERLEAVE_CONTROLS[result_index]},
                    form=DescriptorEmitForm.CONST,
                ),
                EmitDescriptorOp(
                    descriptor=shuffle,
                    operands={
                        "s1": ValueRef.operand("source"),
                        "s2": ValueRef.operand("source"),
                        "mod": ValueRef.temporary(control_name),
                    },
                    results={"dst": ValueRef.result("results", element=result_index)},
                    form=DescriptorEmitForm.OP,
                ),
            )
        )

    return DescriptorRule(
        source_op=vector.vector_deinterleave,
        descriptor=shuffle,
        guards=(
            Guard.value_type("source", _I8X64_VECTOR),
            Guard.value_type("results", _I8X32_VECTOR),
            Guard.attr_kind("axis", "i64"),
            Guard.i64_range("axis", 0, 0),
        ),
        emit=tuple(emits),
    )


def _vector_interleave_16bit_rule() -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    input_type = Vector(("i16", "f16", "bf16"), lanes=16)
    result_type = Vector(("i16", "f16", "bf16"), lanes=32)
    control = ValueRef.temporary("control")
    return DescriptorRule(
        source_op=vector.vector_interleave,
        descriptor=shuffle,
        guards=(
            Guard.value_type("even", input_type),
            Guard.value_type("odd", input_type),
            Guard.value_type("result", result_type),
            Guard.i64_range("axis", 0, 0),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": control},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": _I16_INTERLEAVE_CONTROL},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=shuffle,
                operands={
                    "s1": ValueRef.operand("even"),
                    "s2": ValueRef.operand("odd"),
                    "mod": control,
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _vector_transpose_i32_f32_4x4_rule() -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    control = ValueRef.temporary("control")
    return DescriptorRule(
        source_op=vector.vector_transpose,
        descriptor=shuffle,
        guards=(
            Guard.value_type("source", _I32_F32_4X4_VECTOR),
            Guard.value_type("result", _I32_F32_4X4_VECTOR),
            Guard.i64_array_count("permutation", 2),
            Guard.i64_array_element_range(
                "permutation", element=0, minimum=1, maximum=1
            ),
            Guard.i64_array_element_range(
                "permutation", element=1, minimum=0, maximum=0
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": control},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": _I32_F32_TRANSPOSE_4X4_CONTROL},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=shuffle,
                operands={
                    "s1": ValueRef.operand("source"),
                    "s2": ValueRef.operand("source"),
                    "mod": control,
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _vector_transpose_16bit_8x8_rule() -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    low = ValueRef.temporary("low")
    high = ValueRef.temporary("high")
    emits: list[ContractEmit] = [
        EmitRegisterSlice(
            source=ValueRef.operand("source"),
            result=low,
            unit_count=2,
        ),
        EmitRegisterSlice(
            source=ValueRef.operand("source"),
            result=high,
            unit_offset=2,
            unit_count=2,
        ),
    ]
    halves = []
    for name, mode in zip(("low", "high"), _I16_TRANSPOSE_8X8_CONTROLS, strict=True):
        control = ValueRef.temporary(f"{name}_control")
        result = ValueRef.temporary(f"{name}_transposed")
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=constant,
                    results={"dst": control},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"i": mode},
                    form=DescriptorEmitForm.CONST,
                ),
                EmitDescriptorOp(
                    descriptor=shuffle,
                    operands={"s1": low, "s2": high, "mod": control},
                    results={"dst": result},
                    result_types={"dst": DescriptorResultType()},
                    form=DescriptorEmitForm.OP,
                ),
            )
        )
        halves.append(result)
    emits.append(EmitRegisterConcat(sources=halves, result=ValueRef.result("result")))
    return DescriptorRule(
        source_op=vector.vector_transpose,
        descriptor=shuffle,
        guards=(
            Guard.value_type("source", _I16_F16_BF16_8X8_VECTOR),
            Guard.value_type("result", _I16_F16_BF16_8X8_VECTOR),
            Guard.i64_array_count("permutation", 2),
            Guard.i64_array_element_range(
                "permutation", element=0, minimum=1, maximum=1
            ),
            Guard.i64_array_element_range(
                "permutation", element=1, minimum=0, maximum=0
            ),
        ),
        emit=tuple(emits),
    )


def _wide_vector_bitcast_alias_rules() -> tuple[ValueAliasRule, ...]:
    # Equal-width bitcasts preserve the ordered four-W payload independently
    # of logical shape and element interpretation.
    return tuple(
        ValueAliasRule(
            source_op=vector.vector_bitcast,
            source=ValueRef.operand("input"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type("input", source_type),
                Guard.value_type("result", result_type),
                Guard.low_value_register_unit_count_eq("input", "result"),
            ),
        )
        for source_type in _WIDE_VECTOR_BITCAST_TYPES
        for result_type in _WIDE_VECTOR_BITCAST_TYPES
    )


def _vector_slice_guards(
    source_type: TypePattern,
    result_type: TypePattern,
    offset_minimum: int,
    offset_maximum: int,
) -> tuple[Guard, ...]:
    return (
        Guard.value_type("source", source_type),
        Guard.value_type("result", result_type),
        Guard.operand_segment_count("offsets", 0),
        Guard.i64_array_count("static_offsets", 1),
        Guard.i64_array_element_range(
            "static_offsets",
            element=0,
            minimum=offset_minimum,
            maximum=offset_maximum,
        ),
    )


def _vector_slice_alias_rule(
    source_type: TypePattern,
    result_type: TypePattern,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_slice,
        source=ValueRef.operand("source"),
        result=ValueRef.result("result"),
        guards=_vector_slice_guards(source_type, result_type, 0, 0),
    )


def _vector_slice_shift_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    element_byte_count: int,
    offset_minimum: int,
    offset_maximum: int,
    *,
    base_byte_offset: int,
    source_unit_offset: int = 0,
    crossing_carriers: bool = False,
) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    source = ValueRef.operand("source")
    low = source
    high = source
    emits: list[ContractEmit] = []
    if source_unit_offset or crossing_carriers:
        low = ValueRef.temporary("low")
        emits.append(
            EmitRegisterSlice(
                source=source,
                result=low,
                unit_offset=source_unit_offset,
                unit_count=2,
            )
        )
        high = low
    if crossing_carriers:
        high = ValueRef.temporary("high")
        emits.append(
            EmitRegisterSlice(
                source=source,
                result=high,
                unit_offset=2,
                unit_count=2,
            )
        )
    byte_offset = ValueRef.temporary("byte_offset")
    emits.extend(
        (
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": byte_offset},
                result_types={"dst": DescriptorResultType()},
                immediates={
                    "i": AttrProject.i64_array_lane_byte_offset(
                        "static_offsets",
                        element=0,
                        bytes_per_lane=element_byte_count,
                        base_byte_offset=base_byte_offset,
                    )
                },
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=shift,
                operands={"s1": low, "s2": high, "shift": byte_offset},
                results={"d": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        )
    )
    return DescriptorRule(
        source_op=vector.vector_slice,
        descriptor=shift,
        guards=_vector_slice_guards(
            source_type,
            result_type,
            offset_minimum,
            offset_maximum,
        ),
        emit=tuple(emits),
    )


def _vector_slice_carrier_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    offset: int,
    *,
    source_unit_offset: int,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=vector.vector_slice,
        guards=_vector_slice_guards(source_type, result_type, offset, offset),
        emit=(
            EmitRegisterSlice(
                source=ValueRef.operand("source"),
                result=ValueRef.result("result"),
                unit_offset=source_unit_offset,
            ),
        ),
    )


def _vector_slice_wide_shift_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    element_byte_count: int,
    carrier_lane_count: int,
) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    source = ValueRef.operand("source")
    low = ValueRef.temporary("low")
    high = ValueRef.temporary("high")
    byte_offset = ValueRef.temporary("byte_offset")
    result_low = ValueRef.temporary("result_low")
    result_high = ValueRef.temporary("result_high")
    return DescriptorRule(
        source_op=vector.vector_slice,
        descriptor=shift,
        guards=_vector_slice_guards(
            source_type,
            result_type,
            1,
            carrier_lane_count - 1,
        ),
        emit=(
            EmitRegisterSlice(source=source, result=low, unit_count=2),
            EmitRegisterSlice(
                source=source,
                result=high,
                unit_offset=2,
                unit_count=2,
            ),
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": byte_offset},
                result_types={"dst": DescriptorResultType()},
                immediates={
                    "i": AttrProject.i64_array_lane_byte_offset(
                        "static_offsets",
                        element=0,
                        bytes_per_lane=element_byte_count,
                    )
                },
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=shift,
                operands={"s1": low, "s2": high, "shift": byte_offset},
                results={"d": result_low},
                result_types={"d": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=shift,
                operands={"s1": high, "s2": high, "shift": byte_offset},
                results={"d": result_high},
                result_types={"d": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            EmitRegisterConcat(
                sources=(result_low, result_high),
                result=ValueRef.result("result"),
            ),
        ),
    )


def _vector_slice_rules(
    element_types: tuple[str, ...],
    element_byte_count: int,
    wide_lane_maximum: int,
) -> tuple[ValueAliasRule | DescriptorRule, ...]:
    carrier_lane_count = 64 // element_byte_count
    narrow_type = Vector(
        element_types,
        minimum_lanes=1,
        maximum_lanes=carrier_lane_count,
    )
    wide_type = Vector(
        element_types,
        minimum_lanes=carrier_lane_count + 1,
        maximum_lanes=wide_lane_maximum,
    )
    rules: list[ValueAliasRule | DescriptorRule] = [
        _vector_slice_alias_rule(narrow_type, narrow_type),
        _vector_slice_shift_rule(
            narrow_type,
            narrow_type,
            element_byte_count,
            1,
            carrier_lane_count - 1,
            base_byte_offset=0,
        ),
        _vector_slice_carrier_rule(
            wide_type,
            narrow_type,
            0,
            source_unit_offset=0,
        ),
        _vector_slice_shift_rule(
            wide_type,
            narrow_type,
            element_byte_count,
            1,
            carrier_lane_count - 1,
            base_byte_offset=0,
            crossing_carriers=True,
        ),
        _vector_slice_carrier_rule(
            wide_type,
            narrow_type,
            carrier_lane_count,
            source_unit_offset=2,
        ),
        _vector_slice_alias_rule(wide_type, wide_type),
        _vector_slice_wide_shift_rule(
            wide_type,
            wide_type,
            element_byte_count,
            carrier_lane_count,
        ),
    ]
    if wide_lane_maximum > carrier_lane_count + 1:
        rules.insert(
            5,
            _vector_slice_shift_rule(
                wide_type,
                narrow_type,
                element_byte_count,
                carrier_lane_count + 1,
                wide_lane_maximum - 1,
                base_byte_offset=-64,
                source_unit_offset=2,
            ),
        )
    return tuple(rules)


def _vector_concat_half_carrier_rule(
    element_types: tuple[str, ...],
    element_byte_count: int,
) -> DescriptorRule:
    half_lane_count = 32 // element_byte_count
    carrier_lane_count = 64 // element_byte_count
    return DescriptorRule(
        source_op=vector.vector_concat,
        guards=(
            Guard.i64_range("axis", 0, 0),
            Guard.operand_segment_count("inputs", 2),
            Guard.value_type("inputs", Vector(element_types, lanes=half_lane_count)),
            Guard.value_type(
                "result",
                Vector(
                    element_types,
                    minimum_lanes=half_lane_count + 1,
                    maximum_lanes=carrier_lane_count,
                ),
            ),
        ),
        emit=(
            EmitRegisterSlice(
                source=ValueRef.operand("inputs", element=0),
                result=ValueRef.temporary("low"),
                unit_count=1,
            ),
            EmitRegisterSlice(
                source=ValueRef.operand("inputs", element=1),
                result=ValueRef.temporary("high"),
                unit_count=1,
            ),
            EmitRegisterConcat(
                sources=(
                    ValueRef.temporary("low"),
                    ValueRef.temporary("high"),
                ),
                result=ValueRef.result("result"),
            ),
        ),
        priority=1,
    )


def _vector_concat_merge_emits(
    left: ValueRef,
    right: ValueRef,
    result: ValueRef,
    *,
    left_byte_count: ValueTypeProject,
    remaining_byte_count: ValueTypeProject,
    temporary_prefix: str = "",
    result_type: DescriptorResultType | None = None,
) -> tuple[ContractEmit, ...]:
    """Joins the suffix-aligned left payload to the right carrier prefix."""

    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    left_bytes = ValueRef.temporary(f"{temporary_prefix}left_bytes")
    rotated_left = ValueRef.temporary(f"{temporary_prefix}rotated_left")
    remaining_bytes = ValueRef.temporary(f"{temporary_prefix}remaining_bytes")
    return (
        EmitDescriptorOp(
            descriptor=constant,
            results={"dst": left_bytes},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": left_byte_count},
            form=DescriptorEmitForm.CONST,
        ),
        EmitDescriptorOp(
            descriptor=shift,
            operands={"s1": left, "s2": left, "shift": left_bytes},
            results={"d": rotated_left},
            result_types={"d": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        ),
        EmitDescriptorOp(
            descriptor=constant,
            results={"dst": remaining_bytes},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": remaining_byte_count},
            form=DescriptorEmitForm.CONST,
        ),
        EmitDescriptorOp(
            descriptor=shift,
            operands={
                "s1": rotated_left,
                "s2": right,
                "shift": remaining_bytes,
            },
            results={"d": result},
            result_types=({"d": result_type} if result_type is not None else None),
            form=DescriptorEmitForm.OP,
        ),
    )


def _vector_concat_shift_rule(
    element_types: tuple[str, ...],
    element_byte_count: int,
) -> DescriptorRule:
    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    left = ValueRef.operand("inputs", element=0)
    right = ValueRef.operand("inputs", element=1)
    carrier_lane_count = 64 // element_byte_count
    return DescriptorRule(
        source_op=vector.vector_concat,
        descriptor=shift,
        guards=(
            Guard.i64_range("axis", 0, 0),
            Guard.operand_segment_count("inputs", 2),
            Guard.value_type(
                "inputs",
                Vector(
                    element_types,
                    minimum_lanes=1,
                    maximum_lanes=carrier_lane_count - 1,
                ),
            ),
            Guard.value_type(
                "result",
                Vector(
                    element_types,
                    minimum_lanes=2,
                    maximum_lanes=carrier_lane_count,
                ),
            ),
        ),
        emit=_vector_concat_merge_emits(
            left,
            right,
            ValueRef.result("result"),
            left_byte_count=ValueTypeProject.static_dim_scaled(
                left,
                scale=element_byte_count,
            ),
            remaining_byte_count=(
                ValueTypeProject.literal_minus_static_dim_scaled(
                    left,
                    scale=element_byte_count,
                    literal=64,
                )
            ),
        ),
    )


def _vector_concat_narrow_left_wide_result_rule(
    element_types: tuple[str, ...],
    element_byte_count: int,
    wide_lane_maximum: int,
    *,
    right_type: TypePattern,
    result_lane_minimum: int,
    right_low: ValueRef,
    right_high: ValueRef,
    prepare_right: tuple[ContractEmit, ...],
) -> DescriptorRule:
    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    carrier_lane_count = 64 // element_byte_count
    left = ValueRef.operand("inputs", element=0)
    low = ValueRef.temporary("low")
    high = ValueRef.temporary("high")
    remaining_bytes = ValueRef.temporary("low_remaining_bytes")
    return DescriptorRule(
        source_op=vector.vector_concat,
        descriptor=shift,
        guards=(
            Guard.i64_range("axis", 0, 0),
            Guard.operand_segment_count("inputs", 2),
            Guard.value_type(
                "inputs",
                Vector(
                    element_types,
                    minimum_lanes=1,
                    maximum_lanes=carrier_lane_count - 1,
                ),
            ),
            Guard.value_type("inputs", right_type, element=1),
            Guard.value_type(
                "result",
                Vector(
                    element_types,
                    minimum_lanes=result_lane_minimum,
                    maximum_lanes=wide_lane_maximum,
                ),
            ),
        ),
        emit=(
            *prepare_right,
            *_vector_concat_merge_emits(
                left,
                right_low,
                low,
                left_byte_count=ValueTypeProject.static_dim_scaled(
                    left,
                    scale=element_byte_count,
                ),
                remaining_byte_count=(
                    ValueTypeProject.literal_minus_static_dim_scaled(
                        left,
                        scale=element_byte_count,
                        literal=64,
                    )
                ),
                temporary_prefix="low_",
                result_type=DescriptorResultType(),
            ),
            EmitDescriptorOp(
                descriptor=shift,
                operands={
                    "s1": right_low,
                    "s2": right_high,
                    "shift": remaining_bytes,
                },
                results={"d": high},
                result_types={"d": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
            EmitRegisterConcat(
                sources=(low, high),
                result=ValueRef.result("result"),
            ),
        ),
    )


def _vector_concat_wide_left_rule(
    element_types: tuple[str, ...],
    element_byte_count: int,
    wide_lane_maximum: int,
) -> DescriptorRule:
    carrier_lane_count = 64 // element_byte_count
    left = ValueRef.operand("inputs", element=0)
    right = ValueRef.operand("inputs", element=1)
    left_low = ValueRef.temporary("left_low")
    left_high = ValueRef.temporary("left_high")
    result_high = ValueRef.temporary("result_high")
    return DescriptorRule(
        source_op=vector.vector_concat,
        descriptor=_descriptor("amd.xdna.aie2p.shift.bytes.x.configured"),
        guards=(
            Guard.i64_range("axis", 0, 0),
            Guard.operand_segment_count("inputs", 2),
            Guard.value_type(
                "inputs",
                Vector(
                    element_types,
                    minimum_lanes=carrier_lane_count + 1,
                    maximum_lanes=wide_lane_maximum - 1,
                ),
            ),
            Guard.value_type(
                "inputs",
                Vector(
                    element_types,
                    minimum_lanes=1,
                    maximum_lanes=carrier_lane_count - 1,
                ),
                element=1,
            ),
            Guard.value_type(
                "result",
                Vector(
                    element_types,
                    minimum_lanes=carrier_lane_count + 2,
                    maximum_lanes=wide_lane_maximum,
                ),
            ),
        ),
        emit=(
            EmitRegisterSlice(source=left, result=left_low, unit_count=2),
            EmitRegisterSlice(
                source=left,
                result=left_high,
                unit_offset=2,
                unit_count=2,
            ),
            *_vector_concat_merge_emits(
                left_high,
                right,
                result_high,
                left_byte_count=ValueTypeProject.static_dim_scaled(
                    left,
                    scale=element_byte_count,
                    addend=-64,
                ),
                remaining_byte_count=(
                    ValueTypeProject.literal_minus_static_dim_scaled(
                        left,
                        scale=element_byte_count,
                        literal=128,
                    )
                ),
                temporary_prefix="high_",
                result_type=DescriptorResultType(),
            ),
            EmitRegisterConcat(
                sources=(left_low, result_high),
                result=ValueRef.result("result"),
            ),
        ),
    )


def _vector_concat_split_carrier_rules(
    element_types: tuple[str, ...],
    element_byte_count: int,
    wide_lane_maximum: int,
) -> tuple[DescriptorRule, ...]:
    carrier_lane_count = 64 // element_byte_count
    right = ValueRef.operand("inputs", element=1)
    right_low = ValueRef.temporary("right_low")
    right_high = ValueRef.temporary("right_high")
    return (
        _vector_concat_narrow_left_wide_result_rule(
            element_types,
            element_byte_count,
            wide_lane_maximum,
            right_type=Vector(
                element_types,
                minimum_lanes=1,
                maximum_lanes=carrier_lane_count,
            ),
            result_lane_minimum=carrier_lane_count + 1,
            right_low=right,
            right_high=right,
            prepare_right=(),
        ),
        _vector_concat_narrow_left_wide_result_rule(
            element_types,
            element_byte_count,
            wide_lane_maximum,
            right_type=Vector(
                element_types,
                minimum_lanes=carrier_lane_count + 1,
                maximum_lanes=wide_lane_maximum - 1,
            ),
            result_lane_minimum=carrier_lane_count + 2,
            right_low=right_low,
            right_high=right_high,
            prepare_right=(
                EmitRegisterSlice(source=right, result=right_low, unit_count=2),
                EmitRegisterSlice(
                    source=right,
                    result=right_high,
                    unit_offset=2,
                    unit_count=2,
                ),
            ),
        ),
        _vector_concat_wide_left_rule(
            element_types,
            element_byte_count,
            wide_lane_maximum,
        ),
    )


def _wide_vector_concat_pair_rule(
    input_type: TypePattern,
    result_type: TypePattern,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=vector.vector_concat,
        guards=(
            Guard.i64_range("axis", 0, 0),
            Guard.operand_segment_count("inputs", 2),
            Guard.value_type("inputs", input_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            EmitRegisterConcat(
                sources=(
                    ValueRef.operand("inputs", element=0),
                    ValueRef.operand("inputs", element=1),
                ),
                result=ValueRef.result("result"),
            ),
        ),
    )


AIE2P_STRUCTURAL_RULES = (
    *(
        rule
        for (
            source_type,
            result_type,
            half_lane_count,
            storage,
            result_unit_count,
        ) in _WIDE_VECTOR_EXTRACT_SPECS
        for rule in (
            _wide_vector_extract_static_rule(
                source_type,
                result_type,
                half_lane_count,
                storage,
                high_half=False,
            ),
            _wide_vector_extract_static_rule(
                source_type,
                result_type,
                half_lane_count,
                storage,
                high_half=True,
            ),
            _wide_vector_extract_dynamic_rule(
                source_type,
                result_type,
                half_lane_count,
                storage,
                result_unit_count,
            ),
        )
    ),
    *(
        rule
        for element_types, element_byte_count, wide_lane_maximum in (
            _VECTOR_CARRIER_SPECS
        )
        for rule in _vector_slice_rules(
            element_types,
            element_byte_count,
            wide_lane_maximum,
        )
    ),
    *(
        rule
        for element_types, element_byte_count in _PACKED_VECTOR_ELEMENT_TYPES
        for rule in (
            _vector_concat_half_carrier_rule(
                element_types,
                element_byte_count,
            ),
            _vector_concat_shift_rule(
                element_types,
                element_byte_count,
            ),
        )
    ),
    *(
        _wide_vector_concat_pair_rule(input_type, result_type)
        for input_type, result_type in _WIDE_VECTOR_CONCAT_SPECS
    ),
    *(
        rule
        for element_types, element_byte_count, wide_lane_maximum in (
            _VECTOR_CARRIER_SPECS
        )
        for rule in _vector_concat_split_carrier_rules(
            element_types,
            element_byte_count,
            wide_lane_maximum,
        )
    ),
    _vector_deinterleave_i8x64_rule(),
    _vector_interleave_16bit_rule(),
    _vector_transpose_i32_f32_4x4_rule(),
    _vector_transpose_16bit_8x8_rule(),
    *_wide_vector_bitcast_alias_rules(),
)
