# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.avx512_predicate import avx512_predicate_rules
from loom.target.arch.x86.contracts.predicate import avx2_predicate_rules
from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.x86.vector_families import (
    AVX2_VECTOR_BIT_WIDTHS,
    AVX512_SELECT_MNEMONICS,
    FLOAT_ELEMENTS,
    INTEGER_ELEMENTS,
    STORAGE_ELEMENTS,
)
from loom.target.contracts import (
    AttrProject,
    DescriptorRule,
    EmitDescriptorOp,
    GuardKind,
    Scalar,
    Vector,
    descriptor_by_key,
)

_INTEGER_PREDICATES = (
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
_PREDICATE_LANE_COUNTS = (2, 4, 8, 16, 32)
_PREDICATE_SIMD_BIT_WIDTHS = (16, 32, 64, 128, 256)


def _descriptor(key: str):
    return descriptor_by_key(X86_AVX2_DESCRIPTOR_SET, key)


_RULES = tuple(
    case
    for case in avx2_predicate_rules(_descriptor)
    if isinstance(case, DescriptorRule)
)


def _rules_for(source_op) -> tuple[DescriptorRule, ...]:
    return tuple(rule for rule in _RULES if rule.source_op is source_op)


def _type_guard(rule: DescriptorRule, field: str):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind == GuardKind.VALUE_TYPE and guard.field == field
    )


def _carrier_register_class(rule: DescriptorRule, field: str) -> str:
    for guard in rule.guards:
        if guard.kind == GuardKind.LOW_VALUE_REGISTER_CLASS and guard.field == field:
            return guard.register_class
    # Predicate type and representation uniquely determine the physical carrier.
    representation = _representation_guard(rule, field)
    if representation == 1:
        return "x86.k"
    bit_width = _type_guard(rule, field).lanes * representation
    return (
        "x86.xmm" if bit_width <= 128 else "x86.ymm" if bit_width == 256 else "x86.zmm"
    )


def _representation_guard(rule: DescriptorRule, field: str) -> int:
    return next(
        guard.count
        for guard in rule.guards
        if guard.kind == GuardKind.LOW_VALUE_REPRESENTATION and guard.field == field
    )


def _enum_guard(rule: DescriptorRule, field: str) -> str:
    return next(
        guard.enum_keyword
        for guard in rule.guards
        if guard.kind == GuardKind.ENUM_ATTR_EQUALS and guard.field == field
    )


def _representation_bit_widths(lane_count: int) -> tuple[int, ...]:
    return tuple(
        bit_width
        for bit_width in _PREDICATE_SIMD_BIT_WIDTHS
        if bit_width % lane_count == 0 and bit_width // lane_count in (8, 16, 32, 64)
    )


def _register_class(bit_width: int) -> str:
    return {
        16: "x86.xmm",
        32: "x86.xmm",
        64: "x86.xmm",
        128: "x86.xmm",
        256: "x86.ymm",
    }[bit_width]


def _representation(lane_count: int, bit_width: int) -> int:
    return bit_width // lane_count


_PAYLOAD_NAMES_BY_BIT_WIDTH = {
    bit_width: tuple(
        element.name
        for element in (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS, *FLOAT_ELEMENTS)
        if element.bit_width == bit_width
    )
    for bit_width in (8, 16, 32, 64)
}


def test_predicate_construction_covers_every_callable_lane_count() -> None:
    constant_rules = _rules_for(vector.vector_splati)
    assert {
        (
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
            _representation_guard(rule, "result"),
            next(
                (guard.minimum, guard.maximum)
                for guard in rule.guards
                if guard.kind == GuardKind.I64_RANGE and guard.field == "value"
            ),
        )
        for rule in constant_rules
    } == {
        (
            Vector("i1", lanes=lane_count),
            _register_class(representation_bit_width),
            _representation(lane_count, representation_bit_width),
            (value, value),
        )
        for lane_count in _PREDICATE_LANE_COUNTS
        for representation_bit_width in _representation_bit_widths(lane_count)
        for value in (0, 1)
    }

    splat_rules = _rules_for(vector.vector_splat)
    assert {
        (
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
            _representation_guard(rule, "result"),
        )
        for rule in splat_rules
    } == {
        (
            Vector("i1", lanes=lane_count),
            _register_class(representation_bit_width),
            _representation(lane_count, representation_bit_width),
        )
        for lane_count in _PREDICATE_LANE_COUNTS
        for representation_bit_width in _representation_bit_widths(lane_count)
    }
    assert all(_type_guard(rule, "scalar") == Scalar("i1") for rule in splat_rules)


def test_predicate_extensions_cover_every_one_register_integer_result() -> None:
    for source_op in (vector.vector_extsi, vector.vector_extui):
        rules = _rules_for(source_op)
        assert {
            (
                _type_guard(rule, "input"),
                _carrier_register_class(rule, "input"),
                _representation_guard(rule, "input"),
                _type_guard(rule, "result"),
                _carrier_register_class(rule, "result"),
            )
            for rule in rules
        } == {
            (
                Vector("i1", lanes=lane_count),
                _register_class(source_bit_width),
                _representation(lane_count, source_bit_width),
                Vector(element.name, lanes=lane_count),
                _register_class(max(128, lane_count * element.bit_width)),
            )
            for lane_count in _PREDICATE_LANE_COUNTS
            for source_bit_width in _representation_bit_widths(lane_count)
            for element in INTEGER_ELEMENTS
            if lane_count * element.bit_width <= 256
        }
        assert all(
            not emit.descriptor.key.startswith("x86.scalar")
            for rule in rules
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
        )


def test_predicate_truncation_covers_every_one_register_integer_source() -> None:
    rules = _rules_for(vector.vector_trunci)
    assert {
        (
            _type_guard(rule, "input"),
            _carrier_register_class(rule, "input"),
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
            _representation_guard(rule, "result"),
        )
        for rule in rules
    } == {
        (
            Vector(element.name, lanes=lane_count),
            _register_class(max(128, lane_count * element.bit_width)),
            Vector("i1", lanes=lane_count),
            _register_class(result_bit_width),
            _representation(lane_count, result_bit_width),
        )
        for lane_count in _PREDICATE_LANE_COUNTS
        for element in INTEGER_ELEMENTS
        if lane_count * element.bit_width <= 256
        for result_bit_width in _representation_bit_widths(lane_count)
    }
    assert all(
        not emit.descriptor.key.startswith("x86.scalar")
        for rule in rules
        for emit in rule.emit
        if isinstance(emit, EmitDescriptorOp)
    )


def test_predicate_lane_movement_covers_every_representation() -> None:
    assert {
        (
            rule.source_op,
            _type_guard(
                rule,
                "source" if rule.source_op is vector.vector_extract else "dest",
            ),
            _carrier_register_class(
                rule,
                "source" if rule.source_op is vector.vector_extract else "result",
            ),
            _representation_guard(
                rule,
                "source" if rule.source_op is vector.vector_extract else "result",
            ),
        )
        for rule in (
            *_rules_for(vector.vector_extract),
            *_rules_for(vector.vector_insert),
        )
    } == {
        (
            source_op,
            Vector("i1", lanes=lane_count),
            _register_class(representation_bit_width),
            _representation(lane_count, representation_bit_width),
        )
        for source_op in (vector.vector_extract, vector.vector_insert)
        for lane_count in _PREDICATE_LANE_COUNTS
        for representation_bit_width in _representation_bit_widths(lane_count)
    }


def test_integer_comparisons_cover_every_element_predicate_and_carrier() -> None:
    assert {
        (
            _enum_guard(rule, "predicate"),
            _type_guard(rule, "lhs"),
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
            _representation_guard(rule, "result"),
        )
        for rule in _rules_for(vector.vector_cmpi)
    } == {
        (
            predicate,
            Vector(
                element.name,
                lanes=vector_bit_width // element.bit_width,
            ),
            Vector(
                "i1",
                lanes=vector_bit_width // element.bit_width,
            ),
            _register_class(representation_bit_width),
            _representation(
                vector_bit_width // element.bit_width,
                representation_bit_width,
            ),
        )
        for element in INTEGER_ELEMENTS
        for vector_bit_width in _PREDICATE_SIMD_BIT_WIDTHS
        if vector_bit_width // element.bit_width in _PREDICATE_LANE_COUNTS
        for predicate in _INTEGER_PREDICATES
        for representation_bit_width in _representation_bit_widths(
            vector_bit_width // element.bit_width
        )
    }


def test_float_comparisons_cover_every_element_and_carrier() -> None:
    rules = _rules_for(vector.vector_cmpf)
    assert {
        (
            _type_guard(rule, "lhs"),
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
            _representation_guard(rule, "result"),
        )
        for rule in rules
    } == {
        (
            Vector(
                element.name,
                lanes=vector_bit_width // element.bit_width,
            ),
            Vector(
                "i1",
                lanes=vector_bit_width // element.bit_width,
            ),
            _register_class(representation_bit_width),
            _representation(
                vector_bit_width // element.bit_width,
                representation_bit_width,
            ),
        )
        for element in FLOAT_ELEMENTS
        for vector_bit_width in _PREDICATE_SIMD_BIT_WIDTHS
        if vector_bit_width // element.bit_width in _PREDICATE_LANE_COUNTS
        for representation_bit_width in _representation_bit_widths(
            vector_bit_width // element.bit_width
        )
    }
    expected_projection = AttrProject.enum_remap("predicate", _FLOAT_COMPARE_IMMEDIATES)
    assert all(
        rule.emit[0].immediates["predicate"] == expected_projection for rule in rules
    )


def test_vector_select_covers_every_condition_and_payload_carrier_pair() -> None:
    assert {
        (
            _type_guard(rule, "condition"),
            _carrier_register_class(rule, "condition"),
            _representation_guard(rule, "condition"),
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
        )
        for rule in _rules_for(vector.vector_select)
    } == {
        (
            Vector("i1", lanes=lane_count),
            _register_class(condition_bit_width),
            _representation(lane_count, condition_bit_width),
            Vector(
                _PAYLOAD_NAMES_BY_BIT_WIDTH[payload_bit_width // lane_count],
                lanes=lane_count,
            ),
            _register_class(payload_bit_width),
        )
        for lane_count in _PREDICATE_LANE_COUNTS
        for condition_bit_width in _representation_bit_widths(lane_count)
        for payload_bit_width in _representation_bit_widths(lane_count)
    }


def test_whole_vector_select_covers_both_register_widths() -> None:
    rules = _rules_for(scf.scf_select)
    assert {_carrier_register_class(rule, "result") for rule in rules} == {
        "x86.xmm",
        "x86.ymm",
    }
    assert all(_type_guard(rule, "condition") == Scalar("i1") for rule in rules)


def test_predicate_logic_preserves_each_physical_representation() -> None:
    assert {
        (
            rule.source_op,
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
            _representation_guard(rule, "result"),
        )
        for source_op in (vector.vector_andi, vector.vector_ori, vector.vector_xori)
        for rule in _rules_for(source_op)
    } == {
        (
            source_op,
            Vector("i1", lanes=lane_count),
            _register_class(representation_bit_width),
            _representation(lane_count, representation_bit_width),
        )
        for source_op in (vector.vector_andi, vector.vector_ori, vector.vector_xori)
        for lane_count in _PREDICATE_LANE_COUNTS
        for representation_bit_width in _representation_bit_widths(lane_count)
    }


def test_predicate_rule_inventory_has_no_unclassified_rows() -> None:
    expected_count = sum(
        len(_representation_bit_widths(lane_count)) * 8
        for lane_count in _PREDICATE_LANE_COUNTS
    )
    expected_count += sum(
        len(_representation_bit_widths(vector_bit_width // element.bit_width))
        * len(_INTEGER_PREDICATES)
        for element in INTEGER_ELEMENTS
        for vector_bit_width in _PREDICATE_SIMD_BIT_WIDTHS
        if vector_bit_width // element.bit_width in _PREDICATE_LANE_COUNTS
    )
    expected_count += sum(
        len(_representation_bit_widths(vector_bit_width // element.bit_width))
        for element in FLOAT_ELEMENTS
        for vector_bit_width in _PREDICATE_SIMD_BIT_WIDTHS
        if vector_bit_width // element.bit_width in _PREDICATE_LANE_COUNTS
    )
    expected_count += sum(
        len(_representation_bit_widths(lane_count)) ** 2
        for lane_count in _PREDICATE_LANE_COUNTS
    )
    expected_count += sum(
        len(_representation_bit_widths(lane_count))
        * sum(lane_count * element.bit_width <= 256 for element in INTEGER_ELEMENTS)
        * 2
        for lane_count in _PREDICATE_LANE_COUNTS
    )
    expected_count += sum(
        len(_representation_bit_widths(lane_count))
        * sum(lane_count * element.bit_width <= 256 for element in INTEGER_ELEMENTS)
        for lane_count in _PREDICATE_LANE_COUNTS
    )
    expected_count += len(AVX2_VECTOR_BIT_WIDTHS)
    assert len(_RULES) == expected_count


def _avx512_descriptor(key: str):
    return descriptor_by_key(X86_AVX512_CORE_DESCRIPTOR_SET, key)


_AVX512_RULES = tuple(
    case
    for case in avx512_predicate_rules(_avx512_descriptor)
    if isinstance(case, DescriptorRule)
)
_AVX512_LANE_COUNTS = (2, 4, 8, 16, 32, 64)
_AVX512_CARRIER_CLASSES = {
    2: "x86.xmm",
    4: "x86.xmm",
    8: "x86.xmm",
    16: "x86.xmm",
    32: "x86.ymm",
    64: "x86.zmm",
}
_AVX512_CARRIER_REPRESENTATIONS = {
    2: 64,
    4: 32,
    8: 16,
    16: 8,
    32: 8,
    64: 8,
}


def _avx512_representation(lane_count: int, register_class: str) -> int:
    return (
        1 if register_class == "x86.k" else _AVX512_CARRIER_REPRESENTATIONS[lane_count]
    )


def _avx512_rules_for(source_op) -> tuple[DescriptorRule, ...]:
    return tuple(rule for rule in _AVX512_RULES if rule.source_op is source_op)


def _enum_set_guard(rule: DescriptorRule, field: str) -> frozenset[str]:
    return frozenset(
        next(
            guard.enum_keywords
            for guard in rule.guards
            if guard.kind == GuardKind.ENUM_ATTR_IN and guard.field == field
        )
    )


def _avx512_delta_representations(
    lane_count: int, vector_bit_width: int | None = None
) -> tuple[str, ...]:
    if lane_count == 64 or vector_bit_width == 512:
        return ("x86.k", _AVX512_CARRIER_CLASSES[lane_count])
    return ("x86.k",)


def _avx512_simd_representation_bit_widths(lane_count: int) -> tuple[int, ...]:
    return tuple(
        bit_width
        for bit_width in (16, 32, 64, 128, 256, 512)
        if bit_width % lane_count == 0
        and bit_width // lane_count in (8, 16, 32, 64)
        and (bit_width != 512 or lane_count == 64)
    )


def _avx512_register_class(vector_bit_width: int) -> str:
    return {
        16: "x86.xmm",
        32: "x86.xmm",
        64: "x86.xmm",
        128: "x86.xmm",
        256: "x86.ymm",
        512: "x86.zmm",
    }[vector_bit_width]


def test_avx512_predicate_construction_is_only_the_missing_delta() -> None:
    assert {
        (
            rule.source_op,
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
            _representation_guard(rule, "result"),
        )
        for rule in (
            *_avx512_rules_for(vector.vector_splati),
            *_avx512_rules_for(vector.vector_splat),
        )
    } == {
        (source_op, Vector("i1", minimum_lanes=2, maximum_lanes=64), "x86.k", 1)
        for source_op in (vector.vector_splati, vector.vector_splat)
    } | {
        (source_op, Vector("i1", lanes=64), "x86.zmm", 8)
        for source_op in (vector.vector_splati, vector.vector_splat)
    }


def test_avx512_predicate_lane_movement_is_only_the_missing_delta() -> None:
    assert {
        (
            rule.source_op,
            _type_guard(
                rule,
                "source" if rule.source_op is vector.vector_extract else "dest",
            ),
            _carrier_register_class(
                rule,
                "source" if rule.source_op is vector.vector_extract else "result",
            ),
            _representation_guard(
                rule,
                "source" if rule.source_op is vector.vector_extract else "result",
            ),
        )
        for rule in (
            *_avx512_rules_for(vector.vector_extract),
            *_avx512_rules_for(vector.vector_insert),
        )
    } == {
        (source_op, Vector("i1", minimum_lanes=2, maximum_lanes=64), "x86.k", 1)
        for source_op in (vector.vector_extract, vector.vector_insert)
    } | {
        (source_op, Vector("i1", lanes=64), "x86.zmm", 8)
        for source_op in (vector.vector_extract, vector.vector_insert)
    }


def test_avx512_predicate_extensions_cover_mask_and_512_bit_deltas() -> None:
    for source_op in (vector.vector_extsi, vector.vector_extui):
        rules = _avx512_rules_for(source_op)
        assert {
            (
                _type_guard(rule, "input"),
                _carrier_register_class(rule, "input"),
                _representation_guard(rule, "input"),
                _type_guard(rule, "result"),
                _carrier_register_class(rule, "result"),
            )
            for rule in rules
        } == {
            (
                Vector("i1", lanes=lane_count),
                (
                    "x86.k"
                    if source_bit_width is None
                    else _avx512_register_class(source_bit_width)
                ),
                1 if source_bit_width is None else source_bit_width // lane_count,
                Vector(element.name, lanes=lane_count),
                _avx512_register_class(max(128, lane_count * element.bit_width)),
            )
            for lane_count in _AVX512_LANE_COUNTS
            for source_bit_width in (
                None,
                *_avx512_simd_representation_bit_widths(lane_count),
            )
            for element in INTEGER_ELEMENTS
            if lane_count * element.bit_width <= 512
            if source_bit_width is None
            or lane_count * element.bit_width > 256
            or lane_count == 64
        }
        assert all(
            not emit.descriptor.key.startswith("x86.scalar")
            for rule in rules
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
        )


def test_avx512_predicate_truncation_covers_every_result_representation() -> None:
    rules = _avx512_rules_for(vector.vector_trunci)
    assert {
        (
            _type_guard(rule, "input"),
            _carrier_register_class(rule, "input"),
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
            _representation_guard(rule, "result"),
        )
        for rule in rules
    } == {
        (
            Vector(element.name, lanes=lane_count),
            _avx512_register_class(max(128, lane_count * element.bit_width)),
            Vector("i1", lanes=lane_count),
            (
                "x86.k"
                if result_bit_width is None
                else _avx512_register_class(result_bit_width)
            ),
            1 if result_bit_width is None else result_bit_width // lane_count,
        )
        for lane_count in _AVX512_LANE_COUNTS
        for element in INTEGER_ELEMENTS
        if lane_count * element.bit_width <= 512
        for result_bit_width in (
            None,
            *_avx512_simd_representation_bit_widths(lane_count),
        )
    }
    assert all(
        not emit.descriptor.key.startswith("x86.scalar")
        for rule in rules
        for emit in rule.emit
        if isinstance(emit, EmitDescriptorOp)
    )


def test_avx512_integer_compare_covers_all_widths_elements_and_predicates() -> None:
    assert {
        (
            _enum_set_guard(rule, "predicate"),
            _type_guard(rule, "lhs"),
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
            _representation_guard(rule, "result"),
        )
        for rule in _avx512_rules_for(vector.vector_cmpi)
    } == {
        (
            predicates,
            Vector(element.name, lanes=vector_bit_width // element.bit_width),
            Vector("i1", lanes=vector_bit_width // element.bit_width),
            register_class,
            _avx512_representation(
                vector_bit_width // element.bit_width, register_class
            ),
        )
        for vector_bit_width in (128, 256, 512)
        for element in INTEGER_ELEMENTS
        for predicates in (
            frozenset(_INTEGER_PREDICATES[:6]),
            frozenset(_INTEGER_PREDICATES[6:]),
        )
        for register_class in _avx512_delta_representations(
            vector_bit_width // element.bit_width, vector_bit_width
        )
    }


def test_avx512_float_compare_covers_all_widths_and_elements() -> None:
    assert {
        (
            _type_guard(rule, "lhs"),
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "result"),
            _representation_guard(rule, "result"),
        )
        for rule in _avx512_rules_for(vector.vector_cmpf)
    } == {
        (
            Vector(element.name, lanes=vector_bit_width // element.bit_width),
            Vector("i1", lanes=vector_bit_width // element.bit_width),
            register_class,
            _avx512_representation(
                vector_bit_width // element.bit_width, register_class
            ),
        )
        for vector_bit_width in (128, 256, 512)
        for element in FLOAT_ELEMENTS
        for register_class in _avx512_delta_representations(
            vector_bit_width // element.bit_width, vector_bit_width
        )
    }


def test_avx512_select_covers_all_widths_and_payload_elements() -> None:
    assert {
        (
            _type_guard(rule, "condition"),
            _type_guard(rule, "result"),
            _carrier_register_class(rule, "condition"),
            _representation_guard(rule, "condition"),
        )
        for rule in _avx512_rules_for(vector.vector_select)
    } == {
        (
            Vector("i1", lanes=vector_bit_width // element_bit_width),
            Vector(element_names, lanes=vector_bit_width // element_bit_width),
            register_class,
            _avx512_representation(
                vector_bit_width // element_bit_width, register_class
            ),
        )
        for vector_bit_width in (128, 256, 512)
        for mnemonic in dict.fromkeys(AVX512_SELECT_MNEMONICS.values())
        for element_names, element_bit_width in (
            (
                tuple(
                    element.name
                    for element in (
                        *INTEGER_ELEMENTS,
                        *STORAGE_ELEMENTS,
                        *FLOAT_ELEMENTS,
                    )
                    if AVX512_SELECT_MNEMONICS[element.name] == mnemonic
                ),
                next(
                    element.bit_width
                    for element in (
                        *INTEGER_ELEMENTS,
                        *STORAGE_ELEMENTS,
                        *FLOAT_ELEMENTS,
                    )
                    if AVX512_SELECT_MNEMONICS[element.name] == mnemonic
                ),
            ),
        )
        for register_class in _avx512_delta_representations(
            vector_bit_width // element_bit_width, vector_bit_width
        )
    }


def test_avx512_predicate_logic_covers_mask_and_missing_carrier() -> None:
    for source_op in (vector.vector_andi, vector.vector_ori, vector.vector_xori):
        assert {
            (
                _type_guard(rule, "result"),
                _carrier_register_class(rule, "result"),
                _representation_guard(rule, "result"),
            )
            for rule in _avx512_rules_for(source_op)
        } == {
            (Vector("i1", minimum_lanes=2, maximum_lanes=64), "x86.k", 1),
            (Vector("i1", lanes=64), "x86.zmm", 8),
        }


def test_avx512_predicate_rule_inventory_has_no_unclassified_rows() -> None:
    extension_count = sum(
        2
        for lane_count in _AVX512_LANE_COUNTS
        for source_bit_width in (
            None,
            *_avx512_simd_representation_bit_widths(lane_count),
        )
        for element in INTEGER_ELEMENTS
        if lane_count * element.bit_width <= 512
        if source_bit_width is None
        or lane_count * element.bit_width > 256
        or lane_count == 64
    )
    truncation_count = sum(
        1
        for lane_count in _AVX512_LANE_COUNTS
        for element in INTEGER_ELEMENTS
        if lane_count * element.bit_width <= 512
        for result_bit_width in (
            None,
            *_avx512_simd_representation_bit_widths(lane_count),
        )
    )
    assert len(_AVX512_RULES) == 80 + extension_count + truncation_count
