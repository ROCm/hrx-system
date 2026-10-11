# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.target.arch.x86.contracts.avx2 import X86_AVX2_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.avx512 import X86_AVX512_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.scalar import X86_SCALAR_CONTRACT_FRAGMENT
from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.x86.vector_families import (
    AVX2_FLOAT_REDUCTION_OPERATIONS,
    AVX2_INTEGER_REDUCTION_FAMILIES,
    AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS,
    AVX2_PAYLOAD_ELEMENT_NAMES,
    AVX2_SCALAR_FLOAT_BINARY_FAMILIES,
    AVX2_SCALAR_FLOAT_FMA_MNEMONICS,
    AVX2_SHUFFLE_FAMILIES,
    AVX2_VECTOR_BIT_WIDTHS,
    AVX512_BITWISE_FAMILIES,
    AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS,
    AVX512_FLOAT_BINARY_FAMILIES,
    AVX512_FLOAT_FMA_MNEMONICS,
    AVX512_INTEGER_BINARY_FAMILIES,
    AVX512_INTEGER_REDUCTION_FAMILIES,
    AVX512_VECTOR_BIT_WIDTHS,
    AVX512VL_INTEGER_BINARY_FAMILIES,
    AVX512VL_VECTOR_BIT_WIDTHS,
    FLOAT_ELEMENTS,
    FLOAT_EXTREMA_OPERATIONS,
    INTEGER_ELEMENTS,
    STORAGE_ELEMENTS,
    X86_LANE_FAMILIES,
    X86_VECTOR_STORAGE_FAMILIES,
)
from loom.target.contracts import (
    DescriptorRule,
    GuardKind,
    OperandSegmentProject,
    Scalar,
    ValueAliasRule,
    ValueProject,
    ValueRef,
    Vector,
)


def _value_type_guard(rule: DescriptorRule, field: str):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind == GuardKind.VALUE_TYPE and guard.field == field
    )


def _static_index_range(rule: DescriptorRule) -> tuple[int, int]:
    guard = next(
        guard
        for guard in rule.guards
        if guard.kind == GuardKind.I64_ARRAY_ELEMENT_RANGE
        and guard.field == "static_indices"
        and guard.element == 0
    )
    assert guard.minimum is not None
    assert guard.maximum is not None
    return guard.minimum, guard.maximum


def test_scalar_signed_narrow_extensions_use_native_carriers() -> None:
    rules = tuple(
        case
        for case in X86_SCALAR_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op is scalar_conversion.scalar_extsi
        and _value_type_guard(case, "input") in (Scalar("i8"), Scalar("i16"))
    )
    assert {
        (
            _value_type_guard(rule, "input"),
            _value_type_guard(rule, "result"),
            rule.descriptor.key,
        )
        for rule in rules
    } == {
        (Scalar("i8"), Scalar("i32"), "x86.scalar.movsx.i8.gpr32"),
        (Scalar("i16"), Scalar("i32"), "x86.scalar.movsx.i16.gpr32"),
    }


def test_scalar_storage_float_constants_share_one_bit_preserving_rule() -> None:
    storage_type = Scalar(tuple(element.name for element in STORAGE_ELEMENTS))
    rules = tuple(
        case
        for case in X86_SCALAR_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op is scalar_conversion.scalar_constant
        and _value_type_guard(case, "result") == storage_type
    )
    assert len(rules) == 1
    rule = rules[0]
    assert rule.descriptor.key == "x86.scalar.movimm.gpr32"
    assert rule.emit[0].immediates == {"imm32": ValueProject.float_bits("result")}


def test_uniform_shift_rules_cover_native_types_widths_and_count_forms() -> None:
    operations = {vector.vector_shli, vector.vector_shrsi, vector.vector_shrui}
    for fragment, widths in (
        (X86_AVX2_CONTRACT_FRAGMENT, (128, 256)),
        (X86_AVX512_CONTRACT_FRAGMENT, (512,)),
    ):
        rules = tuple(
            case
            for case in fragment.cases
            if isinstance(case, DescriptorRule)
            and case.source_op in operations
            and any(guard.kind == GuardKind.VALUE_I64_RANGE for guard in case.guards)
        )
        actual = set()
        for rule in rules:
            count_range = next(
                guard
                for guard in rule.guards
                if guard.kind == GuardKind.VALUE_I64_RANGE
            )
            immediate = any(
                guard.kind == GuardKind.VALUE_EXACT_I64 for guard in rule.guards
            )
            actual.add(
                (
                    rule.source_op,
                    _value_type_guard(rule, "result"),
                    count_range.minimum,
                    count_range.maximum,
                    immediate,
                )
            )
            if immediate:
                assert rule.guards[0].kind == GuardKind.VALUE_EXACT_I64
                assert len(rule.emit) == 1
                assert rule.priority == 2
            else:
                origin = ValueRef.uniform_element_origin_operand("rhs")
                origin_guard = next(
                    guard for guard in rule.guards if guard.value_ref == origin
                )
                assert rule.guards[0] == origin_guard
                assert origin_guard.type_pattern == Scalar(
                    f"i{count_range.maximum + 1}"
                )
                first = rule.emit[0]
                if count_range.maximum == 15:
                    assert first.descriptor.key == "x86.scalar.movzx.u16.gpr32"
                    assert first.operands == {"src": origin}
                    assert len(rule.emit) == 3
                else:
                    assert first.descriptor.key == (
                        "x86.avx2.vmovq.xmm.gpr64"
                        if count_range.maximum == 63
                        else "x86.avx2.vmovd.xmm.gpr32"
                    )
                    assert first.operands == {"input": origin}
                    assert len(rule.emit) == 2
        expected = {
            (operation, Vector(f"i{bits}", lanes=width // bits), 0, bits - 1, immediate)
            for operation in operations
            for bits in (16, 32, 64)
            for width in widths
            for immediate in (
                (False, True) if fragment is X86_AVX2_CONTRACT_FRAGMENT else (True,)
            )
            if not (
                fragment is X86_AVX2_CONTRACT_FRAGMENT
                and operation is vector.vector_shrsi
                and bits == 64
            )
        }
        if fragment is X86_AVX512_CONTRACT_FRAGMENT:
            expected.update(
                (vector.vector_shrsi, Vector("i64", lanes=width // 64), 0, 63, True)
                for width in (128, 256)
            )
        assert actual == expected
        assert len(rules) == len(expected)


def test_scalar_float_memory_matches_integer_addressing_family() -> None:
    for source_op, field in ((view.view_load, "result"), (view.view_store, "value")):
        for float_type, integer_type, mnemonic in (
            ("f32", "i32", "vmovss"),
            ("f64", "i64", "vmovsd"),
        ):
            float_rules = tuple(
                case
                for case in X86_AVX2_CONTRACT_FRAGMENT.cases
                if isinstance(case, DescriptorRule)
                and case.source_op is source_op
                and _value_type_guard(case, field) == Scalar(float_type)
            )
            integer_rules = tuple(
                case
                for case in X86_SCALAR_CONTRACT_FRAGMENT.cases
                if isinstance(case, DescriptorRule)
                and case.source_op is source_op
                and _value_type_guard(case, field) == Scalar(integer_type)
            )
            # Static, direct, three factored scales, preserved source index,
            # materialized byte offset, and two full-width displacement forms.
            assert len(float_rules) == len(integer_rules) == 9
            for float_rule, integer_rule in zip(
                float_rules, integer_rules, strict=True
            ):
                assert replace(float_rule.emit[-1].source_memory, diagnostic=None) == (
                    replace(integer_rule.emit[-1].source_memory, diagnostic=None)
                )
                assert float_rule.descriptor.mnemonic == mnemonic
                assert len(float_rule.emit) == len(integer_rule.emit)
                for descriptors in (
                    X86_AVX2_DESCRIPTOR_SET,
                    X86_AVX512_CORE_DESCRIPTOR_SET,
                ):
                    assert float_rule.descriptor in descriptors.descriptors


def test_avx2_scalar_float_arithmetic_covers_f32_and_f64() -> None:
    source_operations = {
        scalar_arithmetic.scalar_addf: "addf",
        scalar_arithmetic.scalar_subf: "subf",
        scalar_arithmetic.scalar_mulf: "mulf",
        scalar_arithmetic.scalar_divf: "divf",
    }
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op in (*source_operations, scalar_math.scalar_fmaf)
    )
    assert {
        (
            source_operations[rule.source_op],
            _value_type_guard(rule, "result"),
        )
        for rule in rules
        if rule.source_op in source_operations
    } == {
        (row.source_operation, Scalar(row.element.name))
        for row in AVX2_SCALAR_FLOAT_BINARY_FAMILIES
    }
    assert {
        _value_type_guard(rule, "result")
        for rule in rules
        if rule.source_op is scalar_math.scalar_fmaf
    } == {Scalar(element.name) for element in FLOAT_ELEMENTS}
    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX2_DESCRIPTOR_SET.descriptors
    }
    assert {
        f"x86.avx2.{row.mnemonic}.xmm" for row in AVX2_SCALAR_FLOAT_BINARY_FAMILIES
    } <= descriptor_keys
    assert {
        f"x86.avx2.{mnemonic}.xmm"
        for mnemonic in AVX2_SCALAR_FLOAT_FMA_MNEMONICS.values()
    } <= descriptor_keys
    assert sum(rule.source_op in source_operations for rule in rules) == len(
        AVX2_SCALAR_FLOAT_BINARY_FAMILIES
    )
    assert sum(rule.source_op is scalar_math.scalar_fmaf for rule in rules) == len(
        FLOAT_ELEMENTS
    )


def test_avx2_lane_movement_covers_every_payload_type_and_width() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op in (vector.vector_extract, vector.vector_insert)
        and _value_type_guard(
            case, "result" if case.source_op is vector.vector_extract else "value"
        )
        != Scalar("i1")
    )
    expected = {
        (
            source_op.name,
            Vector(row.element_names, lanes=vector_bit_width // row.element_bit_width),
            Scalar(row.element_names),
            (0, vector_bit_width // row.element_bit_width - 1),
        )
        for row in X86_LANE_FAMILIES
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
        for source_op in (vector.vector_extract, vector.vector_insert)
    }
    expected.update(
        (
            source_op.name,
            Vector(element_name, lanes=vector_bit_width // element_bit_width),
            Scalar(element_name),
            (0, vector_bit_width // element_bit_width - 1),
        )
        for element_name, element_bit_width in (("f32", 32), ("f64", 64))
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
        for source_op in (vector.vector_extract, vector.vector_insert)
        if not (
            element_name == "f64"
            and vector_bit_width == 128
            and source_op is vector.vector_insert
        )
    )
    expected.update(
        (
            vector.vector_insert.name,
            Vector("f64", lanes=2),
            Scalar("f64"),
            (lane, lane),
        )
        for lane in range(2)
    )
    actual = {
        (
            rule.source_op.name,
            _value_type_guard(
                rule, "source" if rule.source_op is vector.vector_extract else "dest"
            ),
            _value_type_guard(
                rule, "result" if rule.source_op is vector.vector_extract else "value"
            ),
            _static_index_range(rule),
        )
        for rule in rules
    }
    assert actual == expected
    assert len(rules) == len(expected)


def test_exact_lane_construction_covers_every_payload_type_and_width() -> None:
    register_suffixes = {128: "xmm", 256: "ymm", 512: "zmm"}
    for fragment, vector_bit_widths, descriptor_prefix in (
        (
            X86_AVX2_CONTRACT_FRAGMENT,
            AVX2_VECTOR_BIT_WIDTHS,
            "x86.avx2.vmovdqu.rodata",
        ),
        (
            X86_AVX512_CONTRACT_FRAGMENT,
            AVX512_VECTOR_BIT_WIDTHS,
            "x86.avx512.vmovdqu64.rodata",
        ),
    ):
        rules = tuple(
            case
            for case in fragment.cases
            if isinstance(case, DescriptorRule)
            and case.source_op is vector.vector_from_elements
        )
        expected = {
            (
                Vector(
                    family.element_names,
                    lanes=vector_bit_width // family.element_bit_width,
                ),
                vector_bit_width // family.element_bit_width,
                family.element_bit_width,
                f"{descriptor_prefix}.{register_suffixes[vector_bit_width]}",
            )
            for family in X86_VECTOR_STORAGE_FAMILIES
            for vector_bit_width in vector_bit_widths
        }
        actual = set()
        for rule in rules:
            count_guard = next(
                guard
                for guard in rule.guards
                if guard.kind is GuardKind.OPERAND_SEGMENT_COUNT
            )
            exact_guard = next(
                guard
                for guard in rule.guards
                if guard.kind is GuardKind.OPERAND_SEGMENT_EXACT_BITS
            )
            projection = rule.emit[0].immediates["data"]
            assert isinstance(projection, OperandSegmentProject)
            assert projection.source_operand == "elements"
            assert projection.bit_width == exact_guard.bit_width
            actual.add(
                (
                    _value_type_guard(rule, "result"),
                    count_guard.count,
                    exact_guard.bit_width,
                    rule.descriptor.key,
                )
            )
        assert actual == expected
        assert len(rules) == len(expected)


def test_avx512_lane_movement_covers_every_zmm_payload_type() -> None:
    rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op in (vector.vector_extract, vector.vector_insert)
        and _value_type_guard(
            case, "result" if case.source_op is vector.vector_extract else "value"
        )
        != Scalar("i1")
    )
    expected = {
        (
            source_op.name,
            Vector(row.element_names, lanes=512 // row.element_bit_width),
            Scalar(row.element_names),
            (0, 512 // row.element_bit_width - 1),
        )
        for row in X86_LANE_FAMILIES
        for source_op in (vector.vector_extract, vector.vector_insert)
    }
    expected.update(
        (
            source_op.name,
            Vector(element_name, lanes=512 // element_bit_width),
            Scalar(element_name),
            (0, 512 // element_bit_width - 1),
        )
        for element_name, element_bit_width in (("f32", 32), ("f64", 64))
        for source_op in (vector.vector_extract, vector.vector_insert)
    )
    actual = {
        (
            rule.source_op.name,
            _value_type_guard(
                rule, "source" if rule.source_op is vector.vector_extract else "dest"
            ),
            _value_type_guard(
                rule, "result" if rule.source_op is vector.vector_extract else "value"
            ),
            _static_index_range(rule),
        )
        for rule in rules
    }
    assert actual == expected
    assert len(rules) == len(expected)


def test_avx2_shuffles_cover_every_payload_type_and_width() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_shuffle
    )
    expected = {
        Vector(
            family.element_names,
            lanes=vector_bit_width // family.element_bit_width,
        )
        for family in AVX2_SHUFFLE_FAMILIES
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    }
    assert {_value_type_guard(rule, "result") for rule in rules} == expected
    assert len(rules) == len(expected)
    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX2_DESCRIPTOR_SET.descriptors
    }
    assert {
        "x86.avx2.vpshufb.xmm",
        "x86.avx2.vpshufb.ymm",
        "x86.avx2.vpshufd.xmm",
        "x86.avx2.vpermps.ymm",
        "x86.avx2.vpermilpd.xmm",
        "x86.avx2.vpermq.ymm",
    } <= descriptor_keys


def test_avx512_shuffles_cover_every_payload_type() -> None:
    rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_shuffle
    )
    expected = {
        Vector(family.element_names, lanes=512 // family.element_bit_width)
        for family in AVX2_SHUFFLE_FAMILIES
    }
    assert {_value_type_guard(rule, "result") for rule in rules} == expected
    assert len(rules) == len(expected)
    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
    }
    assert {
        "x86.avx512.vmovdqu64.rodata.zmm",
        "x86.avx512.vpermd.zmm",
        "x86.avx512.vpermq.zmm",
        "x86.avx512.vpermw.zmm",
        "x86.avx512.vpsllw.zmm",
        "x86.avx512.vpsrlw.zmm",
    } <= descriptor_keys


def test_avx2_float_bitcasts_cover_every_storage_family_and_width() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, ValueAliasRule) and case.source_op is vector.vector_bitcast
    )
    expected = {
        (source_type, result_type)
        for float_elements, integer_element, element_bit_width in (
            (("f8E4M3", "f8E5M2"), "i8", 8),
            (("f16", "bf16"), "i16", 16),
            (("f32",), "i32", 32),
            (("f64",), "i64", 64),
        )
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
        for lane_count in (vector_bit_width // element_bit_width,)
        for source_type, result_type in (
            (
                Vector(float_elements, lanes=lane_count),
                Vector(integer_element, lanes=lane_count),
            ),
            (
                Vector(integer_element, lanes=lane_count),
                Vector(float_elements, lanes=lane_count),
            ),
        )
    }
    actual = {
        (
            _value_type_guard(rule, "input"),
            _value_type_guard(rule, "result"),
        )
        for rule in rules
    }
    assert actual == expected
    assert len(rules) == len(expected)


def test_avx512_construction_covers_every_zmm_payload_type() -> None:
    descriptor_rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )
    payload_elements = (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS, *FLOAT_ELEMENTS)
    expected_constant_types = {
        Vector(element.name, lanes=vector_bit_width // element.bit_width)
        for element in payload_elements
        for vector_bit_width in AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS
    }
    constant_rules = tuple(
        rule
        for rule in descriptor_rules
        if rule.source_op is vector.vector_constant
        and rule.descriptor.key.startswith("x86.avx512.vpbroadcast")
    )
    assert {_value_type_guard(rule, "result") for rule in constant_rules} == (
        expected_constant_types
    )
    assert len(constant_rules) == len(expected_constant_types)

    splat_rules = tuple(
        rule
        for rule in descriptor_rules
        if rule.source_op is vector.vector_splat
        and rule.descriptor.key.startswith(
            ("x86.avx512.vpbroadcast", "x86.avx512.vbroadcast")
        )
    )
    expected_splat_types = {
        *(
            Vector(element.name, lanes=vector_bit_width // element.bit_width)
            for element in (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS)
            for vector_bit_width in AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS
        ),
        *(
            Vector(element.name, lanes=512 // element.bit_width)
            for element in FLOAT_ELEMENTS
        ),
    }
    assert {_value_type_guard(rule, "result") for rule in splat_rules} == (
        expected_splat_types
    )
    assert len(splat_rules) == len(expected_splat_types)

    bitcast_rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, ValueAliasRule) and case.source_op is vector.vector_bitcast
    )
    expected_bitcasts = {
        (source_type, result_type)
        for float_elements, integer_element, element_bit_width in (
            (("f8E4M3", "f8E5M2"), "i8", 8),
            (("f16", "bf16"), "i16", 16),
            (("f32",), "i32", 32),
            (("f64",), "i64", 64),
        )
        for lane_count in (512 // element_bit_width,)
        for source_type, result_type in (
            (
                Vector(float_elements, lanes=lane_count),
                Vector(integer_element, lanes=lane_count),
            ),
            (
                Vector(integer_element, lanes=lane_count),
                Vector(float_elements, lanes=lane_count),
            ),
        )
    }
    assert {
        (
            _value_type_guard(rule, "input"),
            _value_type_guard(rule, "result"),
        )
        for rule in bitcast_rules
    } == expected_bitcasts
    assert len(bitcast_rules) == len(expected_bitcasts)


def test_avx512_arithmetic_covers_every_native_family() -> None:
    rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )
    register_suffixes = {128: "xmm", 256: "ymm", 512: "zmm"}
    expected_direct = {
        (
            f"x86.avx512.{family.mnemonic}.zmm",
            Vector(family.element.name, lanes=512 // family.element.bit_width),
        )
        for family in (*AVX512_INTEGER_BINARY_FAMILIES, *AVX512_FLOAT_BINARY_FAMILIES)
    } | {
        (
            f"x86.avx512.{family.mnemonic}.{register_suffixes[vector_bit_width]}",
            Vector(
                family.element.name,
                lanes=vector_bit_width // family.element.bit_width,
            ),
        )
        for family in AVX512VL_INTEGER_BINARY_FAMILIES
        for vector_bit_width in AVX512VL_VECTOR_BIT_WIDTHS
    }
    direct_descriptor_keys = {descriptor_key for descriptor_key, _ in expected_direct}
    actual_direct = {
        (rule.descriptor.key, _value_type_guard(rule, "result"))
        for rule in rules
        if rule.source_op is not vector.vector_reduce
        and rule.descriptor.key in direct_descriptor_keys
    }
    assert actual_direct == expected_direct
    assert len(actual_direct) == len(expected_direct)

    expected_fma = {
        (
            mnemonic,
            Vector(element.name, lanes=512 // element.bit_width),
        )
        for element in FLOAT_ELEMENTS
        for mnemonic in (AVX512_FLOAT_FMA_MNEMONICS[element.name],)
    }
    actual_fma = {
        (rule.descriptor.mnemonic, _value_type_guard(rule, "result"))
        for rule in rules
        if rule.source_op is vector.vector_fmaf
        and rule.descriptor.key.startswith("x86.avx512.")
    }
    assert actual_fma == expected_fma
    assert len(actual_fma) == len(expected_fma)

    bitwise_rules = tuple(
        rule
        for rule in rules
        if rule.source_op is not vector.vector_reduce
        and rule.descriptor.key
        in {f"x86.avx512.{mnemonic}.zmm" for _, mnemonic, _ in AVX512_BITWISE_FAMILIES}
        and _value_type_guard(rule, "result") != Vector("i1", lanes=64)
    )
    assert {
        (rule.source_op.name, rule.descriptor.mnemonic) for rule in bitwise_rules
    } == {
        (f"vector.{source_operation}", mnemonic)
        for source_operation, mnemonic, _ in AVX512_BITWISE_FAMILIES
    }
    assert {_value_type_guard(rule, "result") for rule in bitwise_rules} == {
        Vector(
            AVX2_PAYLOAD_ELEMENT_NAMES,
            minimum_lanes=2,
            maximum_lanes=max(AVX512_VECTOR_BIT_WIDTHS) // 8,
        )
    }


def test_avx2_iotas_cover_every_integer_element_and_width() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_iota
    )
    actual = {
        (
            _value_type_guard(rule, "base"),
            _value_type_guard(rule, "result"),
            rule.priority,
        )
        for rule in rules
    }
    expected = {
        (
            Scalar(element.name),
            Vector(
                element.name,
                lanes=vector_bit_width // element.bit_width,
            ),
            priority,
        )
        for element in INTEGER_ELEMENTS
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
        for priority in (0, 1)
    }
    assert actual == expected
    assert len(rules) == len(expected)
    assert all(
        any(
            guard.kind == GuardKind.VALUE_I64_RANGE
            and guard.field == "step"
            and guard.minimum == 1
            and guard.maximum == 1
            for guard in rule.guards
        )
        for rule in rules
        if rule.priority == 1
    )


def test_avx512_iotas_cover_every_zmm_integer_element() -> None:
    rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_iota
    )
    actual = {
        (
            _value_type_guard(rule, "base"),
            _value_type_guard(rule, "result"),
            rule.priority,
        )
        for rule in rules
    }
    expected = {
        (
            Scalar(element.name),
            Vector(element.name, lanes=512 // element.bit_width),
            priority,
        )
        for element in INTEGER_ELEMENTS
        for priority in (0, 1)
    }
    assert actual == expected
    assert len(rules) == len(expected)
    assert all(
        any(
            guard.kind == GuardKind.VALUE_I64_RANGE
            and guard.field == "step"
            and guard.minimum == 1
            and guard.maximum == 1
            for guard in rule.guards
        )
        for rule in rules
        if rule.priority == 1
    )


def test_float_extrema_cover_every_semantic_and_width() -> None:
    vector_source_operations = {
        vector.vector_minimumf: "minimumf",
        vector.vector_maximumf: "maximumf",
        vector.vector_minnumf: "minnumf",
        vector.vector_maxnumf: "maxnumf",
    }
    for fragment, vector_bit_widths in (
        (X86_AVX2_CONTRACT_FRAGMENT, AVX2_VECTOR_BIT_WIDTHS),
        (X86_AVX512_CONTRACT_FRAGMENT, AVX512_VECTOR_BIT_WIDTHS),
    ):
        extrema_rules = tuple(
            rule
            for rule in fragment.cases
            if isinstance(rule, DescriptorRule)
            and rule.source_op in vector_source_operations
        )
        assert {
            (
                vector_source_operations[rule.source_op],
                _value_type_guard(rule, "result"),
            )
            for rule in extrema_rules
        } == {
            (
                operation,
                Vector(
                    element.name,
                    lanes=vector_bit_width // element.bit_width,
                ),
            )
            for operation in FLOAT_EXTREMA_OPERATIONS
            for element in FLOAT_ELEMENTS
            for vector_bit_width in vector_bit_widths
        }
        priorities_by_cell = {}
        for rule in extrema_rules:
            key = (rule.source_op, _value_type_guard(rule, "result"))
            priorities_by_cell.setdefault(key, set()).add(rule.priority)
        assert {
            frozenset(priorities) for priorities in priorities_by_cell.values()
        } == {frozenset((0, 1))}
        fast_rules = tuple(rule for rule in extrema_rules if rule.priority == 1)
        assert all(
            {
                guard.enum_keyword
                for guard in rule.guards
                if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
            }
            == {"nnan", "nsz"}
            for rule in fast_rules
        )

    scalar_source_operations = {
        scalar_arithmetic.scalar_minimumf: "minimumf",
        scalar_arithmetic.scalar_maximumf: "maximumf",
        scalar_arithmetic.scalar_minnumf: "minnumf",
        scalar_arithmetic.scalar_maxnumf: "maxnumf",
    }
    actual_scalars = {
        (
            scalar_source_operations[rule.source_op],
            _value_type_guard(rule, "result"),
        )
        for rule in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule)
        and rule.source_op in scalar_source_operations
    }
    assert actual_scalars == {
        (operation, Scalar(element.name))
        for operation in FLOAT_EXTREMA_OPERATIONS
        for element in FLOAT_ELEMENTS
    }

    scalar_extrema_rules = tuple(
        rule
        for rule in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule)
        and rule.source_op in scalar_source_operations
    )
    priorities_by_cell = {}
    for rule in scalar_extrema_rules:
        key = (rule.source_op, _value_type_guard(rule, "result"))
        priorities_by_cell.setdefault(key, set()).add(rule.priority)
    assert {frozenset(priorities) for priorities in priorities_by_cell.values()} == {
        frozenset((0, 1))
    }
    fast_rules = tuple(rule for rule in scalar_extrema_rules if rule.priority == 1)
    assert all(
        {
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
        }
        == {"nnan", "nsz"}
        for rule in fast_rules
    )


def test_integer_reductions_cover_every_native_combine_family() -> None:
    for fragment, families, vector_bit_widths in (
        (
            X86_AVX2_CONTRACT_FRAGMENT,
            AVX2_INTEGER_REDUCTION_FAMILIES,
            AVX2_VECTOR_BIT_WIDTHS,
        ),
        (
            X86_AVX512_CONTRACT_FRAGMENT,
            AVX512_INTEGER_REDUCTION_FAMILIES,
            AVX512_VECTOR_BIT_WIDTHS,
        ),
    ):
        expected = {
            (
                row.source_operation,
                Vector(
                    row.element.name,
                    lanes=vector_bit_width // row.element.bit_width,
                ),
            )
            for row in families
            for vector_bit_width in vector_bit_widths
        }
        expected_operations = {row.source_operation for row in families}
        integer_rules = tuple(
            rule
            for rule in fragment.cases
            if isinstance(rule, DescriptorRule)
            and rule.source_op is vector.vector_reduce
            and any(
                guard.kind == GuardKind.ENUM_ATTR_EQUALS
                and guard.field == "kind"
                and guard.enum_keyword in expected_operations
                for guard in rule.guards
            )
        )
        actual = {
            (
                next(
                    guard.enum_keyword
                    for guard in rule.guards
                    if guard.kind == GuardKind.ENUM_ATTR_EQUALS
                    and guard.field == "kind"
                ),
                _value_type_guard(rule, "input"),
            )
            for rule in integer_rules
        }
        assert actual == expected
        assert len(integer_rules) == len(expected)


def test_float_reductions_and_dots_cover_both_types_and_all_widths() -> None:
    for fragment, vector_bit_widths in (
        (X86_AVX2_CONTRACT_FRAGMENT, AVX2_VECTOR_BIT_WIDTHS),
        (X86_AVX512_CONTRACT_FRAGMENT, AVX512_VECTOR_BIT_WIDTHS),
    ):
        reduction_rules = tuple(
            rule
            for rule in fragment.cases
            if isinstance(rule, DescriptorRule)
            and rule.source_op is vector.vector_reduce
            and any(
                guard.kind == GuardKind.ENUM_ATTR_EQUALS
                and guard.enum_keyword in AVX2_FLOAT_REDUCTION_OPERATIONS
                for guard in rule.guards
            )
        )
        packed_reduction_rules = tuple(
            rule
            for rule in reduction_rules
            if next(
                guard.enum_keyword
                for guard in rule.guards
                if guard.kind == GuardKind.ENUM_ATTR_EQUALS
            )
            in AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS
        )
        assert len(packed_reduction_rules) == (
            len(AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS)
            * len(FLOAT_ELEMENTS)
            * len(vector_bit_widths)
            * 2
        )
        assert {
            (
                next(
                    guard.enum_keyword
                    for guard in rule.guards
                    if guard.kind == GuardKind.ENUM_ATTR_EQUALS
                ),
                _value_type_guard(rule, "input"),
                next(
                    guard.kind
                    for guard in rule.guards
                    if guard.kind
                    in (
                        GuardKind.INSTANCE_FLAGS_HAS_ALL,
                        GuardKind.INSTANCE_FLAGS_HAS_NONE,
                    )
                ),
            )
            for rule in packed_reduction_rules
        } == {
            (
                operation,
                Vector(
                    element.name,
                    lanes=vector_bit_width // element.bit_width,
                ),
                fastmath_guard,
            )
            for operation in AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS
            for element in FLOAT_ELEMENTS
            for vector_bit_width in vector_bit_widths
            for fastmath_guard in (
                GuardKind.INSTANCE_FLAGS_HAS_NONE,
                GuardKind.INSTANCE_FLAGS_HAS_ALL,
            )
        }

        fast_extrema_rules = tuple(
            rule
            for rule in reduction_rules
            if next(
                guard.enum_keyword
                for guard in rule.guards
                if guard.kind == GuardKind.ENUM_ATTR_EQUALS
            )
            in FLOAT_EXTREMA_OPERATIONS
        )
        assert len(fast_extrema_rules) == (
            len(FLOAT_EXTREMA_OPERATIONS) * len(FLOAT_ELEMENTS) * len(vector_bit_widths)
        )
        assert {
            (
                next(
                    guard.enum_keyword
                    for guard in rule.guards
                    if guard.kind == GuardKind.ENUM_ATTR_EQUALS
                ),
                _value_type_guard(rule, "input"),
                frozenset(
                    guard.enum_keyword
                    for guard in rule.guards
                    if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
                ),
            )
            for rule in fast_extrema_rules
        } == {
            (
                operation,
                Vector(
                    element.name,
                    lanes=vector_bit_width // element.bit_width,
                ),
                frozenset(("reassoc", "nnan", "nsz")),
            )
            for operation in FLOAT_EXTREMA_OPERATIONS
            for element in FLOAT_ELEMENTS
            for vector_bit_width in vector_bit_widths
        }

        dot_rules = tuple(
            rule
            for rule in fragment.cases
            if isinstance(rule, DescriptorRule) and rule.source_op is vector.vector_dotf
        )
        assert len(dot_rules) == len(FLOAT_ELEMENTS) * len(vector_bit_widths)
        assert {
            (
                _value_type_guard(rule, "lhs"),
                _value_type_guard(rule, "rhs"),
                _value_type_guard(rule, "init"),
                _value_type_guard(rule, "result"),
            )
            for rule in dot_rules
        } == {
            (vector_type, vector_type, scalar_type, scalar_type)
            for element in FLOAT_ELEMENTS
            for vector_bit_width in vector_bit_widths
            for vector_type in (
                Vector(
                    element.name,
                    lanes=vector_bit_width // element.bit_width,
                ),
            )
            for scalar_type in (Scalar(element.name),)
        }
