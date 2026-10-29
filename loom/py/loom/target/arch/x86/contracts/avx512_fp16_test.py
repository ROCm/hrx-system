# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.target.arch.x86.contracts.avx512_fp16 import (
    X86_AVX512_FP16_CONTRACT_DIALECT_OPS,
    X86_AVX512_FP16_CONTRACT_FRAGMENT,
)
from loom.target.arch.x86.descriptors import (
    X86_AVX512_FEATURES_DESCRIPTOR_SET,
    X86_AVX512_FP16_DESCRIPTOR_SET,
)
from loom.target.arch.x86.feature_bits import FEATURE_AVX512_FP16
from loom.target.arch.x86.vector_families import (
    AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS,
    AVX512_FP16_FLOAT_EXTREMA_MNEMONICS,
    AVX512_FP16_SCALAR_FLOAT_BINARY_FAMILIES,
    AVX512_FP16_SCALAR_FLOAT_EXTREMA_MNEMONICS,
    AVX512_FP16_SCALAR_FLOAT_FMA_MNEMONIC,
    AVX512_FP16_VECTOR_BIT_WIDTHS,
    FLOAT_EXTREMA_OPERATIONS,
)
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    GuardKind,
    Scalar,
    ValueAliasRule,
    Vector,
    compile_lower_rule_set,
)

_SCALAR_BINARY_OPS = {
    scalar_arithmetic.scalar_addf: "addf",
    scalar_arithmetic.scalar_subf: "subf",
    scalar_arithmetic.scalar_mulf: "mulf",
    scalar_arithmetic.scalar_divf: "divf",
}
_REGISTER_SUFFIXES = {
    32: "xmm",
    64: "xmm",
    128: "xmm",
    256: "ymm",
    512: "zmm",
}
_SCALAR_EXTREMA_OPS = {
    scalar_arithmetic.scalar_minimumf: "minimumf",
    scalar_arithmetic.scalar_maximumf: "maximumf",
    scalar_arithmetic.scalar_minnumf: "minnumf",
    scalar_arithmetic.scalar_maxnumf: "maxnumf",
}
_VECTOR_EXTREMA_OPS = {
    vector.vector_minimumf: "minimumf",
    vector.vector_maximumf: "maximumf",
    vector.vector_minnumf: "minnumf",
    vector.vector_maxnumf: "maxnumf",
}


def _rules_for(source_op) -> tuple[DescriptorRule, ...]:
    return tuple(
        case
        for case in X86_AVX512_FP16_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is source_op
    )


def _value_type(rule: DescriptorRule, field: str):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind == GuardKind.VALUE_TYPE and guard.field == field
    )


def _register_class(rule: DescriptorRule, field: str) -> str | None:
    return next(
        (
            guard.register_class
            for guard in rule.guards
            if guard.kind == GuardKind.LOW_VALUE_REGISTER_CLASS and guard.field == field
        ),
        None,
    )


def _representation(rule: DescriptorRule, field: str) -> int:
    return next(
        guard.count
        for guard in rule.guards
        if guard.kind == GuardKind.LOW_VALUE_REPRESENTATION and guard.field == field
    )


def _predicate_representations(lane_count: int) -> set[int]:
    representations = {1}
    for vector_bit_width in (16, 32, 64, 128, 256, 512):
        if vector_bit_width % lane_count:
            continue
        element_bit_width = vector_bit_width // lane_count
        if element_bit_width not in (8, 16, 32, 64):
            continue
        if vector_bit_width == 512 and lane_count != 64:
            continue
        representations.add(element_bit_width)
    return representations


def _descriptor_keys(rule: DescriptorRule) -> tuple[str, ...]:
    return tuple(
        emit.descriptor.key for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
    )


def _enum_keyword(rule: DescriptorRule, field: str) -> str:
    return next(
        guard.enum_keyword
        for guard in rule.guards
        if guard.kind == GuardKind.ENUM_ATTR_EQUALS and guard.field == field
    )


def _static_index_lanes(rule: DescriptorRule) -> range:
    guard = next(
        guard
        for guard in rule.guards
        if guard.kind is GuardKind.I64_ARRAY_ELEMENT_RANGE
        and guard.field == "static_indices"
        and guard.element == 0
    )
    assert guard.minimum is not None
    assert guard.maximum is not None
    return range(guard.minimum, guard.maximum + 1)


def _has_positive_zero_guard(rule: DescriptorRule) -> bool:
    return any(
        guard.kind is GuardKind.VALUE_FLOAT_EQUALS
        and guard.field == "dest"
        and guard.f64_value == 0.0
        and math.copysign(1.0, guard.f64_value) > 0.0
        for guard in rule.guards
    )


def test_feature_view_contains_the_complete_fp16_overlay() -> None:
    feature_keys = {
        descriptor.key for descriptor in X86_AVX512_FEATURES_DESCRIPTOR_SET.descriptors
    }
    assert {
        descriptor.key for descriptor in X86_AVX512_FP16_DESCRIPTOR_SET.descriptors
    } <= feature_keys


def test_fragment_compiles_every_authored_rule() -> None:
    compiled = compile_lower_rule_set(
        X86_AVX512_FP16_CONTRACT_FRAGMENT,
        dialect_ops=X86_AVX512_FP16_CONTRACT_DIALECT_OPS,
    )
    assert len(compiled.rules) == len(X86_AVX512_FP16_CONTRACT_FRAGMENT.cases)


def test_base_transport_and_alias_rules_require_the_fp16_representation() -> None:
    for case in X86_AVX512_FP16_CONTRACT_FRAGMENT.cases:
        descriptors = [
            guard.descriptor
            for guard in case.guards
            if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
        ]
        if isinstance(case, DescriptorRule) and case.descriptor is not None:
            descriptors.append(case.descriptor)
        assert any(
            descriptor.feature_mask_words
            and descriptor.feature_mask_words[0] & FEATURE_AVX512_FP16
            for descriptor in descriptors
        ), case


def test_scalar_arithmetic_covers_the_native_fp16_family() -> None:
    expected_binary = {
        (
            getattr(scalar_arithmetic, f"scalar_{row.source_operation}"),
            f"x86.avx512_fp16.{row.mnemonic}.xmm",
        )
        for row in AVX512_FP16_SCALAR_FLOAT_BINARY_FAMILIES
    }
    actual_binary = {
        (rule.source_op, rule.descriptor.key)
        for source_op in _SCALAR_BINARY_OPS
        for rule in _rules_for(source_op)
    }
    assert actual_binary == expected_binary
    assert all(
        _value_type(rule, "result") == Scalar("f16")
        for source_op in _SCALAR_BINARY_OPS
        for rule in _rules_for(source_op)
    )

    fma_rules = _rules_for(scalar_math.scalar_fmaf)
    assert {
        (rule.descriptor.key, _value_type(rule, "result")) for rule in fma_rules
    } == {
        (
            f"x86.avx512_fp16.{AVX512_FP16_SCALAR_FLOAT_FMA_MNEMONIC}.xmm",
            Scalar("f16"),
        )
    }


def test_comparisons_cover_scalar_and_every_vector_representation() -> None:
    scalar_rules = _rules_for(scalar_comparison.scalar_cmpf)
    assert len(scalar_rules) == 1
    assert _value_type(scalar_rules[0], "lhs") == Scalar("f16")
    assert _value_type(scalar_rules[0], "result") == Scalar("i1")

    assert {
        (
            _value_type(rule, "lhs"),
            _value_type(rule, "result"),
            _representation(rule, "result"),
        )
        for rule in _rules_for(vector.vector_cmpf)
    } == {
        (
            Vector("f16", lanes=lane_count),
            Vector("i1", lanes=lane_count),
            representation,
        )
        for lane_count in (2, 4, 8, 16, 32)
        for representation in _predicate_representations(lane_count)
    }

    assert {
        (
            _value_type(rule, "condition"),
            _value_type(rule, "result"),
            _representation(rule, "condition"),
        )
        for rule in _rules_for(vector.vector_select)
    } == {
        (
            Vector("i1", lanes=lane_count),
            Vector("f16", lanes=lane_count),
            representation,
        )
        for lane_count in (2, 4)
        for representation in _predicate_representations(lane_count)
    }


def test_scalar_carrier_rules_preserve_the_existing_fp16_surface() -> None:
    assert {
        (_value_type(rule, "input"), _value_type(rule, "result"))
        for rule in _rules_for(scalar_conversion.scalar_bitcast)
    } == {
        (Scalar("f16"), Scalar("i16")),
        (Scalar("i16"), Scalar("f16")),
    }
    assert {_value_type(rule, "result") for rule in _rules_for(scf.scf_select)} == {
        Scalar("f16")
    }
    assert {
        _value_type(rule, "result")
        for rule in _rules_for(scalar_conversion.scalar_constant)
    } == {Scalar("f16")}
    assert {
        _value_type(rule, "result") for rule in _rules_for(vector.vector_splat)
    } == {
        Vector("f16", lanes=2),
        Vector("f16", lanes=4),
        Vector("f16", lanes=8),
        Vector("f16", lanes=16),
        Vector("f16", lanes=32),
    }
    assert {rule.priority for rule in _rules_for(vector.vector_splat)} == {2}
    constant_rules = (
        *_rules_for(vector.vector_splati),
        *_rules_for(vector.vector_splatf),
    )
    assert {_value_type(rule, "result") for rule in constant_rules} == {
        Vector("i16", lanes=4),
        Vector("f16", lanes=4),
    }
    assert all(
        _descriptor_keys(rule)
        == (
            "x86.scalar.movimm.gpr32",
            "x86.avx2.vmovd.xmm.gpr32",
            "x86.avx2.vpbroadcastw.xmm",
        )
        for rule in constant_rules
    )
    assert {
        (_value_type(rule, "input"), _value_type(rule, "result"))
        for rule in X86_AVX512_FP16_CONTRACT_FRAGMENT.cases
        if isinstance(rule, ValueAliasRule) and rule.source_op is vector.vector_bitcast
    } == {
        (Vector("f16", lanes=4), Vector("i16", lanes=4)),
        (Vector("i16", lanes=4), Vector("f16", lanes=4)),
    }


def test_low_xmm_shuffle_covers_both_logical_payload_types() -> None:
    rules = _rules_for(vector.vector_shuffle)
    assert len(rules) == 1
    rule = rules[0]
    assert _value_type(rule, "source") == Vector(("i16", "f16"), lanes=4)
    assert _value_type(rule, "result") == Vector(("i16", "f16"), lanes=4)
    assert _descriptor_keys(rule) == ("x86.avx2.vpshuflw.xmm",)


def test_conversions_cover_every_representable_native_width() -> None:
    scalar_pairs = {
        source_op: {
            (_value_type(rule, "input"), _value_type(rule, "result"))
            for rule in _rules_for(source_op)
        }
        for source_op in (
            scalar_conversion.scalar_extf,
            scalar_conversion.scalar_fptrunc,
        )
    }
    assert scalar_pairs == {
        scalar_conversion.scalar_extf: {(Scalar("f16"), Scalar("f32"))},
        scalar_conversion.scalar_fptrunc: {(Scalar("f32"), Scalar("f16"))},
    }

    vector_pairs = {
        source_op: {
            (_value_type(rule, "input"), _value_type(rule, "result"))
            for rule in _rules_for(source_op)
        }
        for source_op in (vector.vector_extf, vector.vector_fptrunc)
    }
    assert vector_pairs == {
        vector.vector_extf: {
            (Vector("f16", lanes=2), Vector("f32", lanes=2)),
            (Vector("f16", lanes=4), Vector("f32", lanes=4)),
            (Vector("f16", lanes=8), Vector("f32", lanes=8)),
            (Vector("f16", lanes=16), Vector("f32", lanes=16)),
        },
        vector.vector_fptrunc: {
            (Vector("f32", lanes=2), Vector("f16", lanes=2)),
            (Vector("f32", lanes=4), Vector("f16", lanes=4)),
            (Vector("f32", lanes=8), Vector("f16", lanes=8)),
            (Vector("f32", lanes=16), Vector("f16", lanes=16)),
        },
    }


def test_scalar_memory_preserves_every_addressing_recipe() -> None:
    for source_op, type_field, transport_key in (
        (view.view_load, "result", "x86.avx2.vmovd.xmm.gpr32"),
        (view.view_store, "value", "x86.avx2.vmovd.gpr32.xmm"),
    ):
        rules = _rules_for(source_op)
        assert len(rules) == 9
        assert all(_value_type(rule, type_field) == Scalar("f16") for rule in rules)
        for rule in rules:
            descriptor_keys = _descriptor_keys(rule)
            assert descriptor_keys.count(transport_key) == 1
            memory_emits = tuple(
                emit
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
                and emit.descriptor.semantic_tag.startswith("memory.")
            )
            assert len(memory_emits) == 1
            assert memory_emits[0].source_memory is not None
            assert memory_emits[0].source_memory.element_byte_count == 2
            assert memory_emits[0].source_memory.vector_lane_count == 1


def test_low_xmm_memory_preserves_every_addressing_recipe() -> None:
    for source_op, type_field in (
        (vector.vector_load, "result"),
        (vector.vector_store, "value"),
    ):
        rules = tuple(
            rule
            for rule in _rules_for(source_op)
            if _value_type(rule, type_field) == Vector("f16", lanes=4)
        )
        assert len(rules) == 9
        for rule in rules:
            memory_emits = tuple(
                emit
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
                and emit.descriptor.semantic_tag.startswith("memory.")
            )
            assert len(memory_emits) == 1
            assert memory_emits[0].source_memory is not None
            assert memory_emits[0].source_memory.element_byte_count == 2
            assert memory_emits[0].source_memory.vector_lane_count == 4
            assert memory_emits[0].descriptor.key.startswith("x86.avx2.vmovsd.")
            assert not any(
                key
                in (
                    "x86.avx2.vmovd.xmm.gpr32",
                    "x86.avx2.vmovd.gpr32.xmm",
                )
                for key in _descriptor_keys(rule)
            )


def test_lane_movement_preserves_the_xmm_scalar_carrier() -> None:
    for source_op, vector_field, scalar_field, transport_key in (
        (
            vector.vector_extract,
            "source",
            "result",
            "x86.avx2.vmovd.xmm.gpr32",
        ),
        (
            vector.vector_insert,
            "dest",
            "value",
            "x86.avx2.vmovd.gpr32.xmm",
        ),
    ):
        rules = _rules_for(source_op)
        assert {
            (_value_type(rule, vector_field), _value_type(rule, scalar_field))
            for rule in rules
        } == {
            (Vector("f16", lanes=2), Scalar("f16")),
            (Vector("f16", lanes=4), Scalar("f16")),
            (Vector("f16", lanes=8), Scalar("f16")),
            (Vector("f16", lanes=16), Scalar("f16")),
            (Vector("f16", lanes=32), Scalar("f16")),
        }
        assert all(transport_key in _descriptor_keys(rule) for rule in rules)


def test_zero_insert_covers_every_wide_native_fp16_lane() -> None:
    rules = tuple(
        rule
        for rule in _rules_for(vector.vector_insert)
        if rule.priority == 2 and _has_positive_zero_guard(rule)
    )
    actual_cells = [
        (_value_type(rule, "dest"), lane)
        for rule in rules
        for lane in _static_index_lanes(rule)
    ]
    expected_cells = {
        (Vector("f16", lanes=vector_bit_width // 16), lane)
        for vector_bit_width in AVX512_FP16_VECTOR_BIT_WIDTHS
        if vector_bit_width > 128
        for lane in range(vector_bit_width // 16)
    }
    assert set(actual_cells) == expected_cells
    assert len(actual_cells) == len(expected_cells)
    assert all(_value_type(rule, "value") == Scalar("f16") for rule in rules)
    assert all(_register_class(rule, "value") == "x86.xmm" for rule in rules)
    assert all("x86.avx2.vmovd.gpr32.xmm" in _descriptor_keys(rule) for rule in rules)


def test_extrema_cover_exact_and_native_fast_semantics() -> None:
    expected_cells = {
        (source_op, Scalar("f16"), priority)
        for source_op in _SCALAR_EXTREMA_OPS
        for priority in (1, 2)
    } | {
        (source_op, Vector("f16", lanes=bit_width // 16), priority)
        for source_op in _VECTOR_EXTREMA_OPS
        for bit_width in AVX512_FP16_VECTOR_BIT_WIDTHS
        for priority in (1, 2)
    }
    rules = tuple(
        rule
        for source_op in (*_SCALAR_EXTREMA_OPS, *_VECTOR_EXTREMA_OPS)
        for rule in _rules_for(source_op)
    )
    assert {
        (rule.source_op, _value_type(rule, "result"), rule.priority) for rule in rules
    } == expected_cells

    exact_rules = tuple(rule for rule in rules if rule.priority == 1)
    assert all(len(rule.emit) == 9 for rule in exact_rules)
    assert all(
        rule.descriptor.key.startswith("x86.avx512.vpblendmw.") for rule in exact_rules
    )

    fast_rules = tuple(rule for rule in rules if rule.priority == 2)
    assert all(
        {
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
        }
        == {"nnan", "nsz"}
        for rule in fast_rules
    )
    expected_fast_descriptors = {
        f"x86.avx512_fp16.{AVX512_FP16_SCALAR_FLOAT_EXTREMA_MNEMONICS[operation]}.xmm"
        for operation in FLOAT_EXTREMA_OPERATIONS
    } | {
        "x86.avx512_fp16."
        f"{AVX512_FP16_FLOAT_EXTREMA_MNEMONICS[operation]}."
        f"{register_suffix}"
        for operation in FLOAT_EXTREMA_OPERATIONS
        for register_suffix in {
            _REGISTER_SUFFIXES[bit_width] for bit_width in AVX512_FP16_VECTOR_BIT_WIDTHS
        }
    }
    assert {rule.descriptor.key for rule in fast_rules} == expected_fast_descriptors


def test_reductions_preserve_fp16_accumulation_semantics() -> None:
    rules = _rules_for(vector.vector_reduce)
    packed_rules = tuple(
        rule
        for rule in rules
        if _enum_keyword(rule, "kind") in AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS
    )
    assert {
        (
            _enum_keyword(rule, "kind"),
            _value_type(rule, "input"),
            next(
                guard.kind
                for guard in rule.guards
                if guard.field == "fastmath" and guard.enum_keyword == "reassoc"
            ),
        )
        for rule in packed_rules
    } == {
        (
            operation,
            Vector("f16", lanes=bit_width // 16),
            reassociation_guard,
        )
        for operation in AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS
        for bit_width in AVX512_FP16_VECTOR_BIT_WIDTHS
        for reassociation_guard in (
            GuardKind.INSTANCE_FLAGS_HAS_NONE,
            GuardKind.INSTANCE_FLAGS_HAS_ALL,
        )
    }
    assert all(_value_type(rule, "init") == Scalar("f16") for rule in rules)
    assert all(_value_type(rule, "result") == Scalar("f16") for rule in rules)

    extrema_rules = tuple(
        rule
        for rule in rules
        if _enum_keyword(rule, "kind") in FLOAT_EXTREMA_OPERATIONS
    )
    assert {
        (_enum_keyword(rule, "kind"), _value_type(rule, "input"))
        for rule in extrema_rules
    } == {
        (operation, Vector("f16", lanes=bit_width // 16))
        for operation in FLOAT_EXTREMA_OPERATIONS
        for bit_width in AVX512_FP16_VECTOR_BIT_WIDTHS
    }
    assert all(
        {
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
        }
        == {"reassoc", "nnan", "nsz"}
        for rule in extrema_rules
    )
    assert all(
        not any("vpermil" in key for key in _descriptor_keys(rule)) for rule in rules
    )
    ordered_f16x4 = next(
        rule
        for rule in packed_rules
        if _enum_keyword(rule, "kind") == "addf"
        and _value_type(rule, "input") == Vector("f16", lanes=4)
        and any(
            guard.kind == GuardKind.INSTANCE_FLAGS_HAS_NONE
            and guard.field == "fastmath"
            and guard.enum_keyword == "reassoc"
            for guard in rule.guards
        )
    )
    assert _descriptor_keys(ordered_f16x4).count("x86.avx512_fp16.vaddsh.xmm") == 4
    assert tuple(
        emit.immediates["bytes"]
        for emit in ordered_f16x4.emit
        if emit.descriptor.key == "x86.avx2.vpsrldq.xmm"
    ) == (2, 4, 6)

    reassociated_f16x4 = next(
        rule
        for rule in packed_rules
        if _enum_keyword(rule, "kind") == "addf"
        and _value_type(rule, "input") == Vector("f16", lanes=4)
        and any(
            guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
            and guard.field == "fastmath"
            and guard.enum_keyword == "reassoc"
            for guard in rule.guards
        )
    )
    assert tuple(
        emit.immediates["bytes"]
        for emit in reassociated_f16x4.emit
        if emit.descriptor.key == "x86.avx2.vpsrldq.xmm"
    ) == (4, 2)


def test_dots_accumulate_with_ordered_scalar_fp16_fma() -> None:
    rules = _rules_for(vector.vector_dotf)
    assert {
        (
            _value_type(rule, "lhs"),
            _value_type(rule, "rhs"),
            _value_type(rule, "init"),
            _value_type(rule, "result"),
        )
        for rule in rules
    } == {
        (
            Vector("f16", lanes=bit_width // 16),
            Vector("f16", lanes=bit_width // 16),
            Scalar("f16"),
            Scalar("f16"),
        )
        for bit_width in AVX512_FP16_VECTOR_BIT_WIDTHS
    }
    assert all(
        rule.descriptor.key
        == f"x86.avx512_fp16.{AVX512_FP16_SCALAR_FLOAT_FMA_MNEMONIC}.xmm"
        for rule in rules
    )
    assert all("x86.avx2.vpsrldq.xmm" in _descriptor_keys(rule) for rule in rules)
    assert all(
        not any("vpermil" in key for key in _descriptor_keys(rule)) for rule in rules
    )
    f16x4_rule = next(
        rule for rule in rules if _value_type(rule, "lhs") == Vector("f16", lanes=4)
    )
    assert (
        _descriptor_keys(f16x4_rule).count(
            f"x86.avx512_fp16.{AVX512_FP16_SCALAR_FLOAT_FMA_MNEMONIC}.xmm"
        )
        == 4
    )
