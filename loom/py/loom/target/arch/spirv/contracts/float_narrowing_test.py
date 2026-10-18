# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.scalar import conversion
from loom.ir import ScalarType
from loom.scalar_type import ScalarTypeKind, scalar_type_name
from loom.target.arch.spirv.contracts.bfloat import bfloat_narrow_rules
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards,
    logical_core_descriptor,
)
from loom.target.arch.spirv.contracts.float_width import float_width_narrow_rules
from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.features import feature_bits_value
from loom.target.contracts import DescriptorRule, GuardKind

_FLOAT_KINDS = frozenset(
    {
        ScalarTypeKind.F8E4M3,
        ScalarTypeKind.F8E5M2,
        ScalarTypeKind.F16,
        ScalarTypeKind.BF16,
        ScalarTypeKind.F32,
        ScalarTypeKind.F64,
    }
)
_NON_FLOAT_KINDS = frozenset(
    {
        ScalarTypeKind.INDEX,
        ScalarTypeKind.OFFSET,
        ScalarTypeKind.I1,
        ScalarTypeKind.I8,
        ScalarTypeKind.I16,
        ScalarTypeKind.I32,
        ScalarTypeKind.I64,
    }
)


def _narrowing_pair(rule: DescriptorRule) -> tuple[str, str]:
    type_guards = {
        guard.field: guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE
    }
    input_type = type_guards["input"]
    result_type = type_guards["result"]
    assert input_type is not None and input_type.kind == "scalar"
    assert result_type is not None and result_type.kind == "scalar"
    assert len(input_type.elements) == 1
    assert len(result_type.elements) == 1
    return input_type.elements[0], result_type.elements[0]


def _expected_narrowing_pairs() -> frozenset[tuple[str, str]]:
    return frozenset(
        (scalar_type_name(source), scalar_type_name(result))
        for source in _FLOAT_KINDS
        for result in _FLOAT_KINDS
        if ScalarType(source).bitwidth > ScalarType(result).bitwidth
    )


def test_scalar_float_narrowing_rules_cover_the_complete_type_matrix() -> None:
    assert _FLOAT_KINDS | _NON_FLOAT_KINDS == frozenset(ScalarTypeKind)
    assert _FLOAT_KINDS.isdisjoint(_NON_FLOAT_KINDS)

    expected_pairs = _expected_narrowing_pairs()
    actual_pairs = {
        _narrowing_pair(rule)
        for rule in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule)
        and rule.source_op is conversion.scalar_fptrunc
    }

    assert len(expected_pairs) == 13
    assert actual_pairs == expected_pairs


def test_generic_narrowing_recipes_cover_every_pair() -> None:
    exact_recipe_pairs = {
        _narrowing_pair(rule)
        for rule in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule)
        and rule.source_op is conversion.scalar_fptrunc
        and rule.report_key.startswith("exact_")
    }

    assert exact_recipe_pairs == _expected_narrowing_pairs()


def test_float_width_fallbacks_recover_native_results_from_low_payloads() -> None:
    rules = {_narrowing_pair(rule): rule for rule in float_width_narrow_rules()}
    expected_tails = {
        ("f64", "f32"): ("i64", "v2f32", "f32"),
        ("f64", "f16"): ("i64", "v4f16", "f16"),
        ("f32", "f16"): ("i32", "v2f16", "f16"),
    }

    assert rules.keys() == expected_tails.keys()
    for pair, (carrier, packed, result) in expected_tails.items():
        bitcast, extract = rules[pair].emit[-2:]
        assert bitcast.descriptor.key == f"spirv.op_bitcast.{carrier}.{packed}"
        assert extract.descriptor.key == (
            f"spirv.op_composite_extract.{packed}.{result}"
        )
        assert extract.immediates == {"component_index": 0}


def test_bfloat_narrowing_preserves_the_profile_result_representation() -> None:
    # Native BF16 and Int16 are independent capabilities. Every narrowing row
    # must produce the representation selected by the profile's BF16 support,
    # including when the input is known not to be NaN.
    for source_kind in (ScalarTypeKind.F32, ScalarTypeKind.F64):
        source_type = scalar_type_name(source_kind)
        rules = tuple(
            rule
            for rule in bfloat_narrow_rules()
            if _narrowing_pair(rule) == (source_type, "bf16")
        )
        assert len(rules) == (8 if source_kind is ScalarTypeKind.F32 else 4)
        bit_width = ScalarType(source_kind).bitwidth
        vector_suffix = f"v{bit_width // 16}bf16"
        native_rules = tuple(
            rule for rule in rules if rule.descriptor.key.endswith(".bf16")
        )
        assert len(native_rules) == (4 if source_kind is ScalarTypeKind.F32 else 2)
        for rule in native_rules:
            bitcast, extract = rule.emit[-2:]
            assert bitcast.descriptor.key == (
                f"spirv.op_bitcast.i{bit_width}.{vector_suffix}"
            )
            assert extract.descriptor.key == (
                f"spirv.op_composite_extract.{vector_suffix}.bf16"
            )
            assert extract.immediates == {"component_index": 0}
        for native_bfloat in (False, True):
            for int16 in (False, True):
                features = feature_bits_value(
                    ("vulkan_shader", "float64", "int64")
                    + (("bfloat16_type_khr",) if native_bfloat else ())
                    + (("int16",) if int16 else ())
                )
                for satisfied_fact_guards in (
                    frozenset(),
                    frozenset({GuardKind.VALUE_NOT_NAN}),
                    frozenset({GuardKind.VALUE_FLOAT_EXACTLY_REPRESENTABLE_AS}),
                    frozenset(
                        {
                            GuardKind.VALUE_NOT_NAN,
                            GuardKind.VALUE_FLOAT_EXACTLY_REPRESENTABLE_AS,
                        }
                    ),
                ):
                    applicable = []
                    for rule in rules:
                        feature_masks = (
                            word
                            for guard in rule.guards
                            if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
                            for word in guard.descriptor.feature_mask_words
                        )
                        if any(mask & ~features for mask in feature_masks):
                            continue
                        if any(
                            guard.kind
                            in (
                                GuardKind.VALUE_NOT_NAN,
                                GuardKind.VALUE_FLOAT_EXACTLY_REPRESENTABLE_AS,
                            )
                            and guard.kind not in satisfied_fact_guards
                            for guard in rule.guards
                        ):
                            continue
                        applicable.append(rule)
                    assert applicable
                    first = applicable[0]
                    exact_finite_values = (
                        source_kind is ScalarTypeKind.F32
                        and GuardKind.VALUE_FLOAT_EXACTLY_REPRESENTABLE_AS
                        in satisfied_fact_guards
                    )
                    assert (
                        first.report_key.startswith("exactly_representable_")
                        == exact_finite_values
                    )
                    if exact_finite_values:
                        assert first.report_key.endswith("_preserve_nan") == (
                            GuardKind.VALUE_NOT_NAN not in satisfied_fact_guards
                        )
                    assert applicable[0].descriptor.key.endswith(".bf16") == (
                        native_bfloat
                    ), (
                        source_type,
                        native_bfloat,
                        int16,
                        satisfied_fact_guards,
                    )


def test_bfloat_narrowing_exact_value_rules_use_the_short_recipe() -> None:
    rules = tuple(
        rule
        for rule in bfloat_narrow_rules()
        if rule.report_key == "exactly_representable_binary32_to_bfloat16"
    )

    assert len(rules) == 2
    for rule in rules:
        assert any(
            guard.kind is GuardKind.VALUE_FLOAT_EXACTLY_REPRESENTABLE_AS
            for guard in rule.guards
        )
        assert any(guard.kind is GuardKind.VALUE_NOT_NAN for guard in rule.guards)
        assert tuple(emit.descriptor.key for emit in rule.emit[:3]) == (
            "spirv.op_bitcast.f32.i32",
            "spirv.op_constant.i32",
            "spirv.op_shift_right_logical.i32",
        )
        assert rule.emit[1].immediates == {"i32_value": 16}


def test_descriptor_feature_guards_deduplicate_only_identical_descriptors() -> None:
    f64_constant = logical_core_descriptor("spirv.op_constant.f64")
    f64_add = logical_core_descriptor("spirv.op_fadd.f64")
    featureless_i32_constant = logical_core_descriptor("spirv.op_constant.i32")

    guards = descriptor_feature_guards(
        f64_constant,
        f64_constant,
        f64_add,
        featureless_i32_constant,
    )

    assert tuple(guard.descriptor for guard in guards) == (f64_constant, f64_add)
