# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.contracts.float_width import float_width_widen_rules
from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.features import feature_bits_value
from loom.target.arch.spirv.ordinary_vector_conversion import (
    NATIVE_VECTOR_SCALAR_CONVERSIONS,
)
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    GuardKind,
    ValueRef,
)


def test_numeric_source_rules_cover_native_shapes_and_scalar_singletons() -> None:
    operations = {
        f"vector.{conversion.source_op_key}"
        for conversion in NATIVE_VECTOR_SCALAR_CONVERSIONS
    }
    actual = {}
    for rule in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases:
        if (
            not isinstance(rule, DescriptorRule)
            or rule.source_op.name not in operations
            or rule.report_key
        ):
            continue
        patterns = {
            guard.field: guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE
        }
        source = patterns["input"]
        result = patterns["result"]
        assert source.kind == result.kind == "vector"
        assert source.lanes == result.lanes
        key = (rule.source_op.name, source.element, result.element, source.lanes)
        assert key not in actual
        actual[key] = rule

    expected = {
        (
            f"vector.{row.source_op_key}",
            row.source_type.source_type,
            row.result_type.source_type,
            lane_count,
        ): row
        for row in NATIVE_VECTOR_SCALAR_CONVERSIONS
        for lane_count in (1, 2, 3, 4)
    }
    assert len(expected) == 288
    assert actual.keys() == expected.keys()

    for key, row in expected.items():
        lane_count = key[3]
        prefix = "" if lane_count == 1 else f"v{lane_count}"
        descriptor_key = (
            f"spirv.op_{row.descriptor_suffix}."
            f"{prefix}{row.source_type.suffix}.{prefix}{row.result_type.suffix}"
        )
        rule = actual[key]
        assert rule.descriptor.key == descriptor_key
        assert rule.emit[-1].results == {"dst": ValueRef.result("result")}
        needs_input_view = row.source_op_key in ("extui", "uitofp")
        needs_result_view = row.source_op_key in ("extui", "fptoui")
        assert len(rule.emit) == 1 + needs_input_view + needs_result_view
        conversion = rule.emit[int(needs_input_view)]
        assert conversion.descriptor == rule.descriptor
        if needs_input_view:
            assert conversion.operands == {
                "input": ValueRef.temporary("unsigned_input")
            }
            assert rule.emit[0].result_types == {"dst": DescriptorResultType()}
        else:
            assert conversion.operands == {"input": ValueRef.operand("input")}
        if needs_result_view:
            assert conversion.results == {"dst": ValueRef.temporary("unsigned_result")}
            assert conversion.result_types == {"dst": DescriptorResultType()}
            result_pattern = rule.emit[-1].result_types["dst"]
            assert result_pattern.kind == "vector"
            assert result_pattern.element == row.result_type.source_type
            assert result_pattern.lanes == lane_count
        availability = tuple(
            guard.descriptor.key
            for guard in rule.guards
            if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
        )
        expected_availability = (descriptor_key,) if row.feature_bits else ()
        assert availability == expected_availability


def test_ieee_widening_covers_all_shapes_without_integer_width_features() -> None:
    rules = float_width_widen_rules()
    actual = set()
    for rule in rules:
        rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
        patterns = {
            guard.field: guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE
        }
        source, result = patterns["input"], patterns["result"]
        assert source.kind == result.kind
        assert source.lanes == result.lanes
        actual.add((source.elements[0], result.elements[0], source.lanes))
        features = feature_bits_value(("vulkan_shader", "float16", "float64"))
        for step in rule.emit:
            assert not any(
                mask & ~features for mask in step.descriptor.feature_mask_words
            )
        assert rule.report_key.startswith("exact_binary")
        assert rule.emit[-1].results == {"dst": ValueRef.result("result")}
        # Normalization remains a single native-vector conversion, not one
        # floating conversion per lane or a call back into source legalization.
        conversions = [
            step for step in rule.emit if ".op_convert_s_to_f." in step.descriptor.key
        ]
        assert len(conversions) == 1
        prefix = "" if source.lanes in (None, 1) else f"v{source.lanes}"
        assert (
            conversions[0].descriptor.key
            == f"spirv.op_convert_s_to_f.{prefix}i32.{prefix}f32"
        )
    assert actual == {
        (source, result, lanes)
        for source, result in (("f16", "f32"), ("f16", "f64"), ("f32", "f64"))
        for lanes in (None, 1, 2, 3, 4)
    }


def test_ieee_widening_prefers_native_only_with_every_required_guarantee() -> None:
    for source, result in (("f16", "f32"), ("f16", "f64"), ("f32", "f64")):
        controls = (
            f"{source.replace('f', 'float')}_denorm_preserve",
            f"{source.replace('f', 'float')}_signed_zero_inf_nan_preserve",
            f"{result.replace('f', 'float')}_signed_zero_inf_nan_preserve",
        )
        for lanes in (None, 1, 2, 3, 4):
            matching = []
            for rule in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases:
                if not isinstance(rule, DescriptorRule) or rule.source_op.name != (
                    "scalar.extf" if lanes is None else "vector.extf"
                ):
                    continue
                patterns = {
                    guard.field: guard.type_pattern
                    for guard in rule.guards
                    if guard.kind is GuardKind.VALUE_TYPE
                }
                if (
                    patterns["input"].elements == (source,)
                    and patterns["result"].elements == (result,)
                    and patterns["input"].lanes == lanes
                ):
                    matching.append(rule)
            for missing in (None, *controls):
                features = feature_bits_value(
                    (
                        "vulkan_shader",
                        "float16",
                        "float64",
                        *(control for control in controls if control != missing),
                    )
                )
                selected = next(
                    rule
                    for rule in matching
                    if not any(
                        mask & ~features
                        for guard in rule.guards
                        if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
                        for mask in guard.descriptor.feature_mask_words
                    )
                )
                assert bool(selected.report_key) == (missing is not None)
                assert (len(selected.emit) == 1) == (missing is None)
