# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.contracts.extrema import extrema_rules
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.features import feature_bits_value
from loom.target.contracts import GuardKind


def _pattern(rule):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == "result"
    )


def _mode(rule):
    return next(
        (
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind is GuardKind.ENUM_ATTR_EQUALS
        ),
        None,
    )


def _permissions(rule):
    return {
        guard.enum_keyword
        for guard in rule.guards
        if guard.kind is GuardKind.INSTANCE_FLAGS_HAS_ALL
    }


def test_extrema_covers_all_formats_modes_and_native_shapes() -> None:
    actual = set()
    for rule in extrema_rules():
        rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
        pattern = _pattern(rule)
        actual.add(
            (
                rule.source_op.name,
                _mode(rule),
                pattern.kind,
                pattern.elements,
                pattern.lanes,
            )
        )
    expected = set()
    for operation in ("minimumf", "maximumf", "minnumf", "maxnumf", "clampf"):
        for mode in ("ordered", "number", "ieee") if operation == "clampf" else (None,):
            for element in ("f16", "bf16", "f32", "f64", "f8E4M3", "f8E5M2"):
                expected.add((f"scalar.{operation}", mode, "scalar", (element,), None))
                if element not in ("f8E4M3", "f8E5M2"):
                    for lanes in (1, 2, 3, 4):
                        expected.add(
                            (f"vector.{operation}", mode, "vector", (element,), lanes)
                        )
    assert actual == expected


def test_portable_extrema_never_depend_on_floating_arithmetic_or_controls() -> None:
    for rule in extrema_rules():
        if not rule.report_key:
            continue
        assert rule.report_key == "integer_float_extrema"
        for step in rule.emit:
            key = step.descriptor.key.removeprefix("spirv.op_")
            assert key.startswith(
                (
                    "constant.",
                    "composite_construct.",
                    "composite_extract.",
                    "bitcast.",
                    "bitwise_",
                    "logical_",
                    "shift_left_logical.",
                    "s_less_than.",
                    "s_greater_than.",
                    "i_equal.",
                    "select.",
                )
            ), key
            optional_controls = feature_bits_value(
                (
                    "float32_denorm_preserve",
                    "float16_64_denorm_preserve",
                    "float16_signed_zero_inf_nan_preserve",
                    "float32_signed_zero_inf_nan_preserve",
                    "float64_signed_zero_inf_nan_preserve",
                )
            )
            assert not any(
                mask & optional_controls for mask in step.descriptor.feature_mask_words
            )


def test_extrema_carriers_do_not_infer_int16_from_native_half_support() -> None:
    rules = extrema_rules()
    for element, optional_features, expected_suffix in (
        ("f16", ("float16",), "f16"),
        ("f16", ("float16", "int16"), "f16"),
        ("bf16", (), "i32"),
        ("bf16", ("int16",), "i32"),
        ("bf16", ("bfloat16_type_khr",), "bf16"),
        ("bf16", ("bfloat16_type_khr", "int16"), "bf16"),
        ("f32", (), "f32"),
        ("f64", ("float64", "int64"), "f64"),
        ("f8E4M3", ("int8",), "i8"),
        ("f8E5M2", ("int8",), "i8"),
    ):
        features = feature_bits_value(("vulkan_shader", *optional_features))
        selected = {}
        for rule in rules:
            if (
                not rule.source_op.name.startswith("scalar.")
                or _pattern(rule).elements != (element,)
                or _permissions(rule)
            ):
                continue
            if any(
                mask & ~features
                for guard in rule.guards
                if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
                for mask in guard.descriptor.feature_mask_words
            ):
                continue
            selected.setdefault((rule.source_op.name, _mode(rule)), rule)
        assert len(selected) == 7, (element, optional_features)
        for rule in selected.values():
            assert rule.descriptor.key.endswith(f".{expected_suffix}")
            if element in ("f16", "bf16") and "int16" not in optional_features:
                assert all(".i16" not in step.descriptor.key for step in rule.emit)


def test_ordered_clamp_retains_short_core_comparisons_with_required_controls() -> None:
    rules = [
        rule
        for rule in extrema_rules()
        if not rule.report_key and not _permissions(rule)
    ]
    assert len(rules) == 3 * 5
    for rule in rules:
        assert _mode(rule) == "ordered"
        assert len(rule.emit) == 4
        assert "f_ord_less_than.preserve" in rule.emit[0].descriptor.key
        assert "f_ord_greater_than.preserve" in rule.emit[2].descriptor.key
        element = _pattern(rule).elements[0]
        required = feature_bits_value(
            (
                "float32_denorm_preserve"
                if element == "f32"
                else "float16_64_denorm_preserve",
                f"float{element[1:]}_signed_zero_inf_nan_preserve",
            )
        )
        for step in (rule.emit[0], rule.emit[2]):
            assert any(
                mask & required == required
                for mask in step.descriptor.feature_mask_words
            )


def test_native_extrema_require_precise_permissions_and_preserve_denormals() -> None:
    native = [rule for rule in extrema_rules() if _permissions(rule)]
    assert len(native) == 3 * 5 * 7
    for rule in native:
        assert _permissions(rule) == {"nnan", "ninf", "nsz"}
        element = _pattern(rule).elements[0]
        denorm = feature_bits_value(
            (
                "float32_denorm_preserve"
                if element == "f32"
                else "float16_64_denorm_preserve",
            )
        )
        assert all(
            any(mask & denorm for mask in step.descriptor.feature_mask_words)
            for step in rule.emit
        )
        assert not rule.report_key
        if rule.source_op.name.endswith("clampf"):
            assert len(rule.emit) == 2
            assert "maxnumf" in rule.emit[0].descriptor.key
            assert "minnumf" in rule.emit[1].descriptor.key
        else:
            assert len(rule.emit) == 1
