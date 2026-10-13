# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P native vector reduction contracts."""

from typing import Literal

from loom.target.arch.amd.xdna.aie2p.contracts.reduction import (
    _BF16_REDUCTION_SHAPES,
    _I64_REDUCTION_SHAPES,
    _NATIVE_INTEGER_REDUCTION_FAMILIES,
    AIE2P_REDUCTION_RULES,
    _emit_i64_extremum,
    _reduction_shapes,
    _reduction_shift_byte_counts,
)
from loom.target.contracts import (
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    Scalar,
    ValueRef,
    ValueTypeProject,
    Vector,
)

_U32_MASK = (1 << 32) - 1
_U64_MASK = (1 << 64) - 1


def _signed(value: int, width: int) -> int:
    sign = 1 << (width - 1)
    value &= (1 << width) - 1
    return value - (1 << width) if value & sign else value


def _evaluate_i64_extremum(
    lhs_value: int,
    rhs_value: int,
    *,
    operation: Literal["min", "max"],
    signedness: Literal["signed", "unsigned"],
) -> int:
    lhs = (
        ValueRef.temporary("lhs_low"),
        ValueRef.temporary("lhs_high"),
    )
    rhs = (
        ValueRef.temporary("rhs_low"),
        ValueRef.temporary("rhs_high"),
    )
    emits, result = _emit_i64_extremum(
        lhs,
        rhs,
        operation=operation,
        signedness=signedness,
        temporary_prefix="result_",
    )
    values: dict[ValueRef, tuple[int, ...] | int] = {
        lhs[0]: (lhs_value & _U32_MASK,),
        lhs[1]: ((lhs_value >> 32) & _U32_MASK,),
        rhs[0]: (rhs_value & _U32_MASK,),
        rhs[1]: ((rhs_value >> 32) & _U32_MASK,),
    }
    for emit in emits:
        assert isinstance(emit, EmitDescriptorOp)
        operands = {name: values[ref] for name, ref in emit.operands.items()}
        key = emit.descriptor.key
        if key.startswith(("amd.xdna.aie2p.min.", "amd.xdna.aie2p.max.")):
            lhs_words = operands["s1"]
            rhs_words = operands["s2"]
            assert isinstance(lhs_words, tuple)
            assert isinstance(rhs_words, tuple)
            compare = min if ".min." in key else max
            signed = ".signed." in key
            value: tuple[int, ...] | int = tuple(
                compare(
                    _signed(lhs_word, 32) if signed else lhs_word,
                    _signed(rhs_word, 32) if signed else rhs_word,
                )
                & _U32_MASK
                for lhs_word, rhs_word in zip(lhs_words, rhs_words, strict=True)
            )
        elif ".cmp.lt." in key:
            lhs_words = operands["s1"]
            rhs_words = operands["s2"]
            assert isinstance(lhs_words, tuple)
            assert isinstance(rhs_words, tuple)
            signed = ".signed." in key
            value = sum(
                (
                    (_signed(lhs_word, 32) if signed else lhs_word)
                    < (_signed(rhs_word, 32) if signed else rhs_word)
                )
                << index
                for index, (lhs_word, rhs_word) in enumerate(
                    zip(lhs_words, rhs_words, strict=True)
                )
            )
        elif key == "amd.xdna.aie2p.select.i32x16.mask64":
            false_words = operands["s1"]
            true_words = operands["s2"]
            selector = operands["sel"]
            assert isinstance(false_words, tuple)
            assert isinstance(true_words, tuple)
            assert isinstance(selector, int)
            value = tuple(
                true_word if selector & (1 << index) else false_word
                for index, (false_word, true_word) in enumerate(
                    zip(false_words, true_words, strict=True)
                )
            )
        else:
            raise AssertionError(f"unsupported i64 extremum operation {key}")
        assert len(emit.results) == 1
        values[next(iter(emit.results.values()))] = value

    low_words = values[result[0]]
    high_words = values[result[1]]
    assert isinstance(low_words, tuple)
    assert isinstance(high_words, tuple)
    return low_words[0] | (high_words[0] << 32)


def _input_type(
    element_type: str, minimum_lane_count: int, maximum_lane_count: int
) -> Vector:
    if minimum_lane_count == maximum_lane_count:
        return Vector(element_type, lanes=minimum_lane_count)
    return Vector(
        element_type,
        minimum_static_elements=minimum_lane_count,
        maximum_static_elements=maximum_lane_count,
    )


def test_extremum_reduction_shapes_partition_each_native_carrier() -> None:
    for maximum_lane_count in (8, 16, 32, 64):
        covered_lane_counts = []
        for minimum_lane_count, maximum, tree_lane_count in _reduction_shapes(
            maximum_lane_count
        ):
            covered_lane_counts.extend(range(minimum_lane_count, maximum + 1))
            assert tree_lane_count >= maximum
            assert tree_lane_count & (tree_lane_count - 1) == 0
            assert _reduction_shift_byte_counts(tree_lane_count, 1) == tuple(
                tree_lane_count >> stage
                for stage in range(1, tree_lane_count.bit_length())
            )
        assert covered_lane_counts == list(range(1, maximum_lane_count + 1))


def test_bf16_extremum_reductions_use_direct_and_padded_packed_trees() -> None:
    for kinds, operation, extremum_key in (
        (
            ("minnumf", "minimumf"),
            "minimum",
            "amd.xdna.aie2p.min.ge.bf16x32.native",
        ),
        (
            ("maxnumf", "maximumf"),
            "maximum",
            "amd.xdna.aie2p.max.lt.bf16x32.native",
        ),
    ):
        matching_rules = [
            rule
            for rule in AIE2P_REDUCTION_RULES
            if Guard.enum_attr_in("kind", kinds) in rule.guards
        ]
        assert len(matching_rules) == len(_BF16_REDUCTION_SHAPES)
        for rule, shape in zip(matching_rules, _BF16_REDUCTION_SHAPES, strict=True):
            minimum_lane_count, maximum_lane_count, tree_lane_count = shape
            input_type = _input_type("bf16", minimum_lane_count, maximum_lane_count)
            assert rule.guards == (
                Guard.enum_attr_in("kind", kinds),
                Guard.value_type("input", input_type),
                Guard.value_type("init", Scalar("bf16")),
                Guard.value_type("result", Scalar("bf16")),
                Guard.instance_flags_has_all("fastmath", "reassoc"),
                Guard.instance_flags_has_all("fastmath", "nnan"),
                Guard.instance_flags_has_all("fastmath", "nsz"),
            )

            descriptor_keys = [emit.descriptor.key for emit in rule.emit]
            expected_keys = ["amd.xdna.aie2p.splat.i16x32"]
            exact_tree = minimum_lane_count == maximum_lane_count == tree_lane_count
            if exact_tree:
                expected_keys.append(extremum_key)
            else:
                expected_keys.extend(
                    (
                        "amd.xdna.aie2p.constant.i32.predicate.low32",
                        "amd.xdna.aie2p.predicate.complete.zero.high32",
                        "amd.xdna.aie2p.select.i16x32.mask64",
                    )
                )
                predicate_constant = rule.emit[1]
                assert predicate_constant.immediates == {
                    "i": ValueTypeProject.static_dim_low_bits_mask(
                        ValueRef.operand("input")
                    )
                }
            for _ in _reduction_shift_byte_counts(tree_lane_count, 2):
                expected_keys.extend(
                    (
                        "amd.xdna.aie2p.constant.i32.mova",
                        "amd.xdna.aie2p.shift.bytes.x.configured",
                        extremum_key,
                    )
                )
            expected_keys.append("amd.xdna.aie2p.extract.i16.immediate")
            assert descriptor_keys == expected_keys
            assert [
                emit.immediates["i"]
                for emit in rule.emit
                if emit.descriptor.key == "amd.xdna.aie2p.constant.i32.mova"
            ] == list(_reduction_shift_byte_counts(tree_lane_count, 2))
            assert rule.emit[-1].immediates == {"idx": 0}
            assert rule.emit[-1].results == {"dst": ValueRef.result("result")}
            expected_report_mode = "packed" if exact_tree else "padded"
            assert rule.report_key == (f"bf16_{expected_report_mode}_{operation}_tree")


def test_integer_extremum_reductions_cover_every_kind_type_and_width() -> None:
    for kind, operation, signedness in (
        ("minsi", "min", "signed"),
        ("maxsi", "max", "signed"),
        ("minui", "min", "unsigned"),
        ("maxui", "max", "unsigned"),
    ):
        for element_type, element_width in _NATIVE_INTEGER_REDUCTION_FAMILIES:
            lane_count = 512 // element_width
            element_byte_count = element_width // 8
            scalar_type = Scalar(element_type)
            matching_rules = [
                rule
                for rule in AIE2P_REDUCTION_RULES
                if Guard.enum_attr_equals("kind", kind) in rule.guards
                and Guard.value_type("init", scalar_type) in rule.guards
            ]
            assert len(matching_rules) == len(_reduction_shapes(lane_count))
            extremum_key = (
                f"amd.xdna.aie2p.{operation}.{signedness}.i{element_width}x{lane_count}"
            )
            for rule, shape in zip(
                matching_rules, _reduction_shapes(lane_count), strict=True
            ):
                minimum_lane_count, maximum, tree_lane_count = shape
                assert rule.guards == (
                    Guard.enum_attr_equals("kind", kind),
                    Guard.value_type(
                        "input",
                        _input_type(element_type, minimum_lane_count, maximum),
                    ),
                    Guard.value_type("init", scalar_type),
                    Guard.value_type("result", scalar_type),
                )

                descriptor_keys = [emit.descriptor.key for emit in rule.emit]
                expected_keys = [f"amd.xdna.aie2p.splat.i{element_width}x{lane_count}"]
                exact_tree = minimum_lane_count == maximum == tree_lane_count
                if exact_tree:
                    expected_keys.append(extremum_key)
                else:
                    high_descriptor = (
                        "amd.xdna.aie2p.predicate.complete.constant.high32"
                        if minimum_lane_count > 32
                        else "amd.xdna.aie2p.predicate.complete.zero.high32"
                    )
                    select_suffix = (
                        f"i{element_width}x{lane_count}"
                        if element_width == 8
                        else f"i{element_width}x{lane_count}.mask64"
                    )
                    expected_keys.extend(
                        (
                            "amd.xdna.aie2p.constant.i32.predicate.low32",
                            high_descriptor,
                            f"amd.xdna.aie2p.select.{select_suffix}",
                        )
                    )
                    if minimum_lane_count > 32:
                        assert rule.emit[1].immediates == {"i": -1}
                        assert rule.emit[2].immediates == {
                            "i": ValueTypeProject.static_dim_low_bits_mask(
                                ValueRef.operand("input"), addend=-32
                            )
                        }
                    else:
                        assert rule.emit[1].immediates == {
                            "i": ValueTypeProject.static_dim_low_bits_mask(
                                ValueRef.operand("input")
                            )
                        }
                        assert rule.emit[2].immediates == {"i": 0}
                for _ in _reduction_shift_byte_counts(
                    tree_lane_count, element_byte_count
                ):
                    expected_keys.extend(
                        (
                            "amd.xdna.aie2p.constant.i32.mova",
                            "amd.xdna.aie2p.shift.bytes.x.configured",
                            extremum_key,
                        )
                    )
                expected_keys.append(
                    f"amd.xdna.aie2p.extract.i{element_width}.immediate"
                )
                assert descriptor_keys == expected_keys
                assert [
                    emit.immediates["i"]
                    for emit in rule.emit
                    if emit.descriptor.key == "amd.xdna.aie2p.constant.i32.mova"
                ] == list(
                    _reduction_shift_byte_counts(tree_lane_count, element_byte_count)
                )
                assert rule.emit[-1].immediates == {"idx": 0}
                assert rule.emit[-1].results == {"dst": ValueRef.result("result")}
                report_mode = "packed" if exact_tree else "padded"
                assert rule.report_key == (f"{element_type}_{report_mode}_{kind}_tree")


def test_i64_extremum_recipe_matches_lexicographic_ordering() -> None:
    samples = (
        0,
        1,
        (1 << 32) - 1,
        1 << 32,
        (1 << 32) + 1,
        (1 << 63) - 1,
        1 << 63,
        (1 << 63) + 1,
        (1 << 64) - 1,
    )
    for operation in ("min", "max"):
        combine = min if operation == "min" else max
        for signedness in ("signed", "unsigned"):
            for lhs in samples:
                for rhs in samples:
                    actual = _evaluate_i64_extremum(
                        lhs,
                        rhs,
                        operation=operation,
                        signedness=signedness,
                    )
                    ordered_lhs = _signed(lhs, 64) if signedness == "signed" else lhs
                    ordered_rhs = _signed(rhs, 64) if signedness == "signed" else rhs
                    assert actual == combine(ordered_lhs, ordered_rhs) & _U64_MASK


def test_i64_extremum_reductions_cover_every_kind_and_width() -> None:
    for kind, operation, signedness in (
        ("minsi", "min", "signed"),
        ("maxsi", "max", "signed"),
        ("minui", "min", "unsigned"),
        ("maxui", "max", "unsigned"),
    ):
        matching_rules = [
            rule
            for rule in AIE2P_REDUCTION_RULES
            if Guard.enum_attr_equals("kind", kind) in rule.guards
            and Guard.value_type("init", Scalar("i64")) in rule.guards
        ]
        assert len(matching_rules) == len(_I64_REDUCTION_SHAPES)
        extremum_keys = (
            f"amd.xdna.aie2p.{operation}.{signedness}.i32x16",
            f"amd.xdna.aie2p.{operation}.unsigned.i32x16",
            f"amd.xdna.aie2p.cmp.lt.{signedness}.i32x16.el.low32",
            f"amd.xdna.aie2p.cmp.lt.{signedness}.i32x16.el.low32",
            "amd.xdna.aie2p.select.i32x16.mask64",
            "amd.xdna.aie2p.select.i32x16.mask64",
        )
        for rule, shape in zip(matching_rules, _I64_REDUCTION_SHAPES, strict=True):
            minimum_lane_count, maximum_lane_count, tree_lane_count = shape
            assert rule.guards == (
                Guard.enum_attr_equals("kind", kind),
                Guard.value_type(
                    "input",
                    _input_type("i64", minimum_lane_count, maximum_lane_count),
                ),
                Guard.value_type("init", Scalar("i64")),
                Guard.value_type("result", Scalar("i64")),
            )

            exact_tree = minimum_lane_count == maximum_lane_count == tree_lane_count
            descriptor_keys = [
                emit.descriptor.key
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
            ]
            expected_keys = [
                "amd.xdna.aie2p.constant.i32.mova",
                "amd.xdna.aie2p.constant.i32.mova",
                "amd.xdna.aie2p.shuffle.x.configured",
                "amd.xdna.aie2p.shuffle.x.configured",
                "amd.xdna.aie2p.splat.i32x16",
                "amd.xdna.aie2p.splat.i32x16",
            ]
            if exact_tree:
                expected_keys.extend(extremum_keys)
            else:
                expected_keys.extend(
                    (
                        "amd.xdna.aie2p.constant.i32.predicate.low32",
                        "amd.xdna.aie2p.select.i32x16.mask64",
                        "amd.xdna.aie2p.select.i32x16.mask64",
                    )
                )
                predicate_constant = next(
                    emit
                    for emit in rule.emit
                    if isinstance(emit, EmitDescriptorOp)
                    and emit.descriptor.key
                    == "amd.xdna.aie2p.constant.i32.predicate.low32"
                )
                assert predicate_constant.immediates == {
                    "i": ValueTypeProject.static_dim_low_bits_mask(
                        ValueRef.operand("input")
                    )
                }
            for _ in _reduction_shift_byte_counts(tree_lane_count, 4):
                expected_keys.extend(
                    (
                        "amd.xdna.aie2p.constant.i32.mova",
                        "amd.xdna.aie2p.shift.bytes.x.configured",
                        "amd.xdna.aie2p.shift.bytes.x.configured",
                        *extremum_keys,
                    )
                )
            expected_keys.extend(
                (
                    "amd.xdna.aie2p.extract.i32.immediate",
                    "amd.xdna.aie2p.extract.i32.immediate",
                )
            )
            assert descriptor_keys == expected_keys
            assert [
                emit.immediates["i"]
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
                and emit.descriptor.key == "amd.xdna.aie2p.constant.i32.mova"
            ] == [4, 5, *_reduction_shift_byte_counts(tree_lane_count, 4)]
            assert sum(isinstance(emit, EmitRegisterSlice) for emit in rule.emit) == 2
            assert isinstance(rule.emit[-1], EmitRegisterConcat)
            assert rule.emit[-1].result == ValueRef.result("result")
            report_mode = "packed" if exact_tree else "padded"
            assert rule.report_key == f"i64_{report_mode}_{kind}_tree"
