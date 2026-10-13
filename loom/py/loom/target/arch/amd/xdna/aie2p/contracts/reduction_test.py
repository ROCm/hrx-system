# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P native vector reduction contracts."""

from loom.target.arch.amd.xdna.aie2p.contracts.reduction import (
    _BF16_REDUCTION_SHAPES,
    AIE2P_REDUCTION_RULES,
    _bf16_reduction_shift_byte_counts,
)
from loom.target.contracts import (
    Guard,
    Scalar,
    ValueRef,
    ValueTypeProject,
    Vector,
)


def _input_type(minimum_lane_count: int, maximum_lane_count: int) -> Vector:
    if minimum_lane_count == maximum_lane_count:
        return Vector("bf16", lanes=minimum_lane_count)
    return Vector(
        "bf16",
        minimum_static_elements=minimum_lane_count,
        maximum_static_elements=maximum_lane_count,
    )


def test_bf16_extremum_reduction_shapes_partition_the_native_carrier() -> None:
    covered_lane_counts = []
    for (
        minimum_lane_count,
        maximum_lane_count,
        tree_lane_count,
    ) in _BF16_REDUCTION_SHAPES:
        covered_lane_counts.extend(range(minimum_lane_count, maximum_lane_count + 1))
        assert tree_lane_count >= maximum_lane_count
        assert tree_lane_count & (tree_lane_count - 1) == 0
        assert _bf16_reduction_shift_byte_counts(tree_lane_count) == tuple(
            2 * lane_count
            for lane_count in (
                tree_lane_count >> stage
                for stage in range(1, tree_lane_count.bit_length())
            )
        )
    assert covered_lane_counts == list(range(1, 33))


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
            input_type = _input_type(minimum_lane_count, maximum_lane_count)
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
            for _ in _bf16_reduction_shift_byte_counts(tree_lane_count):
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
            ] == list(_bf16_reduction_shift_byte_counts(tree_lane_count))
            assert rule.emit[-1].immediates == {"idx": 0}
            assert rule.emit[-1].results == {"dst": ValueRef.result("result")}
            expected_report_mode = "packed" if exact_tree else "padded"
            assert rule.report_key == (f"bf16_{expected_report_mode}_{operation}_tree")
