# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.amd.xdna.aie2p.contracts.predicate_range import (
    _COORDINATE_TYPES,
    AIE2P_PREDICATE_RANGE_RULES,
)
from loom.target.contracts import Guard, Vector


def test_predicate_range_rules_cover_one_native_carrier() -> None:
    partitions = ((1, 31), (32, 32), (33, 63), (64, 64))
    assert len(AIE2P_PREDICATE_RANGE_RULES) == len(_COORDINATE_TYPES) * len(partitions)

    for coordinate_index, coordinate in enumerate(_COORDINATE_TYPES):
        coordinate_rules = AIE2P_PREDICATE_RANGE_RULES[
            coordinate_index * len(partitions) : (coordinate_index + 1)
            * len(partitions)
        ]
        for rule, (minimum_lanes, maximum_lanes) in zip(
            coordinate_rules, partitions, strict=True
        ):
            result_type = Vector(
                "i1",
                minimum_static_elements=minimum_lanes,
                maximum_static_elements=maximum_lanes,
            )
            assert rule.guards == (
                Guard.value_type("lower_bound", coordinate.pattern),
                Guard.value_type("upper_bound", coordinate.pattern),
                Guard.value_type("step", coordinate.pattern),
                Guard.value_type("result", result_type),
                Guard.value_i64_range(
                    "lower_bound",
                    0,
                    coordinate.maximum - (maximum_lanes - 1),
                ),
                Guard.value_i64_range("upper_bound", 0, coordinate.maximum),
                Guard.value_i64_range("step", 1, 1),
            )


def test_predicate_range_rules_keep_narrow_path_short() -> None:
    narrow = AIE2P_PREDICATE_RANGE_RULES[0]
    assert [emit.descriptor.key for emit in narrow.emit] == [
        "amd.xdna.aie2p.sub.i32",
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.cmp.slt.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.cmp.slt.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.lshl.i32",
        "amd.xdna.aie2p.add.i32.immediate",
        "amd.xdna.aie2p.predicate.copy.scalar.to.low32",
        "amd.xdna.aie2p.predicate.complete.zero.high32",
    ]


def test_predicate_range_rules_complete_both_words() -> None:
    wide = AIE2P_PREDICATE_RANGE_RULES[2]
    assert wide.emit[-1].descriptor.key == (
        "amd.xdna.aie2p.predicate.copy.scalar.to.high32"
    )
    low_copy = next(
        emit
        for emit in wide.emit
        if emit.descriptor.key == "amd.xdna.aie2p.predicate.copy.scalar.to.low32"
    )
    assert wide.emit[-1].operands["storage"] == low_copy.results["d0"]

    exact_64 = AIE2P_PREDICATE_RANGE_RULES[3]
    assert exact_64.emit[-1].descriptor.key == (
        "amd.xdna.aie2p.predicate.copy.scalar.to.high32"
    )
    assert [
        emit.descriptor.key
        for emit in exact_64.emit
        if emit.descriptor.key == "amd.xdna.aie2p.predicate.copy.scalar.to.high32"
    ] == ["amd.xdna.aie2p.predicate.copy.scalar.to.high32"]
