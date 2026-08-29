# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the AMD XDNA AIE2P core source-to-Low contract."""

from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.core import (
    AIE2P_CORE_CONTRACT_FRAGMENT,
)
from loom.target.contracts import DescriptorRule, ValueAliasRule


def test_core_contract_closes_scalar_and_integer_vector_families() -> None:
    rules = tuple(
        case
        for case in AIE2P_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )
    assert len(rules) == 49

    constant_rules = [
        rule for rule in rules if rule.source_op is scalar_conversion.scalar_constant
    ]
    assert [rule.descriptor.key for rule in constant_rules] == [
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.constant.i32",
    ]

    compare_rules = [
        rule for rule in rules if rule.source_op is scalar_comparison.scalar_cmpi
    ]
    assert len(compare_rules) == 14
    assert [rule.descriptor.key for rule in compare_rules[:4]] == [
        "amd.xdna.aie2p.cmp.eqz.i32",
        "amd.xdna.aie2p.cmp.eqz.i32",
        "amd.xdna.aie2p.cmp.nez.i32",
        "amd.xdna.aie2p.cmp.nez.i32",
    ]

    bitfield_rules = [
        rule
        for rule in rules
        if rule.source_op
        in (
            scalar_bitwise.scalar_bitfield_extractu,
            scalar_bitwise.scalar_bitfield_extracts,
        )
    ]
    assert len(bitfield_rules) == 2
    assert all(len(rule.emit) == 4 for rule in bitfield_rules)

    vector_rules = [
        rule
        for rule in rules
        if rule.source_op in (vector.vector_addi, vector.vector_subi)
    ]
    assert [rule.descriptor.key for rule in vector_rules] == [
        "amd.xdna.aie2p.add.i8x64",
        "amd.xdna.aie2p.sub.i8x64",
        "amd.xdna.aie2p.add.i16x32",
        "amd.xdna.aie2p.sub.i16x32",
        "amd.xdna.aie2p.add.i32x16",
        "amd.xdna.aie2p.sub.i32x16",
    ]

    vector_bitwise_rules = [
        rule
        for rule in rules
        if rule.source_op in (vector.vector_andi, vector.vector_ori, vector.vector_xori)
    ]
    assert [rule.descriptor.key for rule in vector_bitwise_rules] == [
        "amd.xdna.aie2p.and.bits512",
        "amd.xdna.aie2p.and.bits512",
        "amd.xdna.aie2p.and.bits512",
        "amd.xdna.aie2p.or.bits512",
        "amd.xdna.aie2p.or.bits512",
        "amd.xdna.aie2p.or.bits512",
        "amd.xdna.aie2p.sub.i8x64",
        "amd.xdna.aie2p.sub.i16x32",
        "amd.xdna.aie2p.sub.i32x16",
    ]
    assert [len(rule.emit) for rule in vector_bitwise_rules] == [
        1,
        1,
        1,
        1,
        1,
        1,
        3,
        3,
        3,
    ]

    vector_splat_rules = [
        rule for rule in rules if rule.source_op is vector.vector_splat
    ]
    assert [rule.descriptor.key for rule in vector_splat_rules] == [
        "amd.xdna.aie2p.splat.i8x64",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.splat.i32x16",
    ]

    alias_rules = [
        case
        for case in AIE2P_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, ValueAliasRule)
    ]
    assert [rule.source_op for rule in alias_rules] == [vector.vector_bitcast] * 9
