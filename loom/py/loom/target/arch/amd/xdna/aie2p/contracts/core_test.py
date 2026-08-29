# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the AMD XDNA AIE2P scalar source-to-Low contract."""

from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.target.arch.amd.xdna.aie2p.contracts.core import (
    AIE2P_CORE_CONTRACT_FRAGMENT,
)
from loom.target.contracts import DescriptorRule


def test_scalar_contract_closes_template_operation_family() -> None:
    rules = tuple(
        case
        for case in AIE2P_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )
    assert len(rules) == 31

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
