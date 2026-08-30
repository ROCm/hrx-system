# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the AMD XDNA AIE2P core source-to-Low contract."""

from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.core import (
    _I16_ELEMENTWISE_MULTIPLY_CONTROL,
    AIE2P_CORE_CONTRACT_FRAGMENT,
)
from loom.target.contracts import DescriptorResultType, DescriptorRule, ValueAliasRule


def test_core_contract_closes_scalar_and_integer_vector_families() -> None:
    rules = tuple(
        case
        for case in AIE2P_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )
    assert len(rules) == 150

    constant_rules = [
        rule for rule in rules if rule.source_op is scalar_conversion.scalar_constant
    ]
    assert [rule.descriptor.key for rule in constant_rules] == [
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.constant.i32",
    ]

    conversion_rules = [
        rule
        for rule in rules
        if rule.source_op
        in (scalar_conversion.scalar_extsi, scalar_conversion.scalar_extui)
    ]
    assert [rule.descriptor.key for rule in conversion_rules] == [
        "amd.xdna.aie2p.extend.signed.i8",
        "amd.xdna.aie2p.extend.signed.i16",
        "amd.xdna.aie2p.extend.unsigned.i8",
        "amd.xdna.aie2p.extend.unsigned.i16",
    ]

    vector_constant_rules = [
        rule for rule in rules if rule.source_op is vector.vector_constant
    ]
    assert [rule.descriptor.key for rule in vector_constant_rules] == [
        "amd.xdna.aie2p.splat.i8x64",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.splat.i32x16",
        "amd.xdna.aie2p.splat.i32x16",
    ]
    assert all(len(rule.emit) == 2 for rule in vector_constant_rules)

    vector_broadcast_rules = [
        rule for rule in rules if rule.source_op is vector.vector_broadcast
    ]
    assert [rule.descriptor.key for rule in vector_broadcast_rules] == [
        "amd.xdna.aie2p.broadcast.i8x64.from-vector",
        "amd.xdna.aie2p.broadcast.i16x32.from-vector",
        "amd.xdna.aie2p.broadcast.i32x16.from-vector",
    ]

    vector_extract_rules = [
        rule for rule in rules if rule.source_op is vector.vector_extract
    ]
    assert [rule.descriptor.key for rule in vector_extract_rules] == [
        "amd.xdna.aie2p.extract.i8.immediate",
        "amd.xdna.aie2p.extract.i8.register",
        "amd.xdna.aie2p.extract.i16.immediate",
        "amd.xdna.aie2p.extract.i16.register",
        "amd.xdna.aie2p.extract.i32.immediate",
        "amd.xdna.aie2p.extract.i32.register",
    ]

    vector_insert_rules = [
        rule for rule in rules if rule.source_op is vector.vector_insert
    ]
    assert [rule.descriptor.key for rule in vector_insert_rules] == [
        "amd.xdna.aie2p.insert.i8.zero",
        "amd.xdna.aie2p.insert.i8.register",
        "amd.xdna.aie2p.insert.i8.register",
        "amd.xdna.aie2p.insert.i16.zero",
        "amd.xdna.aie2p.insert.i16.register",
        "amd.xdna.aie2p.insert.i16.register",
        "amd.xdna.aie2p.insert.i32.zero",
        "amd.xdna.aie2p.insert.i32.register",
        "amd.xdna.aie2p.insert.i32.register",
    ]
    for rule in vector_insert_rules:
        expected_copy_operands = (
            ("idx",) if rule.descriptor.key.endswith(".register") else ()
        )
        assert rule.emit[-1].copy_operands == expected_copy_operands

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

    shift_rules = [
        rule
        for rule in rules
        if rule.source_op
        in (
            scalar_bitwise.scalar_shli,
            scalar_bitwise.scalar_shrsi,
            scalar_bitwise.scalar_shrui,
        )
    ]
    assert [rule.descriptor.key for rule in shift_rules] == [
        "amd.xdna.aie2p.lshl.i32",
        "amd.xdna.aie2p.ashl.i32",
        "amd.xdna.aie2p.lshl.i32",
        "amd.xdna.aie2p.lshl.i32",
        "amd.xdna.aie2p.ashl.i32",
        "amd.xdna.aie2p.lshl.i32",
        "amd.xdna.aie2p.lshl.i32",
        "amd.xdna.aie2p.ashl.i32",
        "amd.xdna.aie2p.lshl.i32",
    ]
    assert [len(rule.emit) for rule in shift_rules] == [1, 3, 3, 2, 5, 5, 2, 5, 5]
    signed_i16_shift = shift_rules[7]
    assert [emit.descriptor.key for emit in signed_i16_shift.emit] == [
        "amd.xdna.aie2p.extend.signed.i16",
        "amd.xdna.aie2p.extend.unsigned.i16",
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.sub.i32",
        "amd.xdna.aie2p.ashl.i32",
    ]
    assert signed_i16_shift.emit[2].immediates == {"i": 0}

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

    vector_minmax_rules = [
        rule
        for rule in rules
        if rule.source_op
        in (
            vector.vector_minsi,
            vector.vector_maxsi,
            vector.vector_minui,
            vector.vector_maxui,
        )
    ]
    assert [rule.descriptor.key for rule in vector_minmax_rules] == [
        "amd.xdna.aie2p.min.signed.i8x64",
        "amd.xdna.aie2p.max.signed.i8x64",
        "amd.xdna.aie2p.min.unsigned.i8x64",
        "amd.xdna.aie2p.max.unsigned.i8x64",
        "amd.xdna.aie2p.min.signed.i16x32",
        "amd.xdna.aie2p.max.signed.i16x32",
        "amd.xdna.aie2p.min.unsigned.i16x32",
        "amd.xdna.aie2p.max.unsigned.i16x32",
        "amd.xdna.aie2p.min.signed.i32x16",
        "amd.xdna.aie2p.max.signed.i32x16",
        "amd.xdna.aie2p.min.unsigned.i32x16",
        "amd.xdna.aie2p.max.unsigned.i32x16",
    ]
    assert all(
        tuple(rule.emit[0].results) == ("d",)
        and rule.emit[0].results["d"].field == "result"
        for rule in vector_minmax_rules
    )

    vector_multiply_rules = [
        rule for rule in rules if rule.source_op is vector.vector_muli
    ]
    assert len(vector_multiply_rules) == 1
    vector_multiply = vector_multiply_rules[0]
    assert vector_multiply.descriptor.key == (
        "amd.xdna.aie2p.narrow.trunc.signed.i16x32"
    )
    assert [emit.descriptor.key for emit in vector_multiply.emit] == [
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.constant.i32.shift",
        "amd.xdna.aie2p.multiply.i16x32.configured",
        "amd.xdna.aie2p.state.rounding.immediate",
        "amd.xdna.aie2p.state.srs-mode.immediate",
        "amd.xdna.aie2p.state.saturation.immediate",
        "amd.xdna.aie2p.narrow.trunc.signed.i16x32",
    ]
    assert _I16_ELEMENTWISE_MULTIPLY_CONTROL == 0x35A
    assert vector_multiply.emit[0].immediates == {
        "i": _I16_ELEMENTWISE_MULTIPLY_CONTROL
    }
    assert vector_multiply.emit[1].immediates == {"i": 0}
    assert vector_multiply.emit[3].immediates == {"i": 0}
    assert vector_multiply.emit[4].immediates == {"i": 1}
    assert vector_multiply.emit[5].immediates == {"i": 0}

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
        "amd.xdna.aie2p.predicate.and.high32",
        "amd.xdna.aie2p.predicate.or.high32",
        "amd.xdna.aie2p.predicate.xor.high32",
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
        2,
        2,
        2,
    ]

    vector_splat_rules = [
        rule for rule in rules if rule.source_op is vector.vector_splat
    ]
    assert [rule.descriptor.key for rule in vector_splat_rules] == [
        "amd.xdna.aie2p.splat.i8x64",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.splat.i32x16",
        "amd.xdna.aie2p.cmp.lt.unsigned.i8x64",
    ]
    predicate_splat = vector_splat_rules[-1]
    assert [emit.descriptor.key for emit in predicate_splat.emit] == [
        "amd.xdna.aie2p.splat.i8x64",
        "amd.xdna.aie2p.sub.i8x64",
        "amd.xdna.aie2p.cmp.lt.unsigned.i8x64",
    ]
    assert all(
        isinstance(next(iter(emit.result_types.values())), DescriptorResultType)
        for emit in predicate_splat.emit[:2]
    )
    assert predicate_splat.emit[-1].result_types is None

    vector_select_rules = [
        rule for rule in rules if rule.source_op is vector.vector_select
    ]
    assert [rule.descriptor.key for rule in vector_select_rules] == [
        "amd.xdna.aie2p.select.i8x64",
        "amd.xdna.aie2p.select.i16x32.mask64",
        "amd.xdna.aie2p.select.i32x16.mask64",
    ]
    for rule in vector_select_rules:
        select = rule.emit[0]
        assert select.operands["s1"].field == "false_value"
        assert select.operands["s2"].field == "true_value"
        assert select.operands["sel"].field == "condition"

    vector_compare_rules = [
        rule for rule in rules if rule.source_op is vector.vector_cmpi
    ]
    assert len(vector_compare_rules) == 30
    assert [len(rule.emit) for rule in vector_compare_rules] == [
        2,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        3,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]
    assert [emit.descriptor.key for emit in vector_compare_rules[10].emit] == [
        "amd.xdna.aie2p.sub.i16x32",
        "amd.xdna.aie2p.cmp.eqz.i16x32.el.low32",
        "amd.xdna.aie2p.predicate.complete.zero.high32",
    ]
    assert [emit.descriptor.key for emit in vector_compare_rules[11].emit] == [
        "amd.xdna.aie2p.cmp.lt.unsigned.i16x32.el.low32",
        "amd.xdna.aie2p.cmp.lt.unsigned.i16x32.el.low32",
        "amd.xdna.aie2p.predicate.or.low32",
        "amd.xdna.aie2p.predicate.complete.zero.high32",
    ]
    swapped_signed_less_equal = vector_compare_rules[13].emit[0]
    assert swapped_signed_less_equal.operands["s1"].field == "rhs"
    assert swapped_signed_less_equal.operands["s2"].field == "lhs"

    whole_select_rules = [rule for rule in rules if rule.source_op is scf.scf_select]
    assert [rule.descriptor.key for rule in whole_select_rules] == [
        "amd.xdna.aie2p.select.i32x16",
        "amd.xdna.aie2p.select.i32x16",
        "amd.xdna.aie2p.select.i32x16",
    ]
    for rule in whole_select_rules:
        assert [emit.descriptor.key for emit in rule.emit] == [
            "amd.xdna.aie2p.select.mask.i32",
            "amd.xdna.aie2p.select.i32x16",
        ]
        assert rule.emit[-1].copy_operands == ()

    alias_rules = [
        case
        for case in AIE2P_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, ValueAliasRule)
    ]
    assert (
        len([rule for rule in alias_rules if rule.source_op is vector.vector_bitcast])
        == 9
    )
    assert (
        len([rule for rule in alias_rules if rule.source_op is vector.vector_broadcast])
        == 3
    )
    assert (
        len(
            [
                rule
                for rule in alias_rules
                if rule.source_op is scalar_conversion.scalar_trunci
            ]
        )
        == 2
    )
