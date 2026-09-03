# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the AMD XDNA AIE2P core source-to-Low contract."""

from loom.dialect.buffer import defs as buffer
from loom.dialect.index import defs as index
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.target.arch.amd.xdna.aie2p.contracts.core import (
    _BF16_CONVERSION_ROUNDING,
    _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
    _BF16_OUTER_PRODUCT_MULTIPLY_CONTROL,
    _BF16_OUTER_PRODUCT_SHUFFLE_CONTROLS,
    _F32_ACCUMULATOR_ADD_CONTROL,
    _I16_ELEMENTWISE_MULTIPLY_CONTROL,
    AIE2P_CORE_CONTRACT_FRAGMENT,
)
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    EmitRegisterConcat,
    EmitRegisterSlice,
    ValueAliasRule,
    Vector,
)


def test_core_contract_closes_scalar_and_integer_vector_families() -> None:
    rules = tuple(
        case
        for case in AIE2P_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )

    address_constant_rules = [
        rule for rule in rules if rule.source_op is index.index_constant
    ]
    assert [rule.descriptor.key for rule in address_constant_rules] == [
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.constant.i32",
    ]

    index_binary_rules = [
        rule
        for rule in rules
        if rule.source_op
        in (
            index.index_add,
            index.index_sub,
            index.index_mul,
            index.index_andi,
            index.index_ori,
            index.index_xori,
            index.index_shli,
        )
    ]
    assert [rule.descriptor.key for rule in index_binary_rules] == [
        "amd.xdna.aie2p.add.i32",
        "amd.xdna.aie2p.add.i32",
        "amd.xdna.aie2p.sub.i32",
        "amd.xdna.aie2p.sub.i32",
        "amd.xdna.aie2p.mul.i32",
        "amd.xdna.aie2p.and.i32",
        "amd.xdna.aie2p.or.i32",
        "amd.xdna.aie2p.xor.i32",
        "amd.xdna.aie2p.lshl.i32",
    ]

    index_divide_rules = [
        rule for rule in rules if rule.source_op in (index.index_div, index.index_rem)
    ]
    assert [len(rule.emit) for rule in index_divide_rules] == [35, 34]
    for rule in index_divide_rules:
        assert rule.emit[0].descriptor.key == ("amd.xdna.aie2p.move.to.division-state")
        assert [emit.descriptor.key for emit in rule.emit[2:34]] == [
            "amd.xdna.aie2p.divide.step.unsigned.i32"
        ] * 32
        for step, emit in enumerate(rule.emit[2:34], start=1):
            assert emit.operands["sd"].field == f"division_state_{step - 1}"
            assert emit.results["sd_out"].field == f"division_state_{step}"
    assert index_divide_rules[0].emit[-1].descriptor.key == (
        "amd.xdna.aie2p.move.from.division-state"
    )
    assert index_divide_rules[1].emit[-1].results["d0"].field == "result"

    index_minmax_rules = [
        rule for rule in rules if rule.source_op in (index.index_min, index.index_max)
    ]
    assert [
        [emit.descriptor.key for emit in rule.emit] for rule in index_minmax_rules
    ] == [
        [
            "amd.xdna.aie2p.cmp.slt.i32.select",
            "amd.xdna.aie2p.select.nonzero.i32",
        ],
        [
            "amd.xdna.aie2p.cmp.slt.i32.select",
            "amd.xdna.aie2p.select.nonzero.i32",
        ],
    ]
    assert index_minmax_rules[0].emit[0].operands["s0"].field == "lhs"
    assert index_minmax_rules[1].emit[0].operands["s0"].field == "rhs"
    assert all(rule.emit[-1].copy_operands == () for rule in index_minmax_rules)

    index_madd_rule = next(rule for rule in rules if rule.source_op is index.index_madd)
    assert index_madd_rule.descriptor.key == "amd.xdna.aie2p.madd.i32"
    assert index_madd_rule.emit[0].copy_operands == ("a0",)

    index_rotate_rules = [
        rule
        for rule in rules
        if rule.source_op in (index.index_rotli, index.index_rotri)
    ]
    assert [len(rule.emit) for rule in index_rotate_rules] == [9, 9]
    assert all(
        rule.emit[-1].descriptor.key == "amd.xdna.aie2p.or.i32"
        for rule in index_rotate_rules
    )

    index_count_rules = [
        rule
        for rule in rules
        if rule.source_op in (index.index_ctlzi, index.index_cttzi, index.index_ctpopi)
    ]
    assert [rule.descriptor.key for rule in index_count_rules] == [
        "amd.xdna.aie2p.clz.i32",
        "amd.xdna.aie2p.select.zero.i32",
        "amd.xdna.aie2p.popcount.i32",
    ]
    assert [len(rule.emit) for rule in index_count_rules] == [1, 7, 1]

    index_compare_rules = [rule for rule in rules if rule.source_op is index.index_cmp]
    assert len(index_compare_rules) == 20
    assert [rule.descriptor.key for rule in index_compare_rules[:10]] == [
        rule.descriptor.key for rule in index_compare_rules[10:]
    ]

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
        "amd.xdna.aie2p.accumulator.clear.i32x64",
        "amd.xdna.aie2p.accumulator.clear.f32x64",
    ]
    assert [len(rule.emit) for rule in vector_constant_rules] == [2, 2, 2, 2, 2, 1, 1]

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

    scalar_address_alu_rules = [
        rule
        for rule in rules
        if rule.source_op
        in (
            scalar_arithmetic.scalar_divui,
            scalar_arithmetic.scalar_remui,
            scalar_arithmetic.scalar_absi,
            scalar_arithmetic.scalar_minsi,
            scalar_arithmetic.scalar_maxsi,
            scalar_arithmetic.scalar_minui,
            scalar_arithmetic.scalar_maxui,
            scalar_arithmetic.scalar_fmai,
            scalar_bitwise.scalar_rotli,
            scalar_bitwise.scalar_rotri,
            scalar_bitwise.scalar_ctlzi,
            scalar_bitwise.scalar_cttzi,
            scalar_bitwise.scalar_ctpopi,
        )
    ]
    assert [rule.descriptor.key for rule in scalar_address_alu_rules] == [
        "amd.xdna.aie2p.divide.step.unsigned.i32",
        "amd.xdna.aie2p.divide.step.unsigned.i32",
        "amd.xdna.aie2p.abs.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.madd.i32",
        "amd.xdna.aie2p.or.i32",
        "amd.xdna.aie2p.or.i32",
        "amd.xdna.aie2p.clz.i32",
        "amd.xdna.aie2p.select.zero.i32",
        "amd.xdna.aie2p.popcount.i32",
    ]
    assert scalar_address_alu_rules[7].emit[0].copy_operands == ("a0",)

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

    bf16_multiply_rules = [
        rule for rule in rules if rule.source_op is vector.vector_mulf
    ]
    assert len(bf16_multiply_rules) == 1
    bf16_multiply = bf16_multiply_rules[0]
    assert bf16_multiply.descriptor.key == ("amd.xdna.aie2p.convert.f32x32.to.bf16x32")
    assert [
        emit.descriptor.key
        for emit in bf16_multiply.emit
        if not isinstance(emit, EmitRegisterSlice)
    ] == [
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.multiply.bf16x32.configured",
        "amd.xdna.aie2p.state.rounding.immediate",
        "amd.xdna.aie2p.convert.f32x32.to.bf16x32",
    ]
    product_slice = bf16_multiply.emit[2]
    assert isinstance(product_slice, EmitRegisterSlice)
    assert product_slice.source.field == "wide_product"
    assert product_slice.result.field == "product"
    assert product_slice.unit_offset == 0
    assert product_slice.unit_count == 2
    assert product_slice.result_type is None
    assert _BF16_ELEMENTWISE_MULTIPLY_CONTROL == 0x3C
    assert bf16_multiply.emit[0].immediates == {"i": _BF16_ELEMENTWISE_MULTIPLY_CONTROL}
    assert _BF16_CONVERSION_ROUNDING == 12
    assert bf16_multiply.emit[3].immediates == {"i": _BF16_CONVERSION_ROUNDING}

    bf16_outer_product_rules = [
        rule
        for rule in rules
        if rule.source_op is vector.vector_mma and rule.report_key == "bf16bf16_m8n8k1"
    ]
    assert len(bf16_outer_product_rules) == 1
    bf16_outer_product = bf16_outer_product_rules[0]
    assert bf16_outer_product.descriptor.key == (
        "amd.xdna.aie2p.matrix.accumulate.bf16bf16.m8n8k1.configured"
    )
    assert [
        emit.descriptor.key
        for emit in bf16_outer_product.emit
        if not isinstance(emit, EmitRegisterConcat)
    ] == [
        "amd.xdna.aie2p.broadcast.bf16x8.to.bf16x32",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shuffle.bf16x32.configured",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shuffle.bf16x32.configured",
        "amd.xdna.aie2p.broadcast.bf16x8.to.bf16x32",
        "amd.xdna.aie2p.move.bf16x32",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.matrix.accumulate.bf16bf16.m8n8k1.configured",
    ]
    lhs_concat = bf16_outer_product.emit[5]
    rhs_concat = bf16_outer_product.emit[8]
    assert isinstance(lhs_concat, EmitRegisterConcat)
    assert isinstance(rhs_concat, EmitRegisterConcat)
    assert lhs_concat.result_type == Vector("bf16", lanes=64)
    assert rhs_concat.result_type == Vector("bf16", lanes=64)
    assert bf16_outer_product.emit[1].immediates == {
        "i": _BF16_OUTER_PRODUCT_SHUFFLE_CONTROLS[0]
    }
    assert bf16_outer_product.emit[3].immediates == {
        "i": _BF16_OUTER_PRODUCT_SHUFFLE_CONTROLS[1]
    }
    assert bf16_outer_product.emit[9].immediates == {
        "i": _BF16_OUTER_PRODUCT_MULTIPLY_CONTROL
    }

    f32_add_rules = [rule for rule in rules if rule.source_op is vector.vector_addf]
    assert len(f32_add_rules) == 1
    f32_add = f32_add_rules[0]
    assert f32_add.descriptor.key == "amd.xdna.aie2p.add.f32x64.configured"
    assert [emit.descriptor.key for emit in f32_add.emit] == [
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.add.f32x64.configured",
    ]
    assert _F32_ACCUMULATOR_ADD_CONTROL == 0x3C
    assert f32_add.emit[0].immediates == {"i": _F32_ACCUMULATOR_ADD_CONTROL}

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
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.select.nonzero.i32",
        "amd.xdna.aie2p.select.i32x16",
        "amd.xdna.aie2p.select.i32x16",
        "amd.xdna.aie2p.select.i32x16",
    ]
    for rule in whole_select_rules[:6]:
        assert len(rule.emit) == 1
        assert rule.emit[0].copy_operands == ("s2",)
        assert rule.emit[0].operands["s0"].field == "true_value"
        assert rule.emit[0].operands["s1"].field == "false_value"
        assert rule.emit[0].operands["s2"].field == "condition"
    for rule in whole_select_rules[6:]:
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
    buffer_view_rules = [
        rule for rule in alias_rules if rule.source_op is buffer.buffer_view
    ]
    assert len(buffer_view_rules) == 1
    assert buffer_view_rules[0].guards == ()
    assert (
        len([rule for rule in alias_rules if rule.source_op is view.view_subview]) == 1
    )
    assert (
        len([rule for rule in alias_rules if rule.source_op is view.view_refine]) == 1
    )
    bitcast_rules = [
        rule for rule in alias_rules if rule.source_op is vector.vector_bitcast
    ]
    bitcast_types = (
        Vector("i8", minimum_lanes=1, maximum_lanes=64),
        Vector("i16", minimum_lanes=1, maximum_lanes=32),
        Vector("bf16", minimum_lanes=1, maximum_lanes=32),
        Vector("i32", minimum_lanes=1, maximum_lanes=16),
        Vector("f32", minimum_lanes=1, maximum_lanes=16),
    )
    assert [
        (rule.guards[0].type_pattern, rule.guards[1].type_pattern)
        for rule in bitcast_rules
    ] == [
        (source_type, result_type)
        for source_type in bitcast_types
        for result_type in bitcast_types
    ]
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
