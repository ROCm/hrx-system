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
    _I16_ELEMENTWISE_MULTIPLY_CONTROL,
)
from loom.target.arch.amd.xdna.aie2p.contracts.core_contract import (
    AIE2P_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    F32_ACCUMULATOR_ADD_CONTROL,
)
from loom.target.arch.amd.xdna.aie2p.contracts.floating import (
    _BF16_DOT2_DEINTERLEAVE_CONTROLS,
    _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
    _BF16_OUTER_PRODUCT_MULTIPLY_CONTROL,
    _BF16_OUTER_PRODUCT_SHUFFLE_CONTROLS,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packed_dot import (
    _DOT4_GROW_CONTROL,
    _DOT4_TRANSPOSE_CONTROL,
)
from loom.target.arch.amd.xdna.aie2p.contracts.reduction import (
    _F32X16_REDUCTION_CONTROLS,
    _I32_REDUCTION_CONTROLS,
)
from loom.target.arch.amd.xdna.aie2p.contracts.structural import (
    _ACCUMULATOR_BITCAST_TYPE_GROUPS,
    _I8_DEINTERLEAVE_CONTROLS,
    _PREDICATE_VECTOR,
    _WIDE_VECTOR_BITCAST_TYPES,
    AIE2P_STRUCTURAL_RULES,
)
from loom.target.arch.amd.xdna.aie2p.contracts.table import AIE2P_TABLE_RULES
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitRegisterConcat,
    EmitRegisterCopy,
    EmitRegisterMove,
    EmitRegisterSlice,
    Guard,
    Scalar,
    ValueAliasRule,
    ValueProjectKind,
    ValueRef,
    Vector,
)


def test_core_contract_closes_scalar_and_vector_families() -> None:
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
        rule
        for rule in rules
        if rule.source_op in (index.index_div, index.index_rem) and rule.priority == 0
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
    constant_rules_by_type = {
        element: [
            rule
            for rule in constant_rules
            if any(
                guard.type_pattern is not None and guard.type_pattern.element == element
                for guard in rule.guards
            )
        ]
        for element in (
            "i1",
            "i8",
            "i16",
            "i32",
            "i64",
            "f8E4M3",
            "f8E5M2",
            "f16",
            "bf16",
            "f32",
            "f64",
        )
    }
    assert {
        element: [rule.descriptor.key for rule in type_rules]
        for element, type_rules in constant_rules_by_type.items()
    } == {
        "i1": ["amd.xdna.aie2p.constant.i32.short"],
        "i8": ["amd.xdna.aie2p.constant.i32.short"],
        "i16": [
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.constant.i32",
        ],
        "i32": [
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.constant.i32",
        ],
        "i64": ["amd.xdna.aie2p.constant.i32"],
        "f8E4M3": ["amd.xdna.aie2p.constant.i32"],
        "f8E5M2": ["amd.xdna.aie2p.constant.i32"],
        "f16": ["amd.xdna.aie2p.constant.i32"],
        "bf16": ["amd.xdna.aie2p.constant.i32"],
        "f32": ["amd.xdna.aie2p.constant.i32"],
        "f64": ["amd.xdna.aie2p.constant.i32"],
    }
    i64_constant = constant_rules_by_type["i64"][0]
    assert len(i64_constant.emit) == 3
    assert [emit.immediates["i"].kind for emit in i64_constant.emit[:2]] == [
        ValueProjectKind.EXACT_I64_I32_WORD,
        ValueProjectKind.EXACT_I64_I32_WORD,
    ]
    assert [emit.immediates["i"].word_index for emit in i64_constant.emit[:2]] == [
        0,
        1,
    ]
    f64_constant = constant_rules_by_type["f64"][0]
    assert len(f64_constant.emit) == 3
    assert [emit.immediates["i"].kind for emit in f64_constant.emit[:2]] == [
        ValueProjectKind.FLOAT_AS_F64_I32_WORD,
        ValueProjectKind.FLOAT_AS_F64_I32_WORD,
    ]
    assert [emit.immediates["i"].word_index for emit in f64_constant.emit[:2]] == [
        0,
        1,
    ]
    float_constant_rules = [
        constant_rules_by_type[element][0]
        for element in ("f8E4M3", "f8E5M2", "f16", "bf16", "f32")
    ]
    assert [rule.emit[0].immediates["i"].kind for rule in float_constant_rules] == [
        ValueProjectKind.FLOAT_BITS,
        ValueProjectKind.FLOAT_BITS,
        ValueProjectKind.FLOAT_BITS,
        ValueProjectKind.FLOAT_BITS,
        ValueProjectKind.FLOAT_AS_F32_I32,
    ]

    vector_broadcast_rules = [
        rule for rule in rules if rule.source_op is vector.vector_broadcast
    ]
    assert [rule.descriptor.key for rule in vector_broadcast_rules] == [
        "amd.xdna.aie2p.broadcast.i8x64.from-vector",
        "amd.xdna.aie2p.broadcast.i16x32.from-vector",
        "amd.xdna.aie2p.broadcast.i32x16.from-vector",
    ]

    table_lookup_rules = [
        rule for rule in rules if rule.source_op is vector.vector_table_lookup
    ]
    assert table_lookup_rules == list(AIE2P_TABLE_RULES)
    assert [rule.descriptor.key for rule in table_lookup_rules] == [
        "amd.xdna.aie2p.broadcast.i8x64.from-vector",
        "amd.xdna.aie2p.broadcast.i16x32.from-vector",
        "amd.xdna.aie2p.broadcast.i32x16.from-vector",
        "amd.xdna.aie2p.broadcast.i64x8.from-vector",
    ]
    assert [rule.emit[0].immediates["idx"].kind for rule in table_lookup_rules] == [
        ValueProjectKind.EXACT_I64,
    ] * 4

    vector_extract_rules = [
        rule
        for rule in rules
        if rule.source_op is vector.vector_extract
        and rule not in AIE2P_STRUCTURAL_RULES
    ]
    expected_extract_keys = [
        ("i64", "i64", "amd.xdna.aie2p.extract.i64.immediate"),
        ("i64", "i64", "amd.xdna.aie2p.extract.i64.register"),
        ("f64", "f64", "amd.xdna.aie2p.extract.i64.immediate"),
        ("f64", "f64", "amd.xdna.aie2p.extract.i64.register"),
    ]
    for element_type, storage in (
        ("i8", "i8"),
        ("f8E4M3", "i8"),
        ("f8E5M2", "i8"),
        ("i16", "i16"),
        ("f16", "i16"),
        ("bf16", "i16"),
        ("i32", "i32"),
        ("f32", "i32"),
    ):
        expected_extract_keys.extend(
            (
                (
                    element_type,
                    element_type,
                    f"amd.xdna.aie2p.extract.{storage}.immediate",
                ),
                (
                    element_type,
                    element_type,
                    f"amd.xdna.aie2p.extract.{storage}.register",
                ),
            )
        )
    expected_extract_keys.append(
        (
            "i1",
            "i1",
            "amd.xdna.aie2p.extract.i8.immediate",
        )
    )
    expected_extract_keys.append(
        (
            "i1",
            "i1",
            "amd.xdna.aie2p.extract.i8.register",
        )
    )
    expected_extract_keys.append(
        (
            "i1",
            "i1",
            "amd.xdna.aie2p.extract.predicate64.immediate",
        )
    )
    assert [
        (
            rule.guards[0].type_pattern.element,
            rule.guards[1].type_pattern.element,
            rule.descriptor.key,
        )
        for rule in vector_extract_rules
    ] == expected_extract_keys

    predicate_extract_rules = [
        rule
        for rule in vector_extract_rules
        if rule.guards[0].type_pattern.element == "i1"
        and rule.descriptor.key.startswith("amd.xdna.aie2p.extract.i8")
    ]
    assert [
        [emit.descriptor.key for emit in rule.emit] for rule in predicate_extract_rules
    ] == [
        [
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.splat.i8x64",
            "amd.xdna.aie2p.sub.i8x64",
            "amd.xdna.aie2p.select.i8x64",
            "amd.xdna.aie2p.extract.i8.immediate",
        ],
        [
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.splat.i8x64",
            "amd.xdna.aie2p.sub.i8x64",
            "amd.xdna.aie2p.select.i8x64",
            "amd.xdna.aie2p.extract.i8.register",
        ],
    ]

    vector_insert_rules = [
        rule for rule in rules if rule.source_op is vector.vector_insert
    ]
    expected_insert_rows = [
        ("i64", "i64", "amd.xdna.aie2p.insert.i64.zero"),
        ("i64", "i64", "amd.xdna.aie2p.insert.i64.register"),
        ("i64", "i64", "amd.xdna.aie2p.insert.i64.register"),
        ("f64", "f64", "amd.xdna.aie2p.insert.i64.zero"),
        ("f64", "f64", "amd.xdna.aie2p.insert.i64.register"),
        ("f64", "f64", "amd.xdna.aie2p.insert.i64.register"),
    ]
    for element_type, storage in (
        ("i8", "i8"),
        ("f8E4M3", "i8"),
        ("f8E5M2", "i8"),
        ("i16", "i16"),
        ("f16", "i16"),
        ("bf16", "i16"),
        ("i32", "i32"),
        ("f32", "i32"),
    ):
        expected_insert_rows.extend(
            (
                (element_type, element_type, f"amd.xdna.aie2p.insert.{storage}.zero"),
                (
                    element_type,
                    element_type,
                    f"amd.xdna.aie2p.insert.{storage}.register",
                ),
                (
                    element_type,
                    element_type,
                    f"amd.xdna.aie2p.insert.{storage}.register",
                ),
            )
        )
    assert [
        (
            rule.guards[0].type_pattern.element,
            rule.guards[1].type_pattern.element,
            rule.descriptor.key,
        )
        for rule in vector_insert_rules
    ] == expected_insert_rows
    for rule in vector_insert_rules:
        expected_copy_operands = (
            ("idx",) if rule.descriptor.key.endswith(".register") else ()
        )
        assert rule.emit[-1].copy_operands == expected_copy_operands

    compare_rules = [
        rule for rule in rules if rule.source_op is scalar_comparison.scalar_cmpi
    ]
    i64_compare_rules = [
        rule for rule in compare_rules if rule.guards[1].type_pattern.element == "i64"
    ]
    assert sorted(rule.guards[0].enum_keyword for rule in i64_compare_rules) == [
        "eq",
        "ne",
        "sge",
        "sgt",
        "sle",
        "slt",
        "uge",
        "ugt",
        "ule",
        "ult",
    ]
    i32_compare_rules = [
        rule for rule in compare_rules if rule.guards[1].type_pattern.element == "i32"
    ]
    assert sorted(
        (rule.guards[0].enum_keyword, rule.descriptor.key) for rule in i32_compare_rules
    ) == sorted(
        [
            ("eq", "amd.xdna.aie2p.cmp.eq.i32"),
            ("eq", "amd.xdna.aie2p.cmp.eqz.i32"),
            ("eq", "amd.xdna.aie2p.cmp.eqz.i32"),
            ("ne", "amd.xdna.aie2p.cmp.ne.i32"),
            ("ne", "amd.xdna.aie2p.cmp.nez.i32"),
            ("ne", "amd.xdna.aie2p.cmp.nez.i32"),
            ("sge", "amd.xdna.aie2p.cmp.sge.i32"),
            ("sgt", "amd.xdna.aie2p.cmp.slt.i32"),
            ("sle", "amd.xdna.aie2p.cmp.sge.i32"),
            ("slt", "amd.xdna.aie2p.cmp.slt.i32"),
            ("uge", "amd.xdna.aie2p.cmp.uge.i32"),
            ("ugt", "amd.xdna.aie2p.cmp.ult.i32"),
            ("ule", "amd.xdna.aie2p.cmp.uge.i32"),
            ("ult", "amd.xdna.aie2p.cmp.ult.i32"),
        ]
    )

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
    shift_rules_by_type = {
        element: [
            rule
            for rule in shift_rules
            if any(
                guard.field == "result"
                and guard.type_pattern is not None
                and guard.type_pattern.element == element
                for guard in rule.guards
            )
        ]
        for element in ("i8", "i16", "i32", "i64")
    }
    assert {
        element: [rule.descriptor.key for rule in type_rules]
        for element, type_rules in shift_rules_by_type.items()
    } == {
        "i8": [
            "amd.xdna.aie2p.lshl.i32",
            "amd.xdna.aie2p.ashl.i32",
            "amd.xdna.aie2p.lshl.i32",
        ],
        "i16": [
            "amd.xdna.aie2p.lshl.i32",
            "amd.xdna.aie2p.ashl.i32",
            "amd.xdna.aie2p.lshl.i32",
        ],
        "i32": [
            "amd.xdna.aie2p.lshl.i32",
            "amd.xdna.aie2p.ashl.i32",
            "amd.xdna.aie2p.ashl.i32",
            "amd.xdna.aie2p.lshl.i32",
            "amd.xdna.aie2p.lshl.i32",
        ],
        "i64": ["amd.xdna.aie2p.lshl.i32"] * 7 + ["amd.xdna.aie2p.ashl.i32"] * 4,
    }
    assert {
        element: [len(rule.emit) for rule in type_rules]
        for element, type_rules in shift_rules_by_type.items()
    } == {
        "i8": [2, 5, 5],
        "i16": [2, 5, 5],
        "i32": [1, 2, 3, 2, 3],
        "i64": [9, 7, 19, 11, 13, 8, 19, 11, 13, 9, 21],
    }
    signed_i16_shift = shift_rules_by_type["i16"][1]
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
        if rule.priority == 0
        and rule.source_op
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
        "amd.xdna.aie2p.min.signed.i8x64",
        "amd.xdna.aie2p.max.signed.i8x64",
    ]
    assert all(
        tuple(rule.emit[0].results) == ("d",)
        and rule.emit[0].results["d"].field == "result"
        for rule in vector_minmax_rules
    )
    assert all(
        rule.emit[0].form is DescriptorEmitForm.PER_LANE
        and rule.guards[0].type_pattern == Vector("i8", lanes=128)
        for rule in vector_minmax_rules[-2:]
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

    packed_dot_rules = [rule for rule in rules if rule.source_op is vector.vector_dot4i]
    assert len(packed_dot_rules) == 8
    assert [rule.guards[0].enum_keyword for rule in packed_dot_rules] == [
        "u8u8",
        "u8u8",
        "u8s8",
        "u8s8",
        "s8u8",
        "s8u8",
        "s8s8",
        "s8s8",
    ]
    assert all(
        rule.descriptor.key == "amd.xdna.aie2p.dot4i.i8x64.configured"
        for rule in packed_dot_rules
    )
    dot4_chunk_descriptors = [
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.move.vector512",
        "amd.xdna.aie2p.dot4i.i8x64.configured",
        "amd.xdna.aie2p.move.accumulator512.to.vector512",
    ]
    for kind_index, (kind, multiply_control) in enumerate(
        (("u8u8", 0x48), ("u8s8", 0x248), ("s8u8", 0x148), ("s8s8", 0x348))
    ):
        low_rule, high_rule = packed_dot_rules[kind_index * 2 : kind_index * 2 + 2]
        assert (
            Guard.value_type(
                "lhs",
                Vector("i8", minimum_static_elements=4, maximum_static_elements=32),
            )
            in low_rule.guards
        )
        assert (
            Guard.value_type(
                "acc",
                Vector("i32", minimum_static_elements=1, maximum_static_elements=8),
            )
            in low_rule.guards
        )
        assert (
            Guard.value_type(
                "lhs",
                Vector("i8", minimum_static_elements=36, maximum_static_elements=64),
            )
            in high_rule.guards
        )
        assert (
            Guard.value_type(
                "acc",
                Vector("i32", minimum_static_elements=9, maximum_static_elements=16),
            )
            in high_rule.guards
        )
        for rule, chunk_count in ((low_rule, 1), (high_rule, 2)):
            assert rule.report_key == f"dot4i_{kind}_{chunk_count}x256"
            descriptor_emits = [
                emit
                for emit in rule.emit
                if not isinstance(emit, (EmitRegisterConcat, EmitRegisterSlice))
            ]
            assert [emit.descriptor.key for emit in descriptor_emits] == [
                "amd.xdna.aie2p.constant.i32.mova",
                "amd.xdna.aie2p.splat.i8x64",
                "amd.xdna.aie2p.constant.i32.mova",
                "amd.xdna.aie2p.constant.i32.mova",
                "amd.xdna.aie2p.constant.i32.mova",
                *dot4_chunk_descriptors * chunk_count,
                "amd.xdna.aie2p.add.i32x16",
            ]
            assert descriptor_emits[2].immediates == {"i": _DOT4_TRANSPOSE_CONTROL}
            assert descriptor_emits[3].immediates == {"i": _DOT4_GROW_CONTROL}
            assert descriptor_emits[4].immediates == {"i": multiply_control}
            assert (
                sum(isinstance(emit, EmitRegisterSlice) for emit in rule.emit)
                == 1 + 5 * chunk_count
            )
            assert (
                sum(isinstance(emit, EmitRegisterConcat) for emit in rule.emit)
                == 1 + 4 * chunk_count
            )

    reduction_rules = [rule for rule in rules if rule.source_op is vector.vector_reduce]
    f32_reductions = [
        rule
        for rule in reduction_rules
        if Guard.enum_attr_equals("kind", "addf") in rule.guards
        and Guard.value_type("input", Vector("f32", lanes=16)) in rule.guards
    ]
    assert len(f32_reductions) == 1
    f32_reduction = f32_reductions[0]
    assert f32_reduction.descriptor.key == "amd.xdna.aie2p.add.f32x64.configured"
    assert f32_reduction.report_key == "f32x16_accumulator_tree"
    assert Guard.value_float_equals("init", 0.0) in f32_reduction.guards
    assert {
        Guard.instance_flags_has_all("fastmath", flag)
        for flag in ("reassoc", "nnan", "ninf", "nsz")
    } <= set(f32_reduction.guards)
    assert [
        emit.immediates["i"]
        for emit in f32_reduction.emit
        if not isinstance(
            emit,
            (
                EmitRegisterConcat,
                EmitRegisterCopy,
                EmitRegisterMove,
                EmitRegisterSlice,
            ),
        )
        and emit.descriptor.key == "amd.xdna.aie2p.constant.i32.mova"
    ] == [F32_ACCUMULATOR_ADD_CONTROL, *_F32X16_REDUCTION_CONTROLS]
    assert [
        emit.descriptor.key
        for emit in f32_reduction.emit
        if not isinstance(
            emit,
            (
                EmitRegisterConcat,
                EmitRegisterCopy,
                EmitRegisterMove,
                EmitRegisterSlice,
            ),
        )
    ] == [
        "amd.xdna.aie2p.accumulator.clear.f32x64",
        "amd.xdna.aie2p.accumulator.clear.f32x64",
        "amd.xdna.aie2p.move.vector512.to.accumulator512",
        "amd.xdna.aie2p.constant.i32.mova",
        *(
            key
            for _ in _F32X16_REDUCTION_CONTROLS
            for key in (
                "amd.xdna.aie2p.constant.i32.mova",
                "amd.xdna.aie2p.shuffle.x.to.accumulator512.configured",
                "amd.xdna.aie2p.add.f32x64.configured",
                "amd.xdna.aie2p.move.accumulator512.to.vector512",
            )
        ),
        "amd.xdna.aie2p.extract.i32.immediate",
    ]
    assert sum(isinstance(emit, EmitRegisterConcat) for emit in f32_reduction.emit) == 5
    assert sum(isinstance(emit, EmitRegisterCopy) for emit in f32_reduction.emit) == 4
    assert sum(isinstance(emit, EmitRegisterMove) for emit in f32_reduction.emit) == 2
    assert sum(isinstance(emit, EmitRegisterSlice) for emit in f32_reduction.emit) == 10

    for lane_count, controls in _I32_REDUCTION_CONTROLS:
        lane_rules = [
            rule
            for rule in reduction_rules
            if Guard.enum_attr_equals("kind", "addi") in rule.guards
            and Guard.value_type("input", Vector("i32", lanes=lane_count))
            in rule.guards
        ]
        zero_guard = Guard.value_i64_range("init", 0, 0)
        zero_rule = next(rule for rule in lane_rules if zero_guard in rule.guards)
        general_rule = next(
            rule for rule in lane_rules if zero_guard not in rule.guards
        )
        assert reduction_rules.index(zero_rule) < reduction_rules.index(general_rule)
        for rule, zero_init in ((zero_rule, True), (general_rule, False)):
            assert [
                emit.immediates["i"]
                for emit in rule.emit
                if emit.descriptor.key == "amd.xdna.aie2p.constant.i32.mova"
            ] == list(controls)
            expected_keys = [
                key
                for _ in controls
                for key in (
                    "amd.xdna.aie2p.constant.i32.mova",
                    "amd.xdna.aie2p.shuffle.x.configured",
                    "amd.xdna.aie2p.add.i32x16",
                )
            ]
            expected_keys.append("amd.xdna.aie2p.extract.i32.immediate")
            if not zero_init:
                expected_keys.append("amd.xdna.aie2p.add.i32")
            assert [emit.descriptor.key for emit in rule.emit] == expected_keys

    bitunpack_rules = [
        rule
        for rule in rules
        if rule.source_op in (vector.vector_bitunpacku, vector.vector_bitunpacks)
    ]
    expected_bitunpack_keys = {
        (
            f"amd.xdna.aie2p.unpack.{source_kind}4x{source_lane_count * 2}.to."
            f"{source_kind}8x{source_lane_count * 2}.configured"
        )
        for source_kind in ("u", "s")
        for source_lane_count in (32, 64)
    }
    assert {rule.descriptor.key for rule in bitunpack_rules} == (
        expected_bitunpack_keys
    )
    for source_op, source_kind in (
        (vector.vector_bitunpacku, "u"),
        (vector.vector_bitunpacks, "s"),
    ):
        for source_lane_count in (32, 64):
            result_lane_count = source_lane_count * 2
            descriptor_key = (
                f"amd.xdna.aie2p.unpack.{source_kind}4x{result_lane_count}.to."
                f"{source_kind}8x{result_lane_count}.configured"
            )
            rule = next(
                rule
                for rule in bitunpack_rules
                if rule.descriptor.key == descriptor_key
            )
            assert rule.source_op is source_op
            emits = list(rule.emit)
            source_field = "source"
            if source_lane_count == 32:
                packed_slice = emits.pop(0)
                assert isinstance(packed_slice, EmitRegisterSlice)
                assert packed_slice.source.field == "source"
                assert packed_slice.result.field == "packed_source"
                assert packed_slice.unit_count == 1
                source_field = "packed_source"
            set_unpack_size, unpack = emits
            assert set_unpack_size.descriptor.key == (
                "amd.xdna.aie2p.state.unpack-size.immediate"
            )
            assert set_unpack_size.immediates == {"i": 0}
            assert unpack.descriptor is rule.descriptor
            assert unpack.operands["src"].field == source_field
            assert unpack.results["dst"].field == "result"

    bf16_dot2_rules = [
        rule
        for rule in rules
        if rule.source_op is vector.vector_dot2f and not rule.source_nodes
    ]
    assert len(bf16_dot2_rules) == 2
    bf16_dot2_zero, bf16_dot2 = bf16_dot2_rules
    assert bf16_dot2_zero.guards == (
        *bf16_dot2.guards,
        Guard.value_float_equals("acc", 0.0),
    )
    assert not any(isinstance(emit, EmitRegisterConcat) for emit in bf16_dot2_zero.emit)
    accumulates = [
        emit
        for emit in bf16_dot2_zero.emit
        if not isinstance(emit, EmitRegisterSlice)
        and emit.descriptor.key == "amd.xdna.aie2p.accumulate.bf16x32.configured"
    ]
    assert len(accumulates) == 2
    assert accumulates[0].operands["acc1"].field == "zero_accumulator"
    assert accumulates[1].operands["acc1"].field == "even_accumulator"
    assert bf16_dot2.descriptor.key == ("amd.xdna.aie2p.accumulate.bf16x32.configured")
    assert [
        emit.descriptor.key
        for emit in bf16_dot2.emit
        if not isinstance(emit, (EmitRegisterConcat, EmitRegisterSlice))
    ] == [
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.accumulator.clear.f32x64",
        "amd.xdna.aie2p.move.vector512.to.accumulator512",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.accumulate.bf16x32.configured",
        "amd.xdna.aie2p.accumulate.bf16x32.configured",
        "amd.xdna.aie2p.move.accumulator512.to.vector512",
    ]
    assert [bf16_dot2.emit[index].immediates["i"] for index in (0, 3)] == list(
        _BF16_DOT2_DEINTERLEAVE_CONTROLS
    )
    assert bf16_dot2.emit[12].immediates == {"i": _BF16_ELEMENTWISE_MULTIPLY_CONTROL}
    assert sum(isinstance(emit, EmitRegisterSlice) for emit in bf16_dot2.emit) == 4
    assert sum(isinstance(emit, EmitRegisterConcat) for emit in bf16_dot2.emit) == 1
    bf16_dot2_input = Vector(
        "bf16", minimum_static_elements=2, maximum_static_elements=32
    )
    bf16_dot2_result = Vector(
        "f32", minimum_static_elements=1, maximum_static_elements=16
    )
    assert Guard.value_type("lhs", bf16_dot2_input) in bf16_dot2.guards
    assert Guard.value_type("rhs", bf16_dot2_input) in bf16_dot2.guards
    assert Guard.value_type("acc", bf16_dot2_result) in bf16_dot2.guards
    assert Guard.value_type("result", bf16_dot2_result) in bf16_dot2.guards

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
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.broadcast.bf16x8.to.bf16x32",
        "amd.xdna.aie2p.move.vector512",
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

    deinterleave_rules = [
        rule for rule in rules if rule.source_op is vector.vector_deinterleave
    ]
    assert len(deinterleave_rules) == 1
    deinterleave = deinterleave_rules[0]
    assert deinterleave.descriptor.key == "amd.xdna.aie2p.shuffle.x.configured"
    assert [emit.descriptor.key for emit in deinterleave.emit] == [
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shuffle.x.configured",
    ]
    shuffles = tuple(
        emit
        for emit in deinterleave.emit
        if emit.descriptor.key == "amd.xdna.aie2p.shuffle.x.configured"
    )
    assert tuple(emit.results["dst"].field for emit in shuffles) == (
        "results",
        "results",
    )
    assert tuple(emit.results["dst"].element for emit in shuffles) == (0, 1)
    assert (
        tuple(
            emit.immediates["i"]
            for emit in deinterleave.emit
            if emit.descriptor.key == "amd.xdna.aie2p.constant.i32.mova"
        )
        == _I8_DEINTERLEAVE_CONTROLS
    )

    f32_add_rules = [rule for rule in rules if rule.source_op is vector.vector_addf]
    assert len(f32_add_rules) == 2
    f32_add = f32_add_rules[0]
    assert f32_add.descriptor.key == "amd.xdna.aie2p.add.f32x64.configured"
    assert [emit.descriptor.key for emit in f32_add.emit] == [
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.add.f32x64.configured",
    ]
    assert F32_ACCUMULATOR_ADD_CONTROL == 0x3C
    assert f32_add.emit[0].immediates == {"i": F32_ACCUMULATOR_ADD_CONTROL}

    f32_vector_carrier_rules = (
        f32_add_rules[1],
        next(rule for rule in rules if rule.source_op is vector.vector_subf),
    )
    for rule, arithmetic_key in zip(
        f32_vector_carrier_rules,
        (
            "amd.xdna.aie2p.add.f32x64.configured",
            "amd.xdna.aie2p.sub.f32x64.configured",
        ),
        strict=True,
    ):
        assert [
            emit.descriptor.key
            for emit in rule.emit
            if not isinstance(emit, (EmitRegisterConcat, EmitRegisterSlice))
        ] == [
            "amd.xdna.aie2p.accumulator.clear.f32x64",
            "amd.xdna.aie2p.move.vector512.to.accumulator512",
            "amd.xdna.aie2p.move.vector512.to.accumulator512",
            "amd.xdna.aie2p.constant.i32.mova",
            arithmetic_key,
            "amd.xdna.aie2p.move.accumulator512.to.vector512",
        ]
        assert sum(isinstance(emit, EmitRegisterConcat) for emit in rule.emit) == 2
        assert sum(isinstance(emit, EmitRegisterSlice) for emit in rule.emit) == 2

    f32_scalar_carrier_rules = (
        next(rule for rule in rules if rule.source_op is scalar_arithmetic.scalar_addf),
        next(rule for rule in rules if rule.source_op is scalar_arithmetic.scalar_subf),
    )
    for rule, arithmetic_key in zip(
        f32_scalar_carrier_rules,
        (
            "amd.xdna.aie2p.add.f32x64.configured",
            "amd.xdna.aie2p.sub.f32x64.configured",
        ),
        strict=True,
    ):
        assert [
            emit.descriptor.key
            for emit in rule.emit
            if not isinstance(emit, (EmitRegisterConcat, EmitRegisterSlice))
        ] == [
            "amd.xdna.aie2p.splat.i32x16",
            "amd.xdna.aie2p.splat.i32x16",
            "amd.xdna.aie2p.accumulator.clear.f32x64",
            "amd.xdna.aie2p.move.vector512.to.accumulator512",
            "amd.xdna.aie2p.move.vector512.to.accumulator512",
            "amd.xdna.aie2p.constant.i32.mova",
            arithmetic_key,
            "amd.xdna.aie2p.move.accumulator512.to.vector512",
            "amd.xdna.aie2p.extract.i32.immediate",
        ]

    payload_bitwise_rules = [
        rule
        for rule in rules
        if rule.source_op in (vector.vector_andi, vector.vector_ori, vector.vector_xori)
        and not rule.descriptor.key.startswith("amd.xdna.aie2p.predicate.")
    ]
    assert [rule.descriptor.key for rule in payload_bitwise_rules] == [
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
    assert [len(rule.emit) for rule in payload_bitwise_rules] == [
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
    pair_splat_rules = [
        rule
        for rule in vector_splat_rules
        if rule.guards[0].type_pattern.element in ("i64", "f64")
    ]
    assert [
        (
            rule.guards[0].type_pattern.element,
            rule.guards[1].type_pattern.element,
            rule.descriptor.key,
        )
        for rule in pair_splat_rules
    ] == [
        (element_type, element_type, "amd.xdna.aie2p.splat.i64x8")
        for element_type in ("i64", "f64")
    ]
    assert all(len(rule.emit) == 1 for rule in pair_splat_rules)
    native_splat_rules = [
        rule
        for rule in vector_splat_rules
        if rule.guards[0].type_pattern.element not in ("i64", "f64")
    ]
    assert [rule.descriptor.key for rule in native_splat_rules] == [
        "amd.xdna.aie2p.splat.i8x64",
        "amd.xdna.aie2p.splat.i8x64",
        "amd.xdna.aie2p.splat.i8x64",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.splat.i32x16",
        "amd.xdna.aie2p.splat.i32x16",
        "amd.xdna.aie2p.cmp.lt.unsigned.i8x64",
    ]
    assert [
        (rule.guards[0].type_pattern.element, rule.guards[1].type_pattern.element)
        for rule in native_splat_rules
    ] == [
        ("i8", "i8"),
        ("f8E4M3", "f8E4M3"),
        ("f8E5M2", "f8E5M2"),
        ("i16", "i16"),
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("i32", "i32"),
        ("f32", "f32"),
        ("i1", "i1"),
    ]
    predicate_splat = native_splat_rules[-1]
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

    payload_select_rules = [
        rule
        for rule in rules
        if rule.source_op is vector.vector_select
        and rule.guards[1].type_pattern.element != "i1"
    ]
    native_payload_select_rules = [
        rule
        for rule in payload_select_rules
        if rule.descriptor.key != "amd.xdna.aie2p.select.i32x16"
    ]
    assert [rule.descriptor.key for rule in native_payload_select_rules] == [
        "amd.xdna.aie2p.select.i8x64",
        "amd.xdna.aie2p.select.i8x64",
        "amd.xdna.aie2p.select.i8x64",
        "amd.xdna.aie2p.select.i16x32.mask64",
        "amd.xdna.aie2p.select.i16x32.mask64",
        "amd.xdna.aie2p.select.i16x32.mask64",
        "amd.xdna.aie2p.select.i32x16.mask64",
        "amd.xdna.aie2p.select.i32x16.mask64",
    ]
    pair_payload_select_rules = [
        rule
        for rule in payload_select_rules
        if rule.descriptor.key == "amd.xdna.aie2p.select.i32x16"
    ]
    assert [
        (
            rule.guards[1].type_pattern.element,
            rule.guards[1].type_pattern.minimum_static_elements,
            rule.guards[1].type_pattern.maximum_static_elements,
        )
        for rule in pair_payload_select_rules
    ] == [
        (element_type, minimum_lanes, maximum_lanes)
        for element_type in ("i64", "f64")
        for minimum_lanes, maximum_lanes in (
            (1, 1),
            (2, 2),
            (3, 4),
            (5, 8),
        )
    ]
    for rule in payload_select_rules:
        select = rule.emit[-1]
        assert select.operands["s1"].field == "false_value"
        assert select.operands["s2"].field == "true_value"
    for rule in native_payload_select_rules:
        select = rule.emit[-1]
        assert select.operands["sel"].field == "condition"
    assert all(
        rule.emit[-1].copy_operands == ("sel",) for rule in pair_payload_select_rules
    )

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
    assert [emit.descriptor.key for emit in vector_compare_rules[1].emit[-2:]] == [
        "amd.xdna.aie2p.predicate.or.low32.rhs_tied",
        "amd.xdna.aie2p.predicate.or.high32.rhs_tied",
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
    scalar_select_rules = [rule for rule in whole_select_rules if len(rule.emit) == 1]
    assert [rule.guards[1].type_pattern for rule in scalar_select_rules] == [
        Scalar("i1"),
        Scalar("i8"),
        Scalar("f8E4M3"),
        Scalar("f8E5M2"),
        Scalar("i16"),
        Scalar("f16"),
        Scalar("bf16"),
        Scalar("i32"),
        Scalar("f32"),
        Scalar("index"),
        Scalar("offset"),
    ]
    assert all(
        rule.descriptor.key == "amd.xdna.aie2p.select.nonzero.i32"
        for rule in scalar_select_rules
    )
    for rule in scalar_select_rules:
        assert len(rule.emit) == 1
        assert rule.emit[0].copy_operands == ("s2",)
        assert rule.emit[0].operands["s0"].field == "true_value"
        assert rule.emit[0].operands["s1"].field == "false_value"
        assert rule.emit[0].operands["s2"].field == "condition"
    pair_scalar_select_rules = [
        rule
        for rule in whole_select_rules
        if rule.descriptor.key == "amd.xdna.aie2p.select.nonzero.i32"
        and rule.guards[1].type_pattern.kind == "scalar"
        and len(rule.emit) != 1
    ]
    assert [
        rule.guards[1].type_pattern.element for rule in pair_scalar_select_rules
    ] == [
        "i64",
        "f64",
    ]
    assert all(len(rule.emit) == 7 for rule in pair_scalar_select_rules)
    vector_select_rules = [
        rule
        for rule in whole_select_rules
        if rule.descriptor.key == "amd.xdna.aie2p.select.i32x16"
    ]
    assert [rule.guards[1].type_pattern for rule in vector_select_rules] == [
        Vector("i8", minimum_static_elements=1, maximum_static_elements=64),
        Vector("f8E4M3", minimum_static_elements=1, maximum_static_elements=64),
        Vector("f8E5M2", minimum_static_elements=1, maximum_static_elements=64),
        Vector("i16", minimum_static_elements=1, maximum_static_elements=32),
        Vector("f16", minimum_static_elements=1, maximum_static_elements=32),
        Vector("bf16", minimum_static_elements=1, maximum_static_elements=32),
        Vector("i32", minimum_static_elements=1, maximum_static_elements=16),
        Vector("f32", minimum_static_elements=1, maximum_static_elements=16),
        Vector("i64", minimum_static_elements=1, maximum_static_elements=8),
        Vector("f64", minimum_static_elements=1, maximum_static_elements=8),
    ]
    for rule in vector_select_rules:
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
    scalar_bitcast_rules = [
        rule
        for rule in alias_rules
        if rule.source_op is scalar_conversion.scalar_bitcast
    ]
    scalar_bitcast_type_groups = (
        (Scalar("i8"), Scalar("f8E4M3"), Scalar("f8E5M2")),
        (Scalar("i16"), Scalar("f16"), Scalar("bf16")),
        (Scalar("i32"), Scalar("f32")),
        (Scalar("i64"), Scalar("f64")),
    )
    assert [
        (rule.guards[0].type_pattern, rule.guards[1].type_pattern)
        for rule in scalar_bitcast_rules
    ] == [
        (source_type, result_type)
        for type_group in scalar_bitcast_type_groups
        for source_type in type_group
        for result_type in type_group
        if source_type != result_type
    ]
    vector_bitcast_rules = [
        rule for rule in alias_rules if rule.source_op is vector.vector_bitcast
    ]
    bitcast_types = (
        Vector("i8", minimum_static_elements=1, maximum_static_elements=64),
        Vector("f8E4M3", minimum_static_elements=1, maximum_static_elements=64),
        Vector("f8E5M2", minimum_static_elements=1, maximum_static_elements=64),
        Vector("i16", minimum_static_elements=1, maximum_static_elements=32),
        Vector("f16", minimum_static_elements=1, maximum_static_elements=32),
        Vector("bf16", minimum_static_elements=1, maximum_static_elements=32),
        Vector("i32", minimum_static_elements=1, maximum_static_elements=16),
        Vector("f32", minimum_static_elements=1, maximum_static_elements=16),
        Vector("i64", minimum_static_elements=1, maximum_static_elements=8),
        Vector("f64", minimum_static_elements=1, maximum_static_elements=8),
    )
    assert [
        (rule.guards[0].type_pattern, rule.guards[1].type_pattern)
        for rule in vector_bitcast_rules
    ] == [
        (source_type, result_type)
        for type_group in _ACCUMULATOR_BITCAST_TYPE_GROUPS
        for source_type in type_group
        for result_type in type_group
    ] + [
        (source_type, result_type)
        for source_type in _WIDE_VECTOR_BITCAST_TYPES
        for result_type in _WIDE_VECTOR_BITCAST_TYPES
    ] + [(_PREDICATE_VECTOR, _PREDICATE_VECTOR)] + [
        (source_type, result_type)
        for source_type in bitcast_types
        for result_type in bitcast_types
    ]
    packed_predicate_aliases = [
        rule for rule in alias_rules if rule.source_op is vector.vector_bitunpacku
    ]
    assert len(packed_predicate_aliases) == 1
    packed_predicate_alias = packed_predicate_aliases[0]
    assert [guard.type_pattern for guard in packed_predicate_alias.guards[:2]] == [
        Vector("i8", lanes=16),
        Vector("i1", dims=(2, 64)),
    ]
    assert packed_predicate_alias.guards[3].minimum == 1
    assert packed_predicate_alias.guards[3].maximum == 1
    assert (
        len([rule for rule in alias_rules if rule.source_op is vector.vector_broadcast])
        == 3
    )


def test_vector_constant_rules_materialize_each_register_carrier() -> None:
    rules = [
        case
        for case in AIE2P_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_constant
    ]

    def result_shape(rule: DescriptorRule):
        pattern = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.field == "result" and guard.type_pattern is not None
        )
        return (
            pattern.element,
            pattern.minimum_static_elements,
            pattern.maximum_static_elements,
            pattern.lanes,
        )

    predicate = next(
        rule
        for rule in rules
        if rule.descriptor.key == "amd.xdna.aie2p.cmp.lt.unsigned.i8x64"
    )
    assert (predicate.guards[-1].minimum, predicate.guards[-1].maximum) == (0, 1)
    assert [emit.descriptor.key for emit in predicate.emit] == [
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.splat.i8x64",
        "amd.xdna.aie2p.sub.i8x64",
        "amd.xdna.aie2p.cmp.lt.unsigned.i8x64",
    ]
    assert predicate.emit[0].immediates["i"].kind is ValueProjectKind.EXACT_I64

    packet_families = (
        ("i8", 64, 1),
        ("i16", 32, 2),
        ("i32", 16, 2),
        ("f8E4M3", 64, 1),
        ("f8E5M2", 64, 1),
        ("f16", 32, 1),
        ("bf16", 32, 1),
        ("f32", 16, 1),
    )
    native = [
        rule
        for rule in rules
        if len(rule.emit) == 2
        and rule.descriptor.key.startswith("amd.xdna.aie2p.splat")
    ]
    assert sorted(result_shape(rule) for rule in native) == sorted(
        (element, 1, lanes, None)
        for element, lanes, rule_count in packet_families
        for _ in range(rule_count)
    )

    wide = [
        rule
        for rule in rules
        if len(rule.emit) == 3 and isinstance(rule.emit[-1], EmitRegisterConcat)
    ]
    assert sorted(result_shape(rule) for rule in wide) == sorted(
        (element, lanes + 1, lanes * 2, None)
        for element, lanes, rule_count in packet_families
        for _ in range(rule_count)
    )
    assert all(
        rule.emit[-1].sources == (rule.emit[-2].results["dst"],) * 2 for rule in wide
    )
    f32_wide = next(rule for rule in wide if result_shape(rule)[0] == "f32")
    assert Guard.low_value_register_class("result", "aie2p.vec256") in f32_wide.guards
    for rule in native + wide:
        element = result_shape(rule)[0]
        if element.startswith("f") or element == "bf16":
            expected_kind = (
                ValueProjectKind.FLOAT_AS_F32_I32
                if element == "f32"
                else ValueProjectKind.FLOAT_BITS
            )
            assert rule.emit[0].immediates["i"].kind is expected_kind

    pair = [rule for rule in rules if result_shape(rule)[0] in ("i64", "f64")]
    assert {result_shape(rule) for rule in pair} == {
        ("i64", 1, 8, None),
        ("f64", 1, 8, None),
        ("i64", 9, 16, None),
        ("f64", 9, 16, None),
        ("i64", None, None, 32),
    }
    assert all(
        [emit.immediates["i"].word_index for emit in rule.emit[:2]] == [0, 1]
        and isinstance(rule.emit[2], EmitRegisterConcat)
        and rule.emit[3].descriptor.key == "amd.xdna.aie2p.splat.i64x8"
        for rule in pair
    )

    accumulator = [
        rule
        for rule in rules
        if isinstance(rule.emit[-1], EmitRegisterConcat)
        and rule.emit[-2].descriptor.key
        == "amd.xdna.aie2p.move.vector512.to.accumulator512"
    ]
    assert sorted(
        (result_shape(rule)[0], result_shape(rule)[3], len(rule.emit[-1].sources))
        for rule in accumulator
    ) == sorted(
        [
            ("i32", 64, 4),
            ("i32", 64, 4),
            ("f32", 32, 2),
            ("f32", 64, 4),
            ("i64", 32, 4),
        ]
    )


def test_predicate_binary_rules_reuse_only_a_dead_source_operand() -> None:
    rules = tuple(
        case
        for case in AIE2P_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )
    for source_op, operation in (
        (vector.vector_andi, "and"),
        (vector.vector_ori, "or"),
        (vector.vector_xori, "xor"),
    ):
        predicate_rules = [
            rule
            for rule in rules
            if rule.source_op is source_op
            and rule.descriptor.key.startswith(f"amd.xdna.aie2p.predicate.{operation}.")
        ]
        assert [rule.descriptor.key for rule in predicate_rules] == [
            f"amd.xdna.aie2p.predicate.{operation}.high32.rhs_tied",
            f"amd.xdna.aie2p.predicate.{operation}.high32.rhs_tied",
            f"amd.xdna.aie2p.predicate.{operation}.high32",
        ]

        update_rhs, update_lhs, preserve_both = predicate_rules
        assert update_rhs.guards[-1] == Guard.value_no_uses_after("rhs")
        assert update_lhs.guards[-1] == Guard.value_no_uses_after("lhs")
        assert len(preserve_both.guards) == 3

        tied_keys = [
            f"amd.xdna.aie2p.predicate.{operation}.low32.rhs_tied",
            f"amd.xdna.aie2p.predicate.{operation}.high32.rhs_tied",
        ]
        assert [emit.descriptor.key for emit in update_rhs.emit] == tied_keys
        assert update_rhs.emit[0].operands == {
            "s0": ValueRef.operand("lhs"),
            "s1": ValueRef.operand("rhs"),
        }
        assert update_rhs.emit[1].operands == {
            "s0": ValueRef.operand("lhs"),
            "s1": ValueRef.temporary("predicate_low32"),
        }

        assert [emit.descriptor.key for emit in update_lhs.emit] == tied_keys
        assert update_lhs.emit[0].operands == {
            "s0": ValueRef.operand("rhs"),
            "s1": ValueRef.operand("lhs"),
        }
        assert update_lhs.emit[1].operands == {
            "s0": ValueRef.operand("rhs"),
            "s1": ValueRef.temporary("predicate_low32"),
        }

        assert [emit.descriptor.key for emit in preserve_both.emit] == [
            f"amd.xdna.aie2p.predicate.{operation}.low32",
            f"amd.xdna.aie2p.predicate.{operation}.high32",
        ]
        assert preserve_both.emit[1].operands == {
            "s0": ValueRef.operand("lhs"),
            "s1": ValueRef.operand("rhs"),
            "storage": ValueRef.temporary("predicate_low32"),
        }


def test_predicate_select_reuses_true_value_only_at_its_last_use() -> None:
    rules = [
        case
        for case in AIE2P_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op is vector.vector_select
        and case.guards[1].type_pattern.element == "i1"
    ]
    assert len(rules) == 2
    update_true, preserve_true = rules
    assert update_true.guards[-1] == Guard.value_no_uses_after("true_value")
    assert len(preserve_true.guards) == 4

    tied_suffix = [
        "amd.xdna.aie2p.predicate.and.low32.rhs_tied",
        "amd.xdna.aie2p.predicate.and.high32.rhs_tied",
        "amd.xdna.aie2p.predicate.xor.low32.rhs_tied",
        "amd.xdna.aie2p.predicate.xor.high32.rhs_tied",
    ]
    assert [emit.descriptor.key for emit in update_true.emit] == [
        "amd.xdna.aie2p.predicate.xor.low32.rhs_tied",
        "amd.xdna.aie2p.predicate.xor.high32.rhs_tied",
        *tied_suffix,
    ]
    assert update_true.emit[0].operands == {
        "s0": ValueRef.operand("false_value"),
        "s1": ValueRef.operand("true_value"),
    }
    assert [emit.descriptor.key for emit in preserve_true.emit] == [
        "amd.xdna.aie2p.predicate.xor.low32",
        "amd.xdna.aie2p.predicate.xor.high32",
        *tied_suffix,
    ]
    assert preserve_true.emit[1].operands == {
        "s0": ValueRef.operand("false_value"),
        "s1": ValueRef.operand("true_value"),
        "storage": ValueRef.temporary("difference_low32"),
    }


def test_predicate_not_equal_reuses_only_complete_comparison_storage() -> None:
    rules = [
        case
        for case in AIE2P_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op is vector.vector_cmpi
        and case.guards[0].enum_keyword == "ne"
    ]
    assert [rule.guards[1].type_pattern.element for rule in rules] == [
        "i8",
        "i16",
        "i32",
    ]
    assert rules[0].emit[2].descriptor.key == (
        "amd.xdna.aie2p.predicate.or.low32.rhs_tied"
    )
    assert rules[0].emit[2].operands == {
        "s0": ValueRef.temporary("comparison_forward"),
        "s1": ValueRef.temporary("comparison_reverse"),
    }
    assert rules[0].emit[3].descriptor.key == (
        "amd.xdna.aie2p.predicate.or.high32.rhs_tied"
    )
    assert all(
        rule.emit[2].descriptor.key == "amd.xdna.aie2p.predicate.or.low32"
        for rule in rules[1:]
    )
    assert all(
        rule.emit[3].descriptor.key == "amd.xdna.aie2p.predicate.complete.zero.high32"
        for rule in rules[1:]
    )
