# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.cfg import cfg_cond_br
from loom.dialect.index import defs as index
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.target.arch.x86.contracts.avx2 import X86_AVX2_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.avx512 import X86_AVX512_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.scalar import X86_SCALAR_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.scalar_read_memory import (
    X86_SCALAR_READ_MEMORY_CONTRACT_FRAGMENT,
)
from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
    X86_INTEGER_CONDITION_CODES,
)
from loom.target.arch.x86.scalar_families import (
    X86_SCALAR_READ_MEMORY_FAMILIES,
)
from loom.target.arch.x86.vector_families import (
    AVX2_CARRIER_VECTOR_BIT_WIDTHS,
    AVX2_FLOAT_BINARY_FAMILIES,
    AVX2_FLOAT_FMA_MNEMONICS,
    AVX2_FLOAT_REDUCTION_OPERATIONS,
    AVX2_FLOAT_VECTOR_BIT_WIDTHS,
    AVX2_INTEGER_REDUCTION_FAMILIES,
    AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS,
    AVX2_PAYLOAD_ELEMENT_NAMES,
    AVX2_SCALAR_FLOAT_BINARY_FAMILIES,
    AVX2_SCALAR_FLOAT_FMA_MNEMONICS,
    AVX2_SHUFFLE_FAMILIES,
    AVX2_VECTOR_BIT_WIDTHS,
    AVX512_BITWISE_FAMILIES,
    AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS,
    AVX512_FLOAT_BINARY_FAMILIES,
    AVX512_FLOAT_FMA_MNEMONICS,
    AVX512_INTEGER_BINARY_FAMILIES,
    AVX512_INTEGER_REDUCTION_FAMILIES,
    AVX512_VECTOR_BIT_WIDTHS,
    AVX512VL_INTEGER_BINARY_FAMILIES,
    AVX512VL_VECTOR_BIT_WIDTHS,
    FLOAT_ELEMENTS,
    FLOAT_EXTREMA_OPERATIONS,
    INTEGER_ELEMENTS,
    STORAGE_ELEMENTS,
    X86_LANE_FAMILIES,
    X86_VECTOR_STORAGE_FAMILIES,
)
from loom.target.contracts import (
    AttrProject,
    AttrProjectKind,
    Buffer,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    GuardKind,
    OperandSegmentProject,
    Scalar,
    SourceMemoryOperation,
    SourceNodeRelation,
    ValueAliasRule,
    ValueProject,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import DescriptorOpKind


def _value_type_guard(rule: DescriptorRule, field: str):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind == GuardKind.VALUE_TYPE and guard.field == field
    )


def _source_node_value_type_guard(node, field: str):
    return next(
        guard.type_pattern
        for guard in node.guards
        if guard.kind == GuardKind.VALUE_TYPE and guard.field == field
    )


def _static_index_range(rule: DescriptorRule) -> tuple[int, int]:
    guard = next(
        guard
        for guard in rule.guards
        if guard.kind == GuardKind.I64_ARRAY_ELEMENT_RANGE
        and guard.field == "static_indices"
        and guard.element == 0
    )
    assert guard.minimum is not None
    assert guard.maximum is not None
    return guard.minimum, guard.maximum


def test_scalar_integer_to_float_family_coverage_and_native_overrides() -> None:
    operations = (scalar_conversion.scalar_sitofp, scalar_conversion.scalar_uitofp)
    base_rules = {
        (
            case.source_op,
            _value_type_guard(case, "input"),
            _value_type_guard(case, "result"),
        ): case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op in operations
    }
    assert set(base_rules) == {
        (operation, Scalar(f"i{width}"), Scalar(f"f{float_width}"))
        for operation in operations
        for width in (1, 8, 16, 32, 64)
        for float_width in (32, 64)
    }
    native_rules = {
        (
            case.source_op,
            _value_type_guard(case, "input"),
            _value_type_guard(case, "result"),
        ): case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op in operations
    }
    assert set(native_rules) == {
        (
            scalar_conversion.scalar_uitofp,
            Scalar(f"i{width}"),
            Scalar(f"f{float_width}"),
        )
        for width in (32, 64)
        for float_width in (32, 64)
    }
    for key, rule in native_rules.items():
        assert rule.priority > base_rules[key].priority
        assert len(rule.emit) == 2
        assert rule.emit[0].descriptor.key == "x86.avx2.vxorps.zero.xmm"
        assert rule.emit[1].operands == {
            "passthrough": ValueRef.temporary("passthrough"),
            "input": ValueRef.operand("input"),
        }
    for float_width in (32, 64):
        unsigned32 = base_rules[
            (scalar_conversion.scalar_uitofp, Scalar("i32"), Scalar(f"f{float_width}"))
        ]
        assert unsigned32.emit[0].descriptor.key == "x86.scalar.movzx.gpr64.gpr32"
        assert unsigned32.emit[0].result_types == {"dst": Scalar("i64")}
        unsigned64 = base_rules[
            (scalar_conversion.scalar_uitofp, Scalar("i64"), Scalar(f"f{float_width}"))
        ]
        assert [step.descriptor.mnemonic for step in unsigned64.emit] == [
            "vxorps",
            "vcvtsi2ss" if float_width == 32 else "vcvtsi2sd",
            "shr",
            "and",
            "or",
            "vcvtsi2ss" if float_width == 32 else "vcvtsi2sd",
            "vaddss" if float_width == 32 else "vaddsd",
            "vmovq",
            "vblendvpd",
        ]
        assert unsigned64.emit[2].immediates == {"shift": 1}
        assert unsigned64.emit[3].immediates == {"imm32": 1}


def test_scalar_integer_to_float_native_encoding_family() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
        if descriptor.mnemonic in ("vcvtsi2ss", "vcvtsi2sd", "vcvtusi2ss", "vcvtusi2sd")
    }
    assert len(descriptors) == 8
    for unsigned in (False, True):
        for width in (32, 64):
            for float_width in (32, 64):
                feature = "avx512" if unsigned else "avx2"
                mnemonic = ("vcvtusi2" if unsigned else "vcvtsi2") + (
                    "ss" if float_width == 32 else "sd"
                )
                descriptor = descriptors[f"x86.{feature}.{mnemonic}.xmm.gpr{width}"]
                # ModRM.reg = result, VEX/EVEX.vvvv = passthrough, ModRM.rm = GPR.
                assert descriptor.encoding_format_id == 0x8210
                expected = (0x7B if unsigned else 0x2A) | 0x100
                expected |= (2 if float_width == 32 else 3) << 10
                expected |= int(width == 64) << 12
                expected |= int(unsigned) << 13
                assert descriptor.encoding_id == expected
                assert [operand.field_name for operand in descriptor.operands] == [
                    "dst",
                    "passthrough",
                    "input",
                ]


def test_scalar_float_comparisons_cover_types_predicates_and_selects() -> None:
    expected_immediates = {
        "oeq": 0,
        "ogt": 30,
        "oge": 29,
        "olt": 17,
        "ole": 18,
        "one": 12,
        "ord": 7,
        "ueq": 8,
        "ugt": 22,
        "uge": 21,
        "ult": 25,
        "ule": 26,
        "une": 4,
        "uno": 3,
    }
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op is scalar_comparison.scalar_cmpf
    )
    assert len(rules) == 4
    for element, mnemonic in (("f32", "vcmpps"), ("f64", "vcmppd")):
        typed_rules = tuple(
            rule for rule in rules if _value_type_guard(rule, "lhs") == Scalar(element)
        )
        assert len(typed_rules) == 2
        for rule in typed_rules:
            assert _value_type_guard(rule, "rhs") == Scalar(element)
            assert _value_type_guard(rule, "result") == Scalar("i1")
            projection = rule.emit[0].immediates["predicate"]
            assert isinstance(projection, AttrProject)
            assert projection.kind is AttrProjectKind.ENUM_REMAP
            assert dict(projection.enum_values) == expected_immediates

        (materialized,) = (rule for rule in typed_rules if not rule.source_nodes)
        assert [emit.descriptor.key for emit in materialized.emit] == [
            f"x86.avx2.{mnemonic}.xmm",
            "x86.avx2.vmovd.gpr32.xmm",
            "x86.scalar.and.imm.gpr32",
        ]
        assert materialized.emit[-1].immediates == {"imm32": 1}
        assert materialized.emit[-1].results == {"dst": ValueRef.result("result")}

        (selected,) = (rule for rule in typed_rules if rule.source_nodes)
        assert not selected.target_query
        assert [emit.descriptor.key for emit in selected.emit] == [
            f"x86.avx2.{mnemonic}.xmm",
            "x86.avx2.vpblendvb.xmm",
        ]
        (select,) = selected.source_nodes
        assert select.source_op is scf.scf_select
        assert select.relation is SourceNodeRelation.EXCLUSIVE_USER
        assert _source_node_value_type_guard(select, "condition") == Scalar("i1")
        for field in ("true_value", "false_value", "result"):
            assert _source_node_value_type_guard(select, field) == Scalar(element)
        assert selected.emit[-1].results == {
            "dst": ValueRef.result("result", source_node="select")
        }


def test_scalar_integer_conversions_cover_native_carrier_matrix() -> None:
    conversion_ops = (
        scalar_conversion.scalar_trunci,
        scalar_conversion.scalar_extui,
        scalar_conversion.scalar_extsi,
    )
    descriptor_rules = tuple(
        case
        for case in X86_SCALAR_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op in conversion_ops
    )
    assert {
        (
            rule.source_op,
            _value_type_guard(rule, "input"),
            _value_type_guard(rule, "result"),
            rule.descriptor.key,
        )
        for rule in descriptor_rules
    } == (
        {
            (
                scalar_conversion.scalar_trunci,
                input_type,
                Scalar("i1"),
                "x86.scalar.and.imm.gpr32",
            )
            for input_type in (
                Scalar("i8"),
                Scalar("i16"),
                Scalar("i32"),
                Scalar("i64"),
            )
        }
        | {
            (
                scalar_conversion.scalar_trunci,
                Scalar("i64"),
                result_type,
                "x86.scalar.mov.trunc.gpr32.gpr64",
            )
            for result_type in (Scalar("i8"), Scalar("i16"), Scalar("i32"))
        }
        | {
            (
                scalar_conversion.scalar_extui,
                Scalar("i1"),
                Scalar("i64"),
                "x86.scalar.movzx.gpr64.gpr32",
            )
        }
        | {
            (
                scalar_conversion.scalar_extui,
                input_type,
                result_type,
                f"x86.scalar.movzx.u{input_width}.gpr{carrier_width}",
            )
            for input_type, input_width, result_type, carrier_width in (
                (Scalar("i8"), 8, Scalar("i16"), 32),
                (Scalar("i8"), 8, Scalar("i32"), 32),
                (Scalar("i8"), 8, Scalar("i64"), 64),
                (Scalar("i16"), 16, Scalar("i32"), 32),
                (Scalar("i16"), 16, Scalar("i64"), 64),
            )
        }
        | {
            (
                scalar_conversion.scalar_extui,
                Scalar("i32"),
                Scalar("i64"),
                "x86.scalar.movzx.gpr64.gpr32",
            )
        }
        | {
            (
                scalar_conversion.scalar_extsi,
                Scalar("i1"),
                result_type,
                f"x86.scalar.neg.gpr{carrier_width}",
            )
            for result_type, carrier_width in (
                (Scalar("i8"), 32),
                (Scalar("i16"), 32),
                (Scalar("i32"), 32),
                (Scalar("i64"), 64),
            )
        }
        | {
            (
                scalar_conversion.scalar_extsi,
                input_type,
                result_type,
                f"x86.scalar.movsx.i{input_width}.gpr{carrier_width}",
            )
            for input_type, input_width, result_type, carrier_width in (
                (Scalar("i8"), 8, Scalar("i16"), 32),
                (Scalar("i8"), 8, Scalar("i32"), 32),
                (Scalar("i8"), 8, Scalar("i64"), 64),
                (Scalar("i16"), 16, Scalar("i32"), 32),
                (Scalar("i16"), 16, Scalar("i64"), 64),
            )
        }
        | {
            (
                scalar_conversion.scalar_extsi,
                Scalar("i32"),
                Scalar("i64"),
                "x86.scalar.movsxd.gpr64.gpr32",
            )
        }
    )

    alias_rules = tuple(
        case
        for case in X86_SCALAR_CONTRACT_FRAGMENT.cases
        if isinstance(case, ValueAliasRule) and case.source_op in conversion_ops
    )
    assert {
        (
            rule.source_op,
            _value_type_guard(rule, "input"),
            _value_type_guard(rule, "result"),
        )
        for rule in alias_rules
    } == {
        *(
            (scalar_conversion.scalar_trunci, input_type, result_type)
            for input_type, result_type in (
                (Scalar("i16"), Scalar("i8")),
                (Scalar("i32"), Scalar("i8")),
                (Scalar("i32"), Scalar("i16")),
            )
        ),
        *(
            (scalar_conversion.scalar_extui, Scalar("i1"), result_type)
            for result_type in (Scalar("i8"), Scalar("i16"), Scalar("i32"))
        ),
    }


def _integer_immediate_rule_cells(source_ops) -> set[tuple]:
    cells = set()
    for rule in X86_SCALAR_CONTRACT_FRAGMENT.cases:
        if (
            not isinstance(rule, DescriptorRule)
            or rule.source_op not in source_ops
            or ".imm.gpr" not in rule.descriptor.key
        ):
            continue
        exact_guard = next(
            guard for guard in rule.guards if guard.kind == GuardKind.VALUE_EXACT_I64
        )
        assert any(
            guard.kind == GuardKind.VALUE_I64_RANGE
            and guard.field == exact_guard.field
            and (guard.minimum, guard.maximum) == (-(2**31), (2**31) - 1)
            for guard in rule.guards
        )
        assert rule.emit[0].immediates == {
            "imm32": ValueProject.exact_i64(exact_guard.field),
        }
        cells.add(
            (
                rule.source_op,
                _value_type_guard(rule, "result"),
                exact_guard.field,
                rule.descriptor.key,
            )
        )
    return cells


def test_scalar_integer_arithmetic_immediates_cover_native_carriers() -> None:
    operation_names = {
        scalar_arithmetic.scalar_addi: "add",
        scalar_arithmetic.scalar_subi: "sub",
        scalar_arithmetic.scalar_muli: "imul",
    }
    assert _integer_immediate_rule_cells(operation_names) == {
        (
            source_op,
            Scalar(f"i{width}"),
            constant_field,
            f"x86.scalar.{operation}.imm.gpr{width}",
        )
        for source_op, operation in operation_names.items()
        for width in (32, 64)
        for constant_field in (
            ("rhs",) if source_op is scalar_arithmetic.scalar_subi else ("rhs", "lhs")
        )
    }
    assert all(
        rule.priority == 3
        for rule in X86_SCALAR_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule)
        and rule.source_op in operation_names
        and ".imm.gpr" in rule.descriptor.key
    )


def test_scalar_integer_comparisons_cover_native_carriers() -> None:
    rules = tuple(
        case
        for case in X86_SCALAR_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op in (scalar_comparison.scalar_cmpi, index.index_cmp)
        and case.descriptor.key.startswith("x86.scalar.cmp.")
        and ".select." not in case.descriptor.key
        and case.descriptor.key.rsplit(".", 1)[-1].startswith("gpr")
    )
    actual_cells = set()
    for rule in rules:
        emit = rule.emit[0]
        predicate = emit.immediates["predicate"]
        assert isinstance(predicate, AttrProject)
        assert predicate.kind is AttrProjectKind.ENUM_REMAP
        assert dict(predicate.enum_values) == dict(X86_INTEGER_CONDITION_CODES)
        immediate_fields = tuple(
            field for field in emit.immediates if field.startswith("imm")
        )
        immediate = bool(immediate_fields)
        assert immediate == (".imm." in rule.descriptor.key)
        if immediate:
            (immediate_field,) = immediate_fields
            bit_count = int(rule.descriptor.key.rsplit("gpr", 1)[1])
            immediate_bit_count = min(bit_count, 32)
            assert immediate_field == f"imm{immediate_bit_count}"
            range_guard = next(
                guard
                for guard in rule.guards
                if guard.kind is GuardKind.VALUE_I64_RANGE and guard.field == "rhs"
            )
            assert (range_guard.minimum, range_guard.maximum) == (
                -(2 ** (immediate_bit_count - 1)),
                2 ** (immediate_bit_count - 1) - 1,
            )
        actual_cells.add(
            (
                rule.source_op,
                _value_type_guard(rule, "lhs"),
                int(rule.descriptor.key.rsplit("gpr", 1)[1]),
                immediate,
            )
        )
    assert actual_cells == {
        *(
            (
                scalar_comparison.scalar_cmpi,
                Scalar(f"i{bit_count}"),
                bit_count,
                immediate,
            )
            for bit_count in (8, 16, 32, 64)
            for immediate in (False, True)
        ),
        *(
            (
                index.index_cmp,
                Scalar(("index", "offset")),
                64,
                immediate,
            )
            for immediate in (False, True)
        ),
    }
    assert len(rules) == len(actual_cells)


def test_scalar_compare_branches_cover_native_carrier_matrix() -> None:
    rules = tuple(
        case
        for case in X86_SCALAR_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op is cfg_cond_br
        and case.descriptor.key.startswith("x86.scalar.cmp.branch.")
    )
    actual_cells = set()
    for rule in rules:
        assert not rule.target_query
        assert rule.descriptor.op_kind is DescriptorOpKind.COND_BR
        assert len(rule.source_nodes) == 1
        compare = rule.source_nodes[0]
        assert compare.source_op in (scalar_comparison.scalar_cmpi, index.index_cmp)
        assert compare.relation is SourceNodeRelation.EXCLUSIVE_DEFINITION
        assert compare.parent_value == ValueRef.operand("condition")
        assert compare.node_value == ValueRef.result("result")
        compare_type = _source_node_value_type_guard(compare, "lhs")
        assert _source_node_value_type_guard(compare, "rhs") == compare_type
        assert _source_node_value_type_guard(compare, "result") == Scalar("i1")
        compare_width = int(rule.descriptor.key.rsplit("gpr", 1)[1])
        emit = rule.emit[0]
        assert isinstance(emit, EmitDescriptorOp)
        assert emit.form is DescriptorEmitForm.COND_BR
        assert not emit.results
        assert emit.operands["lhs"] == ValueRef.operand("lhs", source_node="compare")
        predicate = emit.immediates["predicate"]
        assert isinstance(predicate, AttrProject)
        assert predicate.kind is AttrProjectKind.ENUM_REMAP
        assert predicate.source_node == "compare"
        assert dict(predicate.enum_values) == dict(X86_INTEGER_CONDITION_CODES)
        immediate_fields = tuple(
            field for field in emit.immediates if field.startswith("imm")
        )
        immediate = bool(immediate_fields)
        assert ("rhs" not in emit.operands) == immediate
        assert immediate == (".imm." in rule.descriptor.key)
        assert (rule.priority > 0) == immediate
        if immediate:
            (immediate_field,) = immediate_fields
            assert immediate_field == f"imm{min(compare_width, 32)}"
            immediate_project = emit.immediates[immediate_field]
            assert isinstance(immediate_project, ValueProject)
            assert immediate_project.source_node == "compare"
            range_guard = next(
                guard
                for guard in compare.guards
                if guard.kind is GuardKind.VALUE_I64_RANGE and guard.field == "rhs"
            )
            immediate_bit_count = min(compare_width, 32)
            assert (range_guard.minimum, range_guard.maximum) == (
                -(2 ** (immediate_bit_count - 1)),
                2 ** (immediate_bit_count - 1) - 1,
            )
        else:
            assert emit.operands["rhs"] == ValueRef.operand(
                "rhs", source_node="compare"
            )
        actual_cells.add((compare.source_op, compare_type, compare_width, immediate))
    assert actual_cells == {
        *(
            (
                scalar_comparison.scalar_cmpi,
                Scalar(f"i{bit_count}"),
                bit_count,
                immediate,
            )
            for bit_count in (8, 16, 32, 64)
            for immediate in (False, True)
        ),
        *(
            (
                index.index_cmp,
                Scalar(("index", "offset")),
                64,
                immediate,
            )
            for immediate in (False, True)
        ),
    }
    assert len(rules) == len(actual_cells)


def test_scalar_compare_select_rules_cover_native_carrier_matrix() -> None:
    gpr32_result_types = Scalar(
        ("i1", "i8", "i16", "i32", "f8E4M3", "f8E5M2", "f16", "bf16")
    )
    rules = tuple(
        case
        for case in X86_SCALAR_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op in (scalar_comparison.scalar_cmpi, index.index_cmp)
        and case.descriptor is not None
        and case.descriptor.key.startswith("x86.scalar.cmp.select.")
    )
    expected_cells = {
        (source_op, compare_type, compare_width, result_type, result_width, immediate)
        for source_op, compare_type, compare_width in (
            (scalar_comparison.scalar_cmpi, Scalar("i8"), 8),
            (scalar_comparison.scalar_cmpi, Scalar("i16"), 16),
            (scalar_comparison.scalar_cmpi, Scalar("i32"), 32),
            (scalar_comparison.scalar_cmpi, Scalar("i64"), 64),
            (index.index_cmp, Scalar(("index", "offset")), 64),
        )
        for result_type, result_width in (
            (gpr32_result_types, 32),
            (Scalar(("i64", "index", "offset")), 64),
            (Buffer(), 64),
        )
        for immediate in (False, True)
    }
    actual_cells = set()
    for rule in rules:
        assert not rule.target_query
        assert len(rule.source_nodes) == 1
        select = rule.source_nodes[0]
        assert select.source_op is scf.scf_select
        assert select.relation is SourceNodeRelation.EXCLUSIVE_USER
        assert select.parent_value == ValueRef.result("result")
        assert select.node_value == ValueRef.operand("condition")
        compare_type = _value_type_guard(rule, "lhs")
        compare_width = {
            Scalar("i8"): 8,
            Scalar("i16"): 16,
            Scalar("i32"): 32,
        }.get(compare_type, 64)
        assert _value_type_guard(rule, "rhs") == compare_type
        assert _value_type_guard(rule, "result") == Scalar("i1")
        result_type = _source_node_value_type_guard(select, "result")
        emit = rule.emit[0]
        assert isinstance(emit, EmitDescriptorOp)
        assert emit.results["dst"] == ValueRef.result("result", source_node="select")
        assert emit.operands["lhs"] == ValueRef.operand("lhs")
        assert emit.operands["true_value"] == ValueRef.operand(
            "true_value", source_node="select"
        )
        assert emit.operands["false_value"] == ValueRef.operand(
            "false_value", source_node="select"
        )
        predicate = emit.immediates["predicate"]
        assert predicate.kind is AttrProjectKind.ENUM_REMAP
        assert predicate.source_attr == "predicate"
        assert dict(predicate.enum_values) == dict(X86_INTEGER_CONDITION_CODES)
        immediate_fields = tuple(
            field for field in emit.immediates if field.startswith("imm")
        )
        immediate = bool(immediate_fields)
        assert ("rhs" not in emit.operands) == immediate
        assert immediate == (".imm." in rule.descriptor.key)
        assert (rule.priority > 0) == immediate
        if immediate:
            (immediate_field,) = immediate_fields
            assert immediate_field == f"imm{min(compare_width, 32)}"
            immediate_project = emit.immediates[immediate_field]
            assert isinstance(immediate_project, ValueProject)
            assert not immediate_project.source_node
        else:
            assert emit.operands["rhs"] == ValueRef.operand("rhs")
        result_width = 32 if result_type == gpr32_result_types else 64
        actual_cells.add(
            (
                rule.source_op,
                compare_type,
                compare_width,
                result_type,
                result_width,
                immediate,
            )
        )
    assert actual_cells == expected_cells
    assert len(rules) == len(expected_cells)


def test_scalar_select_rules_cover_native_carrier_families() -> None:
    rules = tuple(
        case
        for case in X86_SCALAR_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is scf.scf_select
    )
    assert {
        (_value_type_guard(rule, "result"), rule.descriptor.key) for rule in rules
    } == {
        (
            Scalar(
                (
                    "i1",
                    "i8",
                    "i16",
                    "i32",
                    "f8E4M3",
                    "f8E5M2",
                    "f16",
                    "bf16",
                )
            ),
            "x86.scalar.select.gpr32",
        ),
        (Scalar(("i64", "index", "offset")), "x86.scalar.select.gpr64"),
        (Buffer(), "x86.scalar.select.gpr64"),
    }


def test_scalar_integer_read_memory_rules_cover_native_carriers() -> None:
    assert not X86_SCALAR_READ_MEMORY_CONTRACT_FRAGMENT.target_contract_query
    source_operations = {
        "addi": scalar_arithmetic.scalar_addi,
        "subi": scalar_arithmetic.scalar_subi,
        "muli": scalar_arithmetic.scalar_muli,
        "andi": scalar_bitwise.scalar_andi,
        "ori": scalar_bitwise.scalar_ori,
        "xori": scalar_bitwise.scalar_xori,
    }
    assert {
        (
            family.source_operation,
            family.descriptor_operation,
            family.commutative,
        )
        for family in X86_SCALAR_READ_MEMORY_FAMILIES
    } == {
        ("addi", "add", True),
        ("subi", "sub", False),
        ("muli", "imul", True),
        ("andi", "and", True),
        ("ori", "or", True),
        ("xori", "xor", True),
    }
    expected_reports = {
        (
            f"native_memory_integer_{family.descriptor_operation}_i{width}_"
            f"{loaded_operand}",
            source_operations[family.source_operation],
            family.descriptor_operation,
            width,
            loaded_operand,
        )
        for family in X86_SCALAR_READ_MEMORY_FAMILIES
        for width in (32, 64)
        for loaded_operand in (("lhs", "rhs") if family.commutative else ("rhs",))
    }
    report_groups = {}
    for rule in X86_SCALAR_READ_MEMORY_CONTRACT_FRAGMENT.cases:
        if (
            not isinstance(rule, DescriptorRule)
            or rule.source_op is not view.view_load
            or not rule.report_key.startswith("native_memory_integer_")
        ):
            continue
        report_groups.setdefault(rule.report_key, []).append(rule)

    assert set(report_groups) == {report for report, *_ in expected_reports}
    for report, source_op, operation, width, loaded_operand in expected_reports:
        rules = report_groups[report]
        assert len(rules) == 9
        for rule in rules:
            assert rule.priority == 1
            assert len(rule.source_nodes) == 1
            consumer = rule.source_nodes[0]
            assert consumer.name == "consumer"
            assert consumer.source_op is source_op
            assert consumer.relation is SourceNodeRelation.EXCLUSIVE_USER
            assert consumer.node_value.field == loaded_operand
            assert all(
                guard.type_pattern == Scalar(f"i{width}")
                for guard in consumer.guards
                if guard.kind is GuardKind.VALUE_TYPE
            )
            memory_emits = tuple(
                emit
                for emit in rule.emit
                if isinstance(emit, EmitDescriptorOp)
                and emit.descriptor is rule.descriptor
            )
            assert len(memory_emits) == 1
            memory_emit = memory_emits[0]
            assert memory_emit.descriptor.key.startswith(f"x86.scalar.{operation}.load")
            assert memory_emit.descriptor.key.endswith(f".gpr{width}")
            assert memory_emit.source_memory.operation is SourceMemoryOperation.LOAD
            assert memory_emit.results["dst"].source_node == "consumer"
            accumulator_operand = "rhs" if loaded_operand == "lhs" else "lhs"
            assert memory_emit.operands["lhs"] == ValueRef.operand(
                accumulator_operand, source_node="consumer"
            )


def test_address_arithmetic_uses_immediates_after_specialized_lea_forms() -> None:
    operations = (index.index_mul, index.index_scale)
    assert _integer_immediate_rule_cells(operations) == {
        *(
            (
                index.index_mul,
                Scalar("index"),
                constant_field,
                "x86.scalar.imul.imm.gpr64",
            )
            for constant_field in ("rhs", "lhs")
        ),
        *(
            (
                index.index_scale,
                Scalar("offset"),
                "stride",
                "x86.scalar.imul.imm.gpr64",
            ),
        ),
    }
    authored_cases = tuple(X86_SCALAR_CONTRACT_FRAGMENT.cases)
    for source_op, specialized_key, immediate_key in (
        (index.index_mul, "x86.scalar.lea.scale.gpr64", "x86.scalar.imul.imm.gpr64"),
        (index.index_scale, "x86.scalar.lea.scale.gpr64", "x86.scalar.imul.imm.gpr64"),
    ):
        specialized_indices = tuple(
            ordinal
            for ordinal, case in enumerate(authored_cases)
            if isinstance(case, DescriptorRule)
            and case.source_op is source_op
            and case.descriptor.key == specialized_key
        )
        immediate_indices = tuple(
            ordinal
            for ordinal, case in enumerate(authored_cases)
            if isinstance(case, DescriptorRule)
            and case.source_op is source_op
            and case.descriptor.key == immediate_key
        )
        assert specialized_indices
        assert immediate_indices
        assert max(specialized_indices) < min(immediate_indices)


def test_scalar_storage_float_constants_share_one_bit_preserving_rule() -> None:
    storage_type = Scalar(tuple(element.name for element in STORAGE_ELEMENTS))
    rules = tuple(
        case
        for case in X86_SCALAR_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op is scalar_conversion.scalar_constant
        and _value_type_guard(case, "result") == storage_type
    )
    assert len(rules) == 1
    rule = rules[0]
    assert rule.descriptor.key == "x86.scalar.movimm.gpr32"
    assert rule.emit[0].immediates == {"imm32": ValueProject.float_bits("result")}


def test_uniform_shift_rules_cover_native_types_widths_and_count_forms() -> None:
    operations = {vector.vector_shli, vector.vector_shrsi, vector.vector_shrui}
    for fragment, widths in (
        (X86_AVX2_CONTRACT_FRAGMENT, (128, 256)),
        (X86_AVX512_CONTRACT_FRAGMENT, (512,)),
    ):
        rules = tuple(
            case
            for case in fragment.cases
            if isinstance(case, DescriptorRule)
            and case.source_op in operations
            and any(guard.kind == GuardKind.VALUE_I64_RANGE for guard in case.guards)
        )
        actual = set()
        for rule in rules:
            count_range = next(
                guard
                for guard in rule.guards
                if guard.kind == GuardKind.VALUE_I64_RANGE
            )
            immediate = any(
                guard.kind == GuardKind.VALUE_EXACT_I64 for guard in rule.guards
            )
            actual.add(
                (
                    rule.source_op,
                    _value_type_guard(rule, "result"),
                    count_range.minimum,
                    count_range.maximum,
                    immediate,
                )
            )
            if immediate:
                assert rule.guards[0].kind == GuardKind.VALUE_EXACT_I64
                assert len(rule.emit) == 1
                assert rule.priority == 2
            else:
                origin = ValueRef.uniform_element_origin_operand("rhs")
                origin_guard = next(
                    guard for guard in rule.guards if guard.value_ref == origin
                )
                assert rule.guards[0] == origin_guard
                assert origin_guard.type_pattern == Scalar(
                    f"i{count_range.maximum + 1}"
                )
                first = rule.emit[0]
                if count_range.maximum == 15:
                    assert first.descriptor.key == "x86.scalar.movzx.u16.gpr32"
                    assert first.operands == {"src": origin}
                    assert len(rule.emit) == 3
                else:
                    assert first.descriptor.key == (
                        "x86.avx2.vmovq.xmm.gpr64"
                        if count_range.maximum == 63
                        else "x86.avx2.vmovd.xmm.gpr32"
                    )
                    assert first.operands == {"input": origin}
                    assert len(rule.emit) == 2
        expected = {
            (operation, Vector(f"i{bits}", lanes=width // bits), 0, bits - 1, immediate)
            for operation in operations
            for bits in (16, 32, 64)
            for width in widths
            for immediate in (
                (False, True) if fragment is X86_AVX2_CONTRACT_FRAGMENT else (True,)
            )
            if not (
                fragment is X86_AVX2_CONTRACT_FRAGMENT
                and operation is vector.vector_shrsi
                and bits == 64
            )
        }
        if fragment is X86_AVX512_CONTRACT_FRAGMENT:
            expected.update(
                (vector.vector_shrsi, Vector("i64", lanes=width // 64), 0, 63, True)
                for width in (128, 256)
            )
        assert actual == expected
        assert len(rules) == len(expected)


def test_avx2_scalar_float_arithmetic_covers_f32_and_f64() -> None:
    source_operations = {
        scalar_arithmetic.scalar_addf: "addf",
        scalar_arithmetic.scalar_subf: "subf",
        scalar_arithmetic.scalar_mulf: "mulf",
        scalar_arithmetic.scalar_divf: "divf",
    }
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op in (*source_operations, scalar_math.scalar_fmaf)
    )
    assert {
        (
            source_operations[rule.source_op],
            _value_type_guard(rule, "result"),
        )
        for rule in rules
        if rule.source_op in source_operations
    } == {
        (row.source_operation, Scalar(row.element.name))
        for row in AVX2_SCALAR_FLOAT_BINARY_FAMILIES
    }
    assert {
        _value_type_guard(rule, "result")
        for rule in rules
        if rule.source_op is scalar_math.scalar_fmaf
    } == {Scalar(element.name) for element in FLOAT_ELEMENTS}
    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX2_DESCRIPTOR_SET.descriptors
    }
    assert {
        f"x86.avx2.{row.mnemonic}.xmm" for row in AVX2_SCALAR_FLOAT_BINARY_FAMILIES
    } <= descriptor_keys
    assert {
        f"x86.avx2.{mnemonic}.xmm"
        for mnemonic in AVX2_SCALAR_FLOAT_FMA_MNEMONICS.values()
    } <= descriptor_keys
    assert sum(rule.source_op in source_operations for rule in rules) == len(
        AVX2_SCALAR_FLOAT_BINARY_FAMILIES
    )
    assert sum(rule.source_op is scalar_math.scalar_fmaf for rule in rules) == len(
        FLOAT_ELEMENTS
    )


def test_scalar_float_memory_covers_types_and_addressing_without_bitcasts() -> None:
    for source_op, type_field in (
        (view.view_load, "result"),
        (view.view_store, "value"),
    ):
        for element_type, byte_count, mnemonic in (
            ("f32", 4, "vmovss"),
            ("f64", 8, "vmovsd"),
        ):
            rules = tuple(
                case
                for case in X86_AVX2_CONTRACT_FRAGMENT.cases
                if isinstance(case, DescriptorRule)
                and case.source_op is source_op
                and _value_type_guard(case, type_field) == Scalar(element_type)
            )
            assert len(rules) == 9
            for rule in rules:
                emitted = tuple(
                    emit for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
                )
                memory = tuple(
                    emit
                    for emit in emitted
                    if emit.descriptor.semantic_tag.startswith("memory.")
                )
                assert len(memory) == 1
                assert memory[0].descriptor.key.startswith(f"x86.avx2.{mnemonic}.")
                assert memory[0].source_memory is not None
                assert memory[0].source_memory.element_byte_count == byte_count
                assert memory[0].source_memory.vector_lane_count == 1
                assert not any(
                    emit.descriptor.key.startswith(
                        ("x86.avx2.vmovd.", "x86.avx2.vmovq.")
                    )
                    for emit in emitted
                )


def test_vector_memory_covers_storage_families_and_carrier_widths() -> None:
    for source_op, type_field in (
        (vector.vector_load, "result"),
        (vector.vector_store, "value"),
    ):
        for family in X86_VECTOR_STORAGE_FAMILIES:
            for vector_bit_width in AVX2_CARRIER_VECTOR_BIT_WIDTHS:
                if vector_bit_width < family.element_bit_width:
                    continue
                value_type = Vector(
                    family.element_names,
                    lanes=vector_bit_width // family.element_bit_width,
                )
                rules = tuple(
                    case
                    for case in X86_AVX2_CONTRACT_FRAGMENT.cases
                    if isinstance(case, DescriptorRule)
                    and case.source_op is source_op
                    and _value_type_guard(case, type_field) == value_type
                )
                assert len(rules) == 9
                for rule in rules:
                    memory = tuple(
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                        and emit.descriptor.semantic_tag.startswith("memory.")
                    )
                    assert len(memory) == 1
                    assert memory[0].source_memory is not None
                    assert memory[0].source_memory.element_byte_count == (
                        family.element_bit_width // 8
                    )
                    assert memory[0].source_memory.vector_lane_count == (
                        vector_bit_width // family.element_bit_width
                    )
                    if vector_bit_width <= 16:
                        descriptor_prefix = (
                            "x86.scalar.movzx."
                            if source_op is vector.vector_load
                            else "x86.scalar.mov."
                        )
                        assert any(
                            emit.descriptor.key.startswith("x86.avx2.vmovd.")
                            for emit in rule.emit
                            if isinstance(emit, EmitDescriptorOp)
                        )
                    elif vector_bit_width == 32:
                        descriptor_prefix = "x86.avx2.vmovss."
                    elif vector_bit_width == 64:
                        descriptor_prefix = "x86.avx2.vmovsd."
                    else:
                        descriptor_prefix = "x86.avx2.vmovdqu32."
                    assert memory[0].descriptor.key.startswith(descriptor_prefix)


def test_avx2_lane_movement_covers_every_payload_type_and_width() -> None:
    assert AVX2_CARRIER_VECTOR_BIT_WIDTHS == (8, 16, 32, 64, 128, 256)
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op in (vector.vector_extract, vector.vector_insert)
        and _value_type_guard(
            case, "result" if case.source_op is vector.vector_extract else "value"
        )
        != Scalar("i1")
    )
    expected = {
        (
            source_op.name,
            Vector(row.element_names, lanes=vector_bit_width // row.element_bit_width),
            Scalar(row.element_names),
            (0, vector_bit_width // row.element_bit_width - 1),
        )
        for row in X86_LANE_FAMILIES
        for vector_bit_width in (8, 16, 32, 64, 128, 256)
        for source_op in (vector.vector_extract, vector.vector_insert)
        if vector_bit_width >= row.element_bit_width
    }
    expected.update(
        (
            source_op.name,
            Vector(element_name, lanes=vector_bit_width // element_bit_width),
            Scalar(element_name),
            (0, vector_bit_width // element_bit_width - 1),
        )
        for element_name, element_bit_width in (("f32", 32), ("f64", 64))
        for vector_bit_width in (8, 16, 32, 64, 128, 256)
        for source_op in (vector.vector_extract, vector.vector_insert)
        if vector_bit_width >= element_bit_width
        if not (
            element_name == "f64"
            and vector_bit_width == 128
            and source_op is vector.vector_insert
        )
    )
    expected.update(
        (
            vector.vector_insert.name,
            Vector("f64", lanes=2),
            Scalar("f64"),
            (lane, lane),
        )
        for lane in range(2)
    )
    expected.add(
        (
            vector.vector_insert.name,
            Vector("f64", lanes=1),
            Scalar("f64"),
            (0, 0),
        )
    )
    actual = {
        (
            rule.source_op.name,
            _value_type_guard(
                rule, "source" if rule.source_op is vector.vector_extract else "dest"
            ),
            _value_type_guard(
                rule, "result" if rule.source_op is vector.vector_extract else "value"
            ),
            _static_index_range(rule),
        )
        for rule in rules
    }
    assert actual == expected
    assert len(rules) == len(expected)


def test_avx2_splats_cover_every_register_carried_payload() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op is vector.vector_splat
        and _value_type_guard(case, "scalar") != Scalar("i1")
    )
    expected = {
        Vector(element.name, lanes=vector_bit_width // element.bit_width)
        for element in (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS, *FLOAT_ELEMENTS)
        for vector_bit_width in (8, 16, 32, 64, 128, 256)
        if vector_bit_width >= element.bit_width
    }
    assert {_value_type_guard(rule, "result") for rule in rules} == expected
    assert len(rules) == len(expected)


def test_avx2_memory_covers_every_register_carried_payload() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op in (vector.vector_load, vector.vector_store)
    )
    actual = set()
    for rule in rules:
        type_field = "result" if rule.source_op is vector.vector_load else "value"
        memory_emits = tuple(
            emit
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
            and emit.source_memory is not None
            and emit.descriptor.semantic_tag.startswith("memory.")
        )
        assert len(memory_emits) == 1
        memory_emit = memory_emits[0]
        source_memory = memory_emit.source_memory
        assert source_memory is not None
        vector_bit_width = (
            source_memory.element_byte_count * source_memory.vector_lane_count * 8
        )
        actual.add(
            (
                rule.source_op,
                _value_type_guard(rule, type_field),
                vector_bit_width,
            )
        )
        descriptor_prefix = {
            8: "x86.scalar.",
            16: "x86.scalar.",
            32: "x86.avx2.vmovss.",
            64: "x86.avx2.vmovsd.",
            128: "x86.avx2.vmovdqu32.",
            256: "x86.avx2.vmovdqu32.",
        }[vector_bit_width]
        assert memory_emit.descriptor.key.startswith(descriptor_prefix)
        transport_keys = {
            emit.descriptor.key
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
            and emit.descriptor.key.startswith("x86.avx2.vmovd.")
        }
        if vector_bit_width <= 16:
            assert transport_keys == {
                (
                    "x86.avx2.vmovd.xmm.gpr32"
                    if rule.source_op is vector.vector_load
                    else "x86.avx2.vmovd.gpr32.xmm"
                )
            }
        else:
            assert not transport_keys

    expected = {
        (
            source_op,
            Vector(
                family.element_names,
                lanes=vector_bit_width // family.element_bit_width,
            ),
            vector_bit_width,
        )
        for family in X86_VECTOR_STORAGE_FAMILIES
        for vector_bit_width in (8, 16, 32, 64, 128, 256)
        for source_op in (vector.vector_load, vector.vector_store)
        if vector_bit_width >= family.element_bit_width
    }
    assert actual == expected


def test_exact_lane_construction_covers_every_payload_type_and_width() -> None:
    register_suffixes = {128: "xmm", 256: "ymm", 512: "zmm"}
    for fragment, vector_bit_widths, descriptor_prefix in (
        (
            X86_AVX2_CONTRACT_FRAGMENT,
            AVX2_VECTOR_BIT_WIDTHS,
            "x86.avx2.vmovdqu.rodata",
        ),
        (
            X86_AVX512_CONTRACT_FRAGMENT,
            AVX512_VECTOR_BIT_WIDTHS,
            "x86.avx512.vmovdqu64.rodata",
        ),
    ):
        rules = tuple(
            case
            for case in fragment.cases
            if isinstance(case, DescriptorRule)
            and case.source_op is vector.vector_from_elements
        )
        expected = {
            (
                Vector(
                    family.element_names,
                    lanes=vector_bit_width // family.element_bit_width,
                ),
                vector_bit_width // family.element_bit_width,
                family.element_bit_width,
                f"{descriptor_prefix}.{register_suffixes[vector_bit_width]}",
            )
            for family in X86_VECTOR_STORAGE_FAMILIES
            for vector_bit_width in vector_bit_widths
        }
        actual = set()
        for rule in rules:
            count_guard = next(
                guard
                for guard in rule.guards
                if guard.kind is GuardKind.OPERAND_SEGMENT_COUNT
            )
            exact_guard = next(
                guard
                for guard in rule.guards
                if guard.kind is GuardKind.OPERAND_SEGMENT_EXACT_BITS
            )
            projection = rule.emit[0].immediates["data"]
            assert isinstance(projection, OperandSegmentProject)
            assert projection.source_operand == "elements"
            assert projection.bit_width == exact_guard.bit_width
            actual.add(
                (
                    _value_type_guard(rule, "result"),
                    count_guard.count,
                    exact_guard.bit_width,
                    rule.descriptor.key,
                )
            )
        assert actual == expected
        assert len(rules) == len(expected)


def test_avx512_lane_movement_covers_every_zmm_payload_type() -> None:
    rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op in (vector.vector_extract, vector.vector_insert)
        and _value_type_guard(
            case, "result" if case.source_op is vector.vector_extract else "value"
        )
        != Scalar("i1")
    )
    expected = {
        (
            source_op.name,
            Vector(row.element_names, lanes=512 // row.element_bit_width),
            Scalar(row.element_names),
            (0, 512 // row.element_bit_width - 1),
        )
        for row in X86_LANE_FAMILIES
        for source_op in (vector.vector_extract, vector.vector_insert)
    }
    expected.update(
        (
            source_op.name,
            Vector(element_name, lanes=512 // element_bit_width),
            Scalar(element_name),
            (0, 512 // element_bit_width - 1),
        )
        for element_name, element_bit_width in (("f32", 32), ("f64", 64))
        for source_op in (vector.vector_extract, vector.vector_insert)
    )
    actual = {
        (
            rule.source_op.name,
            _value_type_guard(
                rule, "source" if rule.source_op is vector.vector_extract else "dest"
            ),
            _value_type_guard(
                rule, "result" if rule.source_op is vector.vector_extract else "value"
            ),
            _static_index_range(rule),
        )
        for rule in rules
    }
    assert actual == expected
    assert len(rules) == len(expected)


def test_avx2_shuffles_cover_every_payload_type_and_width() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_shuffle
    )
    expected = {
        Vector(
            family.element_names,
            lanes=vector_bit_width // family.element_bit_width,
        )
        for family in AVX2_SHUFFLE_FAMILIES
        for vector_bit_width in AVX2_VECTOR_BIT_WIDTHS
    }
    assert {_value_type_guard(rule, "result") for rule in rules} == expected
    assert len(rules) == len(expected)
    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX2_DESCRIPTOR_SET.descriptors
    }
    assert {
        "x86.avx2.vpshufb.xmm",
        "x86.avx2.vpshufb.ymm",
        "x86.avx2.vpshufd.xmm",
        "x86.avx2.vpermps.ymm",
        "x86.avx2.vpermilpd.xmm",
        "x86.avx2.vpermq.ymm",
    } <= descriptor_keys


def test_avx512_shuffles_cover_every_payload_type() -> None:
    rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_shuffle
    )
    expected = {
        Vector(family.element_names, lanes=512 // family.element_bit_width)
        for family in AVX2_SHUFFLE_FAMILIES
    }
    assert {_value_type_guard(rule, "result") for rule in rules} == expected
    assert len(rules) == len(expected)
    descriptor_keys = {
        descriptor.key for descriptor in X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
    }
    assert {
        "x86.avx512.vmovdqu64.rodata.zmm",
        "x86.avx512.vpermd.zmm",
        "x86.avx512.vpermq.zmm",
        "x86.avx512.vpermw.zmm",
        "x86.avx512.vpsllw.zmm",
        "x86.avx512.vpsrlw.zmm",
    } <= descriptor_keys


def test_avx2_float_bitcasts_cover_every_storage_family_and_width() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, ValueAliasRule) and case.source_op is vector.vector_bitcast
    )
    expected = {
        (source_type, result_type)
        for float_elements, integer_element, element_bit_width in (
            (("f8E4M3", "f8E5M2"), "i8", 8),
            (("f16", "bf16"), "i16", 16),
            (("f32",), "i32", 32),
            (("f64",), "i64", 64),
        )
        for vector_bit_width in AVX2_CARRIER_VECTOR_BIT_WIDTHS
        if vector_bit_width >= element_bit_width
        for lane_count in (vector_bit_width // element_bit_width,)
        for source_type, result_type in (
            (
                Vector(float_elements, lanes=lane_count),
                Vector(integer_element, lanes=lane_count),
            ),
            (
                Vector(integer_element, lanes=lane_count),
                Vector(float_elements, lanes=lane_count),
            ),
        )
    }
    actual = {
        (
            _value_type_guard(rule, "input"),
            _value_type_guard(rule, "result"),
        )
        for rule in rules
    }
    assert actual == expected
    assert len(rules) == len(expected)


def test_avx2_float_arithmetic_covers_every_carrier_width() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )
    register_suffixes = {32: "xmm", 64: "xmm", 128: "xmm", 256: "ymm"}
    expected_direct = {
        (
            f"vector.{family.source_operation}",
            f"x86.avx2.{family.mnemonic}.{register_suffixes[vector_bit_width]}",
            Vector(
                family.element.name,
                lanes=vector_bit_width // family.element.bit_width,
            ),
        )
        for family in AVX2_FLOAT_BINARY_FAMILIES
        for vector_bit_width in AVX2_FLOAT_VECTOR_BIT_WIDTHS
        if vector_bit_width >= family.element.bit_width
    }
    assert {
        (rule.source_op.name, rule.descriptor.key, _value_type_guard(rule, "result"))
        for rule in rules
        if rule.source_op.name
        in {
            f"vector.{family.source_operation}" for family in AVX2_FLOAT_BINARY_FAMILIES
        }
    } == expected_direct

    expected_fma = {
        (
            f"x86.avx2.{AVX2_FLOAT_FMA_MNEMONICS[element.name]}."
            f"{register_suffixes[vector_bit_width]}",
            Vector(element.name, lanes=vector_bit_width // element.bit_width),
        )
        for element in FLOAT_ELEMENTS
        for vector_bit_width in AVX2_FLOAT_VECTOR_BIT_WIDTHS
        if vector_bit_width >= element.bit_width
    }
    assert {
        (rule.descriptor.key, _value_type_guard(rule, "result"))
        for rule in rules
        if rule.source_op is vector.vector_fmaf
    } == expected_fma

    expected_constant_types = {
        Vector(element.name, lanes=vector_bit_width // element.bit_width)
        for element in FLOAT_ELEMENTS
        for vector_bit_width in AVX2_FLOAT_VECTOR_BIT_WIDTHS
        if vector_bit_width >= element.bit_width
    }
    assert {
        _value_type_guard(rule, "result")
        for rule in rules
        if rule.source_op is vector.vector_constant
        and _value_type_guard(rule, "result") in expected_constant_types
    } == expected_constant_types


def test_avx512_construction_covers_every_zmm_payload_type() -> None:
    descriptor_rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )
    payload_elements = (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS, *FLOAT_ELEMENTS)
    expected_constant_types = {
        Vector(element.name, lanes=vector_bit_width // element.bit_width)
        for element in payload_elements
        for vector_bit_width in AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS
    }
    constant_rules = tuple(
        rule
        for rule in descriptor_rules
        if rule.source_op is vector.vector_constant
        and rule.descriptor.key.startswith("x86.avx512.vpbroadcast")
    )
    assert {_value_type_guard(rule, "result") for rule in constant_rules} == (
        expected_constant_types
    )
    assert len(constant_rules) == len(expected_constant_types)

    splat_rules = tuple(
        rule
        for rule in descriptor_rules
        if rule.source_op is vector.vector_splat
        and rule.descriptor.key.startswith(
            ("x86.avx512.vpbroadcast", "x86.avx512.vbroadcast")
        )
    )
    expected_splat_types = {
        *(
            Vector(element.name, lanes=vector_bit_width // element.bit_width)
            for element in (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS)
            for vector_bit_width in AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS
        ),
        *(
            Vector(element.name, lanes=512 // element.bit_width)
            for element in FLOAT_ELEMENTS
        ),
    }
    assert {_value_type_guard(rule, "result") for rule in splat_rules} == (
        expected_splat_types
    )
    assert len(splat_rules) == len(expected_splat_types)

    bitcast_rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, ValueAliasRule) and case.source_op is vector.vector_bitcast
    )
    expected_bitcasts = {
        (source_type, result_type)
        for float_elements, integer_element, element_bit_width in (
            (("f8E4M3", "f8E5M2"), "i8", 8),
            (("f16", "bf16"), "i16", 16),
            (("f32",), "i32", 32),
            (("f64",), "i64", 64),
        )
        for lane_count in (512 // element_bit_width,)
        for source_type, result_type in (
            (
                Vector(float_elements, lanes=lane_count),
                Vector(integer_element, lanes=lane_count),
            ),
            (
                Vector(integer_element, lanes=lane_count),
                Vector(float_elements, lanes=lane_count),
            ),
        )
    }
    assert {
        (
            _value_type_guard(rule, "input"),
            _value_type_guard(rule, "result"),
        )
        for rule in bitcast_rules
    } == expected_bitcasts
    assert len(bitcast_rules) == len(expected_bitcasts)


def test_avx512_arithmetic_covers_every_native_family() -> None:
    rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.descriptor is not None
    )
    register_suffixes = {128: "xmm", 256: "ymm", 512: "zmm"}
    expected_direct = {
        (
            f"x86.avx512.{family.mnemonic}.zmm",
            Vector(family.element.name, lanes=512 // family.element.bit_width),
        )
        for family in (*AVX512_INTEGER_BINARY_FAMILIES, *AVX512_FLOAT_BINARY_FAMILIES)
    } | {
        (
            f"x86.avx512.{family.mnemonic}.{register_suffixes[vector_bit_width]}",
            Vector(
                family.element.name,
                lanes=vector_bit_width // family.element.bit_width,
            ),
        )
        for family in AVX512VL_INTEGER_BINARY_FAMILIES
        for vector_bit_width in AVX512VL_VECTOR_BIT_WIDTHS
    }
    direct_descriptor_keys = {descriptor_key for descriptor_key, _ in expected_direct}
    actual_direct = {
        (rule.descriptor.key, _value_type_guard(rule, "result"))
        for rule in rules
        if rule.source_op is not vector.vector_reduce
        and rule.descriptor.key in direct_descriptor_keys
    }
    assert actual_direct == expected_direct
    assert len(actual_direct) == len(expected_direct)

    expected_fma = {
        (
            mnemonic,
            Vector(element.name, lanes=512 // element.bit_width),
        )
        for element in FLOAT_ELEMENTS
        for mnemonic in (AVX512_FLOAT_FMA_MNEMONICS[element.name],)
    }
    actual_fma = {
        (rule.descriptor.mnemonic, _value_type_guard(rule, "result"))
        for rule in rules
        if rule.source_op is vector.vector_fmaf
        and rule.descriptor.key.startswith("x86.avx512.")
    }
    assert actual_fma == expected_fma
    assert len(actual_fma) == len(expected_fma)

    bitwise_rules = tuple(
        rule
        for rule in rules
        if rule.source_op is not vector.vector_reduce
        and rule.descriptor.key
        in {f"x86.avx512.{mnemonic}.zmm" for _, mnemonic, _ in AVX512_BITWISE_FAMILIES}
        and _value_type_guard(rule, "result") != Vector("i1", lanes=64)
    )
    assert {
        (rule.source_op.name, rule.descriptor.mnemonic) for rule in bitwise_rules
    } == {
        (f"vector.{source_operation}", mnemonic)
        for source_operation, mnemonic, _ in AVX512_BITWISE_FAMILIES
    }
    assert {_value_type_guard(rule, "result") for rule in bitwise_rules} == {
        Vector(
            AVX2_PAYLOAD_ELEMENT_NAMES,
            minimum_lanes=2,
            maximum_lanes=max(AVX512_VECTOR_BIT_WIDTHS) // 8,
        )
    }


def test_avx2_iotas_cover_every_integer_element_and_width() -> None:
    rules = tuple(
        case
        for case in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_iota
    )
    actual = {
        (
            _value_type_guard(rule, "base"),
            _value_type_guard(rule, "result"),
            rule.priority,
        )
        for rule in rules
    }
    expected = {
        (
            Scalar(element.name),
            Vector(
                element.name,
                lanes=vector_bit_width // element.bit_width,
            ),
            priority,
        )
        for element in INTEGER_ELEMENTS
        for vector_bit_width in AVX2_CARRIER_VECTOR_BIT_WIDTHS
        for priority in (0, 1)
        if vector_bit_width >= element.bit_width
        if vector_bit_width != element.bit_width or priority == 0
    }
    assert actual == expected
    assert len(rules) == len(expected)
    assert all(
        any(
            guard.kind == GuardKind.VALUE_I64_RANGE
            and guard.field == "step"
            and guard.minimum == 1
            and guard.maximum == 1
            for guard in rule.guards
        )
        for rule in rules
        if rule.priority == 1
    )


def test_avx512_iotas_cover_every_zmm_integer_element() -> None:
    rules = tuple(
        case
        for case in X86_AVX512_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op is vector.vector_iota
    )
    actual = {
        (
            _value_type_guard(rule, "base"),
            _value_type_guard(rule, "result"),
            rule.priority,
        )
        for rule in rules
    }
    expected = {
        (
            Scalar(element.name),
            Vector(element.name, lanes=512 // element.bit_width),
            priority,
        )
        for element in INTEGER_ELEMENTS
        for priority in (0, 1)
    }
    assert actual == expected
    assert len(rules) == len(expected)
    assert all(
        any(
            guard.kind == GuardKind.VALUE_I64_RANGE
            and guard.field == "step"
            and guard.minimum == 1
            and guard.maximum == 1
            for guard in rule.guards
        )
        for rule in rules
        if rule.priority == 1
    )


def test_float_extrema_cover_every_semantic_and_width() -> None:
    vector_source_operations = {
        vector.vector_minimumf: "minimumf",
        vector.vector_maximumf: "maximumf",
        vector.vector_minnumf: "minnumf",
        vector.vector_maxnumf: "maxnumf",
    }
    for fragment, vector_bit_widths in (
        (X86_AVX2_CONTRACT_FRAGMENT, AVX2_FLOAT_VECTOR_BIT_WIDTHS),
        (X86_AVX512_CONTRACT_FRAGMENT, AVX512_VECTOR_BIT_WIDTHS),
    ):
        extrema_rules = tuple(
            rule
            for rule in fragment.cases
            if isinstance(rule, DescriptorRule)
            and rule.source_op in vector_source_operations
        )
        assert {
            (
                vector_source_operations[rule.source_op],
                _value_type_guard(rule, "result"),
            )
            for rule in extrema_rules
        } == {
            (
                operation,
                Vector(
                    element.name,
                    lanes=vector_bit_width // element.bit_width,
                ),
            )
            for operation in FLOAT_EXTREMA_OPERATIONS
            for element in FLOAT_ELEMENTS
            for vector_bit_width in vector_bit_widths
            if vector_bit_width >= element.bit_width
        }
        priorities_by_cell = {}
        for rule in extrema_rules:
            key = (rule.source_op, _value_type_guard(rule, "result"))
            priorities_by_cell.setdefault(key, set()).add(rule.priority)
        assert {
            frozenset(priorities) for priorities in priorities_by_cell.values()
        } == {frozenset((0, 1))}
        fast_rules = tuple(rule for rule in extrema_rules if rule.priority == 1)
        assert all(
            {
                guard.enum_keyword
                for guard in rule.guards
                if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
            }
            == {"nnan", "nsz"}
            for rule in fast_rules
        )

    scalar_source_operations = {
        scalar_arithmetic.scalar_minimumf: "minimumf",
        scalar_arithmetic.scalar_maximumf: "maximumf",
        scalar_arithmetic.scalar_minnumf: "minnumf",
        scalar_arithmetic.scalar_maxnumf: "maxnumf",
    }
    actual_scalars = {
        (
            scalar_source_operations[rule.source_op],
            _value_type_guard(rule, "result"),
        )
        for rule in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule)
        and rule.source_op in scalar_source_operations
    }
    assert actual_scalars == {
        (operation, Scalar(element.name))
        for operation in FLOAT_EXTREMA_OPERATIONS
        for element in FLOAT_ELEMENTS
    }

    scalar_extrema_rules = tuple(
        rule
        for rule in X86_AVX2_CONTRACT_FRAGMENT.cases
        if isinstance(rule, DescriptorRule)
        and rule.source_op in scalar_source_operations
    )
    priorities_by_cell = {}
    for rule in scalar_extrema_rules:
        key = (rule.source_op, _value_type_guard(rule, "result"))
        priorities_by_cell.setdefault(key, set()).add(rule.priority)
    assert {frozenset(priorities) for priorities in priorities_by_cell.values()} == {
        frozenset((0, 1))
    }
    fast_rules = tuple(rule for rule in scalar_extrema_rules if rule.priority == 1)
    assert all(
        {
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
        }
        == {"nnan", "nsz"}
        for rule in fast_rules
    )


def test_integer_reductions_cover_every_native_combine_family() -> None:
    for fragment, families, vector_bit_widths in (
        (
            X86_AVX2_CONTRACT_FRAGMENT,
            AVX2_INTEGER_REDUCTION_FAMILIES,
            AVX2_VECTOR_BIT_WIDTHS,
        ),
        (
            X86_AVX512_CONTRACT_FRAGMENT,
            AVX512_INTEGER_REDUCTION_FAMILIES,
            AVX512_VECTOR_BIT_WIDTHS,
        ),
    ):
        expected = {
            (
                row.source_operation,
                Vector(
                    row.element.name,
                    lanes=vector_bit_width // row.element.bit_width,
                ),
            )
            for row in families
            for vector_bit_width in vector_bit_widths
        }
        expected_operations = {row.source_operation for row in families}
        integer_rules = tuple(
            rule
            for rule in fragment.cases
            if isinstance(rule, DescriptorRule)
            and rule.source_op is vector.vector_reduce
            and any(
                guard.kind == GuardKind.ENUM_ATTR_EQUALS
                and guard.field == "kind"
                and guard.enum_keyword in expected_operations
                for guard in rule.guards
            )
        )
        actual = {
            (
                next(
                    guard.enum_keyword
                    for guard in rule.guards
                    if guard.kind == GuardKind.ENUM_ATTR_EQUALS
                    and guard.field == "kind"
                ),
                _value_type_guard(rule, "input"),
            )
            for rule in integer_rules
        }
        assert actual == expected
        assert len(integer_rules) == len(expected)


def test_float_reductions_and_dots_cover_both_types_and_all_widths() -> None:
    for fragment, vector_bit_widths in (
        (X86_AVX2_CONTRACT_FRAGMENT, AVX2_FLOAT_VECTOR_BIT_WIDTHS),
        (X86_AVX512_CONTRACT_FRAGMENT, AVX512_VECTOR_BIT_WIDTHS),
    ):
        valid_shapes = tuple(
            (element, vector_bit_width)
            for element in FLOAT_ELEMENTS
            for vector_bit_width in vector_bit_widths
            if vector_bit_width >= element.bit_width
        )
        reduction_rules = tuple(
            rule
            for rule in fragment.cases
            if isinstance(rule, DescriptorRule)
            and rule.source_op is vector.vector_reduce
            and any(
                guard.kind == GuardKind.ENUM_ATTR_EQUALS
                and guard.enum_keyword in AVX2_FLOAT_REDUCTION_OPERATIONS
                for guard in rule.guards
            )
        )
        packed_reduction_rules = tuple(
            rule
            for rule in reduction_rules
            if next(
                guard.enum_keyword
                for guard in rule.guards
                if guard.kind == GuardKind.ENUM_ATTR_EQUALS
            )
            in AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS
        )
        assert len(packed_reduction_rules) == (
            len(AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS) * len(valid_shapes) * 2
        )
        assert {
            (
                next(
                    guard.enum_keyword
                    for guard in rule.guards
                    if guard.kind == GuardKind.ENUM_ATTR_EQUALS
                ),
                _value_type_guard(rule, "input"),
                next(
                    guard.kind
                    for guard in rule.guards
                    if guard.kind
                    in (
                        GuardKind.INSTANCE_FLAGS_HAS_ALL,
                        GuardKind.INSTANCE_FLAGS_HAS_NONE,
                    )
                ),
            )
            for rule in packed_reduction_rules
        } == {
            (
                operation,
                Vector(
                    element.name,
                    lanes=vector_bit_width // element.bit_width,
                ),
                fastmath_guard,
            )
            for operation in AVX2_PACKED_FLOAT_REDUCTION_OPERATIONS
            for element, vector_bit_width in valid_shapes
            for fastmath_guard in (
                GuardKind.INSTANCE_FLAGS_HAS_NONE,
                GuardKind.INSTANCE_FLAGS_HAS_ALL,
            )
        }

        fast_extrema_rules = tuple(
            rule
            for rule in reduction_rules
            if next(
                guard.enum_keyword
                for guard in rule.guards
                if guard.kind == GuardKind.ENUM_ATTR_EQUALS
            )
            in FLOAT_EXTREMA_OPERATIONS
        )
        assert len(fast_extrema_rules) == (
            len(FLOAT_EXTREMA_OPERATIONS) * len(valid_shapes)
        )
        assert {
            (
                next(
                    guard.enum_keyword
                    for guard in rule.guards
                    if guard.kind == GuardKind.ENUM_ATTR_EQUALS
                ),
                _value_type_guard(rule, "input"),
                frozenset(
                    guard.enum_keyword
                    for guard in rule.guards
                    if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
                ),
            )
            for rule in fast_extrema_rules
        } == {
            (
                operation,
                Vector(
                    element.name,
                    lanes=vector_bit_width // element.bit_width,
                ),
                frozenset(("reassoc", "nnan", "nsz")),
            )
            for operation in FLOAT_EXTREMA_OPERATIONS
            for element, vector_bit_width in valid_shapes
        }

        dot_rules = tuple(
            rule
            for rule in fragment.cases
            if isinstance(rule, DescriptorRule) and rule.source_op is vector.vector_dotf
        )
        assert len(dot_rules) == len(valid_shapes)
        assert {
            (
                _value_type_guard(rule, "lhs"),
                _value_type_guard(rule, "rhs"),
                _value_type_guard(rule, "init"),
                _value_type_guard(rule, "result"),
            )
            for rule in dot_rules
        } == {
            (vector_type, vector_type, scalar_type, scalar_type)
            for element, vector_bit_width in valid_shapes
            for vector_type in (
                Vector(
                    element.name,
                    lanes=vector_bit_width // element.bit_width,
                ),
            )
            for scalar_type in (Scalar(element.name),)
        }
