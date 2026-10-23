# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.contracts.ordinary_vector import (
    SPIRV_ORDINARY_VECTOR_CONTRACT_CASES,
)
from loom.target.arch.spirv.ordinary_vector import (
    NATIVE_ORDINARY_VECTOR_LANE_COUNTS,
    ORDINARY_VECTOR_COMPONENT_TYPES,
    ORDINARY_VECTOR_CONCAT_SHAPES,
    ORDINARY_VECTOR_SOURCE_SHUFFLE_INSTRUCTIONS,
    ORDINARY_VECTOR_TYPES,
    OrdinaryVectorComponentKind,
)
from loom.target.arch.spirv.ordinary_vector_bit_layout import (
    ORDINARY_VECTOR_BIT_LAYOUT_ALIAS_CASES,
    ORDINARY_VECTOR_BIT_LAYOUT_CASES,
    ORDINARY_VECTOR_BIT_LAYOUT_DIRECT_CASES,
)
from loom.target.arch.spirv.scalar_alu import (
    FLOAT_BINARY_OPERATIONS,
    FLOAT_SCALAR_ALU_TYPES,
)
from loom.target.contracts import (
    DescriptorRule,
    GuardKind,
    SourceOpProject,
    SourceValueKind,
    TypePattern,
    ValueAliasRule,
    ValueRef,
)


def test_every_owned_source_operation_has_a_closed_shape_matrix() -> None:
    case_counts = Counter(
        contract_case.source_op.name
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
    )
    assert case_counts == {
        vector.vector_constant.name: 55,
        vector.vector_bitcast.name: 140,
        vector.vector_splat.name: 32,
        vector.vector_from_elements.name: 31,
        vector.vector_extract.name: 61,
        vector.vector_insert.name: 61,
        vector.vector_shuffle.name: 30,
        vector.vector_concat.name: 66,
        scf.scf_select.name: 40,
    }
    assert len(SPIRV_ORDINARY_VECTOR_CONTRACT_CASES) == 516


def test_aliases_cover_singletons_uniform_carriers_and_identity_bit_layouts() -> None:
    alias_cases = tuple(
        contract_case
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
        if isinstance(contract_case, ValueAliasRule)
    )
    assert {contract_case.source_op for contract_case in alias_cases} == {
        vector.vector_splat,
        vector.vector_from_elements,
        vector.vector_extract,
        vector.vector_insert,
        vector.vector_bitcast,
    }
    assert len(alias_cases) == 37


def test_fp8_structural_aliases_use_singleton_byte_carriers() -> None:
    fp8_types = {"f8E4M3", "f8E5M2"}
    covered_operations = set()
    for case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES:
        patterns = tuple(
            guard.type_pattern
            for guard in case.guards
            if guard.kind is GuardKind.VALUE_TYPE
        )
        fp8_patterns = tuple(
            pattern for pattern in patterns if fp8_types.intersection(pattern.elements)
        )
        if not fp8_patterns:
            continue
        assert all(fp8_types.issubset(pattern.elements) for pattern in fp8_patterns)
        assert all(pattern.lanes in (None, 1) for pattern in fp8_patterns)
        covered_operations.add(case.source_op)
        if isinstance(case, DescriptorRule):
            assert case.source_op is scf.scf_select
            assert case.descriptor.key == "spirv.op_select.i8"
        else:
            assert isinstance(case, ValueAliasRule)
    assert covered_operations == {
        vector.vector_splat,
        vector.vector_from_elements,
        vector.vector_extract,
        vector.vector_insert,
        scf.scf_select,
    }


def test_bit_layout_contract_covers_the_exact_generated_matrix() -> None:
    bitcast_cases = tuple(
        contract_case
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
        if contract_case.source_op is vector.vector_bitcast
    )
    actual_cases = {}
    for contract_case in bitcast_cases:
        patterns = {
            guard.field: guard.type_pattern
            for guard in contract_case.guards
            if guard.kind == GuardKind.VALUE_TYPE
        }
        assert patterns.keys() == {"input", "result"}
        source_pattern = patterns["input"]
        result_pattern = patterns["result"]
        key = (
            source_pattern.element,
            source_pattern.lanes,
            result_pattern.element,
            result_pattern.lanes,
        )
        assert key not in actual_cases
        actual_cases[key] = contract_case

    expected_cases = {
        (
            case.source.element_type,
            case.source.lane_count,
            case.result.element_type,
            case.result.lane_count,
        ): case
        for case in ORDINARY_VECTOR_BIT_LAYOUT_CASES
    }
    assert actual_cases.keys() == expected_cases.keys()

    for key, contract_case in actual_cases.items():
        layout_case = expected_cases[key]
        if layout_case in ORDINARY_VECTOR_BIT_LAYOUT_ALIAS_CASES:
            assert isinstance(contract_case, ValueAliasRule)
            assert contract_case.source == ValueRef.operand("input")
            assert contract_case.result == ValueRef.result("result")
        else:
            assert layout_case in ORDINARY_VECTOR_BIT_LAYOUT_DIRECT_CASES
            assert isinstance(contract_case, DescriptorRule)
            assert contract_case.descriptor is not None
            assert contract_case.descriptor.key == layout_case.key
            availability_guards = [
                guard
                for guard in contract_case.guards
                if guard.kind == GuardKind.DESCRIPTOR_AVAILABLE
            ]
            assert bool(availability_guards) == bool(
                contract_case.descriptor.feature_mask_words
            )


def test_each_native_type_drives_all_descriptor_backed_structural_rules() -> None:
    descriptor_rule_counts = Counter(
        (contract_case.source_op.name, contract_case.descriptor.key)
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
        if isinstance(contract_case, DescriptorRule)
    )
    for vector_type in ORDINARY_VECTOR_TYPES:
        construct_key = f"spirv.op_composite_construct.{vector_type.suffix}"
        extract_key = (
            f"spirv.op_composite_extract.{vector_type.suffix}."
            f"{vector_type.component_type.suffix}"
        )
        insert_key = (
            f"spirv.op_composite_insert.{vector_type.component_type.suffix}."
            f"{vector_type.suffix}"
        )
        dynamic_extract_key = (
            f"spirv.op_vector_extract_dynamic.{vector_type.suffix}."
            f"{vector_type.component_type.suffix}"
        )
        dynamic_insert_key = (
            f"spirv.op_vector_insert_dynamic.{vector_type.component_type.suffix}."
            f"{vector_type.suffix}"
        )
        select_key = f"spirv.op_select.{vector_type.suffix}"
        shuffle_key = (
            f"spirv.op_vector_shuffle.{vector_type.suffix}.{vector_type.suffix}"
        )
        constant_count = (
            2
            if vector_type.component_type.kind == OrdinaryVectorComponentKind.BOOLEAN
            else 1
        )
        assert descriptor_rule_counts[(vector.vector_splat.name, construct_key)] == 1
        assert (
            descriptor_rule_counts[(vector.vector_from_elements.name, construct_key)]
            == 1
        )
        assert (
            descriptor_rule_counts[(vector.vector_constant.name, construct_key)]
            == constant_count
        )
        assert descriptor_rule_counts[(vector.vector_extract.name, extract_key)] == 1
        assert descriptor_rule_counts[(vector.vector_insert.name, insert_key)] == 1
        assert (
            descriptor_rule_counts[(vector.vector_extract.name, dynamic_extract_key)]
            == 1
        )
        assert (
            descriptor_rule_counts[(vector.vector_insert.name, dynamic_insert_key)] == 1
        )
        assert descriptor_rule_counts[(vector.vector_shuffle.name, shuffle_key)] == 1
        assert descriptor_rule_counts[(scf.scf_select.name, select_key)] == 1


def test_dynamic_index_rules_bind_one_runtime_index_for_every_native_type() -> None:
    rules = tuple(
        contract_case
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
        if isinstance(contract_case, DescriptorRule)
        and contract_case.descriptor.key.startswith(
            ("spirv.op_vector_extract_dynamic.", "spirv.op_vector_insert_dynamic.")
        )
    )
    assert len(rules) == 2 * len(ORDINARY_VECTOR_TYPES)

    for rule in rules:
        index_type_guard = next(
            guard
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.field == "indices"
        )
        assert index_type_guard.type_pattern == TypePattern.scalar("index")
        assert any(
            guard.kind is GuardKind.OPERAND_SEGMENT_COUNT
            and guard.field == "indices"
            and guard.count == 1
            for guard in rule.guards
        )
        assert any(
            guard.kind is GuardKind.I64_ARRAY_COUNT
            and guard.field == "static_indices"
            and guard.count == 1
            for guard in rule.guards
        )
        assert any(
            guard.kind is GuardKind.I64_ARRAY_ELEMENT_RANGE
            and guard.field == "static_indices"
            and guard.element == 0
            and guard.minimum == -(2**63)
            and guard.maximum == -(2**63)
            for guard in rule.guards
        )

        emit = rule.emit[0]
        if rule.source_op is vector.vector_extract:
            assert emit.operands == {
                "vector": ValueRef.operand("source"),
                "index": ValueRef.operand("indices"),
            }
        else:
            assert rule.source_op is vector.vector_insert
            assert emit.operands == {
                "vector": ValueRef.operand("dest"),
                "component": ValueRef.operand("value"),
                "index": ValueRef.operand("indices"),
            }


def test_native_shuffle_rules_bind_each_selector_and_duplicate_the_source() -> None:
    rules = {
        contract_case.descriptor.key: contract_case
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
        if isinstance(contract_case, DescriptorRule)
        and contract_case.source_op is vector.vector_shuffle
    }
    assert rules.keys() == {
        instruction.key for instruction in ORDINARY_VECTOR_SOURCE_SHUFFLE_INSTRUCTIONS
    }

    for instruction in ORDINARY_VECTOR_SOURCE_SHUFFLE_INSTRUCTIONS:
        rule = rules[instruction.key]
        emit = rule.emit[0]
        assert emit.operands == {
            "source0": ValueRef.operand("source"),
            "source1": ValueRef.operand("source"),
        }
        assert tuple(emit.immediates) == instruction.selector_names
        assert tuple(
            projection.element for projection in emit.immediates.values()
        ) == tuple(range(instruction.result_type.lane_count))


def test_native_concat_rules_cover_every_component_and_binary_partition() -> None:
    rules = tuple(
        contract_case
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
        if isinstance(contract_case, DescriptorRule)
        and contract_case.source_op is vector.vector_concat
    )
    expected = {
        (
            source_type,
            shape.left_lane_count,
            shape.right_lane_count,
            shape.result_lane_count,
        )
        for component_type in ORDINARY_VECTOR_COMPONENT_TYPES
        for source_type in component_type.source_types
        for shape in ORDINARY_VECTOR_CONCAT_SHAPES
    }
    actual = {}
    for rule in rules:
        patterns = {
            (guard.field, guard.element): guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE
        }
        left = patterns[("inputs", 0)]
        right = patterns[("inputs", 1)]
        result = patterns[("result", 0)]
        key = (left.element, left.lanes, right.lanes, result.lanes)
        assert right.element == result.element == left.element
        assert key not in actual
        actual[key] = rule
        assert any(
            guard.kind is GuardKind.I64_RANGE
            and guard.field == "axis"
            and guard.minimum == 0
            and guard.maximum == 0
            for guard in rule.guards
        )
        assert any(
            guard.kind is GuardKind.OPERAND_SEGMENT_COUNT
            and guard.field == "inputs"
            and guard.count == 2
            for guard in rule.guards
        )

    assert actual.keys() == expected
    shapes = {
        (
            shape.left_lane_count,
            shape.right_lane_count,
            shape.result_lane_count,
        ): shape
        for shape in ORDINARY_VECTOR_CONCAT_SHAPES
    }
    for (_, left_lanes, right_lanes, result_lanes), rule in actual.items():
        shape = shapes[(left_lanes, right_lanes, result_lanes)]
        if result_lanes == 2:
            assert len(rule.emit) == 1
            assert rule.emit[0].descriptor.key.startswith(
                "spirv.op_composite_construct.v2"
            )
            assert rule.emit[0].operands == {
                "component0": ValueRef.operand("inputs", element=0),
                "component1": ValueRef.operand("inputs", element=1),
            }
            continue

        assert len(rule.emit) == 1 + (left_lanes == 1) + (right_lanes == 1)
        final_emit = rule.emit[-1]
        assert final_emit.descriptor.key.startswith("spirv.op_vector_shuffle.")
        assert tuple(final_emit.immediates.values()) == shape.selectors
        assert final_emit.operands == {
            "source0": (
                ValueRef.temporary("source0")
                if left_lanes == 1
                else ValueRef.operand("inputs", element=0)
            ),
            "source1": (
                ValueRef.temporary("source1")
                if right_lanes == 1
                else ValueRef.operand("inputs", element=1)
            ),
        }


def test_vector_contract_is_bound_into_the_shipping_fragment() -> None:
    fragment_case_ids = {
        id(contract_case)
        for contract_case in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
    }
    assert all(
        id(contract_case) in fragment_case_ids
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
    )


def test_vector_scale_uses_same_type_scalar_origins_on_either_side() -> None:
    rules = [
        case
        for case in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.descriptor.key.startswith("spirv.op_vector_times_scalar.")
    ]
    expected = {
        (f"spirv.op_vector_times_scalar.v{lanes}{element}", broadcast)
        for element in ("f16", "f32", "f64")
        for lanes in (2, 3, 4)
        for broadcast in ("lhs", "rhs")
    }
    actual = set()
    for rule in rules:
        assert rule.source_op is vector.vector_mulf
        origin_guard = next(
            guard
            for guard in rule.guards
            if guard.value_ref is not None
            and guard.value_ref.kind == SourceValueKind.UNIFORM_ELEMENT_ORIGIN_OPERAND
        )
        broadcast = origin_guard.field
        actual.add((rule.descriptor.key, broadcast))
        emit = rule.emit[0]
        assert emit.operands == {
            "vector": ValueRef.operand("rhs" if broadcast == "lhs" else "lhs"),
            "scalar": ValueRef.uniform_element_origin_operand(broadcast),
        }
        assert emit.immediates == {
            "allow_contraction": SourceOpProject.instance_flag("fastmath", "contract"),
        }
        vector_guard = next(
            guard
            for guard in rule.guards
            if guard.kind == GuardKind.VALUE_TYPE and guard.field == "result"
        )
        assert origin_guard.type_pattern == TypePattern.scalar(
            vector_guard.type_pattern.element
        )
        availability = [
            guard
            for guard in rule.guards
            if guard.kind == GuardKind.DESCRIPTOR_AVAILABLE
        ]
        assert bool(availability) == bool(rule.descriptor.feature_mask_words)
    assert actual == expected
    assert len(rules) == len(expected)


def test_synthesized_results_preserve_exact_source_types() -> None:
    actual_types: Counter[tuple[str, str, TypePattern]] = Counter()
    for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES:
        if not isinstance(contract_case, DescriptorRule):
            continue
        for emit in contract_case.emit:
            for descriptor_field, result in emit.results.items():
                if result.kind != SourceValueKind.TEMPORARY:
                    continue
                assert emit.result_types is not None
                result_type = emit.result_types[descriptor_field]
                assert isinstance(result_type, TypePattern)
                actual_types[(emit.descriptor.key, descriptor_field, result_type)] += 1

    expected_types: Counter[tuple[str, str, TypePattern]] = Counter()
    for vector_type in ORDINARY_VECTOR_TYPES:
        expected_types[
            (
                f"spirv.op_composite_construct.v{vector_type.lane_count}bool",
                "dst",
                TypePattern.vector("i1", lanes=vector_type.lane_count),
            )
        ] += 1
    for component_type in ORDINARY_VECTOR_COMPONENT_TYPES:
        if component_type.kind == OrdinaryVectorComponentKind.BOOLEAN:
            descriptor_keys = (
                "spirv.op_constant_false.bool",
                "spirv.op_constant_true.bool",
            )
        elif component_type.kind == OrdinaryVectorComponentKind.OFFSET:
            descriptor_keys = ("spirv.op_constant.offset64",)
        else:
            descriptor_keys = (f"spirv.op_constant.{component_type.suffix}",)
        for descriptor_key in descriptor_keys:
            expected_types[
                (
                    descriptor_key,
                    "dst",
                    TypePattern.scalar(component_type.source_types[0]),
                )
            ] += len(NATIVE_ORDINARY_VECTOR_LANE_COUNTS)

        concat_construct_key = f"spirv.op_composite_construct.v2{component_type.suffix}"
        for source_type in component_type.source_types:
            expected_types[
                (
                    concat_construct_key,
                    "dst",
                    TypePattern.vector(source_type, lanes=2),
                )
            ] += 4

    assert actual_types == expected_types


def test_vector_contract_contains_only_structural_and_bit_layout_rows() -> None:
    descriptor_keys = {
        contract_case.descriptor.key
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
        if isinstance(contract_case, DescriptorRule)
    }
    structural_stems = (
        "spirv.op_constant",
        "spirv.op_composite_construct",
        "spirv.op_composite_extract",
        "spirv.op_composite_insert",
        "spirv.op_vector_extract_dynamic",
        "spirv.op_vector_insert_dynamic",
        "spirv.op_vector_shuffle",
        "spirv.op_select",
        "spirv.op_bitcast",
    )
    assert all(key.startswith(structural_stems) for key in descriptor_keys)
    assert len(ORDINARY_VECTOR_COMPONENT_TYPES) == 10


def test_shipping_float_arithmetic_covers_widths_lanes_and_capabilities() -> None:
    operations = {
        "vector.addf",
        "vector.subf",
        "vector.mulf",
        "vector.divf",
        "vector.remf",
    }
    rules = [
        case
        for case in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
        and case.source_op.name in operations
        and not case.descriptor.key.startswith("spirv.op_vector_times_scalar.")
    ]
    actual = set()
    for rule in rules:
        patterns = {
            guard.field: guard.type_pattern
            for guard in rule.guards
            if guard.kind == GuardKind.VALUE_TYPE
        }
        assert patterns.keys() == {"lhs", "rhs", "result"}
        assert patterns["lhs"] == patterns["rhs"] == patterns["result"]
        pattern = patterns["result"]
        actual.add((rule.source_op.name, pattern.element, pattern.lanes))
        availability_guards = [
            guard
            for guard in rule.guards
            if guard.kind == GuardKind.DESCRIPTOR_AVAILABLE
        ]
        assert bool(availability_guards) == (pattern.element != "f32")
        assert rule.emit[0].immediates == {
            "allow_contraction": SourceOpProject.instance_flag("fastmath", "contract"),
        }
    assert len(rules) == 60
    assert actual == {
        (operation, element, lanes)
        for operation in operations
        for element in ("f16", "f32", "f64")
        for lanes in (1, 2, 3, 4)
    }


def test_scalar_float_arithmetic_projects_contraction_control() -> None:
    source_ops = {
        getattr(scalar_arithmetic, f"scalar_{operation.source_op_key}")
        for operation in FLOAT_BINARY_OPERATIONS
    }
    rules = tuple(
        case
        for case in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule) and case.source_op in source_ops
    )
    actual = set()
    for rule in rules:
        pattern = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind == GuardKind.VALUE_TYPE and guard.field == "result"
        )
        actual.add((rule.source_op.name, pattern.element))
        assert rule.emit[0].immediates == {
            "allow_contraction": SourceOpProject.instance_flag("fastmath", "contract"),
        }

    assert len(rules) == 15
    assert actual == {
        (f"scalar.{operation.source_op_key}", scalar.source_type)
        for operation in FLOAT_BINARY_OPERATIONS
        for scalar in FLOAT_SCALAR_ALU_TYPES
    }
