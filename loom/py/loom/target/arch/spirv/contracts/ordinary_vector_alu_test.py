# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

from loom.dialect.vector import defs as vector
from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.contracts.ordinary_vector_alu import (
    SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES,
)
from loom.target.arch.spirv.ordinary_vector import (
    BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE,
    ORDINARY_VECTOR_COMPONENT_TYPES,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.ordinary_vector_float import (
    ORDINARY_VECTOR_FLOAT_COMPARE_INSTRUCTIONS,
    ORDINARY_VECTOR_FLOAT_PRESERVE_COMPARE_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer import (
    ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS,
)
from loom.target.arch.spirv.scalar_alu import (
    BOOLEAN_BINARY_OPERATIONS,
    FLOAT_COMPARE_PREDICATES,
    FLOAT_COMPARE_TYPES,
    INTEGER_BITWISE_BINARY_OPERATIONS,
    ORDERED_CLAMP_COMPARE_PREDICATES,
    SIGNED_INTEGER_BINARY_OPERATIONS,
    SIGNED_INTEGER_COMPARE_PREDICATES,
    UNSIGNED_INTEGER_BINARY_OPERATIONS,
    UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
    float_preservation_feature_atoms,
)
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    GuardKind,
    TypePattern,
    ValueRef,
)

_LANE_COUNTS = (1, 2, 3, 4)


def _suffix(component_suffix: str, lane_count: int) -> str:
    return component_suffix if lane_count == 1 else f"v{lane_count}{component_suffix}"


def _type_patterns(rule: DescriptorRule) -> dict[str, TypePattern]:
    return {
        guard.field: guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE
    }


def _predicate(rule: DescriptorRule) -> str:
    guards = tuple(
        guard for guard in rule.guards if guard.kind is GuardKind.ENUM_ATTR_EQUALS
    )
    assert len(guards) == 1
    return guards[0].enum_keyword


def _available_descriptor_keys(rule: DescriptorRule) -> tuple[str, ...]:
    return tuple(
        guard.descriptor.key
        for guard in rule.guards
        if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
    )


def _expected_available_descriptor_keys(*descriptors) -> tuple[str, ...]:
    keys: list[str] = []
    for descriptor in descriptors:
        if descriptor.feature_mask_words and descriptor.key not in keys:
            keys.append(descriptor.key)
    return tuple(keys)


def test_contract_has_a_closed_operation_matrix() -> None:
    assert all(
        isinstance(case, DescriptorRule)
        for case in SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES
    )
    assert Counter(
        case.source_op.name for case in SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES
    ) == {
        vector.vector_addi.name: 16,
        vector.vector_subi.name: 16,
        vector.vector_muli.name: 16,
        vector.vector_divsi.name: 16,
        vector.vector_remsi.name: 16,
        vector.vector_divui.name: 16,
        vector.vector_remui.name: 16,
        vector.vector_andi.name: 20,
        vector.vector_ori.name: 20,
        vector.vector_xori.name: 20,
        vector.vector_shli.name: 16,
        vector.vector_shrsi.name: 16,
        vector.vector_shrui.name: 16,
        vector.vector_minsi.name: 16,
        vector.vector_maxsi.name: 16,
        vector.vector_minui.name: 16,
        vector.vector_maxui.name: 16,
        vector.vector_cmpi.name: 160,
        vector.vector_cmpf.name: 168,
        vector.vector_select.name: 40,
    }
    assert len(SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES) == 652


def test_integer_binary_rules_cover_types_lanes_and_unsigned_views() -> None:
    source_ops = {
        getattr(vector, f"vector_{operation.source_op_key}")
        for operation in (
            *SIGNED_INTEGER_BINARY_OPERATIONS,
            *UNSIGNED_INTEGER_BINARY_OPERATIONS,
            *INTEGER_BITWISE_BINARY_OPERATIONS,
            *BOOLEAN_BINARY_OPERATIONS,
        )
    }
    actual = {}
    for case in SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES:
        if case.source_op not in source_ops:
            continue
        patterns = _type_patterns(case)
        assert patterns.keys() == {"lhs", "rhs", "result"}
        assert patterns["lhs"] == patterns["rhs"] == patterns["result"]
        pattern = patterns["result"]
        key = (case.source_op.name, pattern.element, pattern.lanes)
        assert key not in actual
        actual[key] = case

    expected = {}
    unsigned_keys = set()
    for component_pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS:
        for lane_count in _LANE_COUNTS:
            for operation in (
                *SIGNED_INTEGER_BINARY_OPERATIONS,
                *INTEGER_BITWISE_BINARY_OPERATIONS,
            ):
                key = (
                    f"vector.{operation.source_op_key}",
                    component_pair.source_type,
                    lane_count,
                )
                expected[key] = (
                    f"spirv.op_{operation.descriptor_suffix}."
                    f"{_suffix(component_pair.signed.suffix, lane_count)}"
                )
            for operation in UNSIGNED_INTEGER_BINARY_OPERATIONS:
                key = (
                    f"vector.{operation.source_op_key}",
                    component_pair.source_type,
                    lane_count,
                )
                expected[key] = (
                    f"spirv.op_{operation.descriptor_suffix}."
                    f"{_suffix(component_pair.unsigned.suffix, lane_count)}"
                )
                unsigned_keys.add(key)
    for lane_count in _LANE_COUNTS:
        for operation in BOOLEAN_BINARY_OPERATIONS:
            key = (f"vector.{operation.source_op_key}", "i1", lane_count)
            expected[key] = (
                f"spirv.op_{operation.descriptor_suffix}."
                f"{_suffix(BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE.suffix, lane_count)}"
            )

    assert actual.keys() == expected.keys()
    for key, descriptor_key in expected.items():
        rule = actual[key]
        assert rule.descriptor.key == descriptor_key
        if key not in unsigned_keys:
            assert len(rule.emit) == 1
            assert rule.emit[0].descriptor is rule.descriptor
            assert rule.emit[0].operands == {
                "lhs": ValueRef.operand("lhs"),
                "rhs": ValueRef.operand("rhs"),
            }
            assert rule.emit[0].results == {"dst": ValueRef.result("result")}
            assert _available_descriptor_keys(rule) == (
                _expected_available_descriptor_keys(rule.descriptor)
            )
            continue

        lhs_view, rhs_view, operation, result_view = rule.emit
        assert lhs_view.operands == {"input": ValueRef.operand("lhs")}
        assert lhs_view.results == {"dst": ValueRef.temporary("unsigned_lhs")}
        assert lhs_view.result_types == {"dst": DescriptorResultType()}
        assert rhs_view.operands == {"input": ValueRef.operand("rhs")}
        assert rhs_view.results == {"dst": ValueRef.temporary("unsigned_rhs")}
        assert rhs_view.result_types == {"dst": DescriptorResultType()}
        assert operation.descriptor is rule.descriptor
        assert operation.operands == {
            "lhs": ValueRef.temporary("unsigned_lhs"),
            "rhs": ValueRef.temporary("unsigned_rhs"),
        }
        assert operation.results == {"dst": ValueRef.temporary("unsigned_result")}
        assert operation.result_types == {"dst": DescriptorResultType()}
        assert result_view.operands == {"input": ValueRef.temporary("unsigned_result")}
        assert result_view.results == {"dst": ValueRef.result("result")}
        assert _available_descriptor_keys(rule) == (
            _expected_available_descriptor_keys(
                lhs_view.descriptor, operation.descriptor, result_view.descriptor
            )
        )


def test_integer_extrema_rules_cover_types_lanes_ordering_and_selection() -> None:
    operation_properties = {
        vector.vector_minsi: ("s_less_than", False),
        vector.vector_maxsi: ("s_greater_than", False),
        vector.vector_minui: ("u_less_than", True),
        vector.vector_maxui: ("u_greater_than", True),
    }
    actual = {}
    for case in SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES:
        if case.source_op not in operation_properties:
            continue
        patterns = _type_patterns(case)
        assert patterns.keys() == {"lhs", "rhs", "result"}
        assert patterns["lhs"] == patterns["rhs"] == patterns["result"]
        pattern = patterns["result"]
        key = (case.source_op.name, pattern.element, pattern.lanes)
        assert key not in actual
        actual[key] = case

    expected = {
        (source_op.name, component_pair.source_type, lane_count): (
            component_pair,
            compare_suffix,
            uses_unsigned_ordering,
        )
        for component_pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS
        for lane_count in _LANE_COUNTS
        for source_op, (compare_suffix, uses_unsigned_ordering) in (
            operation_properties.items()
        )
    }
    assert len(actual) == len(expected) == 64
    assert actual.keys() == expected.keys()

    for key, (
        component_pair,
        compare_suffix,
        uses_unsigned_ordering,
    ) in expected.items():
        rule = actual[key]
        lane_count = key[2]
        signed_suffix = _suffix(component_pair.signed.suffix, lane_count)
        compare_component = (
            component_pair.unsigned if uses_unsigned_ordering else component_pair.signed
        )
        compare_key = (
            f"spirv.op_{compare_suffix}.{_suffix(compare_component.suffix, lane_count)}"
        )
        assert rule.descriptor.key == f"spirv.op_select.{signed_suffix}"

        if uses_unsigned_ordering:
            lhs_view, rhs_view, comparison, selection = rule.emit
            unsigned_suffix = _suffix(component_pair.unsigned.suffix, lane_count)
            view_key = f"spirv.op_bitcast.{signed_suffix}.{unsigned_suffix}"
            assert lhs_view.descriptor.key == view_key
            assert lhs_view.operands == {"input": ValueRef.operand("lhs")}
            assert lhs_view.results == {"dst": ValueRef.temporary("unsigned_lhs")}
            assert lhs_view.result_types == {"dst": DescriptorResultType()}
            assert rhs_view.descriptor.key == view_key
            assert rhs_view.operands == {"input": ValueRef.operand("rhs")}
            assert rhs_view.results == {"dst": ValueRef.temporary("unsigned_rhs")}
            assert rhs_view.result_types == {"dst": DescriptorResultType()}
            assert comparison.operands == {
                "lhs": ValueRef.temporary("unsigned_lhs"),
                "rhs": ValueRef.temporary("unsigned_rhs"),
            }
        else:
            comparison, selection = rule.emit
            assert comparison.operands == {
                "lhs": ValueRef.operand("lhs"),
                "rhs": ValueRef.operand("rhs"),
            }

        assert comparison.descriptor.key == compare_key
        assert comparison.results == {"dst": ValueRef.temporary("condition")}
        assert comparison.result_types == {
            "dst": TypePattern.vector("i1", lanes=lane_count)
        }
        assert selection.descriptor is rule.descriptor
        assert selection.operands == {
            "condition": ValueRef.temporary("condition"),
            "true_value": ValueRef.operand("lhs"),
            "false_value": ValueRef.operand("rhs"),
        }
        assert selection.results == {"dst": ValueRef.result("result")}
        assert _available_descriptor_keys(rule) == (
            _expected_available_descriptor_keys(
                *(emit.descriptor for emit in rule.emit)
            )
        )


def test_comparison_rules_cover_types_lanes_predicates_and_views() -> None:
    rules = tuple(
        case
        for case in SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES
        if case.source_op in (vector.vector_cmpi, vector.vector_cmpf)
    )
    actual = {}
    for rule in rules:
        patterns = _type_patterns(rule)
        assert patterns.keys() == {"lhs", "rhs", "result"}
        assert patterns["lhs"] == patterns["rhs"]
        assert patterns["result"] == TypePattern.vector(
            "i1", lanes=patterns["lhs"].lanes
        )
        key = (
            rule.source_op.name,
            patterns["lhs"].element,
            patterns["lhs"].lanes,
            _predicate(rule),
        )
        assert key not in actual
        actual[key] = rule

    expected = {}
    unsigned_keys = set()
    for component_pair in ORDINARY_VECTOR_INTEGER_COMPONENT_TYPE_PAIRS:
        for lane_count in _LANE_COUNTS:
            for predicate in SIGNED_INTEGER_COMPARE_PREDICATES:
                key = (
                    vector.vector_cmpi.name,
                    component_pair.source_type,
                    lane_count,
                    predicate.source_predicate,
                )
                expected[key] = (
                    f"spirv.op_{predicate.descriptor_suffix}."
                    f"{_suffix(component_pair.signed.suffix, lane_count)}"
                )
            for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES:
                key = (
                    vector.vector_cmpi.name,
                    component_pair.source_type,
                    lane_count,
                    predicate.source_predicate,
                )
                expected[key] = (
                    f"spirv.op_{predicate.descriptor_suffix}."
                    f"{_suffix(component_pair.unsigned.suffix, lane_count)}"
                )
                unsigned_keys.add(key)
    for scalar in FLOAT_COMPARE_TYPES:
        for lane_count in _LANE_COUNTS:
            for predicate in FLOAT_COMPARE_PREDICATES:
                key = (
                    vector.vector_cmpf.name,
                    scalar.source_type,
                    lane_count,
                    predicate.source_predicate,
                )
                expected[key] = (
                    f"spirv.op_{predicate.descriptor_suffix}."
                    f"{_suffix(scalar.suffix, lane_count)}"
                )

    assert actual.keys() == expected.keys()
    for key, descriptor_key in expected.items():
        rule = actual[key]
        assert rule.descriptor.key == descriptor_key
        if key not in unsigned_keys:
            assert len(rule.emit) == 1
            assert rule.emit[0].descriptor is rule.descriptor
            assert rule.emit[0].operands == {
                "lhs": ValueRef.operand("lhs"),
                "rhs": ValueRef.operand("rhs"),
            }
            assert rule.emit[0].results == {"dst": ValueRef.result("result")}
            assert _available_descriptor_keys(rule) == (
                _expected_available_descriptor_keys(rule.descriptor)
            )
            continue

        lhs_view, rhs_view, comparison = rule.emit
        assert lhs_view.results == {"dst": ValueRef.temporary("unsigned_lhs")}
        assert rhs_view.results == {"dst": ValueRef.temporary("unsigned_rhs")}
        assert comparison.descriptor is rule.descriptor
        assert comparison.operands == {
            "lhs": ValueRef.temporary("unsigned_lhs"),
            "rhs": ValueRef.temporary("unsigned_rhs"),
        }
        assert comparison.results == {"dst": ValueRef.result("result")}
        assert _available_descriptor_keys(rule) == (
            _expected_available_descriptor_keys(
                lhs_view.descriptor, comparison.descriptor
            )
        )


def test_float_comparison_descriptors_cover_native_shapes_and_type_features() -> None:
    actual = {
        instruction.key: instruction
        for instruction in ORDINARY_VECTOR_FLOAT_COMPARE_INSTRUCTIONS
    }
    expected = {
        (f"spirv.op_{predicate.descriptor_suffix}.v{lane_count}{scalar.suffix}"): (
            scalar,
            predicate,
            lane_count,
        )
        for scalar in FLOAT_COMPARE_TYPES
        for lane_count in (2, 3, 4)
        for predicate in FLOAT_COMPARE_PREDICATES
    }
    assert len(actual) == len(expected) == 126
    assert actual.keys() == expected.keys()
    for key, (scalar, predicate, lane_count) in expected.items():
        instruction = actual[key]
        operand_type = OrdinaryVectorType(
            next(
                component
                for component in ORDINARY_VECTOR_COMPONENT_TYPES
                if component.suffix == scalar.suffix
            ),
            lane_count,
        )
        assert instruction.mnemonic == f"{predicate.mnemonic}.{operand_type.suffix}"
        assert instruction.opcode == predicate.opcode
        assert instruction.packet_form == "LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE"
        assert instruction.operand_names == ("lhs", "rhs")
        assert instruction.operand_types == (operand_type, operand_type)
        assert instruction.result_type == OrdinaryVectorType(
            BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE, lane_count
        )
        assert instruction.feature_atoms == ()
        assert instruction.feature_bits == scalar.feature_bits


def test_ordered_clamp_compare_descriptors_require_preservation_controls() -> None:
    actual = {
        instruction.key: instruction
        for instruction in ORDINARY_VECTOR_FLOAT_PRESERVE_COMPARE_INSTRUCTIONS
    }
    expected = {
        (
            f"spirv.op_{predicate.descriptor_suffix}.preserve."
            f"v{lane_count}{scalar.suffix}"
        ): (scalar, predicate, lane_count)
        for scalar in FLOAT_COMPARE_TYPES
        for lane_count in (2, 3, 4)
        for predicate in ORDERED_CLAMP_COMPARE_PREDICATES
    }
    assert len(actual) == len(expected) == 18
    assert actual.keys() == expected.keys()
    for key, (scalar, predicate, lane_count) in expected.items():
        instruction = actual[key]
        operand_type = OrdinaryVectorType(
            next(
                component
                for component in ORDINARY_VECTOR_COMPONENT_TYPES
                if component.suffix == scalar.suffix
            ),
            lane_count,
        )
        assert instruction.mnemonic == (
            f"{predicate.mnemonic}.preserve.{operand_type.suffix}"
        )
        assert instruction.opcode == predicate.opcode
        assert instruction.packet_form == "LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE"
        assert instruction.operand_names == ("lhs", "rhs")
        assert instruction.operand_types == (operand_type, operand_type)
        assert instruction.result_type == OrdinaryVectorType(
            BOOLEAN_ORDINARY_VECTOR_COMPONENT_TYPE, lane_count
        )
        assert instruction.feature_atoms == float_preservation_feature_atoms(scalar)


def test_select_rules_cover_every_public_component_and_shape() -> None:
    rules = tuple(
        case
        for case in SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES
        if case.source_op is vector.vector_select
    )
    actual = {}
    for rule in rules:
        patterns = _type_patterns(rule)
        assert patterns.keys() == {
            "condition",
            "true_value",
            "false_value",
            "result",
        }
        assert patterns["condition"] == TypePattern.vector(
            "i1", lanes=patterns["result"].lanes
        )
        assert patterns["true_value"] == patterns["false_value"] == patterns["result"]
        key = (patterns["result"].elements, patterns["result"].lanes)
        assert key not in actual
        actual[key] = rule

    expected = {
        (component.source_types, lane_count): (
            f"spirv.op_select.{_suffix(component.suffix, lane_count)}"
        )
        for component in ORDINARY_VECTOR_COMPONENT_TYPES
        for lane_count in _LANE_COUNTS
    }
    assert actual.keys() == expected.keys()
    for key, descriptor_key in expected.items():
        rule = actual[key]
        assert rule.descriptor.key == descriptor_key
        assert len(rule.emit) == 1
        assert rule.emit[0].operands == {
            "condition": ValueRef.operand("condition"),
            "true_value": ValueRef.operand("true_value"),
            "false_value": ValueRef.operand("false_value"),
        }
        assert rule.emit[0].results == {"dst": ValueRef.result("result")}
        assert _available_descriptor_keys(rule) == (
            _expected_available_descriptor_keys(rule.descriptor)
        )


def test_alu_contract_is_bound_into_the_shipping_fragment() -> None:
    fragment_case_ids = {
        id(contract_case)
        for contract_case in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
    }
    assert all(
        id(contract_case) in fragment_case_ids
        for contract_case in SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES
    )
