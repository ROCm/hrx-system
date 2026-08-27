# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

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
    ORDINARY_VECTOR_TYPES,
    OrdinaryVectorComponentKind,
)
from loom.target.contracts import (
    DescriptorRule,
    SourceValueKind,
    TypePattern,
    ValueAliasRule,
)


def test_every_owned_source_operation_has_a_closed_shape_matrix() -> None:
    case_counts = Counter(
        contract_case.source_op.name
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
    )
    assert case_counts == {
        vector.vector_constant.name: 44,
        vector.vector_splat.name: 31,
        vector.vector_from_elements.name: 31,
        vector.vector_extract.name: 31,
        vector.vector_insert.name: 31,
        scf.scf_select.name: 40,
    }
    assert len(SPIRV_ORDINARY_VECTOR_CONTRACT_CASES) == 208


def test_singleton_representation_is_an_alias_for_each_structural_family() -> None:
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
    }
    assert len(alias_cases) == 4


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
        select_key = f"spirv.op_select.{vector_type.suffix}"
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
        assert descriptor_rule_counts[(scf.scf_select.name, select_key)] == 1


def test_vector_contract_is_bound_into_the_shipping_fragment() -> None:
    fragment_case_ids = {
        id(contract_case)
        for contract_case in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases
    }
    assert all(
        id(contract_case) in fragment_case_ids
        for contract_case in SPIRV_ORDINARY_VECTOR_CONTRACT_CASES
    )


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

    assert actual_types == expected_types


def test_vector_contract_contains_no_arithmetic_or_conversion_rows() -> None:
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
        "spirv.op_select",
    )
    assert all(key.startswith(structural_stems) for key in descriptor_keys)
    assert len(ORDINARY_VECTOR_COMPONENT_TYPES) == 10
