# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.native_contraction_layout import (
    ROLE_ACCUMULATOR,
    ROLE_LHS,
    ROLE_RESULT,
    ROLE_RHS,
    ExactContractionLayout,
    contiguous_element_layout,
    grouped_dot_contraction_layout,
)
from loom.target.native_layout_facts import (
    NativeContractionFacts,
    NativeTransitionOwnerFactor,
    exact_native_contraction_role_facts,
    exact_native_transition_facts,
)


def _grouped_dot_facts() -> tuple[ExactContractionLayout, NativeContractionFacts]:
    layouts = {
        role: contiguous_element_layout(
            key=role,
            element_count=element_count,
            atom_bit_width=element_bit_count,
            physical_dimension_name="value",
        )
        for role, element_count, element_bit_count in (
            (ROLE_LHS, 8, 8),
            (ROLE_RHS, 8, 8),
            (ROLE_ACCUMULATOR, 2, 32),
            (ROLE_RESULT, 2, 32),
        )
    }
    layout = grouped_dot_contraction_layout(
        group_size=4,
        lhs=layouts[ROLE_LHS],
        rhs=layouts[ROLE_RHS],
        accumulator=layouts[ROLE_ACCUMULATOR],
        result=layouts[ROLE_RESULT],
    )
    facts = NativeContractionFacts(
        shape=layout.shape,
        participant_count=1,
        lhs=exact_native_contraction_role_facts(
            layout.lhs.role,
            layout.lhs.coordinate_map,
            element_bit_count=8,
            register_count=1,
            payload_element_count=8,
        ),
        rhs=exact_native_contraction_role_facts(
            layout.rhs.role,
            layout.rhs.coordinate_map,
            element_bit_count=8,
            register_count=1,
            payload_element_count=8,
        ),
        accumulator=exact_native_contraction_role_facts(
            layout.accumulator.role,
            layout.accumulator.coordinate_map,
            element_bit_count=32,
            register_count=1,
            payload_element_count=2,
        ),
        result=exact_native_contraction_role_facts(
            layout.result.role,
            layout.result.coordinate_map,
            element_bit_count=32,
            register_count=1,
            payload_element_count=2,
        ),
    )
    return layout, facts


def test_exact_role_facts_retain_placement_without_the_finite_map() -> None:
    _, facts = _grouped_dot_facts()

    assert facts.shape.block_count == 2
    assert facts.shape.k == 4
    assert facts.lhs.physical_position_count == 8
    assert facts.lhs.logical_coordinate_count == 8
    assert facts.lhs.owner_multiplicity_minimum == 1
    assert facts.lhs.owner_multiplicity_maximum == 1
    assert facts.result.physical_position_count == 2


def test_exact_transition_compiles_destination_to_source_owner_factors() -> None:
    layout, facts = _grouped_dot_facts()

    transition = exact_native_transition_facts(facts, layout.accumulator, layout.result)

    assert transition is not None
    assert transition.destination_position_count == 2
    assert transition.participant_change_count == 0
    assert transition.local_position_change_count == 0
    assert transition.destination_positions_per_source_minimum == 1
    assert transition.destination_positions_per_source_maximum == 1
    assert transition.source_owner_factors == (
        NativeTransitionOwnerFactor(
            destination_dimension="value",
            source_owner_dimension="value",
            destination_divisor=1,
            destination_modulus=0,
            source_owner_multiplier=1,
        ),
    )
