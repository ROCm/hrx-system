# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compact target-independent facts derived from native coordinate maps.

Exact maps remain generation-time proofs. These immutable summaries retain the
placement and source-owner relations that shipping compiler policy and compile
reports can consume directly without enumerating either coordinate domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from loom.target.native_contraction_layout import (
    ROLE_ACCUMULATOR,
    ROLE_LHS,
    ROLE_RESULT,
    ROLE_RHS,
    ContractionShape,
    CoordinateDimension,
    ExactContractionRoleLayout,
    ExactCoordinateMap,
    operation_local_coordinate_map,
    ownership_relation,
    unique_ownership_coordinate_map,
)
from loom.target.native_coordinate_projection import coordinate_projection_plan


class NativeLayoutEvidence(StrEnum):
    """Strength of the structural layout evidence retained in one fact."""

    EXACT = "exact"
    METADATA_DEPENDENT = "metadata-dependent"
    PARAMETRIC = "parametric"
    OPAQUE = "opaque"


@dataclass(frozen=True, slots=True)
class NativeContractionRoleFacts:
    """Compact physical placement facts for one semantic contraction role."""

    role: str
    evidence: NativeLayoutEvidence
    element_bit_count: int
    register_count: int
    payload_element_count: int
    physical_position_count: int
    logical_coordinate_count: int
    owner_multiplicity_minimum: int | None
    owner_multiplicity_maximum: int | None

    def __post_init__(self) -> None:
        if self.role not in (ROLE_LHS, ROLE_RHS, ROLE_ACCUMULATOR, ROLE_RESULT):
            raise ValueError(f"unknown native contraction role '{self.role}'")
        for name, value in (
            ("element bit count", self.element_bit_count),
            ("register count", self.register_count),
            ("payload element count", self.payload_element_count),
            ("physical position count", self.physical_position_count),
            ("logical coordinate count", self.logical_coordinate_count),
        ):
            if value <= 0:
                raise ValueError(f"native {self.role} {name} must be positive")
        multiplicities = (
            self.owner_multiplicity_minimum,
            self.owner_multiplicity_maximum,
        )
        if self.evidence is NativeLayoutEvidence.EXACT:
            if any(value is None or value <= 0 for value in multiplicities):
                raise ValueError(
                    f"exact native {self.role} ownership requires multiplicities"
                )
            if multiplicities[0] > multiplicities[1]:
                raise ValueError(
                    f"native {self.role} ownership multiplicities are reversed"
                )
        elif any(value is not None for value in multiplicities):
            raise ValueError(
                f"non-exact native {self.role} ownership cannot claim multiplicity"
            )


@dataclass(frozen=True, slots=True)
class NativeContractionFacts:
    """Compact placement facts for one target-native contraction primitive."""

    shape: ContractionShape
    participant_count: int
    lhs: NativeContractionRoleFacts
    rhs: NativeContractionRoleFacts
    accumulator: NativeContractionRoleFacts
    result: NativeContractionRoleFacts

    def __post_init__(self) -> None:
        if self.participant_count <= 0:
            raise ValueError("native contraction participant count must be positive")
        for expected_role, facts in zip(
            (ROLE_LHS, ROLE_RHS, ROLE_ACCUMULATOR, ROLE_RESULT),
            (self.lhs, self.rhs, self.accumulator, self.result),
            strict=True,
        ):
            if facts.role != expected_role:
                raise ValueError(
                    f"native {facts.role} facts occupy the {expected_role} slot"
                )
            if facts.physical_position_count % self.participant_count != 0:
                raise ValueError(
                    f"native {facts.role} positions do not divide across participants"
                )

    def role_facts(self, role: str) -> NativeContractionRoleFacts:
        try:
            return {
                ROLE_LHS: self.lhs,
                ROLE_RHS: self.rhs,
                ROLE_ACCUMULATOR: self.accumulator,
                ROLE_RESULT: self.result,
            }[role]
        except KeyError as error:
            raise ValueError(f"unknown native contraction role '{role}'") from error


@dataclass(frozen=True, slots=True)
class NativeTransitionOwnerFactor:
    """One destination-coordinate digit contributing to a source owner."""

    destination_dimension: str
    source_owner_dimension: str
    destination_divisor: int
    destination_modulus: int
    source_owner_multiplier: int


@dataclass(frozen=True, slots=True)
class NativeTransitionFacts:
    """Exact compact source-owner movement for one native layout transition."""

    contraction: NativeContractionFacts
    source_role: str
    destination_role: str
    destination_position_count: int
    participant_change_count: int
    local_position_change_count: int
    destination_positions_per_source_minimum: int
    destination_positions_per_source_maximum: int
    source_owner_factors: tuple[NativeTransitionOwnerFactor, ...]

    def __post_init__(self) -> None:
        source = self.contraction.role_facts(self.source_role)
        destination = self.contraction.role_facts(self.destination_role)
        if source.evidence is not NativeLayoutEvidence.EXACT:
            raise ValueError("native transition source placement must be exact")
        if destination.evidence is not NativeLayoutEvidence.EXACT:
            raise ValueError("native transition destination placement must be exact")
        if source.logical_coordinate_count != destination.logical_coordinate_count:
            raise ValueError("native transition roles have different logical domains")
        if self.destination_position_count != destination.physical_position_count:
            raise ValueError("native transition destination position count disagrees")
        for name, value in (
            ("participant change count", self.participant_change_count),
            ("local position change count", self.local_position_change_count),
        ):
            if value < 0 or value > self.destination_position_count:
                raise ValueError(f"native transition {name} is outside its domain")
        if (
            self.destination_positions_per_source_minimum <= 0
            or self.destination_positions_per_source_maximum
            < self.destination_positions_per_source_minimum
        ):
            raise ValueError("native transition source replication is invalid")
        if not self.source_owner_factors:
            raise ValueError("exact native transition requires source-owner factors")


def exact_native_contraction_role_facts(
    role: str,
    coordinate_map: ExactCoordinateMap,
    *,
    element_bit_count: int,
    register_count: int,
    payload_element_count: int,
) -> NativeContractionRoleFacts:
    """Summarizes one exact role map without retaining its finite domain."""

    multiplicities = tuple(map(len, coordinate_map.sources_by_destination))
    return NativeContractionRoleFacts(
        role=role,
        evidence=NativeLayoutEvidence.EXACT,
        element_bit_count=element_bit_count,
        register_count=register_count,
        payload_element_count=payload_element_count,
        physical_position_count=coordinate_map.source_point_count,
        logical_coordinate_count=coordinate_map.destination_point_count,
        owner_multiplicity_minimum=min(multiplicities),
        owner_multiplicity_maximum=max(multiplicities),
    )


def _coordinate_digit(
    dimensions: tuple[CoordinateDimension, ...], name: str
) -> tuple[int, int]:
    divisor = 1
    for dimension in dimensions:
        if dimension.name == name:
            return divisor, dimension.extent
        divisor *= dimension.extent
    # A missing physical axis has the sole coordinate zero.
    return 1, 1


def exact_native_transition_facts(
    contraction: NativeContractionFacts,
    source_layout: ExactContractionRoleLayout,
    destination_layout: ExactContractionRoleLayout,
) -> NativeTransitionFacts | None:
    """Compiles one exact role transition into bounded source-owner factors."""

    source = operation_local_coordinate_map(source_layout)
    destination = operation_local_coordinate_map(destination_layout)
    relation = ownership_relation(source, destination)
    owner_map = unique_ownership_coordinate_map(relation)
    if owner_map is None:
        return None
    projection = coordinate_projection_plan(owner_map)
    if projection is None:
        return None

    destination_participant_divisor, destination_participant_modulus = (
        _coordinate_digit(owner_map.source_dimensions, "participant")
    )
    destination_position_divisor, destination_position_modulus = _coordinate_digit(
        owner_map.source_dimensions, "value"
    )
    source_participant_divisor, source_participant_modulus = _coordinate_digit(
        owner_map.destination_dimensions, "participant"
    )
    source_position_divisor, source_position_modulus = _coordinate_digit(
        owner_map.destination_dimensions, "value"
    )
    participant_change_count = 0
    local_position_change_count = 0
    for destination_ordinal, source_ordinal in enumerate(
        owner_map.destination_by_source
    ):
        participant_change_count += (
            destination_ordinal // destination_participant_divisor
        ) % destination_participant_modulus != (
            source_ordinal // source_participant_divisor
        ) % source_participant_modulus
        local_position_change_count += (
            destination_ordinal // destination_position_divisor
        ) % destination_position_modulus != (
            source_ordinal // source_position_divisor
        ) % source_position_modulus

    destination_positions_per_source = tuple(map(len, owner_map.sources_by_destination))
    return NativeTransitionFacts(
        contraction=contraction,
        source_role=source_layout.role,
        destination_role=destination_layout.role,
        destination_position_count=owner_map.source_point_count,
        participant_change_count=participant_change_count,
        local_position_change_count=local_position_change_count,
        destination_positions_per_source_minimum=min(destination_positions_per_source),
        destination_positions_per_source_maximum=max(destination_positions_per_source),
        source_owner_factors=tuple(
            NativeTransitionOwnerFactor(
                destination_dimension=term.source_dimension,
                source_owner_dimension=term.destination_dimension,
                destination_divisor=term.source_divisor,
                destination_modulus=term.source_modulus,
                source_owner_multiplier=term.destination_multiplier,
            )
            for term in projection.forward_terms
        ),
    )
