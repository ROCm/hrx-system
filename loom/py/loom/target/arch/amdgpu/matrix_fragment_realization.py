# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact AMDGPU matrix contract realization catalog.

The catalog binds semantic contraction contracts to physical result
representations. Canonical rows use the contract's declared fragment layout.
Dense exact rows additionally admit the operand-exchanged, M/N-transposed
instruction realization when exhaustive coordinate maps prove it equivalent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from loom.target.arch.amdgpu.matrix_contracts import (
    AMDGPU_MATRIX_CONTRACTS,
    AmdgpuMatrixContract,
    AmdgpuMatrixPayload,
)
from loom.target.arch.amdgpu.matrix_fragment_layout import (
    AmdgpuMatrixFragmentLayout,
    layout_roles,
)
from loom.target.arch.amdgpu.matrix_fragment_layout_adaptation import (
    matrix_fragment_native_role_layout,
)
from loom.target.arch.amdgpu.matrix_fragment_layouts import (
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY,
)
from loom.target.native_contraction_layout import (
    ContractionShape,
    ExactContractionLayout,
    ExactContractionRoleLayout,
    ExactCoordinateMap,
    transpose_contraction_layout,
)

MATRIX_RESULT_REPRESENTATION_NONE = 0
MATRIX_CONTRACT_ORDINAL_NONE = 0xFFFF


@dataclass(frozen=True, slots=True)
class MatrixResultRepresentation:
    """One exact result placement and its compact runtime realization."""

    fragment_layout: AmdgpuMatrixFragmentLayout
    transposed: bool
    payload: AmdgpuMatrixPayload
    coordinate_map: ExactCoordinateMap


@dataclass(frozen=True, slots=True)
class MatrixContractRealizationChoices:
    """Compact canonical and operand-symmetric choices for one contract."""

    canonical_result_representation_id: int
    operand_exchanged_contract_ordinal: int
    operand_exchanged_result_representation_id: int


@dataclass(frozen=True, slots=True)
class MatrixFragmentRealizationCatalog:
    """Generated realization choices and their deduplicated result placements."""

    contract_choices: tuple[MatrixContractRealizationChoices, ...]
    result_representations: tuple[MatrixResultRepresentation, ...]

    def result_representation(
        self, representation_id: int
    ) -> MatrixResultRepresentation | None:
        """Returns one 1-based representation row, or None for the zero ID."""

        if representation_id == MATRIX_RESULT_REPRESENTATION_NONE:
            return None
        if representation_id < 0 or representation_id > len(
            self.result_representations
        ):
            raise ValueError(
                f"matrix result representation ID {representation_id} is out of range"
            )
        return self.result_representations[representation_id - 1]


@dataclass(frozen=True, slots=True)
class _ContractSignature:
    family: str
    features: frozenset[str]
    wave_size: str
    tile_shape: tuple[int, int, int]
    lhs: AmdgpuMatrixPayload
    rhs: AmdgpuMatrixPayload
    accumulator: AmdgpuMatrixPayload
    result: AmdgpuMatrixPayload
    scale_kind: str
    block_count: int
    flags: frozenset[str]
    implicit_scale_formats: frozenset[str]
    source_requirements: frozenset[str]


@dataclass(frozen=True, slots=True)
class _PendingChoices:
    canonical_result_representation: MatrixResultRepresentation | None
    operand_exchanged_contract_ordinal: int
    operand_exchanged_result_representation: MatrixResultRepresentation | None


def _contract_signature(contract: AmdgpuMatrixContract) -> _ContractSignature:
    return _ContractSignature(
        family=contract.family,
        features=frozenset(contract.features),
        wave_size=contract.wave_size,
        tile_shape=contract.tile_shape,
        lhs=contract.lhs,
        rhs=contract.rhs,
        accumulator=contract.accumulator,
        result=contract.result,
        scale_kind=contract.scale_kind,
        block_count=contract.block_count,
        flags=frozenset(contract.flags),
        implicit_scale_formats=frozenset(contract.implicit_scale_formats),
        source_requirements=frozenset(contract.source_requirements),
    )


def _transposed_contract_signature(
    contract: AmdgpuMatrixContract,
) -> _ContractSignature:
    row_count, column_count, reduction_count = contract.tile_shape
    return _ContractSignature(
        family=contract.family,
        features=frozenset(contract.features),
        wave_size=contract.wave_size,
        tile_shape=(column_count, row_count, reduction_count),
        lhs=contract.rhs,
        rhs=contract.lhs,
        accumulator=contract.accumulator,
        result=contract.result,
        scale_kind=contract.scale_kind,
        block_count=contract.block_count,
        flags=frozenset(contract.flags),
        implicit_scale_formats=frozenset(contract.implicit_scale_formats),
        source_requirements=frozenset(contract.source_requirements),
    )


def _fragment_layout(
    contract: AmdgpuMatrixContract,
    layouts_by_key: Mapping[str, AmdgpuMatrixFragmentLayout],
) -> AmdgpuMatrixFragmentLayout | None:
    if contract.fragment_layout is None:
        return None
    try:
        return layouts_by_key[contract.fragment_layout]
    except KeyError as error:
        raise ValueError(
            f"AMDGPU matrix contract '{contract.name}' names unknown fragment "
            f"layout '{contract.fragment_layout}'"
        ) from error


def _native_layout(
    contract: AmdgpuMatrixContract,
    layouts_by_key: Mapping[str, AmdgpuMatrixFragmentLayout],
) -> ExactContractionLayout | None:
    fragment_layout = _fragment_layout(contract, layouts_by_key)
    if fragment_layout is None:
        return None
    role_layouts = tuple(
        matrix_fragment_native_role_layout(fragment_layout, role)
        for role in layout_roles(fragment_layout)
    )
    if any(role_layout is None for role_layout in role_layouts):
        return None
    lhs, rhs, accumulator, result = tuple(
        cast(ExactContractionRoleLayout, role_layout) for role_layout in role_layouts
    )
    block_count, row_count, column_count, reduction_count = fragment_layout.tile_shape
    return ExactContractionLayout(
        shape=ContractionShape(
            block_count=block_count,
            m=row_count,
            n=column_count,
            k=reduction_count,
        ),
        lhs=lhs,
        rhs=rhs,
        accumulator=accumulator,
        result=result,
    )


def _result_representation(
    contract: AmdgpuMatrixContract,
    fragment_layout: AmdgpuMatrixFragmentLayout,
    native_layout: ExactContractionLayout,
    *,
    transposed: bool,
) -> MatrixResultRepresentation:
    return MatrixResultRepresentation(
        fragment_layout=fragment_layout,
        transposed=transposed,
        payload=contract.result,
        coordinate_map=native_layout.result.coordinate_map,
    )


def matrix_fragment_realization_catalog(
    contracts: Sequence[AmdgpuMatrixContract] = AMDGPU_MATRIX_CONTRACTS,
    layouts_by_key: Mapping[
        str, AmdgpuMatrixFragmentLayout
    ] = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY,
) -> MatrixFragmentRealizationCatalog:
    """Builds and exhaustively validates all exact contract realizations."""

    if len(contracts) > MATRIX_CONTRACT_ORDINAL_NONE:
        raise ValueError("AMDGPU matrix contract catalog exceeds uint16 ordinals")

    ordinals_by_signature: dict[_ContractSignature, list[int]] = defaultdict(list)
    for contract_ordinal, contract in enumerate(contracts):
        ordinals_by_signature[_contract_signature(contract)].append(contract_ordinal)

    pending_choices: list[_PendingChoices] = []
    for source_contract in contracts:
        source_fragment_layout = _fragment_layout(source_contract, layouts_by_key)
        if source_fragment_layout is None:
            pending_choices.append(
                _PendingChoices(
                    canonical_result_representation=None,
                    operand_exchanged_contract_ordinal=(MATRIX_CONTRACT_ORDINAL_NONE),
                    operand_exchanged_result_representation=None,
                )
            )
            continue

        source_accumulator = matrix_fragment_native_role_layout(
            source_fragment_layout, source_fragment_layout.accumulator
        )
        source_result = matrix_fragment_native_role_layout(
            source_fragment_layout, source_fragment_layout.result
        )
        if source_accumulator is None or source_result is None:
            raise ValueError(
                f"AMDGPU matrix contract '{source_contract.name}' does not have "
                "exact accumulator and result layouts"
            )
        if source_accumulator.coordinate_map != source_result.coordinate_map:
            raise ValueError(
                f"AMDGPU matrix contract '{source_contract.name}' changes "
                "physical representation between accumulator and result"
            )
        canonical_representation = MatrixResultRepresentation(
            fragment_layout=source_fragment_layout,
            transposed=False,
            payload=source_contract.result,
            coordinate_map=source_result.coordinate_map,
        )
        if any(
            role_layout.reduction_group is not None
            for role_layout in layout_roles(source_fragment_layout)
        ):
            pending_choices.append(
                _PendingChoices(
                    canonical_result_representation=canonical_representation,
                    operand_exchanged_contract_ordinal=(MATRIX_CONTRACT_ORDINAL_NONE),
                    operand_exchanged_result_representation=None,
                )
            )
            continue

        source_native_layout = _native_layout(source_contract, layouts_by_key)
        if source_native_layout is None:
            raise ValueError(
                f"dense AMDGPU matrix contract '{source_contract.name}' does not "
                "have an exact fragment layout"
            )

        alternatives: list[tuple[int, MatrixResultRepresentation]] = []
        for native_ordinal in ordinals_by_signature[
            _transposed_contract_signature(source_contract)
        ]:
            native_contract = contracts[native_ordinal]
            native_fragment_layout = _fragment_layout(native_contract, layouts_by_key)
            native_layout = _native_layout(native_contract, layouts_by_key)
            if native_fragment_layout is None or native_layout is None:
                continue
            transposed_layout = transpose_contraction_layout(native_layout)
            if (
                source_native_layout.lhs.coordinate_map
                != transposed_layout.lhs.coordinate_map
            ):
                continue
            if (
                source_native_layout.rhs.coordinate_map
                != transposed_layout.rhs.coordinate_map
            ):
                continue
            if (
                transposed_layout.accumulator.coordinate_map
                != transposed_layout.result.coordinate_map
            ):
                continue
            alternatives.append(
                (
                    native_ordinal,
                    _result_representation(
                        source_contract,
                        native_fragment_layout,
                        transposed_layout,
                        transposed=True,
                    ),
                )
            )

        if len(alternatives) != 1:
            candidate_names = ", ".join(
                contracts[native_ordinal].name for native_ordinal, _ in alternatives
            )
            raise ValueError(
                f"dense AMDGPU matrix contract '{source_contract.name}' has "
                f"{len(alternatives)} exact operand-symmetric realizations: "
                f"{candidate_names or 'none'}"
            )
        alternative_contract_ordinal, alternative_representation = alternatives[0]
        pending_choices.append(
            _PendingChoices(
                canonical_result_representation=canonical_representation,
                operand_exchanged_contract_ordinal=alternative_contract_ordinal,
                operand_exchanged_result_representation=alternative_representation,
            )
        )

    result_representations: list[MatrixResultRepresentation] = []
    representation_ids: dict[tuple[AmdgpuMatrixPayload, ExactCoordinateMap], int] = {}

    def intern_representation(representation: MatrixResultRepresentation) -> int:
        signature = (representation.payload, representation.coordinate_map)
        representation_id = representation_ids.get(signature)
        if representation_id is not None:
            return representation_id
        result_representations.append(representation)
        representation_id = len(result_representations)
        representation_ids[signature] = representation_id
        return representation_id

    canonical_representation_ids = tuple(
        MATRIX_RESULT_REPRESENTATION_NONE
        if choices.canonical_result_representation is None
        else intern_representation(choices.canonical_result_representation)
        for choices in pending_choices
    )
    alternative_representation_ids = tuple(
        MATRIX_RESULT_REPRESENTATION_NONE
        if choices.operand_exchanged_result_representation is None
        else intern_representation(choices.operand_exchanged_result_representation)
        for choices in pending_choices
    )
    if len(result_representations) > 0xFF:
        raise ValueError(
            "AMDGPU matrix result representation catalog exceeds uint8 IDs"
        )

    contract_choices = tuple(
        MatrixContractRealizationChoices(
            canonical_result_representation_id=(canonical_representation_ids[index]),
            operand_exchanged_contract_ordinal=(
                choices.operand_exchanged_contract_ordinal
            ),
            operand_exchanged_result_representation_id=(
                alternative_representation_ids[index]
            ),
        )
        for index, choices in enumerate(pending_choices)
    )
    return MatrixFragmentRealizationCatalog(
        contract_choices=contract_choices,
        result_representations=tuple(result_representations),
    )


AMDGPU_MATRIX_FRAGMENT_REALIZATION_CATALOG = matrix_fragment_realization_catalog()
