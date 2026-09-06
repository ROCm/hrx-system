# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections import Counter

from loom.target.arch.amdgpu.matrix_contracts import AMDGPU_MATRIX_CONTRACTS
from loom.target.arch.amdgpu.matrix_fragment_layout import layout_roles
from loom.target.arch.amdgpu.matrix_fragment_layout_adaptation import (
    matrix_fragment_native_role_layout,
)
from loom.target.arch.amdgpu.matrix_fragment_layouts import (
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY,
)
from loom.target.arch.amdgpu.matrix_fragment_realization import (
    AMDGPU_MATRIX_FRAGMENT_REALIZATION_CATALOG,
    MATRIX_CONTRACT_ORDINAL_NONE,
    MATRIX_RESULT_REPRESENTATION_NONE,
)
from loom.target.native_contraction_layout import (
    ContractionShape,
    ExactContractionLayout,
    transpose_contraction_layout,
)


def _contract_ordinal(name: str) -> int:
    for ordinal, contract in enumerate(AMDGPU_MATRIX_CONTRACTS):
        if contract.name == name:
            return ordinal
    raise ValueError(f"unknown AMDGPU matrix contract '{name}'")


def _exact_layout(contract_ordinal: int) -> ExactContractionLayout:
    contract = AMDGPU_MATRIX_CONTRACTS[contract_ordinal]
    assert contract.fragment_layout is not None
    fragment_layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY[contract.fragment_layout]
    role_layouts = tuple(
        matrix_fragment_native_role_layout(fragment_layout, role)
        for role in layout_roles(fragment_layout)
    )
    assert all(role_layout is not None for role_layout in role_layouts)
    lhs, rhs, accumulator, result = role_layouts
    assert lhs is not None
    assert rhs is not None
    assert accumulator is not None
    assert result is not None
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


def test_catalog_covers_every_contract_family_and_layout_class() -> None:
    catalog = AMDGPU_MATRIX_FRAGMENT_REALIZATION_CATALOG

    assert len(catalog.contract_choices) == 235
    assert len(catalog.result_representations) == 51
    assert Counter(
        AMDGPU_MATRIX_CONTRACTS[choices.operand_exchanged_contract_ordinal].family
        for choices in catalog.contract_choices
        if choices.operand_exchanged_contract_ordinal != MATRIX_CONTRACT_ORDINAL_NONE
    ) == Counter({"mfma": 75, "wmma": 75})

    canonical_ids = set()
    alternative_ids = set()
    for contract, choices in zip(
        AMDGPU_MATRIX_CONTRACTS, catalog.contract_choices, strict=True
    ):
        if contract.fragment_layout is None:
            assert (
                choices.canonical_result_representation_id
                == MATRIX_RESULT_REPRESENTATION_NONE
            )
            assert (
                choices.operand_exchanged_contract_ordinal
                == MATRIX_CONTRACT_ORDINAL_NONE
            )
            continue
        assert (
            choices.canonical_result_representation_id
            != MATRIX_RESULT_REPRESENTATION_NONE
        )
        canonical_ids.add(choices.canonical_result_representation_id)
        if "sparse" in contract.flags:
            assert (
                choices.operand_exchanged_contract_ordinal
                == MATRIX_CONTRACT_ORDINAL_NONE
            )
            continue
        assert (
            choices.operand_exchanged_contract_ordinal != MATRIX_CONTRACT_ORDINAL_NONE
        )
        alternative_ids.add(choices.operand_exchanged_result_representation_id)

    assert len(canonical_ids) == 26
    assert len(alternative_ids) == 25
    assert canonical_ids.isdisjoint(alternative_ids)


def test_every_alternative_preserves_exact_operand_ownership() -> None:
    catalog = AMDGPU_MATRIX_FRAGMENT_REALIZATION_CATALOG

    for source_ordinal, choices in enumerate(catalog.contract_choices):
        if choices.operand_exchanged_contract_ordinal == MATRIX_CONTRACT_ORDINAL_NONE:
            continue
        source_layout = _exact_layout(source_ordinal)
        native_layout = _exact_layout(choices.operand_exchanged_contract_ordinal)
        transposed_layout = transpose_contraction_layout(native_layout)
        assert source_layout.lhs.coordinate_map == (
            transposed_layout.lhs.coordinate_map
        )
        assert source_layout.rhs.coordinate_map == (
            transposed_layout.rhs.coordinate_map
        )
        assert transposed_layout.accumulator.coordinate_map == (
            transposed_layout.result.coordinate_map
        )
        representation = catalog.result_representation(
            choices.operand_exchanged_result_representation_id
        )
        assert representation is not None
        assert representation.payload == AMDGPU_MATRIX_CONTRACTS[source_ordinal].result
        assert representation.coordinate_map == (
            transposed_layout.result.coordinate_map
        )


def test_gfx11_f16_f32_alternative_is_all_lane_scalar_publication() -> None:
    catalog = AMDGPU_MATRIX_FRAGMENT_REALIZATION_CATALOG
    contract_ordinal = _contract_ordinal("wmma.f32.16x16x16.f16")
    choices = catalog.contract_choices[contract_ordinal]

    assert choices.operand_exchanged_contract_ordinal == contract_ordinal
    representation = catalog.result_representation(
        choices.operand_exchanged_result_representation_id
    )
    assert representation is not None
    assert representation.transposed
    assert representation.coordinate_map.is_bijective
    assert representation.coordinate_map.source_dimensions[0].extent == 32
    assert representation.coordinate_map.source_dimensions[1].extent == 8
    assert tuple(
        representation.coordinate_map.evaluate((participant, value))
        for participant, value in ((0, 0), (1, 0), (15, 0), (16, 0), (31, 7))
    ) == (
        (0, 0, 0),
        (0, 1, 0),
        (0, 15, 0),
        (0, 0, 1),
        (0, 15, 15),
    )


def test_compact_runtime_catalog_fits_two_kibibytes() -> None:
    catalog = AMDGPU_MATRIX_FRAGMENT_REALIZATION_CATALOG

    contract_choice_bytes = len(catalog.contract_choices) * 4
    result_representation_bytes = (len(catalog.result_representations) + 1) * 3
    assert contract_choice_bytes + result_representation_bytes == 1096
    assert contract_choice_bytes + result_representation_bytes < 2048
