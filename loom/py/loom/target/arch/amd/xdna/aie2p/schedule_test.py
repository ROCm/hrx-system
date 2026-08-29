# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections import Counter

from loom.target.arch.amd.xdna.aie.schedule import (
    DependencyKind,
    PipelineStageKind,
    dependency_separation,
    itinerary_payload,
    validate_schedule_table,
)
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE
from loom.target.arch.amd.xdna.aie2p.core_schedule_data import CORE_SCHEDULE_TABLE


def test_core_schedule_table_is_structurally_complete() -> None:
    validate_schedule_table(CORE_SCHEDULE_TABLE)

    assert len(CORE_SCHEDULE_TABLE.resources) == 73
    assert len(CORE_SCHEDULE_TABLE.bypasses) == 3
    assert len(CORE_SCHEDULE_TABLE.itineraries) == 4_013
    assert (
        len(
            {
                itinerary_payload(itinerary)
                for itinerary in CORE_SCHEDULE_TABLE.itineraries
            }
        )
        == 601
    )
    assert (
        sum(len(itinerary.stages) for itinerary in CORE_SCHEDULE_TABLE.itineraries)
        == 12_260
    )
    assert Counter(
        stage.kind
        for itinerary in CORE_SCHEDULE_TABLE.itineraries
        for stage in itinerary.stages
    ) == {
        PipelineStageKind.REQUIRED: 11_654,
        PipelineStageKind.RESERVED: 606,
    }
    assert (
        sum(
            itinerary.memory is not None
            for itinerary in CORE_SCHEDULE_TABLE.itineraries
        )
        == 620
    )


def test_every_physical_form_resolves_an_exact_operand_schedule() -> None:
    itineraries = {
        itinerary.name: itinerary for itinerary in CORE_SCHEDULE_TABLE.itineraries
    }
    assert "NoItinerary" not in itineraries

    for form in CORE_MACHINE_TABLE.forms:
        operand_count = (
            len(form.outputs)
            + len(form.inputs)
            + len(form.implicit_defs)
            + len(form.implicit_uses)
        )
        if form.itinerary == "NoItinerary":
            assert operand_count == 0
            continue
        itinerary = itineraries[form.itinerary]
        assert len(itinerary.operand_cycles) == operand_count, form.name


def test_seed_dependency_rows_come_from_general_equations() -> None:
    itineraries = {
        itinerary.name: itinerary for itinerary in CORE_SCHEDULE_TABLE.itineraries
    }
    scalar_add = itineraries["II_ADD_alu_r_rr"]
    scalar_multiply = itineraries["II_MUL"]
    vector_add = itineraries["II_VADD_32"]
    vector_load = itineraries["II_VLDA_dmx_lda_x_idx_imm"]
    vector_store = itineraries["II_VST_dmx_sts_x_idx_imm"]

    assert dependency_separation(vector_load, 0, vector_add, 1, DependencyKind.RAW) == 7
    assert dependency_separation(vector_add, 0, vector_add, 1, DependencyKind.RAW) == 1
    assert (
        dependency_separation(vector_add, 0, vector_store, 0, DependencyKind.RAW) == 2
    )
    assert dependency_separation(vector_add, 1, vector_add, 0, DependencyKind.WAR) == 0
    assert dependency_separation(vector_add, 0, vector_add, 0, DependencyKind.WAW) == 1
    assert dependency_separation(scalar_add, 0, scalar_add, 1, DependencyKind.RAW) == 1
    assert (
        dependency_separation(scalar_multiply, 0, scalar_add, 1, DependencyKind.RAW)
        == 2
    )
