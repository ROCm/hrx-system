# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.target.arch.amd.xdna.aie.schedule import (
    DependencyKind,
    Itinerary,
    MemoryCycles,
    PipelineStage,
    PipelineStageKind,
    ScheduleTable,
    bypass_class,
    dependency_separation,
    memory_separation,
    pipeline_uses,
    validate_schedule_table,
)


def _itinerary(
    name: str,
    operand_cycles: tuple[int, ...],
    bypasses: tuple[str, ...] = (),
) -> Itinerary:
    return Itinerary(
        name=name,
        stages=(),
        operand_cycles=operand_cycles,
        bypasses=bypasses,
    )


def test_pipeline_uses_resolve_stage_time_increments() -> None:
    itinerary = Itinerary(
        name="staged",
        stages=(
            PipelineStage(("a",), 2, -1, PipelineStageKind.REQUIRED),
            PipelineStage(("b",), 3, 0, PipelineStageKind.RESERVED),
            PipelineStage(("c",), 1, 4, PipelineStageKind.REQUIRED),
            PipelineStage(("d",), 2, -1, PipelineStageKind.REQUIRED),
        ),
        operand_cycles=(),
        bypasses=(),
    )

    assert tuple(
        (use.resources, use.start_cycle, use.cycles, use.kind)
        for use in pipeline_uses(itinerary)
    ) == (
        (("a",), 0, 2, PipelineStageKind.REQUIRED),
        (("b",), 2, 3, PipelineStageKind.RESERVED),
        (("c",), 2, 1, PipelineStageKind.REQUIRED),
        (("d",), 6, 2, PipelineStageKind.REQUIRED),
    )


def test_dependency_equations_preserve_signed_separations() -> None:
    producer = _itinerary("producer", (2,), ("VEC_Bypass",))
    forwarded_consumer = _itinerary("forwarded", (1,), ("VEC_Bypass",))
    ordinary_consumer = _itinerary("ordinary", (3,), ("NoBypass",))

    assert (
        dependency_separation(producer, 0, forwarded_consumer, 0, DependencyKind.RAW)
        == 1
    )
    assert (
        dependency_separation(producer, 0, ordinary_consumer, 0, DependencyKind.RAW)
        == 0
    )
    assert (
        dependency_separation(ordinary_consumer, 0, producer, 0, DependencyKind.WAR)
        == 1
    )
    assert (
        dependency_separation(producer, 0, ordinary_consumer, 0, DependencyKind.WAW)
        == 0
    )


def test_memory_equation_uses_last_producer_and_first_consumer_cycle() -> None:
    producer = replace(_itinerary("producer", ()), memory=MemoryCycles((5, 11)))
    consumer = replace(_itinerary("consumer", ()), memory=MemoryCycles((7,)))
    assert memory_separation(producer, consumer) == 5


def test_validation_rejects_duplicate_itinerary_names() -> None:
    itinerary = _itinerary("duplicate", ())
    table = ScheduleTable(
        oracle_source_commit="0" * 40,
        resources=("resource",),
        bypasses=("NoBypass",),
        itineraries=(itinerary, itinerary),
    )
    with pytest.raises(ValueError, match="sorted and unique"):
        validate_schedule_table(table)


def test_negative_bypass_operand_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        bypass_class(_itinerary("producer", (1,), ("VEC_Bypass",)), -1)
