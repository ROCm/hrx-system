# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Owned AIE itinerary model and dependency timing equations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class PipelineStageKind(IntEnum):
    """Functional-unit composition rule for one itinerary stage."""

    REQUIRED = 0
    RESERVED = 1


class DependencyKind(Enum):
    """Register dependency direction evaluated between two operands."""

    RAW = "raw"
    WAR = "war"
    WAW = "waw"


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """One non-pipelined itinerary stage in source order."""

    resources: tuple[str, ...]
    cycles: int
    time_increment: int
    kind: PipelineStageKind


@dataclass(frozen=True, slots=True)
class MemoryCycles:
    """Cycles on which an itinerary accesses memory."""

    cycles: tuple[int, ...]

    @property
    def first_cycle(self) -> int:
        return self.cycles[0]

    @property
    def last_cycle(self) -> int:
        return self.cycles[-1]


@dataclass(frozen=True, slots=True)
class Itinerary:
    """Complete scheduler-visible behavior for one AIE itinerary class."""

    name: str
    stages: tuple[PipelineStage, ...]
    operand_cycles: tuple[int, ...]
    bypasses: tuple[str, ...]
    memory: MemoryCycles | None = None
    micro_ops: int = 1


@dataclass(frozen=True, slots=True)
class PipelineUse:
    """Absolute functional-unit use derived from an itinerary stage."""

    resources: tuple[str, ...]
    start_cycle: int
    cycles: int
    kind: PipelineStageKind


@dataclass(frozen=True, slots=True)
class ScheduleTable:
    """Complete owned schedule table for one AIE target."""

    oracle_source_commit: str
    resources: tuple[str, ...]
    bypasses: tuple[str, ...]
    itineraries: tuple[Itinerary, ...]


NO_ITINERARY = Itinerary(
    name="NoItinerary",
    stages=(),
    operand_cycles=(),
    bypasses=(),
)


def itinerary_payload(itinerary: Itinerary) -> tuple[object, ...]:
    """Returns the complete scheduling payload excluding the itinerary name."""

    return (
        itinerary.stages,
        itinerary.operand_cycles,
        itinerary.bypasses,
        itinerary.memory,
        itinerary.micro_ops,
    )


def pipeline_uses(itinerary: Itinerary) -> tuple[PipelineUse, ...]:
    """Resolves source-ordered stages to absolute issue-relative uses."""

    uses = []
    start_cycle = 0
    for stage in itinerary.stages:
        if stage.cycles:
            uses.append(
                PipelineUse(
                    resources=stage.resources,
                    start_cycle=start_cycle,
                    cycles=stage.cycles,
                    kind=stage.kind,
                )
            )
        start_cycle += (
            stage.cycles if stage.time_increment == -1 else stage.time_increment
        )
    return tuple(uses)


def operand_cycle(itinerary: Itinerary, operand_index: int) -> int:
    """Returns the issue-relative read or write cycle for an MC operand."""

    if operand_index < 0 or operand_index >= len(itinerary.operand_cycles):
        raise ValueError(
            f"operand {operand_index} is outside {itinerary.name}'s "
            f"0..{len(itinerary.operand_cycles) - 1} range"
        )
    return itinerary.operand_cycles[operand_index]


def bypass_class(itinerary: Itinerary, operand_index: int) -> str | None:
    """Returns the physical forwarding class attached to an operand."""

    if operand_index < 0:
        raise ValueError("operand index must not be negative")
    if operand_index >= len(itinerary.bypasses):
        return None
    bypass = itinerary.bypasses[operand_index]
    return None if bypass == "NoBypass" else bypass


def has_forwarding(
    producer: Itinerary,
    producer_operand: int,
    consumer: Itinerary,
    consumer_operand: int,
) -> bool:
    """Returns whether two endpoints share a physical bypass path."""

    producer_bypass = bypass_class(producer, producer_operand)
    return producer_bypass is not None and producer_bypass == bypass_class(
        consumer, consumer_operand
    )


def dependency_separation(
    producer: Itinerary,
    producer_operand: int,
    consumer: Itinerary,
    consumer_operand: int,
    kind: DependencyKind,
) -> int:
    """Evaluates the target's signed register-dependency separation."""

    producer_cycle = operand_cycle(producer, producer_operand)
    consumer_cycle = operand_cycle(consumer, consumer_operand)
    if kind is DependencyKind.RAW:
        forwarded_cycles = int(
            has_forwarding(
                producer,
                producer_operand,
                consumer,
                consumer_operand,
            )
        )
        return producer_cycle - consumer_cycle + 1 - forwarded_cycles
    if kind is DependencyKind.WAR:
        forwarded_cycles = int(
            has_forwarding(
                consumer,
                consumer_operand,
                producer,
                producer_operand,
            )
        )
        return producer_cycle - consumer_cycle + forwarded_cycles
    if kind is DependencyKind.WAW:
        return producer_cycle - consumer_cycle + 1
    raise ValueError(f"unsupported dependency kind {kind!r}")


def memory_separation(producer: Itinerary, consumer: Itinerary) -> int:
    """Evaluates the target's signed memory-dependency separation."""

    if producer.memory is None or consumer.memory is None:
        raise ValueError("both itineraries must define memory cycles")
    return producer.memory.last_cycle - consumer.memory.first_cycle + 1


def validate_schedule_table(table: ScheduleTable) -> None:
    """Validates every schedule-table invariant consumed without C checks."""

    if len(table.oracle_source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in table.oracle_source_commit
    ):
        raise ValueError("schedule oracle commit must be a lowercase Git SHA")
    if tuple(sorted(set(table.resources))) != table.resources:
        raise ValueError("schedule resources must be sorted and unique")
    if any(not resource for resource in table.resources):
        raise ValueError("schedule resource names must not be empty")
    if tuple(sorted(set(table.bypasses))) != table.bypasses:
        raise ValueError("schedule bypasses must be sorted and unique")
    if any(not bypass for bypass in table.bypasses):
        raise ValueError("schedule bypass names must not be empty")
    if "NoBypass" not in table.bypasses:
        raise ValueError("schedule table must declare NoBypass")
    itinerary_names = tuple(row.name for row in table.itineraries)
    if itinerary_names != tuple(sorted(set(itinerary_names))):
        raise ValueError("schedule itineraries must be sorted and unique")

    known_resources = set(table.resources)
    known_bypasses = set(table.bypasses)
    for itinerary in table.itineraries:
        if not itinerary.name:
            raise ValueError("schedule itinerary name must not be empty")
        if itinerary.micro_ops <= 0:
            raise ValueError(f"{itinerary.name}: micro-op count must be positive")
        if any(cycle <= 0 for cycle in itinerary.operand_cycles):
            raise ValueError(f"{itinerary.name}: operand cycles must be positive")
        if len(itinerary.bypasses) > len(itinerary.operand_cycles):
            raise ValueError(f"{itinerary.name}: too many operand bypasses")
        unknown_bypasses = set(itinerary.bypasses) - known_bypasses
        if unknown_bypasses:
            raise ValueError(
                f"{itinerary.name}: unknown bypasses {sorted(unknown_bypasses)}"
            )
        for stage in itinerary.stages:
            if not stage.resources or len(set(stage.resources)) != len(stage.resources):
                raise ValueError(
                    f"{itinerary.name}: stage resources must be nonempty and unique"
                )
            unknown_resources = set(stage.resources) - known_resources
            if unknown_resources:
                raise ValueError(
                    f"{itinerary.name}: unknown stage resources "
                    f"{sorted(unknown_resources)}"
                )
            if stage.cycles < 0 or stage.time_increment < -1:
                raise ValueError(f"{itinerary.name}: invalid stage timing")
            if not isinstance(stage.kind, PipelineStageKind):
                raise ValueError(f"{itinerary.name}: invalid stage kind")
        pipeline_uses(itinerary)
        if itinerary.memory is not None:
            cycles = itinerary.memory.cycles
            if not cycles or tuple(sorted(set(cycles))) != cycles:
                raise ValueError(
                    f"{itinerary.name}: memory cycles must be sorted and unique"
                )
            if cycles[0] <= 0:
                raise ValueError(f"{itinerary.name}: memory cycles must be positive")
