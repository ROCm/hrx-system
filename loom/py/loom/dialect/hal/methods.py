# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-independent HAL method values and static grouping.

These declarations describe semantic values, not VM register selectors or C
struct layout. Source verification and target ABI projections consume the same
records. Optional groups distinguish absence from an explicitly empty range.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Value:
    """One exact scalar or managed-reference value."""

    type: str


@dataclass(frozen=True, slots=True)
class Field:
    """One named field within an ordered record."""

    name: str
    value: "Shape"


@dataclass(frozen=True, slots=True)
class Record:
    """Fixed ordered fields, represented as a tuple in source builders."""

    name: str
    fields: tuple[Field, ...]


@dataclass(frozen=True, slots=True)
class Range:
    """A statically sized list of values with one common element schema."""

    element: "Shape"


@dataclass(frozen=True, slots=True)
class Optional:
    """An absent value or a present value, including a present empty list."""

    value: "Shape"


type Shape = Value | Record | Range | Optional


@dataclass(frozen=True, slots=True)
class Method:
    """A synchronous host entry which may submit asynchronous HAL work."""

    name: str
    version: int
    arguments: Record
    doc: str
    examples: tuple[str, ...]


I32 = Value("i32")
I64 = Value("i64")
BUFFER = Value("hal.buffer")
QUEUE = Value("hal.queue")
SEMAPHORE = Value("hal.semaphore")

TIMEPOINT = Record("timepoint", (Field("semaphore", SEMAPHORE), Field("value", I64)))
BUFFER_RANGE = Record("buffer_range", (Field("buffer", BUFFER), Field("offset", I64), Field("length", I64)))
TRANSITION_OPERATION = Record(
    "transition_operation",
    (
        Field("kind", I32),
        Field("executor", I32),
        Field("operation", I32),
        Field("range_granularity", I64),
        Field("host_instruction", I32),
        Field("host_fence_before", I32),
        Field("host_fence_after", I32),
    ),
)
TRANSITION_RECIPE = Record("transition_recipe", (Field("effects", I32), Field("operations", Range(TRANSITION_OPERATION))))
MEMORY_BARRIER = Record("memory_barrier", (Field("source_scope", I32), Field("target_scope", I32)))
BUFFER_BARRIER = Record(
    "buffer_barrier",
    (
        Field("source_scope", I32),
        Field("target_scope", I32),
        Field("buffer", BUFFER_RANGE),
        Field("recipe", Optional(TRANSITION_RECIPE)),
    ),
)
BARRIER = Record(
    "barrier",
    (
        Field("source_stage", I32),
        Field("target_stage", I32),
        Field("flags", I64),
        Field("effects", I32),
        Field("memory", Range(MEMORY_BARRIER)),
        Field("buffers", Range(BUFFER_BARRIER)),
    ),
)
QUEUE_BARRIERS = Record(
    "queue_barriers",
    (
        Field("before", Optional(Range(BARRIER))),
        Field("after", Optional(Range(BARRIER))),
    ),
)
BUFFER_POSITION = Record("buffer_position", (Field("buffer", BUFFER), Field("offset", I64)))

QUEUE_COPY = Method(
    name="queue.copy",
    version=1,
    arguments=Record(
        "queue_copy",
        (
            Field("queue", QUEUE),
            Field("waits", Range(TIMEPOINT)),
            Field("signals", Range(TIMEPOINT)),
            Field("source", BUFFER_POSITION),
            Field("target", BUFFER_POSITION),
            Field("length", I64),
            Field("barriers", QUEUE_BARRIERS),
            Field("flags", I64),
        ),
    ),
    doc=(
        "Submits a buffer copy with explicit timeline dependencies and visibility "
        "boundaries. The HAL captures descriptors during the call; buffer and "
        "backing lifetimes follow the selected HAL flags. Absent before/after "
        "barriers preserve conservative defaults; empty lists request no extra "
        "payload barriers. Transition recipes contain qualified HAL actions."
    ),
    examples=("hal.queue.copy (%queue, [(%ready, %value)], [(%done, %value)], (%source, %source_offset), (%target, %target_offset), %length, (none, []), %flags)",),
)

ALL_HAL_METHODS = (QUEUE_COPY,)
