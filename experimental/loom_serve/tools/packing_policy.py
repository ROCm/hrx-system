# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Count-only counterpart of scheduling/packing.c for offline experiments.

The native differential test establishes plan equivalence, including row holes,
indivisible verifiers, chunk limits and cursor movement. Costs, arrivals and
model outcomes belong to the caller, not this policy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackingPlan:
    # Ordinal in the caller's ordered (token capacity, span capacity) catalog.
    shape: int
    # Next rotating row cursor; unchanged when no work fits.
    cursor: int
    # Ordered (resident row, granted token count) pairs.
    spans: tuple[tuple[int, int], ...]


def pack_spans(ready, token_capacity, span_capacity, chunk_size, cursor):
    """Packs trusted (available tokens, indivisible minimum) rows.

    This has the same positive-capacity and cursor contract as the C packer.
    Zero available tokens marks an ineligible row; its minimum is not read.
    """
    spans = []
    remaining = token_capacity
    visited = 0
    while visited < len(ready) and len(spans) < span_capacity and remaining:
        row = (cursor + visited) % len(ready)
        count, minimum = ready[row]
        if count and minimum <= remaining:
            spans.append((row, minimum))
            remaining -= minimum
        visited += 1
    for index, (row, count) in enumerate(spans):
        limit = max(count, min(ready[row][0], chunk_size))
        extra = min(remaining, limit - count)
        spans[index] = (row, count + extra)
        remaining -= extra
    if spans:
        cursor = (cursor + (visited if visited < len(ready) else 1)) % len(ready)
    return tuple(spans), cursor


def pack_shapes(ready, shapes, chunk_size, cursor):
    """Chooses occupied tokens, then rows advanced, then smaller capacities."""
    best = None
    best_score = None
    for index, (tokens, spans) in enumerate(shapes):
        packed, next_cursor = pack_spans(ready, tokens, spans, chunk_size, cursor)
        score = (sum(count for _, count in packed), len(packed), -tokens, -spans)
        if best is None or score > best_score:
            best = PackingPlan(index, next_cursor, packed)
            best_score = score
    return best
