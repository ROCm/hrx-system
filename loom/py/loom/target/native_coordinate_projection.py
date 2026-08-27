# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compact executable projections compiled from exact native coordinates.

Exact finite coordinate maps are generation-time proofs. This module compiles
the subset that can be evaluated as separable mixed-radix digit transfers into
small rows suitable for trusted JIT consumers. The finite map remains the
source fact; quotient/remainder terms are its compact executable form.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from itertools import product

from loom.target.native_contraction_layout import (
    CoordinateDimension,
    ExactCoordinateMap,
)


@dataclass(frozen=True, slots=True)
class CoordinateProjectionTerm:
    """One source digit accumulated into one destination coordinate.

    Evaluation adds
    `((source / source_divisor) % source_modulus) * destination_multiplier`.
    A zero modulus omits the remainder operation.
    """

    source_dimension: str
    destination_dimension: str
    source_divisor: int
    source_modulus: int
    destination_multiplier: int

    def __post_init__(self) -> None:
        if not self.source_dimension or not self.destination_dimension:
            raise ValueError("coordinate projection dimensions must be named")
        if self.source_divisor <= 0:
            raise ValueError("coordinate projection divisor must be positive")
        if self.source_modulus < 0 or self.source_modulus == 1:
            raise ValueError(
                "coordinate projection modulus must be zero or at least two"
            )
        if self.destination_multiplier <= 0:
            raise ValueError("coordinate projection multiplier must be positive")


@dataclass(frozen=True, slots=True)
class CoordinateProjectionPlan:
    """Forward and canonical-inverse mixed-radix coordinate projections."""

    forward_terms: tuple[CoordinateProjectionTerm, ...]
    inverse_terms: tuple[CoordinateProjectionTerm, ...]


@cache
def _candidate_digits(
    extent: int,
) -> tuple[tuple[tuple[int, ...], int, int], ...]:
    candidates: dict[tuple[int, ...], tuple[int, int]] = {}
    for divisor in range(1, extent):
        quotient = tuple(value // divisor for value in range(extent))
        quotient_extent = (extent + divisor - 1) // divisor
        if any(quotient):
            candidates[quotient] = (divisor, 0)
        for modulus in range(2, quotient_extent + 1):
            digit = tuple(value % modulus for value in quotient)
            if any(digit):
                candidates.setdefault(digit, (divisor, modulus))
    return tuple(
        (values, divisor, modulus)
        for values, (divisor, modulus) in sorted(
            candidates.items(), key=lambda item: (item[1], item[0])
        )
    )


@cache
def _recognize_digit_sum(
    values: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...] | None:
    """Finds the smallest compact sum of mixed-radix source digits.

    The logarithmic term bound covers the independent source digits exercised
    by native coordinate layouts while keeping rejection of arbitrary finite
    lookup tables cheap. Exact maps outside that executable subset remain
    valid generation-time facts.
    """

    if not any(values):
        return ()

    candidates = _candidate_digits(len(values))

    @cache
    def search(
        residual: tuple[int, ...],
        remaining_term_count: int,
    ) -> tuple[tuple[int, int, int], ...] | None:
        if not any(residual):
            return ()
        if remaining_term_count == 0:
            return None

        pivot = next(
            index for index, residual_value in enumerate(residual) if residual_value
        )
        multiplier = residual[pivot]
        for digit, divisor, modulus in candidates:
            if divisor != pivot:
                continue
            next_residual = tuple(
                residual_value - multiplier * digit_value
                for residual_value, digit_value in zip(residual, digit, strict=True)
            )
            if any(value < 0 for value in next_residual):
                continue
            suffix = search(next_residual, remaining_term_count - 1)
            if suffix is not None:
                return ((divisor, modulus, multiplier), *suffix)
        return None

    maximum_term_count = (len(values) - 1).bit_length()
    for term_count in range(1, maximum_term_count + 1):
        terms = search(values, term_count)
        if terms is not None:
            return terms
    return None


def _compile_separable_projection(
    *,
    source_dimensions: tuple[CoordinateDimension, ...],
    destination_dimensions: tuple[CoordinateDimension, ...],
    evaluate: Callable[[tuple[int, ...]], tuple[int, ...]],
) -> tuple[CoordinateProjectionTerm, ...] | None:
    zero_source = tuple(0 for _ in source_dimensions)
    if evaluate(zero_source) != tuple(0 for _ in destination_dimensions):
        return None

    contributions: dict[tuple[int, int], tuple[int, ...]] = {}
    for source_index, source_dimension in enumerate(source_dimensions):
        for destination_index in range(len(destination_dimensions)):
            values = []
            for source_value in range(source_dimension.extent):
                source = list(zero_source)
                source[source_index] = source_value
                values.append(evaluate(tuple(source))[destination_index])
            contributions[(source_index, destination_index)] = tuple(values)

    for source in product(
        *(range(dimension.extent) for dimension in source_dimensions)
    ):
        expected = [0] * len(destination_dimensions)
        for source_index, source_value in enumerate(source):
            for destination_index in range(len(destination_dimensions)):
                expected[destination_index] += contributions[
                    (source_index, destination_index)
                ][source_value]
        if tuple(expected) != evaluate(source):
            return None

    terms = []
    for destination_index, destination_dimension in enumerate(destination_dimensions):
        for source_index, source_dimension in enumerate(source_dimensions):
            values = contributions[(source_index, destination_index)]
            if not any(values):
                continue
            digits = _recognize_digit_sum(values)
            if digits is None:
                return None
            terms.extend(
                CoordinateProjectionTerm(
                    source_dimension=source_dimension.name,
                    destination_dimension=destination_dimension.name,
                    source_divisor=divisor,
                    source_modulus=modulus,
                    destination_multiplier=multiplier,
                )
                for divisor, modulus, multiplier in digits
            )
    return tuple(terms)


def coordinate_projection_plan(
    coordinate_map: ExactCoordinateMap,
) -> CoordinateProjectionPlan | None:
    """Compiles an exact map into direct forward and canonical inverse terms.

    Each destination must be a separable sum of compact mixed-radix source
    digits. Other exact maps remain valid generation-time facts but do not
    acquire a shipping projection plan.
    """

    forward_terms = _compile_separable_projection(
        source_dimensions=coordinate_map.source_dimensions,
        destination_dimensions=coordinate_map.destination_dimensions,
        evaluate=coordinate_map.evaluate,
    )
    inverse_terms = _compile_separable_projection(
        source_dimensions=coordinate_map.destination_dimensions,
        destination_dimensions=coordinate_map.source_dimensions,
        evaluate=coordinate_map.evaluate_canonical_inverse,
    )
    if forward_terms is None or inverse_terms is None:
        return None

    return CoordinateProjectionPlan(
        forward_terms=forward_terms,
        inverse_terms=inverse_terms,
    )
