# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C table emission for compact native contraction and transition facts."""

from __future__ import annotations

from collections.abc import Sequence

from loom.target.native_layout_facts import (
    NativeContractionFacts,
    NativeContractionRoleFacts,
    NativeLayoutEvidence,
    NativeTransitionFacts,
)

_EVIDENCE_C_NAMES = {
    NativeLayoutEvidence.EXACT: "LOOM_NATIVE_LAYOUT_EVIDENCE_EXACT",
    NativeLayoutEvidence.METADATA_DEPENDENT: ("LOOM_NATIVE_LAYOUT_EVIDENCE_METADATA_DEPENDENT"),
    NativeLayoutEvidence.PARAMETRIC: "LOOM_NATIVE_LAYOUT_EVIDENCE_PARAMETRIC",
    NativeLayoutEvidence.OPAQUE: "LOOM_NATIVE_LAYOUT_EVIDENCE_OPAQUE",
}

_ROLE_C_NAMES = {
    "lhs": "LOOM_CONTRACT_OPERAND_ROLE_LHS",
    "rhs": "LOOM_CONTRACT_OPERAND_ROLE_RHS",
    "accumulator": "LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR",
    "result": "LOOM_CONTRACT_OPERAND_ROLE_RESULT",
}

_PHYSICAL_DIMENSION_C_NAMES = {
    "participant": "LOOM_NATIVE_PHYSICAL_DIMENSION_PARTICIPANT",
    "value": "LOOM_NATIVE_PHYSICAL_DIMENSION_POSITION",
}


def _uint(value: int, maximum: int, field_name: str) -> int:
    if value < 0 or value > maximum:
        raise ValueError(f"native layout {field_name} {value} exceeds C storage")
    return value


def _role_initializer(facts: NativeContractionRoleFacts) -> list[str]:
    return [
        "{",
        f"    .role = {_ROLE_C_NAMES[facts.role]},",
        f"    .evidence = {_EVIDENCE_C_NAMES[facts.evidence]},",
        f"    .element_bit_count = {_uint(facts.element_bit_count, 0xFFFF, 'element bit count')},",
        f"    .register_count = {_uint(facts.register_count, 0xFFFF, 'register count')},",
        f"    .payload_element_count = {_uint(facts.payload_element_count, 0xFFFF, 'payload element count')},",
        f"    .physical_position_count = {_uint(facts.physical_position_count, 0xFFFFFFFF, 'physical position count')},",
        f"    .logical_coordinate_count = {_uint(facts.logical_coordinate_count, 0xFFFFFFFF, 'logical coordinate count')},",
        f"    .owner_multiplicity_minimum = {_uint(facts.owner_multiplicity_minimum or 0, 0xFFFF, 'minimum owner multiplicity')},",
        f"    .owner_multiplicity_maximum = {_uint(facts.owner_multiplicity_maximum or 0, 0xFFFF, 'maximum owner multiplicity')},",
        "},",
    ]


class NativeContractionFactTable:
    """Deduplicates native contraction facts and emits their C table."""

    def __init__(self, symbol: str):
        self._symbol = symbol
        self._facts: list[NativeContractionFacts] = []
        self._ordinals: dict[NativeContractionFacts, int] = {}

    def reference(self, facts: NativeContractionFacts) -> str:
        ordinal = self._ordinals.get(facts)
        if ordinal is None:
            ordinal = len(self._facts)
            self._ordinals[facts] = ordinal
            self._facts.append(facts)
        return f"&{self._symbol}[{ordinal}]"

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    def definition_lines(self) -> list[str]:
        if not self._facts:
            return []
        lines = [
            "static const loom_native_contraction_facts_t",
            f"    {self._symbol}[] = {{",
        ]
        for facts in self._facts:
            shape = facts.shape
            lines.extend(
                [
                    "    {",
                    "        .shape = {",
                    f"            .block_count = {_uint(shape.block_count, 0xFFFFFFFF, 'block count')},",
                    f"            .result_row_count = {_uint(shape.m, 0xFFFFFFFF, 'result row count')},",
                    f"            .result_column_count = {_uint(shape.n, 0xFFFFFFFF, 'result column count')},",
                    f"            .reduction_count = {_uint(shape.k, 0xFFFFFFFF, 'reduction count')},",
                    "        },",
                    f"        .participant_count = {_uint(facts.participant_count, 0xFFFF, 'participant count')},",
                ]
            )
            for field_name, role_facts in (
                ("lhs", facts.lhs),
                ("rhs", facts.rhs),
                ("accumulator", facts.accumulator),
                ("result", facts.result),
            ):
                role_lines = _role_initializer(role_facts)
                lines.append(f"        .{field_name} = {role_lines[0]}")
                lines.extend(f"        {line}" for line in role_lines[1:])
            lines.append("    },")
        lines.extend(["};", ""])
        return lines


def _transition_key(facts: NativeTransitionFacts) -> tuple[object, ...]:
    # The selected contraction is retained independently beside the transition.
    # Deduplicating only the owner transform lets layouts with the same physical
    # movement share one shipping row without conflating their contraction facts.
    return (
        facts.source_role,
        facts.destination_role,
        facts.destination_position_count,
        facts.participant_change_count,
        facts.local_position_change_count,
        facts.destination_positions_per_source_minimum,
        facts.destination_positions_per_source_maximum,
        facts.source_owner_factors,
    )


class NativeTransitionFactTable:
    """Deduplicates native transitions and emits their compact owner algebra."""

    def __init__(self, symbol: str):
        self._symbol = symbol
        self._facts: list[NativeTransitionFacts] = []
        self._ordinals: dict[tuple[object, ...], int] = {}

    def reference(self, facts: NativeTransitionFacts) -> str:
        key = _transition_key(facts)
        ordinal = self._ordinals.get(key)
        if ordinal is None:
            ordinal = len(self._facts)
            self._ordinals[key] = ordinal
            self._facts.append(facts)
        return f"&{self._symbol}[{ordinal}]"

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    def definition_lines(self) -> list[str]:
        if not self._facts:
            return []
        lines: list[str] = []
        for ordinal, facts in enumerate(self._facts):
            lines.extend(
                [
                    "static const loom_native_transition_owner_factor_t",
                    f"    {self._symbol}OwnerFactors{ordinal}[] = {{",
                ]
            )
            for factor in facts.source_owner_factors:
                lines.extend(
                    [
                        "    {",
                        f"        .destination_dimension = {_PHYSICAL_DIMENSION_C_NAMES[factor.destination_dimension]},",
                        f"        .source_owner_dimension = {_PHYSICAL_DIMENSION_C_NAMES[factor.source_owner_dimension]},",
                        f"        .destination_divisor = {_uint(factor.destination_divisor, 0xFFFFFFFF, 'destination divisor')},",
                        f"        .destination_modulus = {_uint(factor.destination_modulus, 0xFFFFFFFF, 'destination modulus')},",
                        f"        .source_owner_multiplier = {_uint(factor.source_owner_multiplier, 0xFFFFFFFF, 'source owner multiplier')},",
                        "    },",
                    ]
                )
            lines.extend(["};", ""])
        lines.extend(
            [
                "static const loom_native_transition_facts_t",
                f"    {self._symbol}[] = {{",
            ]
        )
        for ordinal, facts in enumerate(self._facts):
            lines.extend(
                [
                    "    {",
                    f"        .source_role = {_ROLE_C_NAMES[facts.source_role]},",
                    f"        .destination_role = {_ROLE_C_NAMES[facts.destination_role]},",
                    f"        .destination_position_count = {_uint(facts.destination_position_count, 0xFFFFFFFF, 'destination position count')},",
                    f"        .participant_change_count = {_uint(facts.participant_change_count, 0xFFFFFFFF, 'participant change count')},",
                    f"        .local_position_change_count = {_uint(facts.local_position_change_count, 0xFFFFFFFF, 'local position change count')},",
                    f"        .destination_positions_per_source_minimum = {_uint(facts.destination_positions_per_source_minimum, 0xFFFF, 'minimum source replication')},",
                    f"        .destination_positions_per_source_maximum = {_uint(facts.destination_positions_per_source_maximum, 0xFFFF, 'maximum source replication')},",
                    f"        .source_owner_factors = {self._symbol}OwnerFactors{ordinal},",
                    f"        .source_owner_factor_count = {_uint(len(facts.source_owner_factors), 0xFF, 'source owner factor count')},",
                    "    },",
                ]
            )
        lines.extend(["};", ""])
        return lines


def indent_lines(lines: Sequence[str], prefix: str) -> list[str]:
    """Indents generated initializer lines while retaining empty lines."""

    return [prefix + line if line else line for line in lines]
