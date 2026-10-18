# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compact candidate selection for composed target-contract indices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, unique

from loom.scalar_type import parse_scalar_type_kind
from loom.target.contracts.compile import (
    CompiledContractFragment,
    CompiledContractIndex,
    CompiledIndexCase,
)
from loom.target.contracts.guards import GuardKind
from loom.target.contracts.kinds import ContractSystem, SourceValueKind
from loom.target.contracts.lower_rule_tables import (
    CompiledLowerRuleSet,
    LowerRule,
)
from loom.target.contracts.patterns import TypePattern

# Candidate indices pay for fixed rodata and one root-field lookup. A basic
# rule match costs one work unit; each related node adds three units for its
# use-def resolution, eligibility checks, and connection verification.
CONTRACT_SELECTION_MINIMUM_AVOIDED_WORK = 32
CONTRACT_SELECTION_SOURCE_NODE_WORK = 3
CONTRACT_SELECTION_MAX_FIELD_INDEX = 0xFF
CONTRACT_SELECTION_MAX_ELEMENT_INDEX = 0x3F

_CANDIDATE_BITMAP_BIT = 0x8000
_TYPE_KEY_SCALAR = 1 << 28
_TYPE_KEY_VECTOR = 2 << 28


@unique
class ContractSelectionKind(IntEnum):
    """Root source-op fact used to select composed contract candidates."""

    ENUM_ATTRIBUTE = 1
    OPERAND_TYPE = 2
    RESULT_TYPE = 3


@unique
class ContractCandidateEncoding(IntEnum):
    """Encoding used by one candidate sequence in the shared word pool."""

    ORDINAL_LIST = 0
    PRIORITY_BITMAP = 1


@dataclass(frozen=True, order=True, slots=True)
class ContractSelectionSelector:
    """Normalized root operation field selected by a generated index."""

    kind: ContractSelectionKind
    field_index: int
    element_index: int = 0


@dataclass(frozen=True, slots=True)
class ContractSelectionBucket:
    """Exact selector key and its priority-ordered relative case ordinals."""

    key: int
    candidate_ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ContractSelectionRow:
    """Candidate index for one composed source-op case span."""

    op_kind: int
    case_start: int
    case_count: int
    selector: ContractSelectionSelector
    fallback_ordinals: tuple[int, ...]
    buckets: tuple[ContractSelectionBucket, ...]


@dataclass(frozen=True, slots=True)
class ContractSelectionIndex:
    """Semantic candidate rows ordered by source op kind."""

    rows: tuple[ContractSelectionRow, ...]


@dataclass(frozen=True, slots=True)
class PackedContractCandidateSpan:
    """One interned candidate sequence in the packed word pool."""

    word_start: int
    count: int
    encoding: ContractCandidateEncoding


@dataclass(frozen=True, slots=True)
class PackedContractSelectionBucket:
    """Packed exact-key bucket."""

    key: int
    candidates: PackedContractCandidateSpan


@dataclass(frozen=True, slots=True)
class PackedContractSelectionRow:
    """Packed fixed-size selection row."""

    op_kind: int
    case_start: int
    case_count: int
    selector: int
    bucket_start: int
    bucket_count: int
    fallback_candidates: PackedContractCandidateSpan


@dataclass(frozen=True, slots=True)
class PackedContractSelectionIndex:
    """Relocation-free selection rows, buckets, and candidate words."""

    rows: tuple[PackedContractSelectionRow, ...]
    buckets: tuple[PackedContractSelectionBucket, ...]
    candidate_words: tuple[int, ...]


_DIRECT_VALUE_SELECTOR_KINDS = {
    SourceValueKind.OPERAND: ContractSelectionKind.OPERAND_TYPE,
    SourceValueKind.RESULT: ContractSelectionKind.RESULT_TYPE,
}


def _exact_type_keys(pattern: TypePattern) -> frozenset[int] | None:
    if not pattern.elements:
        return None
    elements = []
    for element in pattern.elements:
        scalar_kind = parse_scalar_type_kind(element)
        if scalar_kind is None:
            return None
        elements.append(int(scalar_kind))
    if pattern.kind == "scalar":
        return frozenset(_TYPE_KEY_SCALAR | (element << 16) for element in elements)
    if pattern.kind != "vector":
        return None
    if pattern.lanes is not None:
        lanes = pattern.lanes
    elif len(pattern.dims) == 1:
        lanes = pattern.dims[0]
    else:
        return None
    if not 0 <= lanes <= 0xFFFF:
        return None
    return frozenset(_TYPE_KEY_VECTOR | (element << 16) | lanes for element in elements)


def _merge_selector_values(
    selector_values: dict[ContractSelectionSelector, frozenset[int]],
    selector: ContractSelectionSelector,
    values: frozenset[int] | None,
) -> None:
    if values is None:
        return
    previous = selector_values.get(selector)
    selector_values[selector] = values if previous is None else previous & values


def _rule_selector_values(
    rule_set: CompiledLowerRuleSet,
    rule: LowerRule,
) -> dict[ContractSelectionSelector, frozenset[int]]:
    selector_values: dict[ContractSelectionSelector, frozenset[int]] = {}
    for guard in rule_set.guards[
        rule.guard_start : rule.guard_start + rule.guard_count
    ]:
        if guard.kind is GuardKind.ENUM_ATTR_EQUALS:
            if guard.u64 <= 0xFFFFFFFF:
                _merge_selector_values(
                    selector_values,
                    ContractSelectionSelector(
                        ContractSelectionKind.ENUM_ATTRIBUTE,
                        guard.attr_index,
                    ),
                    frozenset((guard.u64,)),
                )
            continue
        if guard.kind is not GuardKind.VALUE_TYPE:
            continue
        value_ref = rule_set.value_refs[guard.value_ref_index]
        selector_kind = _DIRECT_VALUE_SELECTOR_KINDS.get(value_ref.kind)
        if (
            selector_kind is None
            or value_ref.source_node_index != 0
            or value_ref.materializer_index != 0
            or value_ref.index > CONTRACT_SELECTION_MAX_FIELD_INDEX
            or value_ref.element_index > CONTRACT_SELECTION_MAX_ELEMENT_INDEX
        ):
            continue
        _merge_selector_values(
            selector_values,
            ContractSelectionSelector(
                selector_kind,
                value_ref.index,
                value_ref.element_index,
            ),
            _exact_type_keys(
                rule_set.type_patterns[guard.type_pattern_index].type_pattern
            ),
        )
    return selector_values


def _case_rule(
    contract_case: CompiledIndexCase,
    fragments: Sequence[CompiledContractFragment],
    lower_rule_sets: Sequence[CompiledLowerRuleSet | None],
) -> tuple[CompiledLowerRuleSet, LowerRule] | None:
    fragment = fragments[contract_case.binding_index]
    if contract_case.system in (
        ContractSystem.DESCRIPTOR_RULE,
        ContractSystem.DESCRIPTOR_RULE_LOWERING_ONLY,
    ):
        rule_index = fragment.descriptor_rules[contract_case.row_index].rule_index
    elif contract_case.system in {
        ContractSystem.VALUE_ALIAS,
        ContractSystem.VALUE_ELIDE,
        ContractSystem.RECIPE_RULE,
    }:
        rule_index = contract_case.row_index
    else:
        return None
    rule_set = lower_rule_sets[contract_case.binding_index]
    if rule_set is None:
        raise ValueError(
            f"contract fragment '{fragment.name}' case requires a lower-rule set"
        )
    return rule_set, rule_set.rules[rule_index]


def _compile_selection_row(
    op_kind: int,
    case_start: int,
    case_count: int,
    index: CompiledContractIndex,
    fragments: Sequence[CompiledContractFragment],
    lower_rule_sets: Sequence[CompiledLowerRuleSet | None],
) -> ContractSelectionRow | None:
    case_selectors = []
    selectors = set()
    case_match_work = []
    for contract_case in index.cases[case_start : case_start + case_count]:
        case_rule = _case_rule(contract_case, fragments, lower_rule_sets)
        if case_rule is None:
            selector_values = {}
            match_work = 1
        else:
            rule_set, rule = case_rule
            selector_values = _rule_selector_values(rule_set, rule)
            match_work = 1 + (
                CONTRACT_SELECTION_SOURCE_NODE_WORK * rule.source_node_count
            )
        case_selectors.append(selector_values)
        case_match_work.append(match_work)
        selectors.update(selector_values)

    choices = []
    for selector in selectors:
        keys = sorted(
            {
                key
                for selector_values in case_selectors
                for key in selector_values.get(selector, ())
            }
        )
        if not keys:
            continue
        fallback_ordinals = tuple(
            case_ordinal
            for case_ordinal, selector_values in enumerate(case_selectors)
            if selector not in selector_values
        )
        buckets = tuple(
            ContractSelectionBucket(
                key,
                tuple(
                    case_ordinal
                    for case_ordinal, selector_values in enumerate(case_selectors)
                    if selector not in selector_values
                    or key in selector_values[selector]
                ),
            )
            for key in keys
        )
        candidate_counts = (
            len(fallback_ordinals),
            *(len(bucket.candidate_ordinals) for bucket in buckets),
        )
        candidate_work = (
            sum(case_match_work[ordinal] for ordinal in fallback_ordinals),
            *(
                sum(case_match_work[ordinal] for ordinal in bucket.candidate_ordinals)
                for bucket in buckets
            ),
        )
        choices.append(
            (
                max(candidate_work),
                candidate_work[0],
                sum(candidate_work),
                max(candidate_counts),
                len(fallback_ordinals),
                sum(candidate_counts),
                selector,
                fallback_ordinals,
                buckets,
            )
        )
    if not choices:
        return None
    choices.sort(key=lambda choice: choice[:7])
    worst_work, _, _, _, _, _, selector, fallback_ordinals, buckets = choices[0]
    if sum(case_match_work) - worst_work < CONTRACT_SELECTION_MINIMUM_AVOIDED_WORK:
        return None
    return ContractSelectionRow(
        op_kind,
        case_start,
        case_count,
        selector,
        fallback_ordinals,
        buckets,
    )


def compile_contract_selection_index(
    index: CompiledContractIndex,
    fragments: Sequence[CompiledContractFragment],
    lower_rule_sets: Sequence[CompiledLowerRuleSet | None],
) -> ContractSelectionIndex:
    """Compiles profitable exact-key candidates for one composed index."""

    if len(fragments) != len(lower_rule_sets):
        raise ValueError("contract fragments and lower-rule sets must align")
    rows = []
    for dialect_ordinal, dialect in enumerate(index.dialects):
        dialect_id = index.dialect_base_id + dialect_ordinal
        for op_index, entry in enumerate(dialect):
            case_start, case_count = entry.case_start, entry.case_count
            if not case_count:
                continue
            op_kind = (dialect_id << 8) | op_index
            row = _compile_selection_row(
                op_kind,
                case_start,
                case_count,
                index,
                fragments,
                lower_rule_sets,
            )
            if row is not None:
                rows.append(row)
    return ContractSelectionIndex(tuple(rows))


def _pack_selector(selector: ContractSelectionSelector) -> int:
    if not 0 <= selector.field_index <= CONTRACT_SELECTION_MAX_FIELD_INDEX:
        raise ValueError("contract selection field index exceeds 8 bits")
    if not 0 <= selector.element_index <= CONTRACT_SELECTION_MAX_ELEMENT_INDEX:
        raise ValueError("contract selection element index exceeds 6 bits")
    return (
        int(selector.kind)
        | (selector.field_index << 2)
        | (selector.element_index << 10)
    )


def _candidate_sequence_words(
    case_count: int,
    candidate_ordinals: tuple[int, ...],
) -> tuple[ContractCandidateEncoding, int, tuple[int, ...]]:
    if any(
        candidate_ref < 0 or candidate_ref >= case_count
        for candidate_ref in candidate_ordinals
    ):
        raise ValueError("contract selection candidate ref is outside its case span")
    if tuple(sorted(set(candidate_ordinals))) != candidate_ordinals:
        raise ValueError("contract selection candidate refs must be unique and ordered")
    list_words = tuple(
        candidate_ordinals[index]
        | (
            candidate_ordinals[index + 1] << 16
            if index + 1 < len(candidate_ordinals)
            else 0
        )
        for index in range(0, len(candidate_ordinals), 2)
    )
    bitmap_words = [0] * ((case_count + 31) // 32)
    for candidate_ref in candidate_ordinals:
        bitmap_words[candidate_ref // 32] |= 1 << (candidate_ref % 32)
    if len(bitmap_words) < len(list_words):
        return (
            ContractCandidateEncoding.PRIORITY_BITMAP,
            len(bitmap_words),
            tuple(bitmap_words),
        )
    return ContractCandidateEncoding.ORDINAL_LIST, len(candidate_ordinals), list_words


def pack_contract_selection_index(
    selection_index: ContractSelectionIndex,
) -> PackedContractSelectionIndex:
    """Packs semantic candidates into one relocation-free word-pool model."""

    if len(selection_index.rows) > 0xFFFF:
        raise ValueError("contract selection row count exceeds uint16_t")
    packed_rows = []
    packed_buckets = []
    candidate_words = []
    sequence_starts: dict[tuple[object, ...], int] = {}
    previous_op_kind = -1

    def intern_candidates(
        case_count: int,
        candidate_ordinals: tuple[int, ...],
    ) -> PackedContractCandidateSpan:
        encoding, count, words = _candidate_sequence_words(
            case_count, candidate_ordinals
        )
        key = encoding, count, words
        word_start = sequence_starts.get(key)
        if word_start is None:
            word_start = len(candidate_words)
            sequence_starts[key] = word_start
            candidate_words.extend(words)
        if word_start > 0xFFFF:
            raise ValueError("contract selection candidate word start exceeds uint16_t")
        if count >= _CANDIDATE_BITMAP_BIT:
            raise ValueError("contract selection candidate count exceeds 15 bits")
        return PackedContractCandidateSpan(word_start, count, encoding)

    for row in selection_index.rows:
        if not 0 <= row.op_kind <= 0xFFFF:
            raise ValueError("contract selection op kind exceeds uint16_t")
        if row.op_kind <= previous_op_kind:
            raise ValueError("contract selection rows must be ordered by op kind")
        previous_op_kind = row.op_kind
        if not 0 <= row.case_start <= 0xFFFF:
            raise ValueError("contract selection case start exceeds uint16_t")
        if not 0 <= row.case_count <= 0xFFFF:
            raise ValueError("contract selection case count exceeds uint16_t")
        bucket_start = len(packed_buckets)
        fallback_candidates = intern_candidates(row.case_count, row.fallback_ordinals)
        previous_key = -1
        for bucket in row.buckets:
            if not 0 <= bucket.key <= 0xFFFFFFFF:
                raise ValueError("contract selection bucket key exceeds uint32_t")
            if bucket.key <= previous_key:
                raise ValueError("contract selection bucket keys must be ordered")
            previous_key = bucket.key
            packed_buckets.append(
                PackedContractSelectionBucket(
                    bucket.key,
                    intern_candidates(row.case_count, bucket.candidate_ordinals),
                )
            )
        bucket_count = len(packed_buckets) - bucket_start
        if bucket_start > 0xFFFF or bucket_count > 0xFFFF:
            raise ValueError("contract selection bucket span exceeds uint16_t")
        packed_rows.append(
            PackedContractSelectionRow(
                row.op_kind,
                row.case_start,
                row.case_count,
                _pack_selector(row.selector),
                bucket_start,
                bucket_count,
                fallback_candidates,
            )
        )
    if len(candidate_words) > 0xFFFF:
        raise ValueError("contract selection candidate word count exceeds uint16_t")
    if len(packed_buckets) > 0xFFFF:
        raise ValueError("contract selection bucket count exceeds uint16_t")
    return PackedContractSelectionIndex(
        tuple(packed_rows), tuple(packed_buckets), tuple(candidate_words)
    )


def _pack_candidate_span(span: PackedContractCandidateSpan) -> int:
    count = span.count
    if span.encoding is ContractCandidateEncoding.PRIORITY_BITMAP:
        count |= _CANDIDATE_BITMAP_BIT
    return span.word_start | (count << 16)


def contract_selection_blob_words(
    selection_index: PackedContractSelectionIndex,
) -> tuple[int, ...]:
    """Returns the exact uint32 blob consumed by the runtime iterator."""

    minimum_case_count = min(
        (row.case_count for row in selection_index.rows), default=0
    )
    words = [
        len(selection_index.rows) | (len(selection_index.buckets) << 16),
        minimum_case_count,
    ]
    for row in selection_index.rows:
        words.extend(
            (
                row.selector | (row.bucket_start << 16),
                row.bucket_count | (row.fallback_candidates.word_start << 16),
                row.fallback_candidates.count
                | (
                    _CANDIDATE_BITMAP_BIT
                    if row.fallback_candidates.encoding
                    is ContractCandidateEncoding.PRIORITY_BITMAP
                    else 0
                )
                | (row.op_kind << 16),
            )
        )
    for bucket in selection_index.buckets:
        words.extend((bucket.key, _pack_candidate_span(bucket.candidates)))
    words.extend(selection_index.candidate_words)
    return tuple(words)
