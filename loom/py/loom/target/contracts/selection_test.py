# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.dialect.scalar import comparison as scalar
from loom.target.contracts.compile import (
    CompiledCase,
    CompiledContractFragment,
    CompiledDescriptorMatrix,
    CompiledDescriptorRule,
    CompiledOpSpan,
    compile_contract_index,
)
from loom.target.contracts.guards import GuardKind
from loom.target.contracts.kinds import ContractSystem, SourceValueKind
from loom.target.contracts.lower_rule_tables import (
    CompiledLowerRuleSet,
    LowerGuard,
    LowerRule,
    LowerTypePattern,
    LowerValueRef,
)
from loom.target.contracts.patterns import Scalar, Vector
from loom.target.contracts.selection import (
    ContractCandidateEncoding,
    ContractSelectionBucket,
    ContractSelectionIndex,
    ContractSelectionKind,
    ContractSelectionRow,
    ContractSelectionSelector,
    compile_contract_selection_index,
    contract_selection_blob_words,
    pack_contract_selection_index,
)

_OP_KIND = 0x703


def _lower_rule_set(
    guard_programs: tuple[tuple[LowerGuard, ...], ...],
    *,
    type_patterns: tuple[LowerTypePattern, ...] = (),
    value_refs: tuple[LowerValueRef, ...] = (),
) -> CompiledLowerRuleSet:
    guards = []
    rules = []
    for program in guard_programs:
        guard_start = len(guards)
        guards.extend(program)
        rules.append(
            LowerRule(
                source_op=scalar.scalar_cmpi,
                temporary_count=0,
                guard_start=guard_start,
                guard_count=len(program),
                emit_start=0,
                emit_count=0,
            )
        )
    return CompiledLowerRuleSet(
        name="test.selection",
        authored_case_indices=tuple(range(len(rules))),
        rules=tuple(rules),
        spans=(),
        type_patterns=type_patterns,
        value_refs=value_refs,
        source_nodes=(),
        nonlocal_source_op_kinds=(),
        source_memories=(),
        guards=tuple(guards),
        attr_copies=(),
        tied_results=(),
        emits=(),
        diagnostics=(),
    )


def _descriptor_fragment(rule_count: int, *, matrix_first: bool = False):
    cases = []
    matrices = ()
    if matrix_first:
        cases.append(CompiledCase(ContractSystem.DESCRIPTOR_MATRIX, 0))
        matrices = (CompiledDescriptorMatrix("vector_mma"),)
    cases.extend(
        CompiledCase(ContractSystem.DESCRIPTOR_RULE, index)
        for index in range(rule_count)
    )
    fragment = CompiledContractFragment(
        name="test.selection",
        target_contract_query=True,
        op_spans=(CompiledOpSpan(_OP_KIND, "scalar.cmpi", 0, len(cases)),),
        cases=tuple(cases),
        descriptor_rules=tuple(
            CompiledDescriptorRule(index) for index in range(rule_count)
        ),
        descriptor_matrices=matrices,
    )
    return fragment, compile_contract_index((fragment,))


def test_compile_selection_preserves_priority_and_metadata_fallbacks() -> None:
    rule_set = _lower_rule_set(
        tuple(
            (
                LowerGuard(
                    kind=GuardKind.ENUM_ATTR_EQUALS,
                    attr_index=0,
                    u64=index % 8,
                ),
            )
            for index in range(64)
        )
    )
    fragment, index = _descriptor_fragment(64, matrix_first=True)

    selection = compile_contract_selection_index(index, (fragment,), (rule_set,))

    assert len(selection.rows) == 1
    row = selection.rows[0]
    assert row.case_start == 0
    assert row.case_count == 65
    assert row.selector == ContractSelectionSelector(
        ContractSelectionKind.ENUM_ATTRIBUTE, 0
    )
    assert row.fallback_ordinals == (0,)
    assert row.buckets[0] == ContractSelectionBucket(
        0, (0, 1, 9, 17, 25, 33, 41, 49, 57)
    )
    assert tuple(bucket.key for bucket in row.buckets) == tuple(range(8))


def test_compile_selection_normalizes_equivalent_value_refs() -> None:
    value_refs = (
        LowerValueRef(SourceValueKind.OPERAND, 1),
        LowerValueRef(SourceValueKind.OPERAND, 1),
    )
    type_patterns = tuple(
        LowerTypePattern(Scalar(element)) for element in ("i8", "i16", "i32", "i64")
    )
    rule_set = _lower_rule_set(
        tuple(
            (
                LowerGuard(
                    kind=GuardKind.VALUE_TYPE,
                    value_ref_index=index % len(value_refs),
                    type_pattern_index=index % len(type_patterns),
                ),
            )
            for index in range(64)
        ),
        type_patterns=type_patterns,
        value_refs=value_refs,
    )
    fragment, index = _descriptor_fragment(64)

    selection = compile_contract_selection_index(index, (fragment,), (rule_set,))

    row = selection.rows[0]
    assert row.selector == ContractSelectionSelector(
        ContractSelectionKind.OPERAND_TYPE, 1
    )
    assert len(row.buckets) == 4
    assert all(len(bucket.candidate_ordinals) == 16 for bucket in row.buckets)


def test_compile_selection_keeps_scalar_and_vector_type_keys_distinct() -> None:
    type_patterns = (
        LowerTypePattern(Scalar("i32")),
        LowerTypePattern(Vector("i32", lanes=4)),
    )
    rule_set = _lower_rule_set(
        tuple(
            (
                LowerGuard(
                    kind=GuardKind.VALUE_TYPE,
                    value_ref_index=0,
                    type_pattern_index=index % 2,
                ),
            )
            for index in range(64)
        ),
        type_patterns=type_patterns,
        value_refs=(LowerValueRef(SourceValueKind.RESULT, 0),),
    )
    fragment, index = _descriptor_fragment(64)

    selection = compile_contract_selection_index(index, (fragment,), (rule_set,))

    row = selection.rows[0]
    assert row.selector == ContractSelectionSelector(
        ContractSelectionKind.RESULT_TYPE, 0
    )
    assert len(row.buckets) == 2
    assert row.buckets[0].key != row.buckets[1].key
    assert all(len(bucket.candidate_ordinals) == 32 for bucket in row.buckets)


@pytest.mark.parametrize(("rule_count", "indexed"), [(40, False), (44, True)])
def test_compile_selection_requires_32_avoided_cases(
    rule_count: int, indexed: bool
) -> None:
    rule_set = _lower_rule_set(
        tuple(
            (
                LowerGuard(
                    kind=GuardKind.ENUM_ATTR_EQUALS,
                    attr_index=0,
                    u64=index % 4,
                ),
            )
            for index in range(rule_count)
        )
    )
    fragment, index = _descriptor_fragment(rule_count)

    selection = compile_contract_selection_index(index, (fragment,), (rule_set,))

    assert bool(selection.rows) is indexed


def test_pack_selection_chooses_and_interns_candidate_encodings() -> None:
    dense_refs = tuple(range(32))
    sparse_refs = (0, 31, 63)
    selection = ContractSelectionIndex(
        (
            ContractSelectionRow(
                op_kind=_OP_KIND,
                case_start=10,
                case_count=64,
                selector=ContractSelectionSelector(
                    ContractSelectionKind.ENUM_ATTRIBUTE, 2
                ),
                fallback_ordinals=(),
                buckets=(
                    ContractSelectionBucket(1, dense_refs),
                    ContractSelectionBucket(2, dense_refs),
                    ContractSelectionBucket(3, sparse_refs),
                ),
            ),
        )
    )

    packed = pack_contract_selection_index(selection)

    assert packed.buckets[0].candidates.encoding is (
        ContractCandidateEncoding.PRIORITY_BITMAP
    )
    assert packed.buckets[0].candidates.word_start == 0
    assert packed.buckets[0].candidates == packed.buckets[1].candidates
    assert packed.buckets[2].candidates.encoding is (
        ContractCandidateEncoding.ORDINAL_LIST
    )
    assert packed.buckets[2].candidates.word_start == 2
    assert packed.candidate_words == (0xFFFFFFFF, 0, 0x001F0000, 63)
    assert contract_selection_blob_words(packed) == (
        1 | (3 << 16),
        64,
        9,
        3,
        _OP_KIND << 16,
        1,
        (2 | 0x8000) << 16,
        2,
        (2 | 0x8000) << 16,
        3,
        2 | (3 << 16),
        0xFFFFFFFF,
        0,
        0x001F0000,
        63,
    )


def test_pack_selection_preserves_full_contract_case_domains() -> None:
    selection = ContractSelectionIndex(
        (
            ContractSelectionRow(
                op_kind=_OP_KIND,
                case_start=0,
                case_count=0xFFFF,
                selector=ContractSelectionSelector(
                    ContractSelectionKind.ENUM_ATTRIBUTE, 0
                ),
                fallback_ordinals=(0, 0xFFFE),
                buckets=(),
            ),
            ContractSelectionRow(
                op_kind=_OP_KIND + 1,
                case_start=0xFFFE,
                case_count=1,
                selector=ContractSelectionSelector(
                    ContractSelectionKind.ENUM_ATTRIBUTE, 0
                ),
                fallback_ordinals=(0,),
                buckets=(),
            ),
        )
    )

    packed = pack_contract_selection_index(selection)

    assert packed.rows[0].case_count == 0xFFFF
    assert packed.rows[1].case_start == 0xFFFE
    assert contract_selection_blob_words(packed)[1] == 1


def test_pack_selection_supports_more_than_32_rows() -> None:
    selection = ContractSelectionIndex(
        tuple(
            ContractSelectionRow(
                op_kind=_OP_KIND + index,
                case_start=index,
                case_count=64,
                selector=ContractSelectionSelector(
                    ContractSelectionKind.ENUM_ATTRIBUTE, 0
                ),
                fallback_ordinals=(),
                buckets=(),
            )
            for index in range(33)
        )
    )

    packed = pack_contract_selection_index(selection)

    assert len(packed.rows) == 33


@pytest.mark.parametrize(
    "candidate_ordinals",
    [
        (0, 64),
        (2, 1),
        (1, 1),
    ],
)
def test_pack_selection_rejects_invalid_candidate_ordinals(
    candidate_ordinals: tuple[int, ...],
) -> None:
    selection = ContractSelectionIndex(
        (
            ContractSelectionRow(
                op_kind=_OP_KIND,
                case_start=0,
                case_count=64,
                selector=ContractSelectionSelector(
                    ContractSelectionKind.ENUM_ATTRIBUTE, 0
                ),
                fallback_ordinals=(),
                buckets=(ContractSelectionBucket(0, candidate_ordinals),),
            ),
        )
    )

    with pytest.raises(ValueError, match="candidate ref"):
        pack_contract_selection_index(selection)
