# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

import pytest

from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.error.target import ERR_TARGET_002
from loom.target.contracts import (
    ContractFragment,
    Guard,
    GuardDiagnostic,
    RecipeRule,
    SourceMemoryConstraint,
    SourceMemoryOperation,
    SourceMemoryRejectionReason,
    SourceValueKind,
    View,
    compile_lower_rule_set,
)
from loom.target.contracts.diagnostics import (
    string_param,
    target_diagnostic,
    value_type_param,
)
from loom.target.contracts.lower_rule_diagnostics import _source_memory_diagnostics
from loom.target.test.descriptors import TEST_LOW_CORE_DESCRIPTOR_SET


def test_value_type_diagnostics_bind_each_source_op() -> None:
    guard = Guard.value_type("view", View("i32"))
    compiled = compile_lower_rule_set(
        ContractFragment(
            name="test.diagnostic-source-bindings",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                RecipeRule(source_op=source_op, guards=(guard,))
                for source_op in (vector.vector_load, vector.vector_store)
            ],
        ),
        dialect_ops={"vector": ALL_VECTOR_OPS},
    )

    bindings = {}
    for rule in compiled.rules:
        diagnostic = compiled.diagnostics[
            compiled.guards[rule.guard_start].diagnostic_index
        ]
        actual_type = next(
            param for param in diagnostic.params if param.name == "actual_type"
        )
        value_ref = compiled.value_refs[actual_type.value_ref_index]
        assert value_ref.kind == SourceValueKind.OPERAND
        bindings[rule.source_op.name] = value_ref.index
    assert bindings == {"vector.load": 0, "vector.store": 1}


@pytest.mark.parametrize(
    ("second_rule_count", "message"),
    [
        (None, "needs an operand_segment_count guard"),
        (1, "exceeds guarded segment count 1"),
    ],
)
def test_value_type_diagnostic_rechecks_each_rule_arity(
    second_rule_count: int | None, message: str
) -> None:
    diagnostic = GuardDiagnostic(
        ref=target_diagnostic(
            ERR_TARGET_002,
            string_param("field_name", "indices[1]"),
            value_type_param("actual_type", "indices", element=1),
            string_param("expected_type", "index"),
        )
    )
    # Only the diagnostic refers to the variadic operand. A previous rule's
    # successful binding must not bypass validation against the next rule.
    guard = Guard.value_type("view", View("i32"), diagnostic=diagnostic)
    rules = [
        RecipeRule(
            source_op=vector.vector_load,
            guards=(Guard.operand_segment_count("indices", 2), guard),
        ),
        RecipeRule(
            source_op=vector.vector_load,
            guards=(guard,)
            if second_rule_count is None
            else (Guard.operand_segment_count("indices", second_rule_count), guard),
        ),
    ]
    with pytest.raises(ValueError, match=message):
        compile_lower_rule_set(
            ContractFragment(
                name="test.diagnostic-variadic-bindings",
                descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
                cases=rules,
            ),
            dialect_ops={"vector": ALL_VECTOR_OPS},
        )


def test_source_memory_alignment_diagnostic_overrides_only_alignment() -> None:
    general_diagnostic = GuardDiagnostic(
        subject_role="source-memory",
        subject_name="general",
        constraint_key="test.general",
    )
    alignment_diagnostic = GuardDiagnostic(
        subject_role="source-memory",
        subject_name="alignment",
        constraint_key="test.alignment",
    )
    constraint = SourceMemoryConstraint(
        operation=SourceMemoryOperation.LOAD,
        memory_spaces=("global",),
        element_byte_count=4,
        vector_lane_count=1,
        vector_lane_byte_stride=4,
        static_byte_offset=0,
        minimum_alignment=4,
        diagnostic=general_diagnostic,
        alignment_diagnostic=alignment_diagnostic,
    )

    diagnostics_by_reason = dict(
        zip(
            SourceMemoryRejectionReason,
            _source_memory_diagnostics(constraint, None),
            strict=True,
        )
    )

    assert (
        diagnostics_by_reason[SourceMemoryRejectionReason.MINIMUM_ALIGNMENT]
        is alignment_diagnostic.ref
    )
    assert (
        diagnostics_by_reason[SourceMemoryRejectionReason.MEMORY_SPACE]
        is general_diagnostic.ref
    )
    default_constraint = replace(
        constraint,
        diagnostic=None,
        alignment_diagnostic=None,
    )
    default_diagnostics = dict(
        zip(
            SourceMemoryRejectionReason,
            _source_memory_diagnostics(default_constraint, None),
            strict=True,
        )
    )
    memory_space_ref = default_diagnostics[SourceMemoryRejectionReason.MEMORY_SPACE]
    assert (
        next(
            param.string_value
            for param in memory_space_ref.params
            if param.name == "constraint_key"
        )
        == "source_memory.memory_space"
    )
