# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.contracts import (
    DescriptorRule,
    GuardKind,
    SourceMemoryOperation,
    Vector,
)


def test_constant_storage_is_readable_across_storage_memory_contracts() -> None:
    operations = set()
    for rule in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases:
        if not isinstance(rule, DescriptorRule):
            continue
        for step in rule.emit:
            memory = step.source_memory
            if memory is None or "global" not in memory.memory_spaces:
                continue
            operations.add(memory.operation)
            assert ("constant" in memory.memory_spaces) == (
                memory.operation is SourceMemoryOperation.LOAD
            ), (rule.source_op.name, step.descriptor.key)
    assert operations == {
        SourceMemoryOperation.LOAD,
        SourceMemoryOperation.STORE,
        SourceMemoryOperation.ATOMIC_REDUCE,
        SourceMemoryOperation.ATOMIC_RMW,
        SourceMemoryOperation.ATOMIC_CMPXCHG,
    }


def test_cooperative_memory_uses_logical_shapes_without_carrier_widths() -> None:
    signatures = set()
    for rule in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases:
        if not isinstance(rule, DescriptorRule) or rule.source_op.name not in (
            "vector.fragment.load",
            "vector.fragment.store",
        ):
            continue
        role = next(
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind == GuardKind.ENUM_ATTR_EQUALS and guard.field == "role"
        )
        payload = "value" if role == "result" else "result"
        payload_type = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind == GuardKind.VALUE_TYPE and guard.field == payload
        )
        assert payload_type is not None
        assert payload_type == Vector(
            payload_type.elements,
            minimum_lanes=1,
            maximum_lanes="LOOM_DIM_MAX_STATIC_SIZE",
        )
        rows, columns = (
            next(
                guard.minimum
                for guard in rule.guards
                if guard.kind == GuardKind.VALUE_I64_RANGE
                and guard.field == dimension
                and guard.minimum == guard.maximum
            )
            for dimension in ("rows", "columns")
        )
        memory = rule.emit[-1].source_memory
        assert memory is not None
        signatures.add(
            (
                rule.descriptor.key.split(".")[3],
                role,
                payload_type.element,
                rows,
                columns,
                "workgroup" in memory.memory_spaces,
                rule.emit[-1].immediates["memory_layout"],
            )
        )
    expected = {
        (family, role, element, rows, columns, workgroup, layout)
        for family, source_element, accumulator_element, k_size in (
            ("f16", "f16", "f32", 16),
            ("bf16", "bf16", "f32", 16),
            ("s8", "i8", "i32", 32),
            ("u8", "i8", "i32", 32),
        )
        for role, element, rows, columns in (
            ("lhs", source_element, 16, k_size),
            ("rhs", source_element, k_size, 16),
            ("init", accumulator_element, 16, 16),
            ("result", accumulator_element, 16, 16),
        )
        for workgroup in (False, True)
        for layout in (0, 1)
    }
    assert signatures == expected
