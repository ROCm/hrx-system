# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.contracts.cooperative_matrix import (
    COOPERATIVE_MATRIX_MEMORY_RULES,
)
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
                memory.operation
                in (
                    SourceMemoryOperation.LOAD,
                    SourceMemoryOperation.VIEW_CARRIER,
                )
            ), (rule.source_op.name, step.descriptor.key)
    assert operations == {
        SourceMemoryOperation.LOAD,
        SourceMemoryOperation.STORE,
        SourceMemoryOperation.ATOMIC_REDUCE,
        SourceMemoryOperation.ATOMIC_RMW,
        SourceMemoryOperation.ATOMIC_CMPXCHG,
        SourceMemoryOperation.VIEW_CARRIER,
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
        (scalar, role, element, rows, columns, workgroup, layout)
        for scalar, element, rows, columns, roles in (
            ("f16", "f16", 16, 16, ("lhs", "rhs", "init", "result")),
            ("bf16", "bf16", 16, 16, ("lhs", "rhs")),
            ("i8", "i8", 16, 16, ("lhs", "rhs")),
            ("u8", "i8", 16, 16, ("lhs", "rhs")),
            ("i8", "i8", 16, 32, ("lhs",)),
            ("i8", "i8", 32, 16, ("rhs",)),
            ("u8", "i8", 16, 32, ("lhs",)),
            ("u8", "i8", 32, 16, ("rhs",)),
            ("f32", "f32", 16, 16, ("init", "result")),
            ("i32", "i32", 16, 16, ("init", "result")),
            ("u32", "i32", 16, 16, ("init", "result")),
        )
        for role in roles
        for workgroup in (False, True)
        for layout in (0, 1)
    }
    assert signatures == expected


def test_cooperative_memory_uses_the_retained_allocation_carrier() -> None:
    signatures = set()
    for rule in COOPERATIVE_MATRIX_MEMORY_RULES:
        memory = rule.emit[-1].source_memory
        if "workgroup" not in memory.memory_spaces:
            continue
        role = next(
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind == GuardKind.ENUM_ATTR_EQUALS
        )
        carrier_guard = next(
            guard
            for guard in rule.guards
            if guard.kind == GuardKind.LOW_VALUE_REGISTER_CLASS
        )
        carrier = carrier_guard.register_class.removeprefix(
            "spirv.ptr.workgroup.array."
        )
        layout = rule.emit[-1].immediates["memory_layout"]
        signature = (
            rule.descriptor.key.split(".")[3],
            rule.descriptor.key.split(".")[4],
            role,
            carrier,
            layout,
        )
        assert signature not in signatures
        signatures.add(signature)
        for step in rule.emit:
            assert step.source_memory is memory
            assert step.source_memory_address_materializer.address.key == (
                f"spirv.op_access_chain.workgroup.{carrier}.byte_offset"
            )
        assert (
            memory.element_byte_count
            == {
                "f16": 2,
                "bf16": 2,
                "i16": 2,
                "f32": 4,
                "i8": 1,
                "u8": 1,
                "i32": 4,
                "u32": 4,
            }[carrier]
        )
    expected = {
        (scalar, shape, role, carrier, layout)
        for scalar, roles, carriers in (
            ("f16", ("lhs", "rhs", "init", "result"), ("f16", "i16")),
            ("bf16", ("lhs", "rhs"), ("bf16", "i16")),
            ("i8", ("lhs", "rhs"), ("i8",)),
            ("u8", ("lhs", "rhs"), ("u8", "i8")),
            ("f32", ("init", "result"), ("f32", "i32")),
            ("i32", ("init", "result"), ("i32",)),
            ("u32", ("init", "result"), ("u32", "i32")),
        )
        for role in roles
        for shape in (
            ("16x16", "16x32" if role == "lhs" else "32x16")
            if scalar in ("i8", "u8")
            else ("16x16",)
        )
        for carrier in carriers
        for layout in (0, 1)
    }
    assert signatures == expected
