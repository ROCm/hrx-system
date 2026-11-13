# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.contracts.packed_memory import packed_storage_buffer_rules
from loom.target.arch.spirv.features import feature_bits_value
from loom.target.contracts import GuardKind, SourceMemoryOperation


def test_packed_transfers_cover_every_scalar_alignment() -> None:
    signatures = []
    for rule in packed_storage_buffer_rules():
        scalar = next(
            guard.type_pattern.element
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.field == "view"
        )
        transfers = [step for step in rule.emit if step.descriptor.effects]
        memory = transfers[0].source_memory
        assert memory is not None
        alignment = memory.minimum_alignment
        signatures.append((scalar, memory.operation.value, alignment))
        expected_bytes = {
            "i16": 2,
            "f16": 2,
            "bf16": 2,
            "i32": 4,
            "f32": 4,
            "i64": 8,
            "f64": 8,
        }[scalar]
        assert (
            sum(step.descriptor.effects[0].width_bits for step in transfers)
            == expected_bytes * 8
        )
        assert len(transfers) == (2 if expected_bytes == 8 and alignment == 1 else 1)
        for step in transfers:
            assert step.source_memory is memory
            assert memory.element_byte_count == expected_bytes
            assert memory.vector_lane_count == 1
            assert memory.vector_lane_byte_stride == expected_bytes
            assert step.source_memory_address_materializer is not None
            assert ("constant" in memory.memory_spaces) == (
                memory.operation is SourceMemoryOperation.LOAD
            )
            assert "workgroup" not in memory.memory_spaces
        if len(transfers) == 2:
            displacements = [
                step.immediates["offset64_value"]
                for step in rule.emit
                if step.descriptor.key == "spirv.op_constant.offset64"
            ]
            assert displacements == [0, 4]

    expected = [
        (scalar, operation, alignment)
        for scalar, alignments in (
            ("i16", (1,)),
            ("f16", (1,)),
            ("bf16", (1,)),
            ("bf16", (1,)),  # Native BF16 and the established i32 carrier.
            ("i32", (2, 1)),
            ("f32", (2, 1)),
            ("i64", (4, 2, 1)),
            ("f64", (4, 2, 1)),
        )
        for operation in ("load", "store")
        for alignment in alignments
    ]
    assert sorted(signatures) == sorted(expected)


def test_packed_transfers_guard_only_the_required_capabilities() -> None:
    int64 = feature_bits_value(("int64",))
    native_bfloat = feature_bits_value(("bfloat16_type_khr",))
    bfloat_paths = {"load": set(), "store": set()}
    for rule in packed_storage_buffer_rules():
        guarded = {
            guard.descriptor.key
            for guard in rule.guards
            if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
        }
        bits = 0
        for step in rule.emit:
            if step.descriptor.feature_mask_words:
                assert step.descriptor.key in guarded
                bits |= step.descriptor.feature_mask_words[0]
        scalar = next(
            guard.type_pattern.element
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.field == "view"
        )
        if scalar == "f64":
            assert not bits & int64
        if scalar == "bf16":
            bfloat_paths[rule.source_op.name.removeprefix("view.")].add(
                bool(bits & native_bfloat)
            )
    assert bfloat_paths == {"load": {False, True}, "store": {False, True}}
