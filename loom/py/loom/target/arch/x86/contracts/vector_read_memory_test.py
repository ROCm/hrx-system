# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.target.arch.x86.contracts.vector_read_memory import (
    X86_VECTOR_READ_MEMORY_CONTRACT_DIALECT_OPS,
    X86_VECTOR_READ_MEMORY_CONTRACT_FRAGMENT,
)
from loom.target.contracts import (
    EmitDescriptorOp,
    GuardKind,
    SourceMemoryOperation,
    SourceNodeRelation,
    compile_lower_rule_set,
)


def test_broadcast_family_compiles_without_speculative_legality() -> None:
    fragment = X86_VECTOR_READ_MEMORY_CONTRACT_FRAGMENT
    assert not fragment.target_contract_query
    compiled = compile_lower_rule_set(
        fragment, dialect_ops=X86_VECTOR_READ_MEMORY_CONTRACT_DIALECT_OPS
    )
    assert len(compiled.rules) == len(fragment.cases)


def test_broadcast_family_preserves_scalar_reads_and_payload_types() -> None:
    elements = {
        1: ("i8", "f8E4M3", "f8E5M2"),
        2: ("i16", "f16", "bf16"),
        4: ("i32", "f32"),
        8: ("i64", "f64"),
    }
    cells = Counter()
    for rule in X86_VECTOR_READ_MEMORY_CONTRACT_FRAGMENT.cases:
        assert rule.source_op is view.view_load
        assert len(rule.source_nodes) == 1
        consumer = rule.source_nodes[0]
        assert consumer.source_op is vector.vector_splat
        assert consumer.relation is SourceNodeRelation.EXCLUSIVE_USER
        assert consumer.parent_value.field == "result"
        assert consumer.node_value.field == "scalar"
        emits = tuple(
            emit
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
            and ".vpbroadcast" in emit.descriptor.key
        )
        assert len(emits) == 1
        emit = emits[0]
        memory = emit.source_memory
        assert memory is not None
        assert memory.operation is SourceMemoryOperation.LOAD
        assert memory.vector_lane_count == 1
        assert emit.results["dst"].source_node == consumer.name
        types = {
            guard.field: guard.type_pattern
            for guard in consumer.guards
            if guard.kind is GuardKind.VALUE_TYPE
        }
        assert types["scalar"].elements == elements[memory.element_byte_count]
        assert types["result"].elements == types["scalar"].elements
        carrier_guards = tuple(
            guard
            for guard in consumer.guards
            if guard.kind is GuardKind.LOW_VALUE_REGISTER_CLASS
        )
        assert len(carrier_guards) == 1
        assert carrier_guards[0].field == "result"
        _, profile, mnemonic, *_ = emit.descriptor.key.split(".")
        suffix = emit.descriptor.key.rsplit(".", 1)[1]
        assert (
            mnemonic
            == {
                1: "vpbroadcastb",
                2: "vpbroadcastw",
                4: "vpbroadcastd",
                8: "vpbroadcastq",
            }[memory.element_byte_count]
        )
        cells[profile, suffix, memory.element_byte_count] += 1
        assert rule.priority == (2 if profile == "avx512" else 1)

    assert cells == Counter(
        {
            (profile, suffix, byte_count): 9
            for profile, suffixes in (
                ("avx2", ("xmm", "ymm")),
                ("avx512", ("xmm", "ymm", "zmm")),
            )
            for suffix in suffixes
            for byte_count in elements
        }
    )
