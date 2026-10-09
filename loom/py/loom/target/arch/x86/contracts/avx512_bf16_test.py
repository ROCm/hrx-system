# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.target.arch.x86.contracts.avx512_bf16 import (
    X86_AVX512_BF16_CONTRACT_DIALECT_OPS,
    X86_AVX512_BF16_CONTRACT_FRAGMENT,
)
from loom.target.arch.x86.descriptors import X86_AVX512_FEATURES_DESCRIPTOR_SET
from loom.target.contracts import (
    LOWER_RULE_FLAG_NONLOCAL_SOURCE_GRAPH,
    DescriptorRule,
    EmitDescriptorOp,
    GuardKind,
    SourceMemoryOperation,
    compile_lower_rule_set,
)


def _rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        case
        for case in X86_AVX512_BF16_CONTRACT_FRAGMENT.cases
        if isinstance(case, DescriptorRule)
    )


def _conversion_descriptor_keys() -> set[str]:
    return {
        descriptor.key
        for descriptor in X86_AVX512_FEATURES_DESCRIPTOR_SET.descriptors
        if descriptor.key.startswith("x86.avx512_bf16.vcvtne")
    }


def _memory_emit(rule: DescriptorRule) -> EmitDescriptorOp | None:
    return next(
        (
            emit
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp) and emit.source_memory is not None
        ),
        None,
    )


def test_fragment_compiles_the_complete_authored_family() -> None:
    compiled = compile_lower_rule_set(
        X86_AVX512_BF16_CONTRACT_FRAGMENT,
        dialect_ops=X86_AVX512_BF16_CONTRACT_DIALECT_OPS,
    )

    assert len(compiled.rules) == len(_rules()) == 570
    assert Counter(
        bool(rule.flags & LOWER_RULE_FLAG_NONLOCAL_SOURCE_GRAPH)
        for rule in compiled.rules
    ) == Counter({True: 567, False: 3})


def test_rules_cover_every_conversion_descriptor() -> None:
    assert {rule.descriptor.key for rule in _rules()} == _conversion_descriptor_keys()
    assert len(_conversion_descriptor_keys()) == 90


def test_rules_cover_every_source_shape_and_mask_representation() -> None:
    assert Counter(rule.source_op for rule in _rules()) == Counter(
        {
            vector.vector_fptrunc: 15,
            vector.vector_concat: 15,
            vector.vector_load: 270,
            view.view_load: 270,
        }
    )
    report_counts = Counter(rule.report_key for rule in _rules())
    assert len(report_counts) == 90
    assert set(report_counts.values()) == {1, 9}
    assert sum(count == 1 for count in report_counts.values()) == 30
    assert sum(count == 9 for count in report_counts.values()) == 60


def test_every_narrowing_source_preserves_the_daz_contract() -> None:
    for rule in _rules():
        narrowing_guards = [
            *([rule.guards] if rule.source_op is vector.vector_fptrunc else []),
            *(
                node.guards
                for node in rule.source_nodes
                if node.source_op
                in (vector.vector_fptrunc, scalar_conversion.scalar_fptrunc)
            ),
        ]
        assert narrowing_guards
        for guards in narrowing_guards:
            permission_guards = tuple(
                guard
                for guard in guards
                if guard.kind is GuardKind.VALUE_NOT_SUBNORMAL_OR_INSTANCE_FLAGS_HAS_ALL
            )
            assert len(permission_guards) == 1
            assert permission_guards[0].enum_keyword == "daz"


def test_masked_rules_cover_native_and_callable_predicates() -> None:
    masked_rules = tuple(
        rule
        for rule in _rules()
        if ".merge." in rule.descriptor.key or ".zero." in rule.descriptor.key
    )
    assert len(masked_rules) == 456
    for rule in masked_rules:
        select = next(
            node for node in rule.source_nodes if node.source_op is vector.vector_select
        )
        condition_class = next(
            guard.register_class
            for guard in select.guards
            if guard.kind is GuardKind.LOW_VALUE_REGISTER_CLASS
            and guard.field == "condition"
        )
        main_emit = next(
            emit
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp) and emit.descriptor == rule.descriptor
        )
        if condition_class == "x86.k":
            assert main_emit.operands["mask"].source_node == "select"
        else:
            assert main_emit.operands["mask"].field == "mask"
            assert any(
                isinstance(emit, EmitDescriptorOp)
                and emit.descriptor.mnemonic.startswith("vpmov")
                for emit in rule.emit
            )
        zero_guards = tuple(
            guard
            for guard in select.guards
            if guard.kind is GuardKind.VALUE_FLOAT_EQUALS
        )
        if ".zero." in rule.descriptor.key:
            assert len(zero_guards) == 1
            assert "passthrough" not in main_emit.operands
        else:
            assert not zero_guards
            assert main_emit.operands["passthrough"].source_node == "select"


def test_memory_rules_preserve_full_and_broadcast_access_shapes() -> None:
    memory_rules = tuple(rule for rule in _rules() if _memory_emit(rule) is not None)
    assert len(memory_rules) == 540
    for rule in memory_rules:
        memory_emit = _memory_emit(rule)
        assert memory_emit is not None
        assert memory_emit.source_memory is not None
        assert memory_emit.source_memory.operation is SourceMemoryOperation.LOAD
        assert memory_emit.source_memory.element_byte_count == 4
        if ".broadcast." in rule.descriptor.key:
            assert rule.source_op is view.view_load
            assert memory_emit.source_memory.vector_lane_count == 1
            assert any(
                node.source_op is scalar_conversion.scalar_fptrunc
                for node in rule.source_nodes
            )
            assert any(
                node.source_op is vector.vector_splat for node in rule.source_nodes
            )
        else:
            assert rule.source_op is vector.vector_load
            assert memory_emit.source_memory.vector_lane_count in (4, 8, 16)


def test_two_source_rules_keep_high_and_low_halves_in_isa_order() -> None:
    binary_rules = tuple(
        rule for rule in _rules() if rule.descriptor.mnemonic == "vcvtne2ps2bf16"
    )
    assert len(binary_rules) == 285
    for rule in binary_rules:
        main_emit = next(
            emit
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp) and emit.descriptor == rule.descriptor
        )
        assert main_emit.operands["high"].source_node == "high"
        if _memory_emit(rule) is None:
            assert main_emit.operands["low"].source_node == "low"
        else:
            assert "low" not in main_emit.operands
