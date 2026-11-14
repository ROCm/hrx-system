# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.avx2 import X86_AVX2_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.avx512 import X86_AVX512_CONTRACT_FRAGMENT
from loom.target.contracts import DescriptorRule, GuardKind


def _type(rule, field):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == field
    )


def test_pack_covers_every_native_predicate_and_storage_shape() -> None:
    for fragment, representations in (
        (X86_AVX2_CONTRACT_FRAGMENT, {8: (8, 16, 32), 16: (8, 16), 32: (8,)}),
        (
            X86_AVX512_CONTRACT_FRAGMENT,
            {8: (1, 8, 16, 32), 16: (1, 8, 16), 32: (1, 8), 64: (1, 8)},
        ),
    ):
        actual = []
        for rule in fragment.cases:
            if (
                not isinstance(rule, DescriptorRule)
                or rule.source_op is not vector.vector_bitpack
            ):
                continue
            source = _type(rule, "source")
            result = _type(rule, "result")
            representation = next(
                guard.count
                for guard in rule.guards
                if guard.kind is GuardKind.LOW_VALUE_REPRESENTATION
                and guard.field == "source"
            )
            actual.append((source.lanes, representation, result.element, result.lanes))
            # A sign mask is correct for normalized i1 carriers, not arbitrary
            # integer values with different high and low bits.
            assert source.elements == ("i1",)
            assert any(
                guard.kind is GuardKind.I64_RANGE
                and guard.field == "width"
                and guard.minimum == guard.maximum == 1
                for guard in rule.guards
            )
            # Pending representation queries must not pin the callable class.
            assert not any(
                guard.kind is GuardKind.LOW_VALUE_REGISTER_CLASS
                for guard in rule.guards
            )
            available = {rule.descriptor.key} | {
                guard.descriptor.key
                for guard in rule.guards
                if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
            }
            assert {emit.descriptor.key for emit in rule.emit} <= available
        expected = {
            (lanes, representation, f"i{width}", lanes // width)
            for lanes, choices in representations.items()
            for representation in choices
            for width in (8, 16, 32, 64)
            if width <= lanes
        }
        assert len(actual) == len(expected)
        assert set(actual) == expected
