# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.features import feature_bits_value
from loom.target.contracts import GuardKind, TypePattern, ValueRef


def test_fused_arithmetic_family_preserves_rounding_and_source_permissions() -> None:
    actual = set()
    for rule in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases:
        if rule.source_op.name not in ("scalar.fmaf", "vector.fmaf"):
            continue
        type_guards = {
            guard.field: guard.type_pattern
            for guard in rule.guards
            if guard.kind == GuardKind.VALUE_TYPE
        }
        assert type_guards.keys() == {"a", "b", "c", "result"}
        assert len(set(type_guards.values())) == 1
        permissions = tuple(
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
        )
        actual.add(
            (
                rule.source_op.name,
                type_guards["result"],
                rule.descriptor.key,
                permissions,
            )
        )
        assert rule.priority == len(permissions)
        assert len(rule.emit) == 1
        assert rule.emit[0].descriptor is rule.descriptor
        assert rule.emit[0].operands == {
            name: ValueRef.operand(name) for name in ("a", "b", "c")
        }
        assert rule.emit[0].results == {"dst": ValueRef.result("result")}

    expected = set()
    descriptors = {}
    for width in (16, 32, 64):
        for lanes in (0, 1, 2, 3, 4):
            source_type = (
                TypePattern.scalar(f"f{width}")
                if not lanes
                else TypePattern.vector(f"f{width}", lanes=lanes)
            )
            carrier = f"v{lanes}f{width}" if lanes > 1 else f"f{width}"
            for suffix, permissions in (
                ("", ()),
                (".afn", ("afn",)),
                (".finite", ("nnan", "ninf", "nsz")),
                (".finite_afn", ("nnan", "ninf", "nsz", "afn")),
            ):
                key = f"spirv.op_fma{suffix}.{carrier}"
                expected.add(
                    (
                        "scalar.fmaf" if not lanes else "vector.fmaf",
                        source_type,
                        key,
                        permissions,
                    )
                )
                features = {f"fma_float{width}", f"float{width}_rounding_rte"}
                if width != 32:
                    features.add(f"float{width}")
                if "afn" not in permissions:
                    features.add(f"float{width}_denorm_preserve")
                if "nnan" not in permissions:
                    features.update(
                        (
                            "float_controls2",
                            f"float{width}_signed_zero_inf_nan_preserve",
                        )
                    )
                descriptors[key] = feature_bits_value(features)
    assert actual == expected
    for rule in SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT.cases:
        if rule.source_op.name in ("scalar.fmaf", "vector.fmaf"):
            assert rule.descriptor.feature_mask_words == (
                descriptors[rule.descriptor.key],
            )
