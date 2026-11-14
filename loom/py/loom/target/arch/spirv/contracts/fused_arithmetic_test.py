# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.contracts.logical_core import (
    SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT,
)
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
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
        rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
        assert rule.emit[0].descriptor is rule.descriptor
        assert rule.emit[0].operands == {
            name: ValueRef.operand(name) for name in ("a", "b", "c")
        }
        if "nnan" in permissions:
            assert len(rule.emit) == 1
            assert rule.emit[0].results == {"dst": ValueRef.result("result")}
        else:
            pattern = type_guards["result"]
            element = pattern.elements[0]
            lanes = pattern.lanes if pattern.kind == "vector" else 1
            carrier = element if lanes == 1 else f"v{lanes}{element}"
            assert len(rule.emit) == (4 if lanes == 1 else 5)
            fused = rule.emit[0].results["dst"]
            classify, literal = rule.emit[1:3]
            assert classify.descriptor.key == f"spirv.op_f_unord_not_equal.{carrier}"
            assert classify.operands == {"lhs": fused, "rhs": fused}
            assert literal.immediates == {
                f"{element}_bits": {
                    "f16": 0x7E00,
                    "f32": 0x7FC00000,
                    "f64": 0x7FF8000000000000,
                }[element]
            }
            quiet = literal.results["dst"]
            if lanes != 1:
                broadcast = rule.emit[3]
                assert broadcast.operands == {
                    f"component{lane}": quiet for lane in range(lanes)
                }
                quiet = broadcast.results["dst"]
            assert rule.emit[-1].descriptor.key == f"spirv.op_select.{carrier}"
            assert rule.emit[-1].operands == {
                "condition": classify.results["dst"],
                "true_value": quiet,
                "false_value": fused,
            }
            assert rule.emit[-1].results == {"dst": ValueRef.result("result")}
        # Result normalization must not add optional integer-width requirements.
        for step in rule.emit:
            assert len(step.descriptor.feature_mask_words) <= len(
                rule.descriptor.feature_mask_words
            )
            assert not any(
                required & ~rule.descriptor.feature_mask_words[index]
                for index, required in enumerate(step.descriptor.feature_mask_words)
            )

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
