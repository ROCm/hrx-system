# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.x86.contracts.avx2 import X86_AVX2_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.avx512 import X86_AVX512_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.avx512_fp16 import X86_AVX512_FP16_CONTRACT_FRAGMENT
from loom.target.arch.x86.feature_bits import FEATURE_AVX512_FP16, FEATURE_AVX512_VL
from loom.target.contracts import DescriptorRule, GuardKind, Scalar, ValueRef, Vector


def test_rounding_family_covers_types_modes_and_native_carriers() -> None:
    modes = {"ceilf": 10, "floorf": 9, "roundevenf": 8, "truncf": 11}
    for fragment, elements, widths, prefix, stem in (
        (
            X86_AVX2_CONTRACT_FRAGMENT,
            (("f32", 32), ("f64", 64)),
            (32, 64, 128, 256),
            "x86.avx2",
            "vround",
        ),
        (
            X86_AVX512_CONTRACT_FRAGMENT,
            (("f32", 32), ("f64", 64)),
            (32, 64, 128, 256, 512),
            "x86.avx512",
            "vrndscale",
        ),
        (
            X86_AVX512_FP16_CONTRACT_FRAGMENT,
            (("f16", 16),),
            (16, 32, 64, 128, 256, 512),
            "x86.avx512_fp16",
            "vrndscale",
        ),
    ):
        actual = set()
        for rule in fragment.cases:
            if not isinstance(rule, DescriptorRule):
                continue
            _domain, operation = rule.source_op.name.split(".", 1)
            if operation not in modes:
                continue
            value_type = next(
                guard.type_pattern
                for guard in rule.guards
                if guard.kind == GuardKind.VALUE_TYPE and guard.field == "result"
            )
            assert not any(
                guard.kind == GuardKind.INSTANCE_FLAGS_HAS_ALL for guard in rule.guards
            )
            assert any(
                guard.kind == GuardKind.DESCRIPTOR_AVAILABLE
                and guard.descriptor == rule.descriptor
                for guard in rule.guards
            )
            assert len(rule.emit) == 1
            emit = rule.emit[0]
            assert emit.immediates == {"rounding": modes[operation]}
            assert emit.operands["input"] == ValueRef.operand("input")
            if rule.descriptor.key.split(".")[-2].startswith(stem + "s"):
                assert emit.operands["passthrough"] == ValueRef.operand("input")
            feature_bits = rule.descriptor.feature_mask_words
            if prefix == "x86.avx512_fp16":
                assert feature_bits[0] & FEATURE_AVX512_FP16
            if prefix != "x86.avx2" and rule.descriptor.key.split(".")[-2].startswith(
                stem + "p"
            ):
                assert bool(feature_bits and feature_bits[0] & FEATURE_AVX512_VL) == (
                    not rule.descriptor.key.endswith(".zmm")
                )
            actual.add((rule.source_op.name, value_type, rule.descriptor.key))
        expected = set()
        for element, bits in elements:
            suffix = {"f16": "h", "f32": "s", "f64": "d"}[element]
            for operation in modes:
                expected.add(
                    (
                        f"scalar.{operation}",
                        Scalar(element),
                        f"{prefix}.{stem}s{suffix}.xmm",
                    )
                )
                for width in widths:
                    if width < bits:
                        continue
                    register = (
                        "zmm" if width == 512 else "ymm" if width == 256 else "xmm"
                    )
                    expected.add(
                        (
                            f"vector.{operation}",
                            Vector(element, lanes=width // bits),
                            f"{prefix}.{stem}{'s' if width == bits else 'p'}{suffix}.{register}",
                        )
                    )
        assert actual == expected
