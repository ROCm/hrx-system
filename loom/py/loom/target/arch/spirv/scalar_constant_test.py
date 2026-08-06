# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.spirv.scalar_constant import FLOAT_CONSTANT_TYPES


def test_float_constant_matrix_covers_every_modeled_spirv_float_type() -> None:
    assert tuple(
        (
            scalar.source_type,
            scalar.scalar_enum,
            scalar.bit_width,
            scalar.literal_word_count,
            scalar.feature_atoms,
        )
        for scalar in FLOAT_CONSTANT_TYPES
    ) == (
        ("f16", "LOOM_SPIRV_SCALAR_TYPE_F16", 16, 1, ("float16",)),
        (
            "bf16",
            "LOOM_SPIRV_SCALAR_TYPE_BF16",
            16,
            1,
            ("bfloat16_type_khr",),
        ),
        ("f32", "LOOM_SPIRV_SCALAR_TYPE_F32", 32, 1, ()),
        ("f64", "LOOM_SPIRV_SCALAR_TYPE_F64", 64, 2, ("float64",)),
    )
