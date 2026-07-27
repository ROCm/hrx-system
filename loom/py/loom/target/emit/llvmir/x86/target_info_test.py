# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.arch.x86.packed_dot_data import (
    FEATURE_AVX512_BF16,
    FEATURE_AVX512_VL,
    FEATURE_AVX_VNNI,
    FEATURE_AVX_VNNI_INT8,
)
from loom.target.emit.llvmir.x86.target_info import (
    x86_llvmir_projection_info_by_profile_key,
    x86_llvmir_target_feature_string,
)


def test_packed_dot_projection_reuses_arch_profile_contract() -> None:
    projection = x86_llvmir_projection_info_by_profile_key("x86.packed_dot")
    target_profile = projection.target_profile

    assert projection.debug_profile_key == "x86_64-packed-dot-object"
    assert target_profile.descriptor_set_key == "x86.packed_dot.core"
    assert target_profile.contract_feature_bits == (
        FEATURE_AVX512_BF16
        | FEATURE_AVX512_VL
        | FEATURE_AVX_VNNI
        | FEATURE_AVX_VNNI_INT8
    )
    assert x86_llvmir_target_feature_string(projection) == (
        "+avx512bf16,+avx512vl,+avxvnni,+avxvnniint8"
    )
