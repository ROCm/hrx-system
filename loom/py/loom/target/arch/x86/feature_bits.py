# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dense x86 feature bits derived from the canonical feature catalog."""

from __future__ import annotations

from loom.target.arch.x86.feature_catalog import X86_FEATURE_INFOS

_FEATURE_BITS = {
    info.spelling: 1 << ordinal for ordinal, info in enumerate(X86_FEATURE_INFOS)
}

FEATURE_AVX512_VNNI = _FEATURE_BITS["avx512vnni"]
FEATURE_AVX512_VL = _FEATURE_BITS["avx512vl"]
FEATURE_AVX_VNNI = _FEATURE_BITS["avxvnni"]
FEATURE_AVX_VNNI_INT8 = _FEATURE_BITS["avxvnniint8"]
FEATURE_AVX_VNNI_INT16 = _FEATURE_BITS["avxvnniint16"]
FEATURE_AVX10_2 = _FEATURE_BITS["avx10.2"]
FEATURE_AVX512_BF16 = _FEATURE_BITS["avx512bf16"]
FEATURE_AVX512_FP16 = _FEATURE_BITS["avx512fp16"]
FEATURE_AVX_NE_CONVERT = _FEATURE_BITS["avxneconvert"]
FEATURE_F16C = _FEATURE_BITS["f16c"]
FEATURE_BMI2 = _FEATURE_BITS["bmi2"]
FEATURE_BMI = _FEATURE_BITS["bmi"]
FEATURE_POPCNT = _FEATURE_BITS["popcnt"]
FEATURE_LZCNT = _FEATURE_BITS["lzcnt"]

X86_KNOWN_FEATURE_BITS = (1 << len(X86_FEATURE_INFOS)) - 1
