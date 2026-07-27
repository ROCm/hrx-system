# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared cache-policy vocabulary for Loom memory operations."""

from loom.dsl import EnumCase, EnumDef

CacheTemporal = EnumDef(
    "CacheTemporal",
    [
        EnumCase("regular", 0, doc="Regular temporal caching behavior."),
        EnumCase(
            "non_temporal",
            1,
            doc="Data is expected to have little or no temporal reuse.",
        ),
        EnumCase(
            "high_temporal",
            2,
            doc="Data is expected to be reused and should be retained when possible.",
        ),
        EnumCase(
            "last_use",
            3,
            doc="Data is not expected to be used again after this memory operation.",
        ),
        EnumCase(
            "writeback",
            4,
            doc="Store-oriented high-temporal write-back policy.",
        ),
        EnumCase(
            "non_temporal_regular",
            5,
            doc="Non-temporal near-cache behavior with regular outer-cache behavior.",
        ),
        EnumCase(
            "regular_non_temporal",
            6,
            doc="Regular near-cache behavior with non-temporal outer-cache behavior.",
        ),
        EnumCase(
            "non_temporal_high_temporal",
            7,
            doc="Non-temporal near-cache behavior with high-temporal outer-cache behavior.",
        ),
        EnumCase(
            "non_temporal_writeback",
            8,
            doc="Store-oriented non-temporal near-cache behavior with write-back outer-cache behavior.",
        ),
        EnumCase(
            "bypass",
            9,
            doc="Bypass caches at the requested cache scope when the target supports it.",
        ),
    ],
    doc="Target-independent temporal cache policy for memory operations.",
    c_type="loom_cache_temporal_t",
    c_const_prefix="LOOM_CACHE_TEMPORAL",
    c_include="loom/ops/cache.h",
)

CacheScope = EnumDef(
    "CacheScope",
    [
        EnumCase("cu", 0, doc="Cache/coherency scope is the compute unit."),
        EnumCase("se", 1, doc="Cache/coherency scope is the shader engine."),
        EnumCase("device", 2, doc="Cache/coherency scope is the current device."),
        EnumCase("system", 3, doc="Cache/coherency scope is the full system."),
    ],
    doc="Target-independent cache scope for memory operations.",
    c_type="loom_cache_scope_t",
    c_const_prefix="LOOM_CACHE_SCOPE",
    c_include="loom/ops/cache.h",
)
