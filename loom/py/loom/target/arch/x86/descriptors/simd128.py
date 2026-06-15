# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SIMD128 x86 descriptor rows and view metadata."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from loom.target.low_descriptors import RegClass, RegClassFlag, SpillSlotSpace

from .common import _REG_XMM
from .scalar import X86_SCALAR_DESCRIPTOR_SET

X86_SIMD128_DESCRIPTOR_SET = replace(
    X86_SCALAR_DESCRIPTOR_SET,
    key="x86.simd128.core",
    feature_key="x86.simd128.v1",
    c_header_path=Path(
        "loom/src/loom/target/arch/x86/descriptors/simd128_descriptors.h"
    ),
    c_source_path=Path("loom/src/loom/target/arch/x86/simd128_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_SIMD128_DESCRIPTORS_H_",
    public_header="loom/target/arch/x86/descriptors/simd128_descriptors.h",
    function_name="loom_x86_simd128_core_descriptor_set",
    c_table_prefix="X86Simd128Core",
    c_enum_prefix="X86_SIMD128_CORE",
    reg_classes=(
        *X86_SCALAR_DESCRIPTOR_SET.reg_classes,
        RegClass(
            _REG_XMM,
            128,
            SpillSlotSpace.STACK,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=16,
            alias_set_id=2,
        ),
    ),
)
