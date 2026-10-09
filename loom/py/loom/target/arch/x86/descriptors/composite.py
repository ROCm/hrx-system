# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Composite x86 descriptor views."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import Descriptor, DescriptorSet

from .avx2 import X86_AVX2_DESCRIPTOR_SET
from .avx512 import X86_AVX512_CORE_DESCRIPTOR_SET
from .avx512_bf16 import X86_AVX512_BF16_DESCRIPTOR_SET
from .avx512_fp16 import X86_AVX512_FP16_DESCRIPTOR_SET
from .avx_ne_convert import X86_AVX_NE_CONVERT_DESCRIPTOR_SET
from .common import _merge_named_items, _qualify_packed_dot_descriptor_asm_forms
from .packed_dot import (
    X86_AVX10_2_DESCRIPTOR_SET,
    X86_AVX512_VNNI_DESCRIPTOR_SET,
    X86_AVX_VNNI_DESCRIPTOR_SET,
    X86_AVX_VNNI_INT8_DESCRIPTOR_SET,
    X86_AVX_VNNI_INT16_DESCRIPTOR_SET,
)

_X86_DESCRIPTOR_SET_COMPONENTS = tuple[tuple[DescriptorSet, frozenset[str]], ...]


_X86_AVX512_FEATURE_COMPONENTS: _X86_DESCRIPTOR_SET_COMPONENTS = (
    (X86_AVX512_CORE_DESCRIPTOR_SET, frozenset()),
    (X86_AVX_NE_CONVERT_DESCRIPTOR_SET, frozenset()),
    (X86_AVX512_FP16_DESCRIPTOR_SET, frozenset()),
    (X86_AVX512_VNNI_DESCRIPTOR_SET, frozenset()),
    (X86_AVX512_BF16_DESCRIPTOR_SET, frozenset()),
    (X86_AVX_VNNI_DESCRIPTOR_SET, frozenset()),
    (X86_AVX_VNNI_INT8_DESCRIPTOR_SET, frozenset()),
    (X86_AVX_VNNI_INT16_DESCRIPTOR_SET, frozenset()),
    (X86_AVX10_2_DESCRIPTOR_SET, frozenset()),
)

_X86_AVX2_FEATURE_COMPONENTS: _X86_DESCRIPTOR_SET_COMPONENTS = (
    (X86_AVX2_DESCRIPTOR_SET, frozenset()),
    (X86_AVX_NE_CONVERT_DESCRIPTOR_SET, frozenset()),
    (X86_AVX_VNNI_DESCRIPTOR_SET, frozenset()),
    (X86_AVX_VNNI_INT8_DESCRIPTOR_SET, frozenset()),
    (X86_AVX_VNNI_INT16_DESCRIPTOR_SET, frozenset()),
)


def _merge_component_descriptors(
    components: tuple[tuple[DescriptorSet, frozenset[str]], ...],
) -> tuple[Descriptor, ...]:
    descriptors = []
    seen_keys: set[str] = set()
    for descriptor_set, excluded_keys in components:
        for descriptor in descriptor_set.descriptors:
            if descriptor.key in excluded_keys:
                continue
            if descriptor.key in seen_keys:
                raise ValueError(
                    "x86 descriptor set component repeats descriptor "
                    f"'{descriptor.key}'"
                )
            descriptors.append(_qualify_packed_dot_descriptor_asm_forms(descriptor))
            seen_keys.add(descriptor.key)
    return tuple(descriptors)


def _composite_descriptor_set(
    *,
    key: str,
    feature_key: str,
    stem: str,
    function_name: str,
    c_table_prefix: str,
    c_enum_prefix: str,
    components: _X86_DESCRIPTOR_SET_COMPONENTS,
) -> DescriptorSet:
    core = components[0][0]
    return DescriptorSet(
        key=key,
        target_key="x86",
        feature_key=feature_key,
        c_header_path=Path(
            f"loom/src/loom/target/arch/x86/descriptors/{stem}_descriptors.h"
        ),
        c_source_path=Path(f"loom/src/loom/target/arch/x86/{stem}_descriptors.c"),
        header_guard=f"LOOM_TARGET_ARCH_X86_{stem.upper()}_DESCRIPTORS_H_",
        public_header=f"loom/target/arch/x86/descriptors/{stem}_descriptors.h",
        function_name=function_name,
        c_table_prefix=c_table_prefix,
        c_enum_prefix=c_enum_prefix,
        generator_version=1,
        supports_native_scheduling=True,
        physical_registers=core.physical_registers,
        reg_classes=_merge_named_items(
            tuple(descriptor_set.reg_classes for descriptor_set, _ in components)
        ),
        resources=_merge_named_items(
            tuple(descriptor_set.resources for descriptor_set, _ in components)
        ),
        schedule_classes=_merge_named_items(
            tuple(descriptor_set.schedule_classes for descriptor_set, _ in components)
        ),
        enum_domains=_merge_named_items(
            tuple(descriptor_set.enum_domains for descriptor_set, _ in components)
        ),
        descriptors=_merge_component_descriptors(components),
    )


X86_AVX2_FEATURES_DESCRIPTOR_SET = _composite_descriptor_set(
    key="x86.avx2_features.core",
    feature_key="x86.avx2_features.v1",
    stem="avx2_features",
    function_name="loom_x86_avx2_features_core_descriptor_set",
    c_table_prefix="X86Avx2FeaturesCore",
    c_enum_prefix="X86_AVX2_FEATURES_CORE",
    components=_X86_AVX2_FEATURE_COMPONENTS,
)

X86_AVX512_FEATURES_DESCRIPTOR_SET = _composite_descriptor_set(
    key="x86.avx512_features.core",
    feature_key="x86.avx512_features.v1",
    stem="avx512_features",
    function_name="loom_x86_avx512_features_core_descriptor_set",
    c_table_prefix="X86Avx512FeaturesCore",
    c_enum_prefix="X86_AVX512_FEATURES_CORE",
    components=_X86_AVX512_FEATURE_COMPONENTS,
)
