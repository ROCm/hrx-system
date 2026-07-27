# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Composite x86 descriptor views."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import Descriptor, DescriptorSet

from .avx512 import X86_AVX512_CORE_DESCRIPTOR_SET
from .common import _T, _qualify_packed_dot_descriptor_asm_forms
from .packed_dot import X86_PACKED_DOT_FEATURE_DESCRIPTOR_SETS

_X86_DESCRIPTOR_SET_COMPONENTS = tuple[tuple[DescriptorSet, frozenset[str]], ...]


_X86_AVX512_PACKED_DOT_COMPONENTS: _X86_DESCRIPTOR_SET_COMPONENTS = (
    (
        X86_AVX512_CORE_DESCRIPTOR_SET,
        frozenset(
            (
                "x86.avx512.vpdpbusd.zmm",
                "x86.avx512.vdpbf16ps.zmm",
            )
        ),
    ),
    *(
        (descriptor_set, frozenset())
        for descriptor_set in X86_PACKED_DOT_FEATURE_DESCRIPTOR_SETS
    ),
)


def _merge_named_items(item_groups: tuple[tuple[_T, ...], ...]) -> tuple[_T, ...]:
    merged_items: list[_T] = []
    seen_names: set[str] = set()
    for items in item_groups:
        for item in items:
            if item.name in seen_names:
                continue
            merged_items.append(item)
            seen_names.add(item.name)
    return tuple(merged_items)


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


X86_AVX512_PACKED_DOT_DESCRIPTOR_SET = DescriptorSet(
    key="x86.avx512_packed_dot.core",
    target_key="x86",
    feature_key="x86.avx512_packed_dot.v1",
    c_header_path=Path(
        "loom/src/loom/target/arch/x86/descriptors/avx512_packed_dot_descriptors.h"
    ),
    c_source_path=Path("loom/src/loom/target/arch/x86/avx512_packed_dot_descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_X86_AVX512_PACKED_DOT_DESCRIPTORS_H_",
    public_header="loom/target/arch/x86/descriptors/avx512_packed_dot_descriptors.h",
    function_name="loom_x86_avx512_packed_dot_core_descriptor_set",
    c_table_prefix="X86Avx512PackedDotCore",
    c_enum_prefix="X86_AVX512_PACKED_DOT_CORE",
    generator_version=1,
    reg_classes=_merge_named_items(
        tuple(
            descriptor_set.reg_classes
            for descriptor_set, _ in _X86_AVX512_PACKED_DOT_COMPONENTS
        )
    ),
    resources=_merge_named_items(
        tuple(
            descriptor_set.resources
            for descriptor_set, _ in _X86_AVX512_PACKED_DOT_COMPONENTS
        )
    ),
    schedule_classes=_merge_named_items(
        tuple(
            descriptor_set.schedule_classes
            for descriptor_set, _ in _X86_AVX512_PACKED_DOT_COMPONENTS
        )
    ),
    enum_domains=_merge_named_items(
        tuple(
            descriptor_set.enum_domains
            for descriptor_set, _ in _X86_AVX512_PACKED_DOT_COMPONENTS
        )
    ),
    descriptors=_merge_component_descriptors(_X86_AVX512_PACKED_DOT_COMPONENTS),
)
