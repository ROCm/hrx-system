# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU descriptor category assignment helpers."""

from __future__ import annotations

from functools import cache

from .common import *


@cache
def _amdgpu_descriptor_category_for_identity(
    key: str, semantic_tag: str | None
) -> DescriptorCategory:
    semantic_tag = semantic_tag or ""
    if semantic_tag.startswith(("matrix.", "dot.")):
        return AMDGPU_MATRIX_DESCRIPTOR_CATEGORY
    if ".atomic." in semantic_tag:
        return AMDGPU_ATOMIC_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith("memory.cache."):
        return AMDGPU_CACHE_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith("memory."):
        return AMDGPU_MEMORY_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith(("control.", "special.")):
        return AMDGPU_CONTROL_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith(("cmp.", "integer.compare.", "float.compare.")):
        return AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith("select."):
        return AMDGPU_COMPARE_SELECT_DESCRIPTOR_CATEGORY
    if semantic_tag.startswith("convert."):
        return AMDGPU_CONVERT_DESCRIPTOR_CATEGORY
    if key.startswith("amdgpu.s_"):
        return AMDGPU_SCALAR_DESCRIPTOR_CATEGORY
    if key.startswith("amdgpu.v_"):
        return AMDGPU_VECTOR_DESCRIPTOR_CATEGORY
    return AMDGPU_MISC_DESCRIPTOR_CATEGORY


def _amdgpu_descriptor_category(descriptor: Descriptor) -> DescriptorCategory:
    return _amdgpu_descriptor_category_for_identity(
        descriptor.key, descriptor.semantic_tag
    )


def _with_amdgpu_descriptor_category(descriptor: Descriptor) -> Descriptor:
    category = _amdgpu_descriptor_category(descriptor)
    if descriptor.category == category:
        return descriptor
    return replace(descriptor, category=category)


def _categorize_amdgpu_descriptors(
    descriptors: tuple[Descriptor, ...],
) -> tuple[Descriptor, ...]:
    return tuple(
        _with_amdgpu_descriptor_category(descriptor) for descriptor in descriptors
    )


def amdgpu_descriptor_category_groups(
    descriptors: tuple[Descriptor, ...],
) -> tuple[tuple[DescriptorCategory, tuple[Descriptor, ...]], ...]:
    """Groups AMDGPU descriptors by stable category while preserving order."""

    grouped: dict[DescriptorCategory, list[Descriptor]] = {
        category: [] for category in AMDGPU_DESCRIPTOR_CATEGORIES
    }
    for descriptor in descriptors:
        category = descriptor.category or _amdgpu_descriptor_category(descriptor)
        grouped[category].append(descriptor)
    return tuple(
        (category, tuple(grouped[category]))
        for category in AMDGPU_DESCRIPTOR_CATEGORIES
        if grouped[category]
    )


__all__ = (
    "_amdgpu_descriptor_category",
    "_categorize_amdgpu_descriptors",
    "amdgpu_descriptor_category_groups",
)
