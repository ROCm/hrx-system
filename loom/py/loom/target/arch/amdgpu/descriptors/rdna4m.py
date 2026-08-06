# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU RDNA 4m descriptor-set base data."""

from __future__ import annotations

from .common import *
from .rdna3 import _AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE


def _without_matrix_schedule_rows(
    descriptor_set: DescriptorSet, key: str
) -> DescriptorSet:
    return _amdgpu_core_descriptor_set(
        key=key,
        reg_classes=descriptor_set.reg_classes,
        register_parts=descriptor_set.register_parts,
        resources=tuple(
            resource
            for resource in descriptor_set.resources
            if resource.kind is not ResourceKind.MATRIX
        ),
        schedule_classes=tuple(
            schedule_class
            for schedule_class in descriptor_set.schedule_classes
            if not any(
                issue_use.resource == _RESOURCE_WMMA
                for issue_use in schedule_class.issue_uses
            )
        ),
    )


_AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE = _without_matrix_schedule_rows(
    _AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE,
    "amdgpu.rdna4m.core",
)

__all__ = ("_AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE",)
