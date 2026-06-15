# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for SPIR-V shader built-in coordinate loads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BuiltinDimension:
    source_keyword: str
    component_index: int


@dataclass(frozen=True, slots=True)
class BuiltinIndexQuery:
    source_op_key: str
    descriptor_suffix: str
    mnemonic_suffix: str
    builtin_enum: str


BUILTIN_DIMENSIONS = (
    BuiltinDimension("x", 0),
    BuiltinDimension("y", 1),
    BuiltinDimension("z", 2),
)

BUILTIN_INDEX_QUERIES = (
    BuiltinIndexQuery(
        source_op_key="workgroup_id",
        descriptor_suffix="workgroup_id",
        mnemonic_suffix="workgroup_id",
        builtin_enum="LOOM_SPIRV_BUILT_IN_WORKGROUP_ID",
    ),
    BuiltinIndexQuery(
        source_op_key="workitem_id",
        descriptor_suffix="workitem_id",
        mnemonic_suffix="local_invocation_id",
        builtin_enum="LOOM_SPIRV_BUILT_IN_LOCAL_INVOCATION_ID",
    ),
    BuiltinIndexQuery(
        source_op_key="workitem_dispatch_id",
        descriptor_suffix="workitem_dispatch_id",
        mnemonic_suffix="global_invocation_id",
        builtin_enum="LOOM_SPIRV_BUILT_IN_GLOBAL_INVOCATION_ID",
    ),
)
