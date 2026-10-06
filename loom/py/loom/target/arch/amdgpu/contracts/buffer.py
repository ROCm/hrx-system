# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU buffer source-to-low contracts."""

from __future__ import annotations

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.globals import ALL_GLOBAL_OPS
from loom.dialect.globals import defs as globals
from loom.error.amdgpu import ERR_AMDGPU_001
from loom.target.arch.amdgpu.contracts.memory import BYTE_ADDRESSABLE_SCALAR_ELEMENTS
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractFragment,
    Guard,
    GuardDiagnostic,
    RecipeRule,
    ValueAliasRule,
    ValueRef,
    View,
    string_param,
    target_diagnostic,
    value_type_param,
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.buffer",
    descriptor_keys=(),
)

_VIEW_TYPE_DIAGNOSTIC = GuardDiagnostic(
    ref=target_diagnostic(
        ERR_AMDGPU_001,
        string_param("field_name", "result"),
        value_type_param("actual_type", "result"),
        string_param("required_storage", "byte-addressable scalar elements"),
    ),
)

AMDGPU_BUFFER_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "global": ALL_GLOBAL_OPS,
}

AMDGPU_BUFFER_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.buffer",
    descriptor_set=_DESCRIPTOR_SET,
    public_header="loom/target/arch/amdgpu/contracts/buffer.h",
    cases=(
        RecipeRule(source_op=globals.global_load),
        ValueAliasRule(
            source_op=buffer.buffer_view,
            source=ValueRef.operand("buffer"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type(
                    "result",
                    View(BYTE_ADDRESSABLE_SCALAR_ELEMENTS),
                    diagnostic=_VIEW_TYPE_DIAGNOSTIC,
                ),
            ),
        ),
    ),
)
