# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU view source-to-low contracts."""

from __future__ import annotations

from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.target.arch.amdgpu.contracts.memory import BYTE_ADDRESSABLE_SCALAR_ELEMENTS
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractFragment,
    Guard,
    ValueAliasRule,
    ValueRef,
    View,
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.view",
    descriptor_keys=(),
)

AMDGPU_VIEW_CONTRACT_DIALECT_OPS = {
    "view": ALL_VIEW_OPS,
}

AMDGPU_VIEW_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.view",
    descriptor_set=_DESCRIPTOR_SET,
    public_header="loom/target/arch/amdgpu/contracts/view.h",
    cases=(
        ValueAliasRule(
            source_op=view.view_subview,
            source=ValueRef.operand("source"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type(
                    "result",
                    View(BYTE_ADDRESSABLE_SCALAR_ELEMENTS),
                ),
            ),
        ),
    ),
)
