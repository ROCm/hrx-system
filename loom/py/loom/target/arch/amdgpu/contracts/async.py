# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU source-level async sequencing contracts."""

from __future__ import annotations

from loom.dialect.kernel import ALL_KERNEL_OPS
from loom.dialect.kernel import defs as kernel
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import ContractFragment, ValueElideRule, ValueRef

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.async",
    descriptor_keys=(),
)

AMDGPU_ASYNC_CONTRACT_DIALECT_OPS = {
    "kernel": ALL_KERNEL_OPS,
}

AMDGPU_ASYNC_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.async",
    descriptor_set=_DESCRIPTOR_SET,
    target_contract_query=False,
    cases=(
        ValueElideRule(
            source_op=kernel.kernel_async_group,
            values=(ValueRef.result("group"),),
        ),
    ),
)
