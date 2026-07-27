# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU config source-to-low contracts."""

from __future__ import annotations

from loom.dialect.config import ALL_CONFIG_OPS
from loom.dialect.config import defs as config
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import (
    ContractFragment,
    Guard,
    ValueElideRule,
    ValueRef,
)

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.config",
    descriptor_keys=(),
)

AMDGPU_CONFIG_CONTRACT_DIALECT_OPS = {
    "config": ALL_CONFIG_OPS,
}

AMDGPU_CONFIG_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.config",
    descriptor_set=_DESCRIPTOR_SET,
    public_header="loom/target/arch/amdgpu/contracts/config.h",
    cases=(
        ValueElideRule(
            source_op=config.config_get,
            values=(ValueRef.result("result"),),
            guards=(Guard.value_no_uses("result"),),
        ),
    ),
)
