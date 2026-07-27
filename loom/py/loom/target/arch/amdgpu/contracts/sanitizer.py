# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU executable sanitizer assertion contracts."""

from __future__ import annotations

from loom.dialect.sanitizer import ALL_SANITIZER_OPS
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import ContractFragment

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.sanitizer",
    descriptor_keys=(),
)

AMDGPU_SANITIZER_CONTRACT_DIALECT_OPS = {
    "sanitizer": ALL_SANITIZER_OPS,
}

AMDGPU_SANITIZER_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.sanitizer",
    descriptor_set=_DESCRIPTOR_SET,
    public_header="loom/target/arch/amdgpu/contracts/sanitizer.h",
    cases=(),
)
