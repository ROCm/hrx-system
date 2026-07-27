# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU matrix source-to-low contracts."""

from __future__ import annotations

from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.target.arch.amdgpu.descriptors import build_amdgpu_contract_descriptor_set
from loom.target.contracts import ContractFragment, DescriptorMatrixRule

_DESCRIPTOR_SET = build_amdgpu_contract_descriptor_set(
    key="amdgpu.matrix",
    descriptor_keys=(),
)

AMDGPU_MATRIX_CONTRACT_FRAGMENT = ContractFragment(
    name="amdgpu.matrix",
    descriptor_set=_DESCRIPTOR_SET,
    public_header="loom/target/arch/amdgpu/contracts/matrix.h",
    cases=[
        DescriptorMatrixRule(
            source_op=vector.vector_mma,
            source="vector_mma",
        ),
    ],
)

AMDGPU_MATRIX_CONTRACT_DIALECT_OPS = {
    "vector": ALL_VECTOR_OPS,
}
