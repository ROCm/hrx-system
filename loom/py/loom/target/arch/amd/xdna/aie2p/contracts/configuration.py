# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile-time integer operands for physical array configuration."""

from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import conversion as scalar
from loom.target.arch.amd.xdna.aie2p.configuration_descriptors import (
    AIE2P_CONFIGURATION_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)

_CONSTANT = descriptor_by_key(
    AIE2P_CONFIGURATION_DESCRIPTOR_SET, "amd.xdna.aie2p.configuration.constant"
)

# Configuration scalars describe nonnegative extents, addresses, and command
# fields. Source calculations fold before this boundary; no runtime arithmetic
# or floating-point carrier exists in the physical configuration contract.
AIE2P_CONFIGURATION_CONTRACT_FRAGMENT = ContractFragment(
    name="amd.xdna.aie2p.configuration",
    descriptor_set=AIE2P_CONFIGURATION_DESCRIPTOR_SET,
    public_header="loom/target/arch/amd/xdna/aie2p/contracts/configuration.h",
    cases=tuple(
        DescriptorRule(
            source_op=source_op,
            descriptor=_CONSTANT,
            guards=(
                Guard.value_type("result", Scalar(types)),
                Guard.value_exact_i64("result"),
                Guard.value_i64_range("result", 0, 2**63 - 1),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=_CONSTANT,
                    results={"result": ValueRef.result("result")},
                    immediates={"value": ValueProject.exact_i64("result")},
                    form=DescriptorEmitForm.CONST,
                ),
            ),
        )
        for source_op, types in (
            (index.index_constant, ("index", "offset")),
            (scalar.scalar_constant, ("i1", "i8", "i16", "i32", "i64")),
        )
    ),
)

AIE2P_CONFIGURATION_CONTRACT_DIALECT_OPS = {
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
}
