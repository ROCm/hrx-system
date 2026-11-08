# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native AIE2P stack state and frame-relative memory transfers."""

from loom.target.arch.amd.xdna.aie2p.core_descriptor_spec import _DescriptorSpec
from loom.target.low_descriptors import DescriptorFlag

STACK_DESCRIPTOR_SPECS = (
    _DescriptorSpec(
        "MOVXM",
        "amd.xdna.aie2p.state.stack-pointer",
        "state.write.stack-pointer",
        "II_MOVXM_mSPm",
        storage_overrides=(("dst", "mSPm"),),
        implicit_outputs=("dst",),
        asm_mnemonic="set.stack-pointer",
        flags=(DescriptorFlag.STATE_ASSIGNMENT,),
    ),
    _DescriptorSpec(
        "PADDXM_pstm_sp_imm",
        "amd.xdna.aie2p.stack.adjust",
        "stack.adjust",
        "II_PADDXM_pstm_sp_imm",
        asm_mnemonic="adjust.stack-pointer",
    ),
)
