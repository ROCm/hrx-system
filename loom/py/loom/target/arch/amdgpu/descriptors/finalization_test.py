# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

from loom.target.arch.amdgpu.descriptors.alignment import _with_operand_alignment
from loom.target.arch.amdgpu.descriptors.categories import (
    _categorize_amdgpu_descriptors,
)
from loom.target.arch.amdgpu.descriptors.common import (
    _REG_SGPR,
    _SCHEDULE_VALU,
)
from loom.target.arch.amdgpu.descriptors.finalization import (
    _finalize_amdgpu_descriptors,
    _with_instruction_classes,
)
from loom.target.arch.amdgpu.descriptors.rdna3 import (
    _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
)
from loom.target.arch.amdgpu.descriptors.timing import (
    _with_state_read_before_clobber_timing,
    _with_valu_sgpr_timing,
)
from loom.target.low_descriptors import (
    Descriptor,
    DescriptorFlag,
    InstructionClass,
    Operand,
    OperandFlag,
    OperandRole,
    RegClassAlt,
)


def test_fused_finalization_matches_independent_projections() -> None:
    descriptor = Descriptor(
        key="amdgpu.v_add_u64",
        mnemonic="v_add_u64",
        semantic_tag="integer.add.u64",
        operands=(
            Operand(
                "result",
                OperandRole.RESULT,
                (RegClassAlt(_REG_SGPR),),
                unit_count=2,
            ),
            Operand(
                "state",
                OperandRole.OPERAND,
                (RegClassAlt(_REG_SGPR),),
                flags=(OperandFlag.STATE_READ,),
            ),
        ),
        schedule_class=_SCHEDULE_VALU,
        instruction_classes=(InstructionClass.VECTOR_ALU,),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )
    source = replace(
        _AMDGPU_RDNA3_CORE_DESCRIPTOR_SET_BASE,
        descriptors=(descriptor,),
    )
    categorized = replace(
        source,
        descriptors=_categorize_amdgpu_descriptors(source.descriptors),
    )
    expected = _with_state_read_before_clobber_timing(
        _with_valu_sgpr_timing(
            _with_operand_alignment(_with_instruction_classes(categorized)),
            5,
        )
    )

    assert (
        _finalize_amdgpu_descriptors(source, valu_sgpr_separation_cycles=5) == expected
    )
