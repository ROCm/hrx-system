# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native SPIR-V ordinary-vector floating-point arithmetic rows."""

from loom.target.arch.spirv.ordinary_vector import (
    ORDINARY_VECTOR_TYPES,
    OrdinaryVectorInstruction,
)
from loom.target.arch.spirv.scalar_alu import (
    FLOAT_BINARY_OPERATIONS,
    FLOAT_SCALAR_ALU_TYPES,
)

_FLOAT_SUFFIXES = frozenset(scalar.suffix for scalar in FLOAT_SCALAR_ALU_TYPES)

ORDINARY_VECTOR_FLOAT_BINARY_INSTRUCTIONS = tuple(
    OrdinaryVectorInstruction(
        key=f"spirv.op_{operation.descriptor_suffix}.{value_type.suffix}",
        mnemonic=f"{operation.mnemonic}.{value_type.suffix}",
        opcode=operation.opcode,
        packet_form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
        result_type=value_type,
        operand_names=("lhs", "rhs"),
        operand_types=(value_type, value_type),
    )
    for value_type in ORDINARY_VECTOR_TYPES
    if value_type.component_type.suffix in _FLOAT_SUFFIXES
    for operation in FLOAT_BINARY_OPERATIONS
)
