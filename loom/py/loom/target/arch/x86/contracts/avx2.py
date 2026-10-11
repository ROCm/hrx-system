# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 AVX2 source-to-low contract fragment."""

from __future__ import annotations

from collections.abc import Sequence

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.view import ALL_VIEW_OPS
from loom.target.arch.x86.contracts.float_narrowing import (
    x86_float_narrowing_rules,
)
from loom.target.arch.x86.contracts.floating_extrema import (
    avx2_float_extrema_rules,
)
from loom.target.arch.x86.contracts.floating_reduction import (
    avx2_float_dot_rules,
    avx2_float_reduction_rules,
)
from loom.target.arch.x86.contracts.integer_reduction import (
    avx2_integer_reduction_rules,
)
from loom.target.arch.x86.contracts.lane_movement import (
    avx2_lane_movement_rules,
)
from loom.target.arch.x86.contracts.memory import (
    x86_scalar_float_memory_rules,
    x86_vector_memory_rules,
)
from loom.target.arch.x86.contracts.predicate import avx2_predicate_rules
from loom.target.arch.x86.contracts.scalar_float import avx2_scalar_float_rules
from loom.target.arch.x86.contracts.shuffle import avx2_shuffle_rules
from loom.target.arch.x86.contracts.vector_arithmetic import (
    avx2_vector_arithmetic_rules,
)
from loom.target.arch.x86.contracts.vector_construction import (
    avx2_vector_construction_rules,
)
from loom.target.arch.x86.descriptors import X86_AVX2_DESCRIPTOR_SET
from loom.target.contracts import (
    ContractCase,
    ContractFragment,
    DescriptorRule,
    GuardDiagnostic,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="x86-avx2",
    constraint_key="x86.avx2.source_memory",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_AVX2_DESCRIPTOR_SET, key)


def _memory_rules() -> tuple[DescriptorRule, ...]:
    return (
        *x86_scalar_float_memory_rules(
            _descriptor,
            diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
        ),
        *x86_vector_memory_rules(
            _descriptor,
            descriptor_key_prefix="x86.avx2",
            vector_bit_widths=(128, 256),
            diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
        ),
    )


def _cases() -> Sequence[ContractCase]:
    return (
        *avx2_scalar_float_rules(_descriptor),
        *x86_float_narrowing_rules(_descriptor),
        *avx2_vector_construction_rules(_descriptor),
        *avx2_predicate_rules(_descriptor),
        *avx2_lane_movement_rules(_descriptor),
        *avx2_shuffle_rules(_descriptor),
        *avx2_vector_arithmetic_rules(_descriptor),
        *avx2_float_extrema_rules(_descriptor),
        *_memory_rules(),
        *avx2_integer_reduction_rules(_descriptor),
        *avx2_float_reduction_rules(_descriptor),
        *avx2_float_dot_rules(_descriptor),
    )


X86_AVX2_CONTRACT_DIALECT_OPS = {
    "scalar": ALL_SCALAR_OPS,
    "scf": ALL_SCF_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

X86_AVX2_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.avx2",
    descriptor_set=X86_AVX2_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/avx2.h",
    cases=_cases(),
)
