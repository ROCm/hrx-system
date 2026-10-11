# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 AVX512 source-to-low contract fragment."""

from __future__ import annotations

from collections.abc import Sequence

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.target.arch.x86.contracts.avx512_predicate import (
    avx512_predicate_rules,
)
from loom.target.arch.x86.contracts.floating_extrema import (
    avx512_float_extrema_rules,
)
from loom.target.arch.x86.contracts.floating_reduction import (
    avx512_float_dot_rules,
    avx512_float_reduction_rules,
)
from loom.target.arch.x86.contracts.integer_float import avx512_integer_float_rules
from loom.target.arch.x86.contracts.integer_reduction import (
    avx512_integer_reduction_rules,
)
from loom.target.arch.x86.contracts.lane_movement import (
    avx512_lane_movement_rules,
)
from loom.target.arch.x86.contracts.memory import x86_vector_memory_rules
from loom.target.arch.x86.contracts.shuffle import avx512_shuffle_rules
from loom.target.arch.x86.contracts.vector_arithmetic import (
    avx512_vector_arithmetic_rules,
)
from loom.target.arch.x86.contracts.vector_construction import (
    avx512_vector_construction_rules,
)
from loom.target.arch.x86.descriptors import X86_AVX512_CORE_DESCRIPTOR_SET
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
    subject_name="x86-avx512",
    constraint_key="x86.avx512.source_memory",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_AVX512_CORE_DESCRIPTOR_SET, key)


def _memory_rules() -> tuple[DescriptorRule, ...]:
    return x86_vector_memory_rules(
        _descriptor,
        descriptor_key_prefix="x86.avx512",
        vector_bit_widths=(512,),
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
    )


def _cases() -> Sequence[ContractCase]:
    return (
        *avx512_integer_float_rules(_descriptor),
        *avx512_vector_construction_rules(_descriptor),
        *avx512_predicate_rules(_descriptor),
        *avx512_vector_arithmetic_rules(_descriptor),
        *avx512_float_extrema_rules(_descriptor),
        *avx512_integer_reduction_rules(_descriptor),
        *avx512_float_reduction_rules(_descriptor),
        *avx512_float_dot_rules(_descriptor),
        *avx512_lane_movement_rules(_descriptor),
        *avx512_shuffle_rules(_descriptor),
        *_memory_rules(),
    )


X86_AVX512_CONTRACT_DIALECT_OPS = {
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
}

X86_AVX512_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.avx512",
    descriptor_set=X86_AVX512_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/avx512.h",
    cases=_cases(),
)
