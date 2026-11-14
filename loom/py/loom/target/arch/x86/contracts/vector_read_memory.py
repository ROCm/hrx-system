# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 lowering-only vector consumers of scalar memory reads."""

from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.target.arch.x86.contracts.memory import x86_fused_load_rules
from loom.target.arch.x86.descriptors import X86_AVX512_CORE_DESCRIPTOR_SET
from loom.target.arch.x86.vector_families import X86_VECTOR_STORAGE_FAMILIES
from loom.target.contracts import (
    ContractFragment,
    DescriptorRule,
    Guard,
    GuardDiagnostic,
    Scalar,
    SourceNode,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="x86-vector",
    constraint_key="x86.vector.source_memory",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_AVX512_CORE_DESCRIPTOR_SET, key)


def _broadcast_rules() -> tuple[DescriptorRule, ...]:
    rules = []
    for family in X86_VECTOR_STORAGE_FAMILIES:
        scalar_type = Scalar(family.element_names)
        mnemonic = {
            8: "vpbroadcastb",
            16: "vpbroadcastw",
            32: "vpbroadcastd",
            64: "vpbroadcastq",
        }[family.element_bit_width]
        for profile, suffixes, priority in (
            ("avx2", ("xmm", "ymm"), 1),
            ("avx512", ("xmm", "ymm", "zmm"), 2),
        ):
            for suffix in suffixes:
                carrier_bits = {"xmm": 128, "ymm": 256, "zmm": 512}[suffix]
                vector_type = Vector(
                    family.element_names,
                    minimum_lanes=1,
                    maximum_lanes=carrier_bits // family.element_bit_width,
                )
                consumer = SourceNode.exclusive_user(
                    "splat",
                    source_op=vector.vector_splat,
                    parent_result=ValueRef.result("result"),
                    node_operand=ValueRef.operand("scalar"),
                    guards=(
                        Guard.value_type("scalar", scalar_type),
                        Guard.value_type("result", vector_type),
                        Guard.low_value_register_class("result", f"x86.{suffix}"),
                    ),
                )
                rules.extend(
                    x86_fused_load_rules(
                        _descriptor,
                        source_op=view.view_load,
                        source_type=scalar_type,
                        source_nodes=(consumer,),
                        result=ValueRef.result("result", source_node="splat"),
                        element_byte_count=family.element_bit_width // 8,
                        lane_count=1,
                        descriptor_key_prefix=f"x86.{profile}.{mnemonic}",
                        register_suffix=suffix,
                        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
                        report_key=f"native_memory_broadcast_{family.element_bit_width}_{suffix}",
                        # EVEX avoids restricting the destination to the lower
                        # sixteen SIMD registers when both forms are available.
                        priority=priority,
                    )
                )
    return tuple(rules)


X86_VECTOR_READ_MEMORY_CONTRACT_DIALECT_OPS = {
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

X86_VECTOR_READ_MEMORY_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.vector.read_memory",
    descriptor_set=X86_AVX512_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/vector_read_memory.h",
    cases=_broadcast_rules(),
    # Both operations are independently legal. Their optional joint realization
    # consumes the retained source graph without speculative legalization work.
    target_contract_query=False,
)
