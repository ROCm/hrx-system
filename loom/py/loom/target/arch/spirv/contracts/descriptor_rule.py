# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builders for rules using the SPIR-V logical core descriptor set."""

from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    DescriptorEmitForm,
    EmitDescriptorOp,
    Guard,
    ResultTypeBinding,
    SourceMemoryAddressMaterializer,
    SourceMemoryConstraint,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor


def logical_core_descriptor(key: str) -> Descriptor:
    return descriptor_by_key(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET, key)


def descriptor_feature_guards(descriptor: Descriptor) -> tuple[Guard, ...]:
    return (
        (Guard.descriptor_available(descriptor),)
        if descriptor.feature_mask_words
        else ()
    )


def emit_descriptor_op(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, ResultTypeBinding] | None = None,
    immediates: dict[str, AttrProject] | None = None,
    source_memory: SourceMemoryConstraint | None = None,
    source_memory_address_materializer: SourceMemoryAddressMaterializer | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        form=DescriptorEmitForm.OP,
        source_memory=source_memory,
        source_memory_address_materializer=source_memory_address_materializer,
    )
