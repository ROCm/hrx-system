# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tied native loads preserve the destination in inactive SIMD lanes."""

from dataclasses import replace

from loom.target.low_descriptors import (
    AsmForm,
    Constraint,
    ConstraintKind,
    Descriptor,
    OperandFlag,
    OperandRole,
)

from .common import _native_operand, _native_result
from .memory_candidates import (
    preserved_memory_descriptor_key,
    preserving_memory_descriptor_keys,
)


def _preserving_asm_form(descriptor: Descriptor, form: AsmForm) -> AsmForm:
    mnemonic = form.mnemonic or descriptor.mnemonic
    assert mnemonic is not None, descriptor.key
    # Explicit native operands keep the old ISA spelling and omit the implicit
    # fallback. Native forms without their own immediates use the existing
    # memory suffix printer, just as the original descriptor did.
    native_values = form.native_assembly_values or (
        *(_native_result(name) for name in form.results),
        *(_native_operand(name) for name in form.operands),
    )
    assert not form.operand_segments, descriptor.key
    return replace(
        form,
        mnemonic=f"{mnemonic}_preserve",
        native_assembly_mnemonic=form.native_assembly_mnemonic or mnemonic,
        operands=("fallback", *form.operands),
        native_assembly_values=native_values,
    )


def with_preserving_memory_loads(
    descriptors: tuple[Descriptor, ...],
) -> tuple[Descriptor, ...]:
    forms = {descriptor.key: descriptor.operand_forms for descriptor in descriptors}
    selected = preserving_memory_descriptor_keys(forms)
    variants = []
    for descriptor in descriptors:
        if descriptor.key not in selected:
            continue
        result = descriptor.operands[0]
        assert result.role is OperandRole.RESULT, descriptor.key
        assert (
            sum(operand.role is OperandRole.RESULT for operand in descriptor.operands)
            == 1
        ), descriptor.key
        fallback = replace(
            result,
            field_name="fallback",
            role=OperandRole.OPERAND,
            flags=(OperandFlag.IMPLICIT,),
        )
        constraints = tuple(
            replace(
                constraint,
                lhs_operand_index=constraint.lhs_operand_index
                + (constraint.lhs_operand_index > 0),
                rhs_operand_index=(
                    None
                    if constraint.rhs_operand_index is None
                    else constraint.rhs_operand_index
                    + (constraint.rhs_operand_index > 0)
                ),
            )
            for constraint in descriptor.constraints
        )
        variants.append(
            replace(
                descriptor,
                key=preserved_memory_descriptor_key(descriptor.key),
                mnemonic=f"{descriptor.mnemonic}_preserve",
                semantic_tag=f"{descriptor.semantic_tag}.preserve_inactive",
                operands=(result, fallback, *descriptor.operands[1:]),
                constraints=(*constraints, Constraint(ConstraintKind.TIED, 0, 1)),
                operand_forms=tuple(
                    replace(
                        form,
                        replacement_descriptor=preserved_memory_descriptor_key(
                            form.replacement_descriptor
                        ),
                    )
                    for form in descriptor.operand_forms
                ),
                asm_forms=tuple(
                    _preserving_asm_form(descriptor, form)
                    for form in descriptor.asm_forms
                ),
            )
        )
    return (*descriptors, *variants)
