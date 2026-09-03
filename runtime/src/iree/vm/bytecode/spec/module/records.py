# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core VM module wire records selected for the projection spine."""

from __future__ import annotations

from iree.vm.bytecode.spec.module import (
    AllowedRangeRule,
    ExactBytesRule,
    FieldRule,
    RecordRule,
    WireField,
    WireRecord,
)
from iree.vm.bytecode.spec.schema import U8, U16, U32, Field
from iree.vm.bytecode.spec.version import CORE_0


def _field(name: str, encoding, summary: str, rule, *, element_count: int = 1):
    return WireField(Field(name, encoding, summary, element_count), rule)


IMAGE_HEADER = WireRecord(
    name="image_header",
    c_type="iree_vm_bytecode_v0_image_header_t",
    since=CORE_0,
    summary="Identifies a VM image and its required Core version.",
    contract=(
        "This fixed prefix is read before any section-dependent allocation or pointer "
        "formation. The exact magic rejects unrelated inputs. The major version must "
        "match the loader, the required minor must not exceed loader support, and the "
        "section count sizes the immediately following directory."
    ),
    fields=(
        _field(
            "magic_u8",
            U8,
            "Exact eight-byte IREE VM image magic.",
            ExactBytesRule(b"IREEVM\x00\x00"),
            element_count=8,
        ),
        _field(
            "core_major_u16",
            U16,
            "Incompatible Core container and ISA version.",
            FieldRule.CORE_MAJOR,
        ),
        _field(
            "core_required_minor_u16",
            U16,
            "Minimum compatible Core minor version.",
            FieldRule.CORE_REQUIRED_MINOR,
        ),
        _field(
            "section_count_u16",
            U16,
            "Number of section-directory rows.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "zero_padding_u16",
            U16,
            "Canonical zero padding.",
            FieldRule.ZERO,
        ),
    ),
)

SIGNATURES_HEADER = WireRecord(
    name="signatures_header",
    c_type="iree_vm_bytecode_v0_signatures_header_t",
    since=CORE_0,
    summary="Counts source-ordered logical function signatures.",
    contract=(
        "The count sizes the signature-row array. Signature ordinal order is stable "
        "within the module and is used directly by functions and callable types."
    ),
    fields=(
        _field(
            "signature_count_u32",
            U32,
            "Number of source-ordered logical signatures.",
            AllowedRangeRule(1, 65536),
        ),
    ),
)

SIGNATURE_ROW = WireRecord(
    name="signature_row",
    c_type="iree_vm_bytecode_v0_signature_row_t",
    since=CORE_0,
    summary="Locates and partitions one logical signature's descriptors.",
    contract=(
        "descriptor_base_u32 is the canonical running base in the section-wide "
        "descriptor array. The six counts partition arguments and results by physical "
        "value, ref, and function carriers while preserving logical descriptor order."
    ),
    fields=(
        _field(
            "descriptor_base_u32",
            U32,
            "Canonical running base in the descriptor array.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "argument_value_count_u16",
            U16,
            "Value argument count.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "result_value_count_u16",
            U16,
            "Value result count.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "argument_ref_count_u16",
            U16,
            "Ref argument count.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "result_ref_count_u16",
            U16,
            "Ref result count.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "argument_function_count_u16",
            U16,
            "Function argument count.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "result_function_count_u16",
            U16,
            "Function result count.",
            FieldRule.ANY_BITS,
        ),
    ),
)

SIGNATURE_DESCRIPTOR_ROW = WireRecord(
    name="signature_descriptor_row",
    c_type="iree_vm_bytecode_v0_signature_descriptor_row_t",
    since=CORE_0,
    summary="Declares one logical signature field's kind and exact type.",
    contract=(
        "Scalar kinds require a zero type ordinal. Ref and function kinds require an "
        "in-range module-local type ordinal from their respective type table."
    ),
    fields=(
        _field(
            "kind_u16",
            U16,
            "Architectural scalar, ref, or function kind.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "type_ordinal_u16",
            U16,
            "Exact ref or callable type ordinal, or zero for scalars.",
            FieldRule.ANY_BITS,
        ),
    ),
    rules=(RecordRule.SIGNATURE_DESCRIPTOR,),
)

RECORDS = (
    IMAGE_HEADER,
    SIGNATURES_HEADER,
    SIGNATURE_ROW,
    SIGNATURE_DESCRIPTOR_ROW,
)
