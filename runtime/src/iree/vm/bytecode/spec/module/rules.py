# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core module field verification rule declarations."""

import enum

from iree.vm.bytecode.spec.schema import U16, U32, U64, RuleKind


class OrdinalDomain(enum.IntEnum):
    STRING = 1
    STRING_NONEMPTY = 2
    REF_TYPE = 3
    SIGNATURE = 4
    CALLABLE_TYPE = 5
    FUNCTION = 6


class FieldRule:
    ANY_BITS = RuleKind("any_bits", summary="Any bit pattern.")
    ZERO = RuleKind("zero", summary="Must be zero.")
    ALLOWED_BITS = RuleKind("allowed_bits", value_count=1)
    ALLOWED_RANGE = RuleKind("allowed_range", value_count=2)
    EXACT_BYTES = RuleKind("exact_bytes", data_type=bytes)
    MULTIPLE = RuleKind("multiple", value_count=1)
    BYTE_ALIGNMENT = RuleKind("byte_alignment", value_count=1)
    CORE_MAJOR = RuleKind(
        "core_major", U16, summary="Must equal the loader's Core major version."
    )
    CORE_REQUIRED_MINOR = RuleKind(
        "core_required_minor",
        U16,
        summary="Must not exceed the loader's supported Core minor version.",
    )
    NONCORE_PAGE = RuleKind(
        "noncore_page",
        U16,
        summary="Must be an architectural extension page ID in 0xF0..0xFD.",
    )
    ORDINAL = RuleKind("ordinal", U16, data_type=OrdinalDomain)
    ORDINAL_OR_NULL = RuleKind(
        "ordinal_or_null", U16, value_count=1, data_type=OrdinalDomain
    )
    PAGE_MAJOR = RuleKind(
        "page_major",
        U16,
        summary="Must exactly match the registered extension page major version.",
    )
    PAGE_REQUIRED_MINOR = RuleKind(
        "page_required_minor",
        U16,
        summary="Must not exceed the registered extension page minor version.",
    )
    SECTION_BYTE_LENGTH = RuleKind(
        "section_byte_length",
        U64,
        summary="Must produce an in-range payload extent under checked arithmetic.",
    )
    SECTION_FLAGS = RuleKind(
        "section_flags",
        U16,
        summary="Must satisfy the known- or unknown-section flag contract.",
    )
    SECTION_TYPE = RuleKind(
        "section_type",
        U16,
        summary="Must name a known section or a valid declared extension authority.",
    )
    SIGNATURE_DESCRIPTOR = RuleKind("signature_descriptor", U16, field_count=1)
    STRING_OFFSET = RuleKind(
        "string_offset",
        U32,
        summary="Must participate in the canonical count-plus-one string offsets.",
    )
    SWITCH_TARGET = RuleKind(
        "switch_target",
        U32,
        summary="Must name an exact control.block in the owning function.",
    )


FIELD_RULES = tuple(value for name, value in vars(FieldRule).items() if name.isupper())
