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
    ANY_BITS = RuleKind("any_bits")
    ZERO = RuleKind("zero")
    ALLOWED_BITS = RuleKind("allowed_bits", value_count=1)
    ALLOWED_RANGE = RuleKind("allowed_range", value_count=2)
    EXACT_BYTES = RuleKind("exact_bytes", data_type=bytes)
    MULTIPLE = RuleKind("multiple", value_count=1)
    BYTE_ALIGNMENT = RuleKind("byte_alignment", value_count=1)
    CORE_MAJOR = RuleKind("core_major", U16)
    CORE_REQUIRED_MINOR = RuleKind("core_required_minor", U16)
    NONCORE_PAGE = RuleKind("noncore_page", U16)
    ORDINAL = RuleKind("ordinal", U16, data_type=OrdinalDomain)
    ORDINAL_OR_NULL = RuleKind(
        "ordinal_or_null", U16, value_count=1, data_type=OrdinalDomain
    )
    PAGE_MAJOR = RuleKind("page_major", U16)
    PAGE_REQUIRED_MINOR = RuleKind("page_required_minor", U16)
    SECTION_BYTE_LENGTH = RuleKind("section_byte_length", U64)
    SECTION_FLAGS = RuleKind("section_flags", U16)
    SECTION_TYPE = RuleKind("section_type", U16)
    SIGNATURE_DESCRIPTOR = RuleKind("signature_descriptor", U16, field_count=1)
    STRING_OFFSET = RuleKind("string_offset", U32)
    SWITCH_TARGET = RuleKind("switch_target", U32)


FIELD_RULES = tuple(value for name, value in vars(FieldRule).items() if name.isupper())
