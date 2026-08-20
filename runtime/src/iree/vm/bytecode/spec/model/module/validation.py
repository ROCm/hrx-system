# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Module validation rules and ordinal domains."""

from __future__ import annotations

from model.module import OrdinalDomain
from model.schema import (
    BASIC_FIELD_RULES,
    ENTITY_ARGUMENT,
    INTEGER_ARGUMENT,
    RuleParameter,
    ValidationRule,
    ValidationScope,
)
from model.specification import CORE_0

STRINGS = OrdinalDomain(
    entity_id="core.module.ordinal.strings",
    since=CORE_0,
    summary="All Strings-section ordinals.",
    maximum_count=65535,
)
NONEMPTY_STRINGS = OrdinalDomain(
    entity_id="core.module.ordinal.nonempty_strings",
    since=CORE_0,
    summary="Strings-section ordinals whose value has nonzero length.",
    maximum_count=65535,
    base_domain_id=STRINGS.entity_id,
    require_nonempty_value=True,
)
REF_TYPES = OrdinalDomain(
    entity_id="core.module.ordinal.ref_types",
    since=CORE_0,
    summary="Flattened module-local ref-type ordinals.",
    maximum_count=65536,
)
SIGNATURES = OrdinalDomain(
    entity_id="core.module.ordinal.signatures",
    since=CORE_0,
    summary="Module-local physical signature ordinals.",
    maximum_count=65536,
)
CALLABLE_TYPES = OrdinalDomain(
    entity_id="core.module.ordinal.callable_types",
    since=CORE_0,
    summary="Module-local structural callable-type ordinals.",
    maximum_count=65536,
)
FUNCTIONS = OrdinalDomain(
    entity_id="core.module.ordinal.functions",
    since=CORE_0,
    summary="Module-local function ordinals.",
    maximum_count=65536,
)

ORDINAL_DOMAINS = (
    STRINGS,
    NONEMPTY_STRINGS,
    REF_TYPES,
    SIGNATURES,
    CALLABLE_TYPES,
    FUNCTIONS,
)
ORDINAL_DOMAINS_BY_KEY = {
    entity.entity_id.removeprefix("core.module.ordinal."): entity
    for entity in ORDINAL_DOMAINS
}


CORE_MAJOR = ValidationRule(
    entity_id="core.validation.module.field.core_major",
    since=CORE_0,
    summary="Validates the incompatible core format major.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "The value must exactly equal the loader's supported core major. A "
        "different major is incompatible rather than an unknown extension."
    ),
)
CORE_REQUIRED_MINOR = ValidationRule(
    entity_id="core.validation.module.field.core_required_minor",
    since=CORE_0,
    summary="Validates the minimum compatible core minor.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "The value must not exceed the loader's supported core minor and must "
        "cover every core feature used by the image."
    ),
)
NONCORE_PAGE = ValidationRule(
    entity_id="core.validation.module.field.noncore_page",
    since=CORE_0,
    summary="Validates an optional architectural page identity.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "The value must be an architectural extension page in 0xF0..0xFD. "
        "Availability is checked separately."
    ),
)
ORDINAL = ValidationRule(
    entity_id="core.validation.module.field.ordinal",
    since=CORE_0,
    summary="Requires a direct ordinal in a declared numbering domain.",
    scope=ValidationScope.FIELD,
    parameters=(RuleParameter("domain", ENTITY_ARGUMENT),),
    normative_text=(
        "The unsigned field value must be less than the materialized count of "
        "the referenced OrdinalDomain. No generic bit pattern is reserved."
    ),
)
ORDINAL_OR_NULL = ValidationRule(
    entity_id="core.validation.module.field.ordinal_or_null",
    since=CORE_0,
    summary="Accepts a direct ordinal or one explicit null sentinel.",
    scope=ValidationScope.FIELD,
    parameters=(
        RuleParameter("domain", ENTITY_ARGUMENT),
        RuleParameter("null_value", INTEGER_ARGUMENT),
    ),
    normative_text=(
        "The value must equal null_value or be a valid direct ordinal in the "
        "referenced OrdinalDomain. null_value must fit the field width."
    ),
)
PAGE_MAJOR = ValidationRule(
    entity_id="core.validation.module.field.page_major",
    since=CORE_0,
    summary="Validates an optional page's incompatible major.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "The value must exactly equal the supported major of the page named by "
        "the same requirement row."
    ),
)
PAGE_REQUIRED_MINOR = ValidationRule(
    entity_id="core.validation.module.field.page_required_minor",
    since=CORE_0,
    summary="Validates an optional page's minimum compatible minor.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "The value must not exceed the supported minor of the page named by "
        "the same requirement row and must cover every used page feature."
    ),
)
SECTION_BYTE_LENGTH = ValidationRule(
    entity_id="core.validation.module.field.section_byte_length",
    since=CORE_0,
    summary="Validates a directory row's exact payload length.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "The value is the exact payload length. Checked packing arithmetic must "
        "prove the resulting aligned slice is inside the image before pointer "
        "formation."
    ),
)
SECTION_FLAGS = ValidationRule(
    entity_id="core.validation.module.field.section_flags",
    since=CORE_0,
    summary="Validates flags against the selected section contract.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "A known section must use exactly its declared required flags. An "
        "unknown section is accepted only when its flags equal SKIPPABLE."
    ),
)
SECTION_TYPE = ValidationRule(
    entity_id="core.validation.module.field.section_type",
    since=CORE_0,
    summary="Validates one architectural section identity.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "The value must be nonzero, its high byte names the owning page, and "
        "the complete directory must be strictly increasing by this value."
    ),
)
SIGNATURE_DESCRIPTOR = ValidationRule(
    entity_id="core.validation.module.field.signature_descriptor",
    since=CORE_0,
    summary="Validates a logical signature descriptor pair.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "The kind must be an available signature kind. Scalar kinds require "
        "type ordinal zero; REF and FUNCTION require valid ordinals in their "
        "corresponding domains."
    ),
)
STRING_OFFSET = ValidationRule(
    entity_id="core.validation.module.field.string_offset",
    since=CORE_0,
    summary="Validates one canonical running string-byte offset.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "Offsets begin at zero, are nondecreasing, remain inside the trailing "
        "UTF-8 byte area, and the terminal offset equals its exact length."
    ),
)
SWITCH_TARGET = ValidationRule(
    entity_id="core.validation.module.field.switch_target",
    since=CORE_0,
    summary="Validates one function-relative switch target.",
    scope=ValidationScope.FIELD,
    parameters=(),
    normative_text=(
        "The u32 word offset must resolve without overflow to a decoded "
        "control.block record inside the owning function."
    ),
)

MODULE_FIELD_RULES = (
    CORE_MAJOR,
    CORE_REQUIRED_MINOR,
    NONCORE_PAGE,
    ORDINAL,
    ORDINAL_OR_NULL,
    PAGE_MAJOR,
    PAGE_REQUIRED_MINOR,
    SECTION_BYTE_LENGTH,
    SECTION_FLAGS,
    SECTION_TYPE,
    SIGNATURE_DESCRIPTOR,
    STRING_OFFSET,
    SWITCH_TARGET,
)
FIELD_RULES = (*BASIC_FIELD_RULES, *MODULE_FIELD_RULES)
ENTITIES = (*BASIC_FIELD_RULES, *ORDINAL_DOMAINS, *MODULE_FIELD_RULES)
