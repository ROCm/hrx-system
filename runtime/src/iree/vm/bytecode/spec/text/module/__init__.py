# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical physical text declarations for VM module records."""

from __future__ import annotations

import enum
from typing import NamedTuple, TypeAlias

from iree.vm.bytecode.spec.module import Section, WireRecord
from iree.vm.bytecode.spec.schema import NumericTable
from iree.vm.bytecode.spec.specification import Specification
from iree.vm.bytecode.spec.text import SymbolDomain


class ScalarRadix(enum.IntEnum):
    """Canonical radix for one physical scalar field."""

    DECIMAL = 0
    HEXADECIMAL = 1


class RowMode(enum.IntEnum):
    """How a grammar invocation acquires its destination record."""

    APPEND = 0
    SINGLETON = 1


class SectionDirectoryValue(enum.IntEnum):
    """One directory field derived from an authored section."""

    TYPE = 0
    FLAGS = 1
    BYTE_LENGTH = 2


class DeclareSymbol(NamedTuple):
    """Declares the next ordinal in a symbol domain."""

    domain: SymbolDomain
    flags_field_name: str | None = None


class SymbolReference(NamedTuple):
    """Reads or writes one required symbolic record field."""

    record: WireRecord
    field_name: str
    domain: SymbolDomain


class NullableSymbolReference(NamedTuple):
    """Reads or writes one nullable symbolic record field."""

    record: WireRecord
    field_name: str
    domain: SymbolDomain


class ScalarField(NamedTuple):
    """Reads or writes one physical scalar record field."""

    record: WireRecord
    field_name: str
    radix: ScalarRadix = ScalarRadix.DECIMAL


class ModuleShell(NamedTuple):
    """Canonical module header and section wrapper syntax."""

    module_prefix: str
    core_major: ScalarField
    version_separator: str
    core_required_minor: ScalarField
    module_open: str
    section_prefix: str
    section_alignment_prefix: str
    section_alignment: ScalarField
    section_open: str
    close: str


class NumericSpelling(NamedTuple):
    """Overrides one numeric value's internal name in canonical text."""

    source_name: str
    text: str


class NumericField(NamedTuple):
    """Reads or writes one enum or flag record field."""

    record: WireRecord
    field_name: str
    table: NumericTable
    spelling_overrides: tuple[NumericSpelling, ...] = ()


class ConstantField(NamedTuple):
    """Writes one mechanically fixed record field."""

    record: WireRecord
    field_name: str
    value: int


class NestedBody(NamedTuple):
    """Runs a nested row grammar and derives its optional range."""

    grammar_name: str
    child_record: WireRecord
    parent_record: WireRecord
    base_field_name: str | None = None
    count_field_name: str | None = None


class QuotedBlob(NamedTuple):
    """Reads or writes one quoted UTF-8 range and its boundary row."""

    tail_name: str
    offset_record: WireRecord


class SignatureBody(NamedTuple):
    """Reads or writes one logical signature and its descriptor rows."""

    row_record: WireRecord
    descriptor_record: WireRecord
    kind_table: NumericTable


class NativeLayoutBody(NamedTuple):
    """Ordered typed fields, with result leaves and referenced range layouts."""

    row_record: WireRecord
    field_record: WireRecord
    kind_table: NumericTable


class NativeCallBody(NamedTuple):
    """Grouped physical selectors encoded using the imported native layout."""

    row_record: WireRecord
    word_record: WireRecord


class FunctionBody(NamedTuple):
    """Reads or writes one function's switch table and instruction stream."""

    row_record: WireRecord
    switch_record: WireRecord
    tail_name: str


class HexBlob(NamedTuple):
    """Reads or writes one aligned opaque byte range."""

    tail_name: str
    record: WireRecord
    length_field_name: str
    alignment_field_name: str


class MetadataValue(NamedTuple):
    """Reads or writes one typed metadata value and its boundary row."""

    tail_name: str
    entry_record: WireRecord
    type_field_name: str
    offset_record: WireRecord


GrammarElement: TypeAlias = (
    str
    | DeclareSymbol
    | SymbolReference
    | NullableSymbolReference
    | ScalarField
    | NumericField
    | ConstantField
    | NestedBody
    | QuotedBlob
    | SignatureBody
    | NativeLayoutBody
    | NativeCallBody
    | FunctionBody
    | HexBlob
    | MetadataValue
)


class RecordGrammar(NamedTuple):
    """Canonical text program for one logical module declaration."""

    name: str
    keyword: str
    record: WireRecord
    row_mode: RowMode
    elements: tuple[GrammarElement, ...]


class FixedRunCount(NamedTuple):
    """Runs a grammar a fixed number of times."""

    count: int


class RecordRunCount(NamedTuple):
    """Runs a grammar once for every mapped record."""

    record: WireRecord


class HeaderFieldRunCount(NamedTuple):
    """Runs a grammar according to a mapped header field."""

    record: WireRecord
    field_name: str


class MatchingFieldRunCount(NamedTuple):
    """Runs a grammar for a contiguous record partition."""

    record: WireRecord
    field_name: str
    value: int


RunCount: TypeAlias = (
    FixedRunCount | RecordRunCount | HeaderFieldRunCount | MatchingFieldRunCount
)


class GrammarRun(NamedTuple):
    """One ordered grammar run within a section."""

    grammar_name: str
    count: RunCount


class FixedLayoutCount(NamedTuple):
    """Maps a fixed number of records."""

    count: int


class RemainingLayoutCount(NamedTuple):
    """Maps every remaining complete record."""


class HeaderFieldLayoutCount(NamedTuple):
    """Maps records according to one preceding header field."""

    record: WireRecord
    field_name: str
    adjustment: int = 0


class SummedFieldLayoutCount(NamedTuple):
    """Maps records using the checked sum of fields in preceding rows."""

    record: WireRecord
    field_names: tuple[str, ...]


class FinalExtentLayoutCount(NamedTuple):
    """Maps records through the final preceding row's base-plus-count."""

    record: WireRecord
    base_field_name: str
    count_field_name: str


LayoutCount: TypeAlias = (
    FixedLayoutCount
    | RemainingLayoutCount
    | HeaderFieldLayoutCount
    | SummedFieldLayoutCount
    | FinalExtentLayoutCount
)


class RecordLayout(NamedTuple):
    """One naturally aligned fixed-record span in a section."""

    record: WireRecord
    count: LayoutCount


class AlignmentLayout(NamedTuple):
    """One canonical zero-padded alignment boundary."""

    alignment: int


class RemainingByteLength(NamedTuple):
    """Consumes every byte remaining in a section."""


class FinalByteExtent(NamedTuple):
    """Consumes the final preceding row's base-plus-length extent."""

    record: WireRecord
    base_field_name: str
    length_field_name: str


TailByteLength: TypeAlias = RemainingByteLength | FinalByteExtent


class TailLayout(NamedTuple):
    """One byte tail in a section."""

    name: str
    byte_length: TailByteLength


SectionLayoutItem: TypeAlias = RecordLayout | AlignmentLayout | TailLayout


class RecordCountDerivation(NamedTuple):
    """Derives a field from the number of parsed records."""

    record: WireRecord
    field_name: str
    source_record: WireRecord


class GrammarCountDerivation(NamedTuple):
    """Derives a field from the number of parsed grammar instances."""

    record: WireRecord
    field_name: str
    grammar_name: str


class StatisticDerivation(NamedTuple):
    """Derives a field from an assembly statistic."""

    record: WireRecord
    field_name: str
    statistic_name: str


class SectionDirectoryFieldDerivation(NamedTuple):
    """Derives a directory field from its authored section."""

    record: WireRecord
    field_name: str
    value: SectionDirectoryValue


DerivedField: TypeAlias = (
    RecordCountDerivation
    | GrammarCountDerivation
    | StatisticDerivation
    | SectionDirectoryFieldDerivation
)


class SectionText(NamedTuple):
    """Canonical grammar and physical layout for one known section."""

    section: Section
    runs: tuple[GrammarRun, ...]
    layout: tuple[SectionLayoutItem, ...]


class ModuleTextFormat(NamedTuple):
    """Complete canonical textual form for one module format."""

    shell: ModuleShell
    grammars: tuple[RecordGrammar, ...]
    sections: tuple[SectionText, ...]
    derived_fields: tuple[DerivedField, ...]
    tail_names: tuple[str, ...]
    statistic_names: tuple[str, ...]


def build_module_text_format(specification: Specification) -> ModuleTextFormat:
    """Builds and validates the complete canonical module text projection."""

    from iree.vm.bytecode.spec.text.module.layout import (
        build_derived_fields,
        build_sections,
    )
    from iree.vm.bytecode.spec.text.module.syntax import build_grammars, build_shell
    from iree.vm.bytecode.spec.text.module.validation import validate_module_text

    text = ModuleTextFormat(
        shell=build_shell(),
        grammars=build_grammars(specification),
        sections=build_sections(specification),
        derived_fields=build_derived_fields(),
        tail_names=("strings", "bytecode", "rodata", "metadata"),
        statistic_names=("maximum_block_count",),
    )
    validate_module_text(specification, text)
    return text
