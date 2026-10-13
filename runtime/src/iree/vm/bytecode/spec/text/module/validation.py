# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validation for the canonical physical VM module text model."""

from iree.vm.bytecode.spec.module import WireRecord, records
from iree.vm.bytecode.spec.module import rules as module_rules
from iree.vm.bytecode.spec.schema import (
    NumericKind,
    is_name,
    is_power_of_two,
    require,
)
from iree.vm.bytecode.spec.specification import Specification
from iree.vm.bytecode.spec.text import SymbolDomain
from iree.vm.bytecode.spec.text.module import (
    AlignmentLayout,
    ConstantField,
    DeclareSymbol,
    FinalByteExtent,
    FinalExtentLayoutCount,
    FixedLayoutCount,
    FixedRunCount,
    FunctionBody,
    GrammarCountDerivation,
    GrammarElement,
    HeaderFieldLayoutCount,
    HeaderFieldRunCount,
    HexBlob,
    MatchingFieldRunCount,
    MetadataValue,
    ModuleShell,
    ModuleTextFormat,
    NativeCallBody,
    NativeLayoutBody,
    NestedBody,
    NullableSymbolReference,
    NumericField,
    QuotedBlob,
    RecordCountDerivation,
    RecordGrammar,
    RecordRunCount,
    RemainingByteLength,
    RemainingLayoutCount,
    RowMode,
    ScalarField,
    ScalarRadix,
    SectionDirectoryFieldDerivation,
    SectionDirectoryValue,
    SignatureBody,
    StatisticDerivation,
    SummedFieldLayoutCount,
    SymbolReference,
    TailLayout,
)

_SYMBOL_DOMAINS = {
    module_rules.OrdinalDomain.STRING: SymbolDomain.STRING,
    module_rules.OrdinalDomain.STRING_NONEMPTY: SymbolDomain.STRING,
    module_rules.OrdinalDomain.REF_TYPE: SymbolDomain.REF_TYPE,
    module_rules.OrdinalDomain.SIGNATURE: SymbolDomain.SIGNATURE,
    module_rules.OrdinalDomain.CALLABLE_TYPE: SymbolDomain.CALLABLE,
    module_rules.OrdinalDomain.FUNCTION: SymbolDomain.FUNCTION,
}


def _field_names(record: WireRecord) -> set[str]:
    return {field.field.name for field in record.fields}


def _wire_field(record: WireRecord, field_name: str):
    return next(field for field in record.fields if field.field.name == field_name)


def _require_field(record: WireRecord, field_name: str) -> None:
    require(
        field_name in _field_names(record),
        f"{record.name}: text references unknown field {field_name}",
    )


def _validate_grammar_element(
    specification: Specification,
    grammar: RecordGrammar,
    grammar_by_name: dict[str, RecordGrammar],
    element: GrammarElement,
) -> None:
    if isinstance(element, str):
        require(bool(element), f"{grammar.name}: empty grammar literal")
    elif isinstance(element, DeclareSymbol):
        if element.flags_field_name:
            _require_field(grammar.record, element.flags_field_name)
            field = _wire_field(grammar.record, element.flags_field_name)
            require(
                field.field.byte_length <= 2
                and any(
                    isinstance(candidate, NumericField)
                    and candidate.record == grammar.record
                    and candidate.field_name == element.flags_field_name
                    and candidate.table.kind == NumericKind.FLAGS
                    for candidate in grammar.elements
                ),
                f"{grammar.name}: declaration flags must name a u16 flags field",
            )
    elif isinstance(element, (SymbolReference, NullableSymbolReference)):
        _require_field(element.record, element.field_name)
        field = _wire_field(element.record, element.field_name)
        expected_rule = (
            module_rules.FieldRule.ORDINAL_OR_NULL
            if isinstance(element, NullableSymbolReference)
            else module_rules.FieldRule.ORDINAL
        )
        if field.rule.kind == expected_rule:
            require(
                _SYMBOL_DOMAINS[field.rule.data] == element.domain,
                f"{grammar.name}: wrong symbol domain for "
                f"{element.record.name}.{element.field_name}",
            )
            if isinstance(element, NullableSymbolReference):
                require(
                    field.rule.values == ((1 << (field.field.byte_length * 8)) - 1,),
                    f"{grammar.name}: nullable symbol does not use the "
                    f"all-ones sentinel for "
                    f"{element.record.name}.{element.field_name}",
                )
        else:
            require(
                field.rule.kind == module_rules.FieldRule.ANY_BITS
                and not isinstance(element, NullableSymbolReference),
                f"{grammar.name}: symbol does not match the field rule for "
                f"{element.record.name}.{element.field_name}",
            )
    elif isinstance(element, ScalarField):
        _require_field(element.record, element.field_name)
    elif isinstance(element, NumericField):
        _require_field(element.record, element.field_name)
        field = _wire_field(element.record, element.field_name)
        require(
            element.table in specification.module_format.numeric_tables
            and field.field.encoding == element.table.encoding,
            f"{grammar.name}: invalid numeric table for "
            f"{element.record.name}.{element.field_name}",
        )
        value_names = {value.name for value in element.table.values}
        source_names = tuple(
            spelling.source_name for spelling in element.spelling_overrides
        )
        spelling_overrides = {
            spelling.source_name: spelling.text
            for spelling in element.spelling_overrides
        }
        text_names = tuple(
            spelling_overrides.get(value.name, value.name)
            for value in element.table.values
        )
        require(
            len(source_names) == len(set(source_names))
            and set(source_names) <= value_names
            and len(text_names) == len(set(text_names))
            and all(is_name(name, qualified=True) for name in text_names),
            f"{grammar.name}: invalid numeric spelling overrides for "
            f"{element.record.name}.{element.field_name}",
        )
        if element.table.kind == NumericKind.FLAGS:
            known_bits = 0
            for value in element.table.values:
                known_bits |= value.value
            require(
                field.rule.kind == module_rules.FieldRule.ALLOWED_BITS
                and element.table.unknown_policy.value == "reject"
                and known_bits == field.rule.values[0],
                f"{grammar.name}: flag spellings do not exactly cover "
                f"{element.record.name}.{element.field_name}",
            )
        elif element.table.unknown_policy.value == "preserve_nonzero":
            maximum = (1 << (field.field.byte_length * 8)) - 1
            require(
                field.rule.kind == module_rules.FieldRule.ALLOWED_RANGE
                and field.rule.values == (1, maximum),
                f"{grammar.name}: open numeric spellings do not cover "
                f"{element.record.name}.{element.field_name}",
            )
    elif isinstance(element, ConstantField):
        _require_field(element.record, element.field_name)
        field = _wire_field(element.record, element.field_name)
        require(
            field.rule.kind == module_rules.FieldRule.ALLOWED_RANGE
            and field.rule.values[0] <= element.value <= field.rule.values[1],
            f"{grammar.name}: constant violates the wire rule for "
            f"{element.record.name}.{element.field_name}",
        )
    elif isinstance(element, NestedBody):
        child_grammar = grammar_by_name.get(element.grammar_name)
        require(
            element.parent_record == grammar.record,
            f"{grammar.name}: nested body has the wrong parent record",
        )
        require(
            child_grammar is not None
            and child_grammar.record == element.child_record
            and child_grammar.row_mode == RowMode.APPEND,
            f"{grammar.name}: invalid nested grammar {element.grammar_name}",
        )
        require(
            bool(element.base_field_name or element.count_field_name),
            f"{grammar.name}: nested body has no physical extent",
        )
        if element.base_field_name:
            _require_field(element.parent_record, element.base_field_name)
        if element.count_field_name:
            _require_field(element.parent_record, element.count_field_name)
    elif isinstance(element, QuotedBlob):
        require(
            element.offset_record == grammar.record,
            f"{grammar.name}: quoted blob has the wrong offset record",
        )
        offset_field = _wire_field(element.offset_record, "byte_offset_u32")
        require(
            offset_field.rule.kind == module_rules.FieldRule.STRING_OFFSET,
            f"{grammar.name}: quoted blob does not own a string offset",
        )
    elif isinstance(element, NativeCallBody):
        require(
            element.row_record == grammar.record == records.NATIVE_CALLSITE_ROW
            and element.word_record == records.NATIVE_CALL_WORD,
            f"{grammar.name}: invalid native-call records",
        )
    elif isinstance(element, NativeLayoutBody):
        require(
            element.row_record == grammar.record == records.NATIVE_LAYOUT_ROW
            and element.field_record == records.NATIVE_FIELD_ROW,
            f"{grammar.name}: invalid native-layout records",
        )
        require(
            element.kind_table in specification.module_format.numeric_tables,
            f"{grammar.name}: invalid native scalar-kind table",
        )
    elif isinstance(element, SignatureBody):
        kind_field = _wire_field(element.descriptor_record, "kind_u16")
        type_field = _wire_field(element.descriptor_record, "type_ordinal_u16")
        kind_values = {value.name: value.value for value in element.kind_table.values}
        scalar_minimum = kind_values.get("i8", 1)
        scalar_maximum = kind_values.get("f64", 0)
        scalar_values = {
            value.value
            for value in element.kind_table.values
            if value.name not in {"invalid", "ref", "function"}
        }
        require(
            element.row_record == grammar.record
            and element.descriptor_record == records.SIGNATURE_DESCRIPTOR_ROW,
            f"{grammar.name}: invalid signature-body records",
        )
        require(
            element.kind_table in specification.module_format.numeric_tables
            and kind_field.field.encoding == element.kind_table.encoding,
            f"{grammar.name}: invalid signature-kind table",
        )
        require(
            kind_field.rule.kind == module_rules.FieldRule.ANY_BITS
            and type_field.rule.kind == module_rules.FieldRule.SIGNATURE_DESCRIPTOR
            and type_field.rule.fields == ("kind_u16",)
            and element.kind_table.kind == NumericKind.ENUM
            and element.kind_table.unknown_policy.value == "reject"
            and kind_values.get("invalid") == 0
            and "i8" in kind_values
            and "f64" in kind_values
            and "ref" in kind_values
            and "function" in kind_values
            and scalar_values == set(range(scalar_minimum, scalar_maximum + 1)),
            f"{grammar.name}: signature kinds do not cover the verifier domain",
        )
    elif isinstance(element, FunctionBody):
        require(
            element.row_record == grammar.record
            and element.switch_record == records.SWITCH_TARGET_ENTRY,
            f"{grammar.name}: invalid function-body records",
        )
    elif isinstance(element, HexBlob):
        require(
            element.record == grammar.record,
            f"{grammar.name}: hex blob has the wrong record",
        )
        _require_field(element.record, element.length_field_name)
        _require_field(element.record, element.alignment_field_name)
    elif isinstance(element, MetadataValue):
        require(
            element.entry_record == grammar.record
            and element.offset_record == records.METADATA_VALUE_OFFSET,
            f"{grammar.name}: invalid metadata-value records",
        )
        type_field = _wire_field(element.entry_record, element.type_field_name)
        require(
            type_field.rule.kind == module_rules.FieldRule.ALLOWED_RANGE
            and type_field.rule.values
            == (1, (1 << (type_field.field.byte_length * 8)) - 1)
            and any(
                isinstance(candidate, NumericField)
                and candidate.record == element.entry_record
                and candidate.field_name == element.type_field_name
                and candidate.table.name == "metadata_value_type"
                for candidate in grammar.elements
            ),
            f"{grammar.name}: metadata values do not own an open type field",
        )
    else:
        raise TypeError(f"{grammar.name}: unknown grammar element {type(element)}")


def _validate_shell(shell: ModuleShell) -> None:
    literals = (
        shell.module_prefix,
        shell.version_separator,
        shell.module_open,
        shell.section_prefix,
        shell.section_alignment_prefix,
        shell.section_open,
        shell.close,
    )
    require(all(literals), "module text shell contains an empty literal")
    fields = (
        (
            shell.core_major,
            records.IMAGE_HEADER,
            "core_major_u16",
        ),
        (
            shell.core_required_minor,
            records.IMAGE_HEADER,
            "core_required_minor_u16",
        ),
        (
            shell.section_alignment,
            records.SECTION_DIRECTORY_ROW,
            "payload_alignment_u32",
        ),
    )
    for field, expected_record, expected_name in fields:
        _require_field(field.record, field.field_name)
        require(
            field.record == expected_record
            and field.field_name == expected_name
            and field.radix == ScalarRadix.DECIMAL,
            f"invalid module shell field {field.record.name}.{field.field_name}",
        )


def _grammar_field_owners(grammar: RecordGrammar) -> dict[WireRecord, set[str]]:
    owners: dict[WireRecord, set[str]] = {}

    def own(record: WireRecord, *field_names: str) -> None:
        owned = owners.setdefault(record, set())
        for field_name in field_names:
            _require_field(record, field_name)
            require(
                field_name not in owned,
                f"{grammar.name}: duplicate owner for {record.name}.{field_name}",
            )
            owned.add(field_name)

    for element in grammar.elements:
        if isinstance(
            element,
            (
                SymbolReference,
                NullableSymbolReference,
                ScalarField,
                NumericField,
                ConstantField,
            ),
        ):
            own(element.record, element.field_name)
        elif isinstance(element, NestedBody):
            if element.base_field_name:
                own(element.parent_record, element.base_field_name)
            if element.count_field_name:
                own(element.parent_record, element.count_field_name)
        elif isinstance(element, QuotedBlob):
            own(element.offset_record, "byte_offset_u32")
        elif isinstance(element, NativeCallBody):
            own(element.row_record, "word_offset_u32", "word_count_u32")
            own(element.word_record, "value_u32")
        elif isinstance(element, NativeLayoutBody):
            own(element.row_record, *_field_names(element.row_record))
            own(element.field_record, *_field_names(element.field_record))
        elif isinstance(element, SignatureBody):
            own(element.row_record, *_field_names(element.row_record))
            own(element.descriptor_record, *_field_names(element.descriptor_record))
        elif isinstance(element, FunctionBody):
            own(
                element.row_record,
                "bytecode_offset_u32",
                "bytecode_length_u32",
                "switch_target_base_u32",
                "switch_target_entry_count_u32",
                "block_count_u32",
            )
            own(element.switch_record, "target_word_offset_u32")
        elif isinstance(element, HexBlob):
            own(element.record, element.length_field_name)
        elif isinstance(element, MetadataValue):
            own(element.offset_record, "byte_offset_u64")
    return owners


def _validate_layout(section) -> None:
    layout_records = []
    preceding_records = set()
    seen_remaining = False
    seen_tail = False
    tail_names = set()
    for item in section.layout:
        require(not seen_tail, f"{section.section.name}: layout follows a tail")
        require(
            not seen_remaining,
            f"{section.section.name}: layout follows a remaining-record span",
        )
        if isinstance(item, AlignmentLayout):
            require(
                is_power_of_two(item.alignment),
                f"{section.section.name}: invalid layout alignment",
            )
            continue
        if isinstance(item, TailLayout):
            require(
                item.name not in tail_names,
                f"{section.section.name}: duplicate tail {item.name}",
            )
            if isinstance(item.byte_length, FinalByteExtent):
                _require_field(
                    item.byte_length.record, item.byte_length.base_field_name
                )
                _require_field(
                    item.byte_length.record, item.byte_length.length_field_name
                )
                require(
                    item.byte_length.record in preceding_records,
                    f"{section.section.name}: tail extent uses a later record",
                )
            elif not isinstance(item.byte_length, RemainingByteLength):
                raise TypeError(
                    f"{section.section.name}: unknown tail byte length "
                    f"{type(item.byte_length)}"
                )
            tail_names.add(item.name)
            seen_tail = True
            continue

        count = item.count
        if isinstance(count, FixedLayoutCount):
            require(count.count > 0, f"{section.section.name}: empty fixed span")
        elif isinstance(count, HeaderFieldLayoutCount):
            _require_field(count.record, count.field_name)
            require(
                count.record in preceding_records,
                f"{section.section.name}: layout count uses a later record",
            )
        elif isinstance(count, SummedFieldLayoutCount):
            require(count.field_names, f"{section.section.name}: empty field sum")
            for field_name in count.field_names:
                _require_field(count.record, field_name)
            require(
                count.record in preceding_records,
                f"{section.section.name}: layout sum uses a later record",
            )
        elif isinstance(count, FinalExtentLayoutCount):
            _require_field(count.record, count.base_field_name)
            _require_field(count.record, count.count_field_name)
            require(
                count.record in preceding_records,
                f"{section.section.name}: layout extent uses a later record",
            )
        elif isinstance(count, RemainingLayoutCount):
            seen_remaining = True
        else:
            raise TypeError(
                f"{section.section.name}: unknown layout count {type(count)}"
            )
        layout_records.append(item.record)
        preceding_records.add(item.record)
    require(
        tuple(layout_records) == section.section.records,
        f"{section.section.name}: layout does not cover its records in order",
    )


def _validate_runs(grammar_by_name: dict[str, RecordGrammar], section) -> None:
    matching_values: dict[tuple[WireRecord, str], set[int]] = {}
    for run in section.runs:
        grammar = grammar_by_name.get(run.grammar_name)
        require(
            grammar is not None,
            f"{section.section.name}: unknown grammar run {run.grammar_name}",
        )
        require(
            grammar.record in section.section.records,
            f"{section.section.name}: grammar uses a foreign record",
        )
        count = run.count
        if isinstance(count, FixedRunCount):
            require(
                count.count == 1 and grammar.row_mode == RowMode.SINGLETON,
                f"{section.section.name}: invalid singleton grammar run",
            )
        elif isinstance(count, RecordRunCount):
            require(
                count.record == grammar.record and grammar.row_mode == RowMode.APPEND,
                f"{section.section.name}: invalid record grammar run",
            )
        elif isinstance(count, HeaderFieldRunCount):
            _require_field(count.record, count.field_name)
            require(
                count.record in section.section.records
                and grammar.row_mode == RowMode.APPEND,
                f"{section.section.name}: invalid header-count grammar run",
            )
        elif isinstance(count, MatchingFieldRunCount):
            _require_field(count.record, count.field_name)
            require(
                count.record == grammar.record and grammar.row_mode == RowMode.APPEND,
                f"{section.section.name}: invalid partitioned grammar run",
            )
            require(
                any(
                    isinstance(element, ConstantField)
                    and element.record == count.record
                    and element.field_name == count.field_name
                    and element.value == count.value
                    for element in grammar.elements
                ),
                f"{section.section.name}: partitioned grammar does not emit "
                f"its matched value",
            )
            matching_values.setdefault((count.record, count.field_name), set()).add(
                count.value
            )
        else:
            raise TypeError(f"{section.section.name}: unknown run count {type(count)}")
    for (record, field_name), values in matching_values.items():
        rule = _wire_field(record, field_name).rule
        require(
            rule.kind == module_rules.FieldRule.ALLOWED_RANGE
            and values == set(range(rule.values[0], rule.values[1] + 1)),
            f"{section.section.name}: partitioned grammars do not cover every "
            f"valid {record.name}.{field_name} value",
        )


def validate_module_text(specification: Specification, text: ModuleTextFormat) -> None:
    """Validates grammar reachability, layout, and complete wire-field ownership."""

    _validate_shell(text.shell)
    grammar_by_name = {grammar.name: grammar for grammar in text.grammars}
    require(
        len(grammar_by_name) == len(text.grammars),
        "duplicate module text grammar name",
    )
    for grammar in text.grammars:
        require(grammar.keyword, f"{grammar.name}: empty grammar keyword")
        require(bool(grammar.elements), f"{grammar.name}: empty module text grammar")
        require(
            isinstance(grammar.elements[0], str)
            and grammar.elements[0].startswith(grammar.keyword),
            f"{grammar.name}: grammar does not begin with its dispatch keyword",
        )
        for element in grammar.elements:
            _validate_grammar_element(specification, grammar, grammar_by_name, element)
        _grammar_field_owners(grammar)

    require(
        tuple(section.section for section in text.sections)
        == specification.module_format.sections,
        "module text does not cover every section in physical order",
    )
    for section in text.sections:
        _validate_layout(section)
        _validate_runs(grammar_by_name, section)
        keywords = tuple(
            grammar_by_name[run.grammar_name].keyword for run in section.runs
        )
        require(
            len(keywords) == len(set(keywords)),
            f"{section.section.name}: ambiguous grammar keyword",
        )

    reachable_grammars = set()
    for section in text.sections:
        section_grammars = {run.grammar_name for run in section.runs}
        worklist = list(section_grammars)
        while worklist:
            grammar = grammar_by_name[worklist.pop()]
            require(
                grammar.record in section.section.records,
                f"{section.section.name}: nested grammar uses a foreign record",
            )
            for element in grammar.elements:
                if (
                    isinstance(element, NestedBody)
                    and element.grammar_name not in section_grammars
                ):
                    section_grammars.add(element.grammar_name)
                    worklist.append(element.grammar_name)
        reachable_grammars.update(section_grammars)
    require(
        reachable_grammars == set(grammar_by_name),
        "module text contains an unreachable grammar",
    )

    derived: dict[WireRecord, set[str]] = {}
    for item in text.derived_fields:
        _require_field(item.record, item.field_name)
        owned = derived.setdefault(item.record, set())
        require(
            item.field_name not in owned,
            f"duplicate derivation for {item.record.name}.{item.field_name}",
        )
        owned.add(item.field_name)
        if isinstance(item, RecordCountDerivation):
            require(
                item.source_record in specification.module_format.records,
                f"unknown derived source record {item.source_record.name}",
            )
        elif isinstance(item, GrammarCountDerivation):
            require(
                item.grammar_name in grammar_by_name,
                f"unknown derived grammar {item.grammar_name}",
            )
        elif isinstance(item, StatisticDerivation):
            require(
                item.statistic_name in text.statistic_names,
                f"unknown derived statistic {item.statistic_name}",
            )
        elif isinstance(item, SectionDirectoryFieldDerivation):
            expected_rules = {
                SectionDirectoryValue.TYPE: module_rules.FieldRule.SECTION_TYPE,
                SectionDirectoryValue.FLAGS: module_rules.FieldRule.SECTION_FLAGS,
                SectionDirectoryValue.BYTE_LENGTH: module_rules.FieldRule.SECTION_BYTE_LENGTH,
            }
            field = _wire_field(item.record, item.field_name)
            require(
                item.record == records.SECTION_DIRECTORY_ROW
                and field.rule.kind == expected_rules[item.value],
                f"invalid section directory derivation for "
                f"{item.record.name}.{item.field_name}",
            )
        else:
            raise TypeError(f"unknown derived field {type(item)}")

    grammar_owners = {
        grammar.name: _grammar_field_owners(grammar) for grammar in text.grammars
    }
    fixed_fields = {
        record: {
            field.field.name
            for field in record.fields
            if field.rule.kind
            in {module_rules.FieldRule.ZERO, module_rules.FieldRule.EXACT_BYTES}
        }
        for record in specification.module_format.records
    }
    singleton_owners: dict[WireRecord, set[str]] = {}
    for grammar in text.grammars:
        expected = (
            _field_names(grammar.record)
            - fixed_fields[grammar.record]
            - derived.get(grammar.record, set())
        )
        actual = grammar_owners[grammar.name].get(grammar.record, set())
        if grammar.row_mode == RowMode.APPEND:
            require(
                actual == expected,
                f"{grammar.name}: grammar field coverage mismatch",
            )
        else:
            accumulated = singleton_owners.setdefault(grammar.record, set())
            require(
                not (accumulated & actual),
                f"{grammar.name}: singleton grammar field ownership overlaps",
            )
            accumulated.update(actual)
    for record, actual in singleton_owners.items():
        expected = (
            _field_names(record) - fixed_fields[record] - derived.get(record, set())
        )
        require(actual == expected, f"{record.name}: singleton field coverage mismatch")

    owners: dict[WireRecord, set[str]] = {}
    for field in (
        text.shell.core_major,
        text.shell.core_required_minor,
        text.shell.section_alignment,
    ):
        owners.setdefault(field.record, set()).add(field.field_name)
    for grammar in text.grammars:
        for record, field_names in grammar_owners[grammar.name].items():
            owners.setdefault(record, set()).update(field_names)
    for record, field_names in derived.items():
        require(
            not (owners.get(record, set()) & field_names),
            f"{record.name}: field is both textual and globally derived",
        )
        owners.setdefault(record, set()).update(field_names)
    for record in specification.module_format.records:
        expected = _field_names(record)
        represented = set(owners.get(record, set()))
        represented.update(fixed_fields[record])
        require(
            represented == expected,
            f"{record.name}: module text field coverage mismatch: "
            f"missing={sorted(expected - represented)}, "
            f"extra={sorted(represented - expected)}",
        )

    referenced_tails = {
        element.tail_name
        for grammar in text.grammars
        for element in grammar.elements
        if isinstance(element, (QuotedBlob, FunctionBody, HexBlob, MetadataValue))
    }
    layout_tails = {
        item.name
        for section in text.sections
        for item in section.layout
        if isinstance(item, TailLayout)
    }
    require(
        referenced_tails == set(text.tail_names) == layout_tails
        and len(text.tail_names) == len(set(text.tail_names)),
        "module text tail declarations are inconsistent",
    )
