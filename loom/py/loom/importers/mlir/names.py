# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source SSA name recovery for MLIR imports.

MLIR preserves semantic value identity, but arbitrary textual SSA names are
printer sugar and are not stored on values. The importer uses this module only
to recover those display names from the input assembly. Conversion semantics
continue to come from the parsed MLIR operation graph.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_FILE_LOCATION_RE = re.compile(
    r'^loc\((?P<source>"(?:\\.|[^"\\])*"):(?P<line>[0-9]+):(?P<col>[0-9]+)\)$'
)


@dataclass(frozen=True, slots=True)
class SourceNameRecord:
    """Recovered names for one source operation line."""

    op_name: str
    result_names: tuple[str, ...]
    block_arg_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FilePosition:
    """One-based source line and column."""

    line: int
    column: int


def build_source_name_overrides(
    root_operation: Any,
    source_text: str,
) -> dict[object, str]:
    """Returns MLIR value objects mapped to their source SSA spelling."""
    from loom.importers.mlir.converter import walk_operations

    lines = source_text.splitlines()
    operations = tuple(walk_operations(root_operation))
    names: dict[object, str] = {}

    for operation in operations:
        record = source_name_record_from_operation_location(operation, lines)
        if record is not None:
            assign_operation_record(operation, record, names)

    source_records = parse_source_name_records(source_text)
    record_index = 0
    for operation in operations:
        if not _operation_needs_names(operation):
            continue
        if record_index >= len(source_records):
            break
        op_name = str(operation.name)
        record = source_records[record_index]
        if record.op_name != op_name:
            continue
        record_index += 1
        if not _operation_has_all_names(operation, names):
            assign_operation_record(operation, record, names)
    return names


def source_name_record_from_operation_location(
    operation: Any,
    lines: Sequence[str],
) -> SourceNameRecord | None:
    position = file_position(getattr(operation, "location", None))
    if position is None or position.line < 1 or position.line > len(lines):
        return None
    line = lines[position.line - 1]
    if position.column < 1 or position.column > len(line) + 1:
        return None
    op_name = str(operation.name)
    if operation_name_at(line, position.column) != op_name:
        return None
    return SourceNameRecord(
        op_name=op_name,
        result_names=operation_result_names(line, position.column),
        block_arg_names=operation_block_arg_names(op_name, line, position.column),
    )


def file_position(location: object | None) -> FilePosition | None:
    if location is None:
        return None
    match = _FILE_LOCATION_RE.match(str(location))
    if match is None:
        return None
    try:
        ast.literal_eval(match.group("source"))
    except (SyntaxError, ValueError):
        return None
    return FilePosition(
        line=int(match.group("line")),
        column=int(match.group("col")),
    )


def operation_result_names(line: str, op_column: int) -> tuple[str, ...]:
    prefix = line[: max(op_column - 1, 0)]
    equal_index = prefix.rfind("=")
    if equal_index < 0:
        return ()
    lhs = prefix[:equal_index].strip()
    if not lhs.startswith("%"):
        return ()
    names, end_index = parse_result_name_list(lhs, 0)
    if names is None or lhs[end_index:].strip():
        return ()
    return names


def parse_source_name_records(source_text: str) -> tuple[SourceNameRecord, ...]:
    records: list[SourceNameRecord] = []
    for line in source_text.splitlines():
        record = source_name_record_from_line(line)
        if record is not None and (record.result_names or record.block_arg_names):
            records.append(record)
    return tuple(records)


def source_name_record_from_line(line: str) -> SourceNameRecord | None:
    text = line.strip()
    if not text or text.startswith(("//", "#")):
        return None

    result_names: tuple[str, ...] = ()
    index = 0
    if text.startswith("%"):
        parsed_names, index = parse_result_name_list(text, index)
        if parsed_names is None:
            return None
        index = skip_spaces(text, index)
        if index >= len(text) or text[index] != "=":
            return None
        result_names = parsed_names
        index = skip_spaces(text, index + 1)
    op_name, _end_index = parse_operation_name(text, index)
    if op_name is None:
        return None
    return SourceNameRecord(
        op_name=op_name,
        result_names=result_names,
        block_arg_names=operation_block_arg_names(op_name, text, index + 1),
    )


def parse_result_name_list(
    text: str,
    start_index: int,
) -> tuple[tuple[str, ...] | None, int]:
    names: list[str] = []
    index = start_index
    while True:
        index = skip_spaces(text, index)
        name, index = parse_value_id(text, index)
        if name is None:
            return None, start_index
        index = skip_spaces(text, index)
        count = 1
        if index < len(text) and text[index] == ":":
            index += 1
            count_start = index
            while index < len(text) and text[index].isdigit():
                index += 1
            if count_start == index:
                return None, start_index
            count = int(text[count_start:index])
        names.extend(name for _ in range(count))
        index = skip_spaces(text, index)
        if index >= len(text) or text[index] != ",":
            return tuple(names), index
        index += 1


def operation_name_at(line: str, column: int) -> str | None:
    index = max(column - 1, 0)
    name, _end_index = parse_operation_name(line, index)
    return name


def parse_operation_name(text: str, start_index: int) -> tuple[str | None, int]:
    index = skip_spaces(text, start_index)
    if index >= len(text):
        return None, index
    if text[index] == '"':
        token, index = parse_quoted_token(text, index)
        return token, index
    start = index
    while index < len(text) and is_operation_name_char(text[index]):
        index += 1
    if start == index:
        return None, start_index
    return text[start:index], index


def operation_block_arg_names(
    op_name: str,
    line: str,
    op_column: int,
) -> tuple[str, ...]:
    op_start = max(op_column - 1, 0)
    text = line[op_start:].strip()
    if op_name == "scf.for":
        return scf_for_block_arg_names(text)
    if op_name == "scf.forall":
        return scf_forall_block_arg_names(text)
    return ()


def scf_for_block_arg_names(text: str) -> tuple[str, ...]:
    index = len("scf.for")
    index = skip_spaces(text, index)
    induction, index = parse_value_id(text, index)
    if induction is None:
        return ()
    index = skip_spaces(text, index)
    if index >= len(text) or text[index] != "=":
        return ()
    names = [induction]
    iter_args = parenthesized_keyword_body(text, "iter_args")
    if iter_args is not None:
        for piece in split_top_level(iter_args, ","):
            arg_name, arg_index = parse_value_id(piece.strip(), 0)
            if arg_name is None:
                return ()
            arg_index = skip_spaces(piece, arg_index)
            if arg_index >= len(piece) or piece[arg_index] != "=":
                return ()
            names.append(arg_name)
    return tuple(names)


def scf_forall_block_arg_names(text: str) -> tuple[str, ...]:
    index = len("scf.forall")
    index = skip_spaces(text, index)
    if index >= len(text) or text[index] != "(":
        return ()
    body, _end_index = parse_balanced_body(text, index)
    if body is None:
        return ()
    names: list[str] = []
    for piece in split_top_level(body, ","):
        value, end_index = parse_value_id(piece.strip(), 0)
        if value is None or piece.strip()[end_index:].strip():
            return ()
        names.append(value)
    return tuple(names)


def parenthesized_keyword_body(text: str, keyword: str) -> str | None:
    keyword_index = text.find(keyword)
    if keyword_index < 0:
        return None
    index = skip_spaces(text, keyword_index + len(keyword))
    if index >= len(text) or text[index] != "(":
        return None
    body, _end_index = parse_balanced_body(text, index)
    return body


def parse_balanced_body(text: str, open_index: int) -> tuple[str | None, int]:
    if open_index >= len(text) or text[open_index] != "(":
        return None, open_index
    depth = 0
    index = open_index
    while index < len(text):
        char = text[index]
        if char == '"':
            _token, next_index = parse_quoted_token(text, index)
            if next_index <= index:
                return None, open_index
            index = next_index
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index], index + 1
        index += 1
    return None, open_index


def parse_value_id(text: str, start_index: int) -> tuple[str | None, int]:
    index = start_index
    if index >= len(text) or text[index] != "%":
        return None, start_index
    index += 1
    if index < len(text) and text[index] == '"':
        token, index = parse_quoted_token(text, index)
        if token is None:
            return None, start_index
        return f"%{token}", index
    start = index
    while index < len(text) and is_value_id_char(text[index]):
        index += 1
    if start == index:
        return None, start_index
    return text[start_index:index], index


def parse_quoted_token(text: str, start_index: int) -> tuple[str | None, int]:
    index = start_index + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return text[start_index + 1 : index], index + 1
        index += 1
    return None, start_index


def split_top_level(text: str, separator: str) -> tuple[str, ...]:
    pieces: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            _token, next_index = parse_quoted_token(text, index)
            if next_index <= index:
                index += 1
            else:
                index = next_index
            continue
        if char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth -= 1
        elif char == separator and depth == 0:
            pieces.append(text[start:index].strip())
            start = index + 1
        index += 1
    pieces.append(text[start:].strip())
    return tuple(piece for piece in pieces if piece)


def skip_spaces(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def is_operation_name_char(char: str) -> bool:
    return char.isalnum() or char in "._$-"


def is_value_id_char(char: str) -> bool:
    return char.isalnum() or char in "._$-"


def assign_operation_record(
    operation: Any,
    record: SourceNameRecord,
    names: dict[object, str],
) -> None:
    results = tuple(operation.results)
    if len(record.result_names) == len(results):
        for result, name in zip(results, record.result_names, strict=True):
            names.setdefault(result, name)
    block_args = first_block_arguments(operation)
    if len(record.block_arg_names) == len(block_args):
        for block_arg, name in zip(block_args, record.block_arg_names, strict=True):
            names.setdefault(block_arg, name)


def first_block_arguments(operation: Any) -> tuple[Any, ...]:
    regions = tuple(operation.regions)
    if not regions:
        return ()
    blocks = tuple(regions[0].blocks)
    if not blocks:
        return ()
    return tuple(blocks[0].arguments)


def _operation_has_all_names(operation: Any, names: dict[object, str]) -> bool:
    values = tuple(operation.results) + first_block_arguments(operation)
    return all(value in names for value in values)


def _operation_needs_names(operation: Any) -> bool:
    return bool(tuple(operation.results) or first_block_arguments(operation))
