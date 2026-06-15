# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source migration rules derived from op legacy-format declarations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from loom.assembly import Attr, FormatElement, Keyword, Ref, TemplateParam, TypeOf
from loom.diagnostics import DiagnosticSeverity
from loom.format.text.tokenizer import ParseError, Token, Tokenizer, TokenKind
from loom.migration.source import (
    MigrationSourceDiagnostic,
    SourceDocument,
    SourceEdit,
)

BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID = "buffer.assume.memory_space.attr_dict"

MigrationRewrite = Callable[[SourceDocument], "MigrationRuleApplication"]


@dataclass(frozen=True, slots=True)
class MigrationRuleApplication:
    """Edits and diagnostics produced by one source migration rule."""

    edits: tuple[SourceEdit, ...] = ()
    diagnostics: tuple[MigrationSourceDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class MigrationRule:
    """One source migration rule registered with the driver."""

    rule_id: str
    rewrite: MigrationRewrite


@dataclass(frozen=True, slots=True)
class _ElementMatch:
    next_index: int
    captured_fields: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _FormatMatch:
    next_index: int
    end_token: Token
    captured_fields: dict[str, str]


def migration_rules_from_ops(
    ops: Iterable[Any],
    *,
    rewrite_hooks: Mapping[str, MigrationRewrite] | None = None,
) -> tuple[MigrationRule, ...]:
    """Builds migration rules from op-authored legacy text formats.

    Legacy formats without a rewrite hook are treated as syntax-only source
    rewrites. The migration layer matches the legacy format, maps parsed
    fields into the current format, and replaces the old op spelling with the
    current spelling. Legacy formats that require semantic construction can
    name a hook and the caller must provide the hook implementation.
    """

    hooks = rewrite_hooks or {}
    rules: list[MigrationRule] = []
    for op in ops:
        for legacy_format in getattr(op, "legacy_formats", ()):
            if legacy_format.rewrite_hook:
                try:
                    rewrite = hooks[legacy_format.rewrite_hook]
                except KeyError as exc:
                    raise ValueError(
                        f"legacy format '{legacy_format.rule_id}' for op "
                        f"'{op.name}' references unknown rewrite hook "
                        f"'{legacy_format.rewrite_hook}'"
                    ) from exc
                rules.append(MigrationRule(legacy_format.rule_id, rewrite))
                continue
            rules.append(_build_structural_rule(op, legacy_format))
    return tuple(rules)


def _build_structural_rule(op: Any, legacy_format: Any) -> MigrationRule:
    _validate_structural_format(op.name, legacy_format.rule_id, legacy_format.format)
    _validate_structural_format(op.name, legacy_format.rule_id, op.format)

    def rewrite(document: SourceDocument) -> MigrationRuleApplication:
        return _rewrite_structural_legacy_format(document, op, legacy_format)

    return MigrationRule(legacy_format.rule_id, rewrite)


def _rewrite_structural_legacy_format(
    document: SourceDocument,
    op: Any,
    legacy_format: Any,
) -> MigrationRuleApplication:
    if op.name not in document.text:
        return MigrationRuleApplication()

    try:
        tokens = _tokenize_source(document)
    except ParseError as exc:
        return MigrationRuleApplication(
            diagnostics=(
                MigrationSourceDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message=str(exc),
                    source_range=document.source_range(exc.location, exc.location),
                    rule_id=legacy_format.rule_id,
                    fixup_hint="Fix source tokenization before running migration.",
                ),
            )
        )

    edits: list[SourceEdit] = []
    diagnostics: list[MigrationSourceDiagnostic] = []
    for index, token in enumerate(tokens):
        if token.kind != TokenKind.OP_NAME or token.text != op.name:
            continue
        current_match = _match_format(
            document,
            tokens,
            index + 1,
            op.format,
        )
        if current_match is not None:
            continue

        legacy_match = _match_format(
            document,
            tokens,
            index + 1,
            legacy_format.format,
        )
        if legacy_match is None:
            diagnostics.append(
                _malformed_structural_diagnostic(document, token, op, legacy_format)
            )
            continue

        start_byte = document.byte_offset_at(token.location)
        end_byte = document.byte_offset_at(legacy_match.end_token.end_location)
        legacy_source = document.text[
            token.location.offset : legacy_match.end_token.end_location.offset
        ]
        if "//" in legacy_source:
            diagnostics.append(
                MigrationSourceDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        "legacy format rewrite would remove an inline comment "
                        "inside the operation"
                    ),
                    source_range=document.source_range(
                        token.location,
                        legacy_match.end_token.end_location,
                    ),
                    rule_id=legacy_format.rule_id,
                    fixup_hint=(
                        "Move comments outside the legacy op spelling before "
                        "running migration."
                    ),
                )
            )
            continue

        replacement_text = _render_current_op_text(op, legacy_format, legacy_match)
        edits.append(
            SourceEdit(
                byte_start=start_byte,
                byte_end=end_byte,
                replacement_text=replacement_text,
                rule_id=legacy_format.rule_id,
            )
        )

    return MigrationRuleApplication(
        edits=tuple(sorted(edits, key=lambda edit: (edit.byte_start, edit.byte_end))),
        diagnostics=tuple(diagnostics),
    )


def _tokenize_source(document: SourceDocument) -> tuple[Token, ...]:
    tokenizer = Tokenizer(document.text, str(document.filename or "<input>"))
    tokens: list[Token] = []
    while not tokenizer.at(TokenKind.EOF):
        tokens.append(tokenizer.next())
    return tuple(tokens)


def _match_format(
    document: SourceDocument,
    tokens: tuple[Token, ...],
    start_index: int,
    elements: tuple[FormatElement, ...],
) -> _FormatMatch | None:
    captured_fields: dict[str, str] = {}
    index = start_index
    end_token = tokens[start_index - 1] if start_index > 0 else None
    for element in elements:
        element_match = _match_format_element(document, tokens, index, element)
        if element_match is None:
            return None
        for field, text in element_match.captured_fields:
            if field in captured_fields and captured_fields[field] != text:
                return None
            captured_fields[field] = text
        end_token = tokens[element_match.next_index - 1]
        index = element_match.next_index
    if end_token is None:
        return None
    return _FormatMatch(
        next_index=index,
        end_token=end_token,
        captured_fields=captured_fields,
    )


def _match_format_element(
    document: SourceDocument,
    tokens: tuple[Token, ...],
    index: int,
    element: FormatElement,
) -> _ElementMatch | None:
    if isinstance(element, Keyword):
        return _match_keyword(tokens, index, element)
    if isinstance(element, Ref):
        return _match_single_token_field(
            document,
            tokens,
            index,
            element.field,
            TokenKind.SSA_VALUE,
        )
    if isinstance(element, Attr):
        return _match_attr(document, tokens, index, element)
    if isinstance(element, TypeOf):
        return _match_type(document, tokens, index, element)
    if isinstance(element, TemplateParam):
        return _match_template_param(document, tokens, index, element)
    return None


def _match_keyword(
    tokens: tuple[Token, ...],
    index: int,
    element: Keyword,
) -> _ElementMatch | None:
    token = _token_at(tokens, index)
    if token is None:
        return None
    expected_kind = _KEYWORD_TOKEN_KINDS.get(element.text)
    if expected_kind is not None:
        if token.kind != expected_kind:
            return None
        return _ElementMatch(next_index=index + 1)
    if token.kind != TokenKind.BARE_IDENT or token.text != element.text:
        return None
    return _ElementMatch(next_index=index + 1)


def _match_single_token_field(
    document: SourceDocument,
    tokens: tuple[Token, ...],
    index: int,
    field: str,
    token_kind: TokenKind,
) -> _ElementMatch | None:
    token = _token_at(tokens, index)
    if token is None or token.kind != token_kind:
        return None
    return _ElementMatch(
        next_index=index + 1,
        captured_fields=((field, _raw_token_text(document, token)),),
    )


def _match_attr(
    document: SourceDocument,
    tokens: tuple[Token, ...],
    index: int,
    element: Attr,
) -> _ElementMatch | None:
    token = _token_at(tokens, index)
    if token is None or token.kind not in _ATTR_TOKEN_KINDS:
        return None
    return _ElementMatch(
        next_index=index + 1,
        captured_fields=((element.field, _raw_token_text(document, token)),),
    )


def _match_type(
    document: SourceDocument,
    tokens: tuple[Token, ...],
    index: int,
    element: TypeOf,
) -> _ElementMatch | None:
    token = _token_at(tokens, index)
    if token is None or token.kind != TokenKind.BARE_IDENT:
        return None
    next_index = _consume_type_tokens(tokens, index)
    end_token = tokens[next_index - 1]
    text = document.text[token.location.offset : end_token.end_location.offset]
    return _ElementMatch(
        next_index=next_index,
        captured_fields=((element.field, text),),
    )


def _match_template_param(
    document: SourceDocument,
    tokens: tuple[Token, ...],
    index: int,
    element: TemplateParam,
) -> _ElementMatch | None:
    open_token = _token_at(tokens, index)
    if open_token is None or open_token.kind != TokenKind.LANGLE:
        return None
    close_index = _consume_balanced_angle_tokens(tokens, index)
    if close_index is None or close_index == index + 1:
        return None
    value_start = tokens[index + 1].location.offset
    value_end = tokens[close_index - 1].location.offset
    return _ElementMatch(
        next_index=close_index,
        captured_fields=((element.field, document.text[value_start:value_end]),),
    )


def _consume_type_tokens(tokens: tuple[Token, ...], start_index: int) -> int:
    next_token = _token_at(tokens, start_index + 1)
    if next_token is None or next_token.kind != TokenKind.LANGLE:
        return start_index + 1
    close_index = _consume_balanced_angle_tokens(tokens, start_index + 1)
    return close_index if close_index is not None else start_index + 1


def _consume_balanced_angle_tokens(
    tokens: tuple[Token, ...],
    open_index: int,
) -> int | None:
    depth = 0
    index = open_index
    while index < len(tokens):
        token = tokens[index]
        if token.kind == TokenKind.LANGLE:
            depth += 1
        elif token.kind == TokenKind.RANGLE:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _render_current_op_text(
    op: Any,
    legacy_format: Any,
    legacy_match: _FormatMatch,
) -> str:
    field_values = _map_legacy_fields_to_current(legacy_format, legacy_match)
    parts = [op.name]
    for element in op.format:
        text, glue = _render_current_element(element, field_values)
        if glue:
            parts[-1] += text
        else:
            parts.append(text)
    return " ".join(parts)


def _map_legacy_fields_to_current(
    legacy_format: Any,
    legacy_match: _FormatMatch,
) -> dict[str, str]:
    field_mapping = {
        field_mapping.legacy: field_mapping.current
        for field_mapping in legacy_format.field_mappings
    }
    field_values = {
        field_mapping.get(field, field): value
        for field, value in legacy_match.captured_fields.items()
    }
    for field_default in legacy_format.field_defaults:
        field_values[field_default.field] = _format_default_value(field_default.value)
    return field_values


def _render_current_element(
    element: FormatElement,
    field_values: Mapping[str, str],
) -> tuple[str, bool]:
    if isinstance(element, Keyword):
        return element.text, False
    if isinstance(element, Ref):
        return _field_value(field_values, element.field), False
    if isinstance(element, Attr):
        return _field_value(field_values, element.field), False
    if isinstance(element, TypeOf):
        return _field_value(field_values, element.field), False
    if isinstance(element, TemplateParam):
        return f"<{_field_value(field_values, element.field)}>", True
    raise TypeError(
        f"unsupported structural migration element: {type(element).__name__}"
    )


def _field_value(field_values: Mapping[str, str], field: str) -> str:
    try:
        return field_values[field]
    except KeyError as exc:
        raise ValueError(
            f"structural migration could not render missing field '{field}'"
        ) from exc


def _format_default_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _validate_structural_format(
    op_name: str,
    rule_id: str,
    format_elements: tuple[FormatElement, ...],
) -> None:
    for element in format_elements:
        if isinstance(element, _STRUCTURAL_ELEMENTS):
            continue
        raise ValueError(
            f"legacy format '{rule_id}' for op '{op_name}' cannot be migrated "
            f"structurally because it uses {type(element).__name__}; provide a "
            "rewrite hook or add structural migration support for that element"
        )


def _malformed_structural_diagnostic(
    document: SourceDocument,
    token: Token,
    op: Any,
    legacy_format: Any,
) -> MigrationSourceDiagnostic:
    return MigrationSourceDiagnostic(
        severity=DiagnosticSeverity.ERROR,
        message=(
            f"expected legacy format '{legacy_format.rule_id}' or current "
            f"syntax for {op.name}"
        ),
        source_range=document.token_source_range(token),
        rule_id=legacy_format.rule_id,
        fixup_hint=f"Rewrite {op.name} using its current source format.",
    )


def _raw_token_text(document: SourceDocument, token: Token) -> str:
    return document.text[token.location.offset : token.end_location.offset]


def _token_at(tokens: tuple[Token, ...], index: int) -> Token | None:
    return tokens[index] if 0 <= index < len(tokens) else None


_KEYWORD_TOKEN_KINDS = {
    ",": TokenKind.COMMA,
    ":": TokenKind.COLON,
    "->": TokenKind.ARROW,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    "=": TokenKind.EQUALS,
}

_ATTR_TOKEN_KINDS = frozenset(
    (
        TokenKind.BARE_IDENT,
        TokenKind.INTEGER,
        TokenKind.FLOAT,
        TokenKind.STRING,
        TokenKind.SYMBOL,
        TokenKind.HASH_ATTR,
    )
)

_STRUCTURAL_ELEMENTS = (
    Keyword,
    Ref,
    Attr,
    TypeOf,
    TemplateParam,
)

__all__ = [
    "BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID",
    "MigrationRule",
    "MigrationRuleApplication",
    "MigrationRewrite",
    "migration_rules_from_ops",
]
