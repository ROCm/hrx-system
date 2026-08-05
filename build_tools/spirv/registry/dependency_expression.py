# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed Vulkan registry dependency expressions.

The Vulkan registry dependency language has two equal-precedence,
left-associative operators: ``+`` is conjunction and ``,`` is disjunction.
Parentheses are the only precedence override. Same-operator chains normalize to
ordered n-ary nodes; mixed operators retain their semantic tree structure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from string import ascii_letters, digits
from typing import TypeAlias

_IDENTIFIER_CHARACTERS = frozenset(ascii_letters + digits + "_:")
_WHITESPACE_CHARACTERS = frozenset(" \t\r\n")


@dataclass(frozen=True, slots=True)
class DependencyAtom:
    """One version, extension, feature, or property dependency name."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or any(
            character not in _IDENTIFIER_CHARACTERS for character in self.name
        ):
            raise ValueError(
                "dependency atom names must contain only ASCII letters, digits, "
                "underscore, and colon"
            )


@dataclass(frozen=True, slots=True)
class DependencyAllOf:
    """An ordered conjunction of two or more dependency expressions."""

    terms: tuple[DependencyExpression, ...]

    def __post_init__(self) -> None:
        _validate_composite_terms(self.terms, DependencyAllOf, "all-of")


@dataclass(frozen=True, slots=True)
class DependencyAnyOf:
    """An ordered disjunction of two or more dependency expressions."""

    terms: tuple[DependencyExpression, ...]

    def __post_init__(self) -> None:
        _validate_composite_terms(self.terms, DependencyAnyOf, "any-of")


DependencyExpression: TypeAlias = DependencyAtom | DependencyAllOf | DependencyAnyOf


def _validate_composite_terms(
    terms: tuple[DependencyExpression, ...],
    composite_type: type[DependencyAllOf] | type[DependencyAnyOf],
    kind: str,
) -> None:
    if not isinstance(terms, tuple):
        raise TypeError(f"dependency {kind} terms must be a tuple")
    if len(terms) < 2:
        raise ValueError(f"dependency {kind} expressions require at least two terms")
    for term in terms:
        if not isinstance(term, (DependencyAtom, DependencyAllOf, DependencyAnyOf)):
            raise TypeError(f"dependency {kind} terms must be dependency expressions")
        if isinstance(term, composite_type):
            raise ValueError(f"nested dependency {kind} expressions must be normalized")


class DependencyExpressionParseError(ValueError):
    """A dependency expression syntax error with source coordinates."""

    def __init__(
        self,
        expression: str,
        position: int,
        reason: str,
        source: str,
    ) -> None:
        self.expression = expression
        self.position = position
        self.reason = reason
        self.source = source

        line = expression.count("\n", 0, position) + 1
        line_start = expression.rfind("\n", 0, position) + 1
        line_end = expression.find("\n", position)
        if line_end < 0:
            line_end = len(expression)
        column = position - line_start + 1
        self.line = line
        self.column = column
        excerpt = expression[line_start:line_end].expandtabs(2)
        caret_prefix = expression[line_start:position].expandtabs(2)
        message = (
            f"{source}:{line}:{column}: {reason}\n{excerpt}\n{' ' * len(caret_prefix)}^"
        )
        super().__init__(message)


class _DependencyExpressionParser:
    def __init__(self, expression: str, source: str) -> None:
        self.expression = expression
        self.source = source
        self.position = 0

    def parse(self) -> DependencyExpression:
        self._skip_whitespace()
        if self._at_end():
            self._fail("expected a dependency atom")
        result = self._parse_expression()
        self._skip_whitespace()
        if not self._at_end():
            if self._peek() == ")":
                self._fail("unexpected ')' without a matching '('")
            self._fail("expected '+', ',', or the end of the expression")
        return result

    def _parse_expression(self) -> DependencyExpression:
        result = self._parse_atom()
        while True:
            self._skip_whitespace()
            if self._at_end() or self._peek() not in "+,":
                return result
            operator = self._peek()
            self.position += 1
            rhs = self._parse_atom()
            result = _combine(operator, result, rhs)

    def _parse_atom(self) -> DependencyExpression:
        self._skip_whitespace()
        if self._at_end():
            self._fail("expected a dependency atom")

        character = self._peek()
        if character == "(":
            self.position += 1
            self._skip_whitespace()
            if not self._at_end() and self._peek() == ")":
                self._fail("empty parenthesized dependency expression")
            result = self._parse_expression()
            self._skip_whitespace()
            if self._at_end() or self._peek() != ")":
                self._fail("expected ')' to close the dependency expression")
            self.position += 1
            return result

        if character in "+,":
            self._fail(f"expected a dependency atom before '{character}'")
        if character == ")":
            self._fail("expected a dependency atom before ')'")

        start = self.position
        while not self._at_end() and self._peek() in _IDENTIFIER_CHARACTERS:
            self.position += 1
        if start == self.position:
            self._fail(f"unsupported character {character!r}")
        return DependencyAtom(self.expression[start : self.position])

    def _skip_whitespace(self) -> None:
        while not self._at_end() and self._peek() in _WHITESPACE_CHARACTERS:
            self.position += 1

    def _peek(self) -> str:
        return self.expression[self.position]

    def _at_end(self) -> bool:
        return self.position == len(self.expression)

    def _fail(self, reason: str) -> None:
        raise DependencyExpressionParseError(
            self.expression,
            self.position,
            reason,
            self.source,
        )


def _combine(
    operator: str,
    lhs: DependencyExpression,
    rhs: DependencyExpression,
) -> DependencyExpression:
    composite_type = DependencyAllOf if operator == "+" else DependencyAnyOf
    terms: list[DependencyExpression] = []
    for term in (lhs, rhs):
        if isinstance(term, composite_type):
            terms.extend(term.terms)
        else:
            terms.append(term)
    return composite_type(tuple(terms))


def parse_dependency_expression(
    expression: str,
    *,
    source: str = "<dependency expression>",
) -> DependencyExpression:
    """Parses one Vulkan registry dependency expression."""

    if not isinstance(expression, str):
        raise TypeError("dependency expression must be a string")
    if not isinstance(source, str):
        raise TypeError("dependency expression source must be a string")
    return _DependencyExpressionParser(expression, source).parse()


def format_dependency_expression(expression: DependencyExpression) -> str:
    """Formats a normalized expression that reparses to structural equality."""

    if isinstance(expression, DependencyAtom):
        return expression.name
    if isinstance(expression, DependencyAllOf):
        separator = "+"
    elif isinstance(expression, DependencyAnyOf):
        separator = ","
    else:
        raise TypeError("expected a dependency expression")

    formatted_terms = []
    for term in expression.terms:
        formatted_term = format_dependency_expression(term)
        if not isinstance(term, DependencyAtom):
            formatted_term = f"({formatted_term})"
        formatted_terms.append(formatted_term)
    return separator.join(formatted_terms)


def evaluate_dependency_expression(
    expression: DependencyExpression,
    is_supported: Callable[[str], bool],
) -> bool:
    """Evaluates an expression with a caller-owned dependency-name query."""

    if isinstance(expression, DependencyAtom):
        return bool(is_supported(expression.name))
    if isinstance(expression, DependencyAllOf):
        return all(
            evaluate_dependency_expression(term, is_supported)
            for term in expression.terms
        )
    if isinstance(expression, DependencyAnyOf):
        return any(
            evaluate_dependency_expression(term, is_supported)
            for term in expression.terms
        )
    raise TypeError("expected a dependency expression")


def dependency_expression_names(expression: DependencyExpression) -> frozenset[str]:
    """Returns every dependency name referenced by an expression."""

    if isinstance(expression, DependencyAtom):
        return frozenset((expression.name,))
    if isinstance(expression, (DependencyAllOf, DependencyAnyOf)):
        return frozenset(
            name
            for term in expression.terms
            for name in dependency_expression_names(term)
        )
    raise TypeError("expected a dependency expression")


__all__ = [
    "DependencyAllOf",
    "DependencyAnyOf",
    "DependencyAtom",
    "DependencyExpression",
    "DependencyExpressionParseError",
    "dependency_expression_names",
    "evaluate_dependency_expression",
    "format_dependency_expression",
    "parse_dependency_expression",
]
