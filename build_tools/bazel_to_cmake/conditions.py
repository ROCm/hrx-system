# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Source-owned selection predicates over native configuration values.

The native bootstrap owns these values. Predicates select BUILD declarations;
they never inspect native targets or interpret compiler generator expressions.
The text projection serves unconfigured package previews.
"""

import re
from dataclasses import dataclass
from typing import Mapping


class Condition:
    def evaluate(self, values: Mapping[str, str]) -> bool:
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return self.render()


@dataclass(frozen=True)
class Option(Condition):
    # Native bootstrap variable, including non-cache parent/toolchain settings.
    name: str

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.name):
            raise ValueError(f"Expected a native setting name, got {self.name!r}")

    def evaluate(self, values):
        value = str(values.get(self.name, "")).upper()
        return value not in (
            "",
            "0",
            "OFF",
            "NO",
            "FALSE",
            "N",
            "IGNORE",
            "NOTFOUND",
        ) and not value.endswith("-NOTFOUND")

    def render(self):
        return self.name


@dataclass(frozen=True)
class Equal(Condition):
    # Native bootstrap variable being selected.
    name: str
    # Required literal value; not another variable or a generator expression.
    value: str

    def evaluate(self, values):
        return values.get(self.name, "") == self.value

    def render(self):
        return f'{self.name} STREQUAL "{self.value}"'


@dataclass(frozen=True)
class Matches(Condition):
    # Native bootstrap variable being selected.
    name: str
    # Source-owned regular expression for platform/compiler identification.
    pattern: str

    def evaluate(self, values):
        return re.search(self.pattern, values.get(self.name, "")) is not None

    def render(self):
        return f'{self.name} MATCHES "{self.pattern}"'


@dataclass(frozen=True)
class Greater(Condition):
    # Native bootstrap numeric option being selected.
    name: str
    # Exclusive lower bound used by the declaration's feature predicate.
    value: int

    def evaluate(self, values):
        return float(values.get(self.name, "0")) > self.value

    def render(self):
        return f"{self.name} GREATER {self.value}"


@dataclass(frozen=True)
class Not(Condition):
    # Predicate whose selection is inverted.
    operand: Condition

    def evaluate(self, values):
        return not self.operand.evaluate(values)

    def render(self):
        return f"NOT {_group(self.operand)}"


@dataclass(frozen=True)
class All(Condition):
    # Predicates that must all hold for the declaration to be available.
    operands: tuple[Condition, ...]

    def evaluate(self, values):
        return all(operand.evaluate(values) for operand in self.operands)

    def render(self):
        return " AND ".join(_group(operand) for operand in self.operands)


@dataclass(frozen=True)
class Any(Condition):
    # Alternative predicates enabling the same source declaration.
    operands: tuple[Condition, ...]

    def evaluate(self, values):
        return any(operand.evaluate(values) for operand in self.operands)

    def render(self):
        return " OR ".join(_group(operand) for operand in self.operands)


def _group(condition):
    text = condition.render()
    return f"({text})" if isinstance(condition, (All, Any)) else text


def condition(value: str | Condition) -> Condition:
    return Option(value) if isinstance(value, str) else value


def all_of(*operands: str | Condition) -> Condition:
    return All(tuple(condition(operand) for operand in operands))


def any_of(*operands: str | Condition) -> Condition:
    return Any(tuple(condition(operand) for operand in operands))
