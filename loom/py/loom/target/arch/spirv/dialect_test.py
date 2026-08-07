# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from typing import ClassVar

from build_tools.spirv.registry.spirv_core_grammar import (
    SpirvCoreGrammar,
    load_spirv_core_grammar,
)

from loom.dsl import EnumCase
from loom.format.text.parser import parse_type_string
from loom.format.text.printer import print_type
from loom.ir import ParameterizedType
from loom.target.arch.spirv.dialect import (
    ALL_SPIRV_TYPES,
    SpirvCooperativeMatrixUse,
    SpirvScalarType,
    SpirvScope,
)

_ENUM_ASSIGNMENT_PATTERN = re.compile(
    r"^\s*(?P<symbol>LOOM_SPIRV_[A-Z0-9_]+)\s*=\s*"
    r"(?P<value>-?(?:0x[0-9A-Fa-f]+|[0-9]+))\s*,"
)


def _literal_enum_assignments(path: Path) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ENUM_ASSIGNMENT_PATTERN.match(line)
        if match is not None:
            assignments[match.group("symbol")] = int(match.group("value"), 0)
    return assignments


class SpirvDialectTest(unittest.TestCase):
    scalar_type_assignments: ClassVar[dict[str, int]]
    isa_assignments: ClassVar[dict[str, int]]
    grammar: ClassVar[SpirvCoreGrammar]

    @classmethod
    def configure_sources(
        cls,
        scalar_types_path: Path,
        isa_path: Path,
        grammar_path: Path,
    ) -> None:
        cls.scalar_type_assignments = _literal_enum_assignments(scalar_types_path)
        cls.isa_assignments = _literal_enum_assignments(isa_path)
        cls.grammar = load_spirv_core_grammar(grammar_path)

    def test_cooperative_matrix_type_roundtrips_exact_payload(self) -> None:
        text = "spirv.cooperative_matrix<16x32xu8, subgroup, matrix_a>"
        type_registry = {type_def.name: type_def for type_def in ALL_SPIRV_TYPES}
        parsed, dynamic_dims = parse_type_string(text, type_registry=type_registry)

        assert dynamic_dims == {}
        assert isinstance(parsed, ParameterizedType)
        assert parsed.get("rows") == 16
        assert parsed.get("columns") == 32
        assert parsed.get("component_type") == 9
        assert parsed.get("scope") == 3
        assert parsed.get("use") == 0
        assert print_type(parsed) == text

    def test_scalar_type_keywords_match_semantic_c_enum(self) -> None:
        symbols = {
            "f16": "LOOM_SPIRV_SCALAR_TYPE_F16",
            "f32": "LOOM_SPIRV_SCALAR_TYPE_F32",
            "f64": "LOOM_SPIRV_SCALAR_TYPE_F64",
            "bf16": "LOOM_SPIRV_SCALAR_TYPE_BF16",
            "s8": "LOOM_SPIRV_SCALAR_TYPE_S8",
            "s16": "LOOM_SPIRV_SCALAR_TYPE_S16",
            "s32": "LOOM_SPIRV_SCALAR_TYPE_S32",
            "s64": "LOOM_SPIRV_SCALAR_TYPE_S64",
            "u8": "LOOM_SPIRV_SCALAR_TYPE_U8",
            "u16": "LOOM_SPIRV_SCALAR_TYPE_U16",
            "u32": "LOOM_SPIRV_SCALAR_TYPE_U32",
            "u64": "LOOM_SPIRV_SCALAR_TYPE_U64",
        }
        actual = {case.keyword: case.value for case in SpirvScalarType.cases}
        expected = {
            keyword: self.scalar_type_assignments[symbol]
            for keyword, symbol in symbols.items()
        }
        assert actual == expected

    def test_scope_keywords_match_c_enum_and_spirv_grammar(self) -> None:
        spellings = {
            "cross_device": ("LOOM_SPIRV_SCOPE_CROSS_DEVICE", "CrossDevice"),
            "device": ("LOOM_SPIRV_SCOPE_DEVICE", "Device"),
            "workgroup": ("LOOM_SPIRV_SCOPE_WORKGROUP", "Workgroup"),
            "subgroup": ("LOOM_SPIRV_SCOPE_SUBGROUP", "Subgroup"),
            "invocation": ("LOOM_SPIRV_SCOPE_INVOCATION", "Invocation"),
            "queue_family": ("LOOM_SPIRV_SCOPE_QUEUE_FAMILY", "QueueFamily"),
            "shader_call": ("LOOM_SPIRV_SCOPE_SHADER_CALL_KHR", "ShaderCallKHR"),
        }
        self._assert_wire_enum_matches(
            SpirvScope.cases,
            c_assignments=self.isa_assignments,
            grammar_kind="Scope",
            spellings=spellings,
        )

    def test_matrix_use_keywords_match_c_enum_and_spirv_grammar(self) -> None:
        spellings = {
            "matrix_a": (
                "LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR",
                "MatrixAKHR",
            ),
            "matrix_b": (
                "LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_BKHR",
                "MatrixBKHR",
            ),
            "matrix_accumulator": (
                "LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_ACCUMULATOR_KHR",
                "MatrixAccumulatorKHR",
            ),
        }
        self._assert_wire_enum_matches(
            SpirvCooperativeMatrixUse.cases,
            c_assignments=self.isa_assignments,
            grammar_kind="CooperativeMatrixUse",
            spellings=spellings,
        )

    def _assert_wire_enum_matches(
        self,
        cases: tuple[EnumCase, ...],
        *,
        c_assignments: dict[str, int],
        grammar_kind: str,
        spellings: dict[str, tuple[str, str]],
    ) -> None:
        operand_kind = self.grammar.operand_kinds_by_name[grammar_kind]
        actual = {case.keyword: case.value for case in cases}
        expected: dict[str, int] = {}
        for keyword, (c_symbol, grammar_spelling) in spellings.items():
            c_value = c_assignments[c_symbol]
            grammar_value = operand_kind.enumerants_by_name[grammar_spelling].value
            assert c_value == grammar_value, c_symbol
            expected[keyword] = grammar_value
        assert actual == expected


def _paths_from_argv(argv: list[str]) -> tuple[Path, Path, Path]:
    if len(argv) != 4:
        raise SystemExit(
            f"usage: {argv[0]} <scalar_types.h> <isa.h> <spirv.core.grammar.json>"
        )
    paths = tuple(Path(value) for value in argv[1:])
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"SPIR-V dialect validation input does not exist: {path}")
    return paths


def _main(argv: list[str]) -> None:
    SpirvDialectTest.configure_sources(*_paths_from_argv(argv))
    unittest.main(argv=[argv[0]])


if __name__ == "__main__":
    _main(sys.argv)
