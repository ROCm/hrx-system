# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Loom DSL declarations -> TextMate grammars.

Reads op/type/error declarations from the Python DSL and emits JSON TextMate
grammars for ordinary `.loom` files and explicit `.loom-test` check files.

The generated grammars are intentionally lexical. They provide fast editor
coloring before semantic tooling is available, but they do not validate op
formats or resolve symbols. That belongs in the parser/document model.

Usage:
    python3 loom/py/loom/gen/run.py textmate
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from loom.dsl import Op, TypeDef
from loom.errors import ErrorDomain
from loom.gen import bootstrap as _bootstrap
from loom.gen.assembly.tokens import KEYWORD_MAP
from loom.gen.support.generated_file import generated_comment


def _regex_alternation(literals: Sequence[str]) -> str:
    """Returns an escaped deterministic alternation for literal tokens."""
    unique = sorted(set(literals), key=lambda text: (-len(text), text))
    return "|".join(re.escape(text) for text in unique)


def _identifier_pattern() -> str:
    """Returns the ASCII spelling for bare identifiers and hash attrs."""
    return r"[A-Za-z_$][A-Za-z0-9_$]*"


def _sigil_name_pattern() -> str:
    """Returns the ASCII spelling allowed after %, @, and ^ sigils."""
    return r"[A-Za-z0-9_$]+"


def _token_boundary_pattern() -> str:
    """Returns a boundary for tokens whose names may end in '$'."""
    return r"(?![A-Za-z0-9_$])"


def _generated_comment() -> str:
    return generated_comment(
        generator="loom.gen.editor.textmate",
        regenerate="python3 loom/py/loom/gen/run.py textmate",
    )


def textmate_output_directory() -> Path:
    return _bootstrap.REPO_ROOT / "loom" / "src" / "loom" / "editor" / "textmate"


def _string_rule(scope_suffix: str) -> dict[str, object]:
    return {
        "name": f"string.quoted.double.{scope_suffix}",
        "begin": r'"',
        "beginCaptures": {
            "0": {"name": f"punctuation.definition.string.begin.{scope_suffix}"},
        },
        "end": r'"',
        "endCaptures": {
            "0": {"name": f"punctuation.definition.string.end.{scope_suffix}"},
        },
        "patterns": [
            {
                "name": f"constant.character.escape.{scope_suffix}",
                "match": r'\\(?:["\\/bfnrt]|u[0-9A-Fa-f]{4})',
            },
            {
                "name": f"invalid.illegal.escape.{scope_suffix}",
                "match": r"\\.",
            },
        ],
    }


def _number_rules() -> list[dict[str, object]]:
    boundary = _token_boundary_pattern()
    return [
        {
            "name": "constant.numeric.float.loom",
            "match": rf"(?<![A-Za-z0-9_$])-?(?:[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+){boundary}",
        },
        {
            "name": "constant.numeric.integer.hex.loom",
            "match": rf"(?<![A-Za-z0-9_$])-?0x[0-9A-Fa-f]+{boundary}",
        },
        {
            "name": "constant.numeric.integer.decimal.loom",
            "match": rf"(?<![A-Za-z0-9_$])-?[0-9]+{boundary}",
        },
    ]


def _op_name_patterns(ops: Sequence[Op]) -> list[dict[str, object]]:
    by_dialect: dict[str, list[str]] = {}
    for op in ops:
        dialect, short_name = op.name.split(".", 1)
        by_dialect.setdefault(dialect, []).append(short_name)

    patterns: list[dict[str, object]] = []
    for dialect in sorted(by_dialect):
        short_names = _regex_alternation(by_dialect[dialect])
        patterns.append(
            {
                "name": "meta.operation-name.loom",
                "match": rf"\b({re.escape(dialect)})\.({short_names})\b",
                "captures": {
                    "1": {"name": "support.namespace.dialect.loom"},
                    "2": {"name": "entity.name.function.operation.loom"},
                },
            }
        )
    return patterns


def _type_name_pattern(type_defs: Sequence[TypeDef]) -> dict[str, object]:
    scalar_types = [
        "index",
        "offset",
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
        "f8E4M3",
        "f8E5M2",
        "f16",
        "bf16",
        "f32",
        "f64",
    ]
    type_names = [type_def.name for type_def in type_defs]
    all_names = _regex_alternation([*scalar_types, *type_names])
    return {
        "name": "support.type.loom",
        "match": rf"\b(?:{all_names})\b",
    }


def _is_textual_keyword(text: str) -> bool:
    if not text:
        return False
    if not (text[0].isalpha() or text[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in text)


def _keyword_pattern() -> dict[str, object]:
    textual_keywords = [text for text in KEYWORD_MAP if _is_textual_keyword(text)]
    keywords = _regex_alternation(textual_keywords)
    return {
        "name": "keyword.other.loom",
        "match": rf"\b(?:{keywords})\b",
    }


def _base_repository(ops: Sequence[Op], type_defs: Sequence[TypeDef]) -> dict[str, object]:
    identifier = _identifier_pattern()
    sigil_name = _sigil_name_pattern()
    boundary = _token_boundary_pattern()
    return {
        "comments": {
            "patterns": [
                {
                    "name": "comment.line.double-slash.loom",
                    "match": r"//.*$",
                },
            ],
        },
        "strings": {
            "patterns": [
                _string_rule("loom"),
            ],
        },
        "op-names": {
            "patterns": _op_name_patterns(ops),
        },
        "references": {
            "patterns": [
                {
                    "name": "variable.other.ssa.loom",
                    "match": rf"(%)({sigil_name}){boundary}",
                    "captures": {
                        "1": {"name": "punctuation.definition.variable.ssa.loom"},
                        "2": {"name": "variable.other.ssa.name.loom"},
                    },
                },
                {
                    "name": "entity.name.label.block.loom",
                    "match": rf"(\^)({sigil_name}){boundary}",
                    "captures": {
                        "1": {"name": "punctuation.definition.label.block.loom"},
                        "2": {"name": "entity.name.label.block.name.loom"},
                    },
                },
                {
                    "name": "variable.other.symbol.loom",
                    "match": rf"(@)({sigil_name}){boundary}",
                    "captures": {
                        "1": {"name": "punctuation.definition.variable.symbol.loom"},
                        "2": {"name": "variable.other.symbol.name.loom"},
                    },
                },
                {
                    "name": "variable.other.attribute.hash.loom",
                    "match": rf"(#)({identifier}){boundary}",
                    "captures": {
                        "1": {"name": "punctuation.definition.attribute.hash.loom"},
                        "2": {"name": "variable.other.attribute.hash.name.loom"},
                    },
                },
            ],
        },
        "attribute-names": {
            "patterns": [
                {
                    "name": "entity.other.attribute-name.loom",
                    "match": rf"(?<![A-Za-z0-9_$]){identifier}(?=\s*=)",
                },
            ],
        },
        "types": {
            "patterns": [
                _type_name_pattern(type_defs),
            ],
        },
        "keywords": {
            "patterns": [
                _keyword_pattern(),
                {
                    "name": "constant.language.boolean.loom",
                    "match": r"\b(?:true|false)\b",
                },
            ],
        },
        "numbers": {
            "patterns": _number_rules(),
        },
        "punctuation": {
            "patterns": [
                {
                    "name": "keyword.operator.arrow.loom",
                    "match": r"->",
                },
                {
                    "name": "punctuation.separator.loom",
                    "match": r"[,|:]",
                },
                {
                    "name": "punctuation.definition.group.loom",
                    "match": r"[(){}\[\]<>]",
                },
                {
                    "name": "keyword.operator.assignment.loom",
                    "match": r"=",
                },
            ],
        },
    }


def generate_loom_grammar(ops: Sequence[Op], type_defs: Sequence[TypeDef]) -> dict[str, object]:
    """Generates the ordinary `.loom` TextMate grammar."""
    return {
        "$schema": "https://raw.githubusercontent.com/martinring/tmlanguage/master/tmlanguage.json",
        "name": "Loom",
        "scopeName": "source.loom",
        "fileTypes": ["loom"],
        "comment": _generated_comment(),
        "patterns": [
            {"include": "#strings"},
            {"include": "#comments"},
            {"include": "#op-names"},
            {"include": "#references"},
            {"include": "#attribute-names"},
            {"include": "#types"},
            {"include": "#keywords"},
            {"include": "#numbers"},
            {"include": "#punctuation"},
        ],
        "repository": _base_repository(ops, type_defs),
    }


def _loom_test_repository() -> dict[str, object]:
    domains = _regex_alternation([domain.name for domain in ErrorDomain])
    return {
        "loom-check": {
            "patterns": [
                {
                    "name": "meta.separator.case.loom-test",
                    "match": r"^\s*(//)\s*(====)(?:\s+(.*))?$",
                    "captures": {
                        "1": {"name": "punctuation.definition.comment.loom-test"},
                        "2": {"name": "keyword.control.separator.case.loom-test"},
                        "3": {"name": "entity.name.section.case.loom-test"},
                    },
                },
                {
                    "name": "meta.separator.expected.loom-test",
                    "match": r"^\s*(//)\s*(----)\s*$",
                    "captures": {
                        "1": {"name": "punctuation.definition.comment.loom-test"},
                        "2": {"name": "keyword.control.separator.expected.loom-test"},
                    },
                },
                {
                    "name": "comment.line.double-slash.directive.run.loom-test",
                    "match": r"^\s*(//)\s*(RUN)(:)\s*(roundtrip|verify|pass|format|emit)\b(?:\s+(.*))?$",
                    "captures": {
                        "1": {"name": "punctuation.definition.comment.loom-test"},
                        "2": {"name": "keyword.control.directive.run.loom-test"},
                        "3": {"name": "punctuation.separator.key-value.loom-test"},
                        "4": {"name": "support.function.mode.loom-test"},
                        "5": {"name": "string.unquoted.directive-argument.loom-test"},
                    },
                },
                {
                    "name": "comment.line.double-slash.directive.xfail.loom-test",
                    "match": r"^\s*(//)\s*(XFAIL)(:)\s*(.*)$",
                    "captures": {
                        "1": {"name": "punctuation.definition.comment.loom-test"},
                        "2": {"name": "keyword.control.directive.xfail.loom-test"},
                        "3": {"name": "punctuation.separator.key-value.loom-test"},
                        "4": {"name": "string.unquoted.xfail-reason.loom-test"},
                    },
                },
                {
                    "name": "comment.line.double-slash.diagnostic.loom-test",
                    "begin": r"^\s*(//)\s*(ERROR|WARNING|REMARK)(@[+-]?[0-9]+)?(:)",
                    "beginCaptures": {
                        "1": {"name": "punctuation.definition.comment.loom-test"},
                        "2": {"name": "keyword.control.diagnostic.severity.loom-test"},
                        "3": {"name": "constant.numeric.annotation-offset.loom-test"},
                        "4": {"name": "punctuation.separator.key-value.loom-test"},
                    },
                    "end": r"$",
                    "patterns": [
                        {
                            "name": "constant.other.error-code.loom-test",
                            "match": rf"\b({domains})(/)([0-9]+)\b",
                            "captures": {
                                "1": {"name": "support.type.error-domain.loom-test"},
                                "2": {"name": "punctuation.separator.error-code.loom-test"},
                                "3": {"name": "constant.numeric.error-code.loom-test"},
                            },
                        },
                        {
                            "name": "support.type.error-domain.loom-test",
                            "match": rf"\b(?:{domains})\b",
                        },
                        {"include": "#strings"},
                    ],
                },
            ],
        },
        "strings": {
            "patterns": [
                _string_rule("loom-test"),
            ],
        },
    }


def generate_loom_test_grammar() -> dict[str, object]:
    """Generates the explicit `.loom-test` TextMate grammar."""
    return {
        "$schema": "https://raw.githubusercontent.com/martinring/tmlanguage/master/tmlanguage.json",
        "name": "Loom Test",
        "scopeName": "source.loom-test",
        "fileTypes": ["loom-test"],
        "comment": _generated_comment(),
        "patterns": [
            {"include": "#loom-check"},
            {"include": "source.loom"},
        ],
        "repository": _loom_test_repository(),
    }


def generate_all_grammars(ops: Sequence[Op], type_defs: Sequence[TypeDef]) -> dict[str, str]:
    """Returns all generated grammar filenames and JSON contents."""
    grammars = {
        "loom.tmLanguage.json": generate_loom_grammar(ops, type_defs),
        "loom-test.tmLanguage.json": generate_loom_test_grammar(),
    }
    return {filename: json.dumps(grammar, indent=2, ensure_ascii=False) + "\n" for filename, grammar in grammars.items()}


def main() -> None:
    """Generate TextMate grammars for all registered dialects."""
    from loom.builtin_types import ALL_BUILTIN_TYPES
    from loom.dialect.buffer import ALL_BUFFER_OPS
    from loom.dialect.check import ALL_CHECK_OPS
    from loom.dialect.config import ALL_CONFIG_OPS
    from loom.dialect.encoding import ALL_ENCODING_OPS
    from loom.dialect.func import ALL_FUNC_OPS
    from loom.dialect.globals import ALL_GLOBAL_OPS
    from loom.dialect.hal import ALL_HAL_TYPES
    from loom.dialect.index import ALL_INDEX_OPS
    from loom.dialect.kernel import ALL_KERNEL_OPS, ALL_KERNEL_TYPES
    from loom.dialect.llvmir import ALL_LLVMIR_OPS
    from loom.dialect.pool import ALL_POOL_OPS
    from loom.dialect.sanitizer import ALL_SANITIZER_OPS
    from loom.dialect.scalar import ALL_SCALAR_OPS
    from loom.dialect.scf import ALL_SCF_OPS
    from loom.dialect.test import ALL_TEST_OPS
    from loom.dialect.vector import ALL_VECTOR_OPS
    from loom.dialect.view import ALL_VIEW_OPS
    from loom.target.arch.ireevm.dialect import ALL_IREEVM_TYPES

    ops = [
        *ALL_TEST_OPS,
        *ALL_SCALAR_OPS,
        *ALL_FUNC_OPS,
        *ALL_ENCODING_OPS,
        *ALL_POOL_OPS,
        *ALL_SANITIZER_OPS,
        *ALL_GLOBAL_OPS,
        *ALL_SCF_OPS,
        *ALL_CHECK_OPS,
        *ALL_CONFIG_OPS,
        *ALL_BUFFER_OPS,
        *ALL_VIEW_OPS,
        *ALL_VECTOR_OPS,
        *ALL_INDEX_OPS,
        *ALL_KERNEL_OPS,
        *ALL_LLVMIR_OPS,
    ]
    type_defs = [
        *ALL_BUILTIN_TYPES,
        *ALL_HAL_TYPES,
        *ALL_KERNEL_TYPES,
        *ALL_IREEVM_TYPES,
    ]

    output_directory = textmate_output_directory()
    output_directory.mkdir(parents=True, exist_ok=True)

    outputs = generate_all_grammars(ops, type_defs)
    for filename, content in outputs.items():
        path = output_directory / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  {filename}")


if __name__ == "__main__":
    main()
