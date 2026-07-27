# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

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
from loom.gen.editor.textmate import (
    generate_all_grammars,
    generate_loom_grammar,
    generate_loom_test_grammar,
)

ALL_OPS = (
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
)

ALL_TYPES = (
    *ALL_BUILTIN_TYPES,
    *ALL_HAL_TYPES,
    *ALL_KERNEL_TYPES,
)


def _walk_regexes(value: Any) -> Iterator[str]:
    if isinstance(value, list):
        for item in value:
            yield from _walk_regexes(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("match", "begin", "end") and isinstance(item, str):
                yield item
            else:
                yield from _walk_regexes(item)


def test_loom_grammar_uses_generated_op_and_type_metadata() -> None:
    grammar = generate_loom_grammar(ALL_OPS, ALL_TYPES)
    serialized = json.dumps(grammar)

    assert "scalar" in serialized
    assert "addi" in serialized
    assert "func" in serialized
    assert "return" in serialized
    type_pattern = grammar["repository"]["types"]["patterns"][0]["match"]  # type: ignore[index]
    assert "hal\\.buffer" in type_pattern
    assert "tile" in serialized


def test_loom_reference_patterns_match_tokenizer_sigiled_name_shape() -> None:
    grammar = generate_loom_grammar(ALL_OPS, ALL_TYPES)
    reference_patterns = {
        pattern["name"]: pattern["match"]
        for pattern in grammar["repository"]["references"]["patterns"]  # type: ignore[index]
    }

    assert re.fullmatch(reference_patterns["variable.other.ssa.loom"], "%1foo")
    assert re.fullmatch(reference_patterns["variable.other.ssa.loom"], "%foo$")
    assert re.fullmatch(reference_patterns["variable.other.symbol.loom"], "@0")
    assert re.fullmatch(reference_patterns["entity.name.label.block.loom"], "^0")
    assert re.fullmatch(reference_patterns["variable.other.attribute.hash.loom"], "#$attr")
    assert not re.fullmatch(reference_patterns["variable.other.attribute.hash.loom"], "#1attr")


def test_loom_test_grammar_prioritizes_check_syntax_before_base_loom() -> None:
    grammar = generate_loom_test_grammar()

    assert grammar["scopeName"] == "source.loom-test"
    assert grammar["fileTypes"] == ["loom-test"]
    assert grammar["patterns"] == [
        {"include": "#loom-check"},
        {"include": "source.loom"},
    ]

    loom_check_patterns = grammar["repository"]["loom-check"]["patterns"]  # type: ignore[index]
    serialized = json.dumps(loom_check_patterns)
    assert "RUN" in serialized
    assert "XFAIL" in serialized
    assert "====" in serialized
    assert "----" in serialized
    assert "ERROR|WARNING|REMARK" in serialized
    assert "TYPE" in serialized
    assert "PARSE" in serialized


def test_regexes_stay_line_local_for_editor_tokenization() -> None:
    outputs = generate_all_grammars(ALL_OPS, ALL_TYPES)
    forbidden_fragments = [
        r"[\s\S]",
        r"(.|\n)",
        r"(?s)",
        "\n",
    ]

    for content in outputs.values():
        grammar = json.loads(content)
        for regex in _walk_regexes(grammar):
            for fragment in forbidden_fragments:
                assert fragment not in regex
