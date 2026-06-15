# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.encoding import ALL_ENCODING_OPS
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.globals import ALL_GLOBAL_OPS
from loom.dialect.hal import ALL_HAL_TYPES
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.kernel import ALL_KERNEL_OPS, ALL_KERNEL_TYPES
from loom.dialect.llvmir import ALL_LLVMIR_OPS
from loom.dialect.low import ALL_LOW_OPS
from loom.dialect.pass_ import ALL_PASS_OPS
from loom.dialect.pool import ALL_POOL_OPS
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.target import ALL_TARGET_OPS
from loom.dialect.test import ALL_TEST_OPS
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.view import ALL_VIEW_OPS
from loom.format.text.parser import ParseError, Parser
from loom.ir import ENCODING_TYPE, I1, INDEX, Module, Operation

_ALL_OPS = (
    *ALL_TEST_OPS,
    *ALL_SCALAR_OPS,
    *ALL_FUNC_OPS,
    *ALL_ENCODING_OPS,
    *ALL_POOL_OPS,
    *ALL_GLOBAL_OPS,
    *ALL_SCF_OPS,
    *ALL_BUFFER_OPS,
    *ALL_VIEW_OPS,
    *ALL_VECTOR_OPS,
    *ALL_INDEX_OPS,
    *ALL_KERNEL_OPS,
    *ALL_LOW_OPS,
    *ALL_PASS_OPS,
    *ALL_LLVMIR_OPS,
    *ALL_TARGET_OPS,
)

_ALL_TYPES = (
    *ALL_BUILTIN_TYPES,
    *ALL_HAL_TYPES,
    *ALL_KERNEL_TYPES,
)


def _parser() -> Parser:
    parser = Parser()
    parser.register_ops(_ALL_OPS)
    parser.register_types(_ALL_TYPES)
    return parser


def _first_body_op(module: Module, symbol_name: str, op_name: str) -> Operation:
    for symbol in module.symbols:
        if symbol.name != symbol_name or symbol.op is None:
            continue
        for region in symbol.op.regions:
            for block in region.blocks:
                for op in block.ops:
                    if op.name == op_name:
                        return op
    raise AssertionError(f"{op_name} not found in @{symbol_name}")


def test_out_of_order_variadic_operands_parse_in_declared_order() -> None:
    source = (
        "func.def @memory(%view: view<16xf32>, %lane: index, "
        "%mask: vector<4xi1>, %old: vector<4xf32>) {\n"
        "  %loaded = vector.load.mask %view[%lane], %mask, %old : "
        "view<16xf32>, vector<4xi1>, vector<4xf32>\n"
        "  func.return\n"
        "}\n"
    )
    module = _parser().parse(source)
    op = _first_body_op(module, "memory", "vector.load.mask")
    operand_names = [module.values[value_id].name for value_id in op.operands]
    assert operand_names == ["view", "mask", "old", "lane"]


def test_inferred_singleton_result_type_is_concrete() -> None:
    source = (
        "func.def @cmp(%lhs: index, %rhs: index) -> (i1) {\n"
        "  %result = index.cmp slt, %lhs, %rhs : index\n"
        "  func.return %result : i1\n"
        "}\n"
    )
    module = _parser().parse(source)
    op = _first_body_op(module, "cmp", "index.cmp")
    assert module.values[op.results[0]].type == I1


def test_dynamic_global_load_co_results_have_metadata_types() -> None:
    source = (
        "global.constant @weights : tile<[%m]xf32, %enc>\n\n"
        "func.def @load() {\n"
        "  %tile, %m, %enc = global.load @weights : tile<[%m]xf32, %enc>\n"
        "  func.return\n"
        "}\n"
    )
    module = _parser().parse(source)
    op = _first_body_op(module, "load", "global.load")
    assert module.values[op.results[1]].type == INDEX
    assert module.values[op.results[2]].type == ENCODING_TYPE


def test_dynamic_global_symbols_are_declaration_local() -> None:
    source = (
        "global.constant @weights : tile<[%m]xf32>\n\n"
        "func.def @bad() {\n"
        "  test.use %m : index\n"
        "  func.return\n"
        "}\n"
    )
    with pytest.raises(ParseError, match="undefined SSA value '%m'"):
        _parser().parse(source)
