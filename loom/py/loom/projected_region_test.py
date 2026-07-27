# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.assembly import FuncArgs, Region, Scope, SymbolRef
from loom.builder import IRBuilder
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.test import ALL_TEST_OPS
from loom.dsl import AttrDef, Op, RegionDef
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.ir import I32, Block
from loom.ir import Region as IRRegion


def test_region_arg_source_accepts_func_args_field() -> None:
    op = Op(
        "test.projected_func",
        attrs=[AttrDef("callee", "symbol")],
        regions=[
            RegionDef("body"),
            RegionDef("config", arg_source="args"),
        ],
        format=[
            SymbolRef("callee"),
            Scope([FuncArgs("args")]),
            Region("body"),
            Region("config"),
        ],
    )

    assert op.regions[1].arg_source == "args"


def test_builder_seeds_projected_func_args_region() -> None:
    builder = IRBuilder(insertion_block=Block())
    builder.register_ops(ALL_TEST_OPS)
    builder.register_types(ALL_BUILTIN_TYPES)

    x = builder.value("x", I32)
    config = IRRegion(blocks=[Block()])
    body = IRRegion(blocks=[Block()])
    builder.build(
        "test.split_func",
        func_args=[x],
        attributes={"callee": "projected"},
        regions=[config, body],
    )

    assert body.blocks[0].arg_ids == [x.id]
    assert len(config.blocks[0].arg_ids) == 1
    config_arg_id = config.blocks[0].arg_ids[0]
    assert config_arg_id != x.id
    assert builder.module.values[config_arg_id].name == "x"
    assert builder.module.values[config_arg_id].type == I32
    assert builder.module.values[config_arg_id].is_block_arg


def test_bytecode_preserves_projected_func_regions() -> None:
    parser = Parser()
    parser.register_ops(ALL_TEST_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    module = parser.parse(
        "test.split_func @projected(%arg: i32, %other: index) {\n"
        "  test.use %arg : i32\n"
        "  test.use %other : index\n"
        "  test.yield\n"
        "} launch {\n"
        "  test.use %arg : i32\n"
        "  test.use %other : index\n"
        "  test.yield\n"
        "}\n"
    )
    loaded = read_module(write_module(module))

    op = loaded.symbols[0].op
    assert op is not None
    assert len(op.regions) == 2
    config_arg_ids = op.regions[0].blocks[0].arg_ids
    body_arg_ids = op.regions[1].blocks[0].arg_ids
    assert len(body_arg_ids) == len(config_arg_ids) == 2
    assert body_arg_ids[0] != config_arg_ids[0]
    assert loaded.values[body_arg_ids[0]].name == "arg"
    assert loaded.values[config_arg_ids[0]].name == "arg"
