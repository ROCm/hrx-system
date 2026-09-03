# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for portable pipeline text and bytecode behavior."""

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.group import ALL_GROUP_OPS, ALL_GROUP_TYPES
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.pipeline import ALL_PIPELINE_OPS, ALL_PIPELINE_TYPES
from loom.dialect.test import ALL_TEST_OPS
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer
from loom.ir import Module

_OPS = (
    *ALL_TEST_OPS,
    *ALL_FUNC_OPS,
    *ALL_BUFFER_OPS,
    *ALL_INDEX_OPS,
    *ALL_GROUP_OPS,
    *ALL_PIPELINE_OPS,
)
_TYPES = (*ALL_BUILTIN_TYPES, *ALL_GROUP_TYPES, *ALL_PIPELINE_TYPES)


def _parse_module(text: str) -> Module:
    parser = Parser()
    parser.register_ops(_OPS)
    parser.register_types(_TYPES)
    return parser.parse(text)


def _print_module(module: Module) -> str:
    printer = Printer()
    printer.register_ops(_OPS)
    printer.register_types(_TYPES)
    return printer.print_module(module)


def _roundtrip(text: str) -> None:
    module = _parse_module(text)
    assert _print_module(module) == text
    assert _print_module(read_module(write_module(module))) == text


def test_split_k_pipeline_roundtrip() -> None:
    text = """test.target<low_core> @array

func.def @product(%lhs: buffer, %rhs: buffer, %partial: buffer) {
  func.return
}

func.def @reduce(%partial0: buffer, %partial1: buffer, %bias: buffer, %output: buffer) {
  func.return
}

pipeline.def<kernel> public target(@array) @split_k() launch(%lhs: buffer, %rhs: buffer, %bias: buffer, %output: buffer) {
  %product_lanes = index.constant 2 : index
  %reducer_lanes = index.constant 1 : index
  %ring_capacity = index.constant 2 : index
  %base = index.constant 0 : offset
  %products = group.create %product_lanes : index -> group
  %reducers = group.create %reducer_lanes : index -> group
  %lhs_view = buffer.view %lhs[%base] : buffer -> view<2x8x8xi8>
  %rhs_view = buffer.view %rhs[%base] : buffer -> view<2x8x8xi8>
  %bias_view = buffer.view %bias[%base] : buffer -> view<8x8xi32>
  %output_view = buffer.view %output[%base] : buffer -> view<8x8xi32>
  %lhs_tiles = pipeline.scatter %lhs_view across %products : view<2x8x8xi8>, group -> pipeline.flow<tile<8x8xi8>>
  %rhs_tiles = pipeline.scatter %rhs_view across %products : view<2x8x8xi8>, group -> pipeline.flow<tile<8x8xi8>>
  %bias_tiles = pipeline.read %bias_view on %reducers : view<8x8xi32>, group -> pipeline.flow<tile<8x8xi32>>
  %partials = pipeline.stage @product on %products(%lhs_tiles, %rhs_tiles) : (group, pipeline.flow<tile<8x8xi8>>, pipeline.flow<tile<8x8xi8>>) -> (pipeline.flow<tile<8x8xi32>>)
  %buffered = pipeline.buffer %partials capacity %ring_capacity : (pipeline.flow<tile<8x8xi32>>, index) -> pipeline.flow<tile<8x8xi32>>
  %result = pipeline.reduce @reduce from %products(%buffered) to %reducers(%bias_tiles) : (group, pipeline.flow<tile<8x8xi32>>) to (group, pipeline.flow<tile<8x8xi32>>) -> (pipeline.flow<tile<8x8xi32>>)
  pipeline.write %result to %output_view : pipeline.flow<tile<8x8xi32>>, view<8x8xi32>
  pipeline.return
}
"""
    module = _parse_module(text)
    definition = next(symbol.op for symbol in module.symbols if symbol.op is not None and symbol.op.name == "pipeline.def")
    assert definition.attributes["scope"] == "kernel"
    assert definition.attributes["specialization_count"] == 0
    _roundtrip(text)


def test_generic_pipeline_roundtrip() -> None:
    _roundtrip("pipeline.def @generic(%batch: index) launch(%input: buffer) {\n  pipeline.return\n}\n")
