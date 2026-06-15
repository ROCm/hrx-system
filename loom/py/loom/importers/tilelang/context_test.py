# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

import loom
from loom.importers.tilelang.context import (
    TileLangConversionContext,
    TileLangReducerInfo,
)
from loom.ir import INDEX


@dataclass
class _UnhashableIndexSource:
    name: str


def test_maps_unhashable_index_sources_by_identity() -> None:
    _, builder = loom.module_builder()
    ref = builder.value("index", INDEX)
    source = _UnhashableIndexSource("threadIdx.x")
    equal_source = _UnhashableIndexSource("threadIdx.x")
    context = TileLangConversionContext(builder=builder)

    context.map_index_value(source, ref)

    assert context.mapped_index_value(source) is ref
    assert context.mapped_index_value(equal_source) is None


@dataclass
class _HandleSource:
    name: str
    dtype: str = "handle"


@dataclass
class Var:
    name: str
    dtype: str = "handle"


@dataclass
class Buffer:
    name: str
    shape: tuple[object, ...]
    dtype: str
    data: object
    _scope: str = "global"

    def scope(self) -> str:
        return self._scope


def test_maps_buffer_wrappers_by_source_semantics() -> None:
    _, builder = loom.module_builder()
    ref = builder.value("view", INDEX)
    data = _HandleSource("src")
    source = Buffer("src", (16,), "float32", data)
    equal_wrapper = Buffer("src", (16,), "float32", _HandleSource("src"))
    different_shape = Buffer("src", (32,), "float32", _HandleSource("src"))
    context = TileLangConversionContext(builder=builder)

    context.map_value(source, ref, "view<1xf32>")

    assert context.mapped(equal_wrapper) is ref
    assert context.mapped_value_type(equal_wrapper) == "view<1xf32>"
    assert context.mapped(different_shape) is None


def test_maps_reducer_info_by_buffer_data_semantics() -> None:
    _, builder = loom.module_builder()
    source = Var("reducer")
    equal_source = Var("reducer")
    different_source = Var("other")
    context = TileLangConversionContext(builder=builder)
    reducer_info = TileLangReducerInfo(operation="sum", replication="none")

    context.map_reducer_info(source, reducer_info)

    assert context.reducer_info(equal_source) == reducer_info
    assert context.reducer_info(different_source) is None
