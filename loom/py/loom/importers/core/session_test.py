# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

import loom
from loom.importers.core import SourceImportSession
from loom.ir import I32


@dataclass
class _UnhashableSource:
    name: str


class _TirLikeSource:
    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return True


def test_maps_unhashable_source_objects_by_identity() -> None:
    _, builder = loom.module_builder()
    ref = builder.value("value", I32)
    source = _UnhashableSource("tilelang-node")
    equal_source = _UnhashableSource("tilelang-node")
    session = SourceImportSession(builder=builder)

    session.map_value(source, ref)

    assert session.mapped(source) is ref
    assert session.mapped(equal_source) is None
    assert session.mapped_value_type(source) == "i32"
    assert session.result_name(source) == "value"


def test_maps_hashable_foreign_source_objects_by_identity() -> None:
    _, builder = loom.module_builder()
    ref = builder.value("value", I32)
    source = _TirLikeSource("num_tokens")
    equalish_source = _TirLikeSource("num_tokens")
    session = SourceImportSession(builder=builder)

    session.map_value(source, ref)

    assert session.mapped(source) is ref
    assert session.mapped(equalish_source) is None
