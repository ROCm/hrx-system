# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.mlir.locations import MlirLocationConverter
from loom.ir import FileLocation, Module, OpaqueLocation


class _Location:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


def test_location_converter_maps_simple_file_location() -> None:
    module = Module()
    converter = MlirLocationConverter(module)

    location_id = converter.opaque(_Location('loc("kernel.mlir":7:8)'))

    location = module.locations.get(location_id)
    assert isinstance(location, FileLocation)
    assert module.sources[location.source_id] == "kernel.mlir"
    assert location.start_line == 7
    assert location.start_col == 8


def test_location_converter_preserves_unknown_shapes_as_opaque() -> None:
    module = Module()
    converter = MlirLocationConverter(module)

    location_id = converter.opaque(_Location('loc(callsite("callee" at "caller"))'))

    location = module.locations.get(location_id)
    assert isinstance(location, OpaqueLocation)
    assert module.sources[location.source_id] == "mlir"
