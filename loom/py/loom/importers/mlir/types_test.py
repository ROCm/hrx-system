# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

from loom.importers.mlir.types import MlirTypeConverter
from loom.ir import INDEX, DynamicDim, ShapedType, StaticDim, TypeKind


def test_type_converter_maps_scalars_through_loom_parser() -> None:
    converter = MlirTypeConverter()

    assert converter.map_text("index") == INDEX


def test_type_converter_maps_memref_to_view() -> None:
    converter = MlirTypeConverter()

    result = converter.map_text("memref<4x?xf32, #hal.descriptor_type<storage_buffer>>")

    assert isinstance(result, ShapedType)
    assert result.type_kind == TypeKind.VIEW
    assert result.dims == (StaticDim(4), DynamicDim())


def test_type_converter_coerces_constants() -> None:
    converter = MlirTypeConverter()

    assert converter.coerce_constant_value("4", "index") == 4
    assert converter.coerce_constant_value("-inf", "f32") == float("-inf")
    assert math.isnan(converter.coerce_constant_value("nan", "f32"))
