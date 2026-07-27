# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for IREE codegen bridge ops at the kernel import boundary."""

from __future__ import annotations

from loom.importers.mlir.converter import ConverterRegistry
from loom.importers.mlir.model import MlirConversionContext, SourceOp


def register(registry: ConverterRegistry) -> None:
    registry.register("iree_codegen.load_from_buffer", convert_tensor_buffer_bridge)
    registry.register("iree_codegen.store_to_buffer", convert_tensor_buffer_bridge)


def convert_tensor_buffer_bridge(
    op: SourceOp,
    context: MlirConversionContext,
) -> bool:
    """Blocks tensor/buffer bridge ops above the kernel import boundary."""

    context.record_blocked(
        op.text,
        "IREE tensor/buffer bridge op is above the post-bufferized kernel boundary",
    )
    return True
