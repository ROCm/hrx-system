# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Converters for MLIR func boundary ops."""

from __future__ import annotations

from loom.importers.mlir.converter import ConverterRegistry
from loom.importers.mlir.model import MlirConversionContext, SourceOp


def register(registry: ConverterRegistry) -> None:
    registry.register("func.return", convert_return)


def convert_return(op: SourceOp, context: MlirConversionContext) -> bool:
    if op.operands:
        context.record_blocked(
            op.text, "func.return with values needs kernel result mapping"
        )
    else:
        context.record_converted(op.text, "kernel.return emitted by kernel shell")
    return True
