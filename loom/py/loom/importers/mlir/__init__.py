# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MLIR/IREE HAL kernel importer for Loom."""

from loom.importers.core import print_loom_module
from loom.importers.mlir.importer import (
    MlirImportOptions,
    format_import_report,
    import_mlir_file,
    import_mlir_module,
)

__all__ = [
    "MlirImportOptions",
    "format_import_report",
    "import_mlir_file",
    "import_mlir_module",
    "print_loom_module",
]
