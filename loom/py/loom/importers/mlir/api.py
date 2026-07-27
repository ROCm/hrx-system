# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optional IREE Python MLIR binding import helpers."""

from __future__ import annotations

import importlib.machinery
import sys
from typing import Any

from loom.diagnostics import (
    Diagnostic,
    DiagnosticSeverity,
    LoomDiagnosticError,
)


def import_iree_ir(*, prefer_abi3_extensions: bool = False) -> Any:
    """Import IREE's MLIR bindings and register dialect modules."""
    if prefer_abi3_extensions:
        _prefer_abi3_extensions()
    try:
        from iree.compiler import ir  # type: ignore[import-not-found]
        from iree.compiler.dialects import (  # type: ignore[import-not-found]
            affine,  # noqa: F401
            amdgpu,  # noqa: F401
            arith,  # noqa: F401
            builtin,  # noqa: F401
            func,  # noqa: F401
            gpu,  # noqa: F401
            hal,  # noqa: F401
            iree_codegen,  # noqa: F401
            iree_gpu,  # noqa: F401
            memref,  # noqa: F401
            scf,  # noqa: F401
            vector,  # noqa: F401
        )

        return ir
    except Exception as exc:
        diagnostic = Diagnostic(
            DiagnosticSeverity.ERROR,
            "failed to import IREE Python MLIR bindings",
            details=(
                "Install iree-base-compiler or put the local IREE compiler "
                "Python bindings on PYTHONPATH.",
                f"import error: {exc!r}",
            ),
        )
        raise LoomDiagnosticError((diagnostic,)) from exc


def _prefer_abi3_extensions() -> None:
    """Prefer freshly built stable-ABI extension modules in local build trees."""
    sys.path_hooks[:] = [
        importlib.machinery.FileFinder.path_hook(
            (importlib.machinery.ExtensionFileLoader, [".abi3.so"]),
            (
                importlib.machinery.SourceFileLoader,
                importlib.machinery.SOURCE_SUFFIXES,
            ),
            (
                importlib.machinery.SourcelessFileLoader,
                importlib.machinery.BYTECODE_SUFFIXES,
            ),
        )
    ]
    sys.path_importer_cache.clear()
