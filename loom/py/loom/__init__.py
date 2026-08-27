# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python support for constructing, inspecting, and serializing Loom IR."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.builders import (
        DialectBuilder,
        LoomBuilder,
        OpCallable,
        default_ops,
        default_types,
        module_builder,
    )
    from loom.diagnostics import (
        Diagnostic,
        DiagnosticEngine,
        DiagnosticFieldRef,
        DiagnosticHighlightRange,
        DiagnosticParam,
        DiagnosticRelatedLocation,
        DiagnosticSeverity,
        LoomDiagnosticError,
        SourceProvenance,
        SourceRange,
    )
    from loom.verify import ModuleVerifier, VerifierRegistry, verify_module

__all__ = [
    "DialectBuilder",
    "Diagnostic",
    "DiagnosticEngine",
    "DiagnosticFieldRef",
    "DiagnosticHighlightRange",
    "DiagnosticParam",
    "DiagnosticRelatedLocation",
    "DiagnosticSeverity",
    "LoomDiagnosticError",
    "LoomBuilder",
    "ModuleVerifier",
    "OpCallable",
    "SourceProvenance",
    "SourceRange",
    "VerifierRegistry",
    "default_ops",
    "default_types",
    "module_builder",
    "verify_module",
]

_LAZY_EXPORT_MODULES = {
    "DialectBuilder": "loom.builders",
    "Diagnostic": "loom.diagnostics",
    "DiagnosticEngine": "loom.diagnostics",
    "DiagnosticFieldRef": "loom.diagnostics",
    "DiagnosticHighlightRange": "loom.diagnostics",
    "DiagnosticParam": "loom.diagnostics",
    "DiagnosticRelatedLocation": "loom.diagnostics",
    "DiagnosticSeverity": "loom.diagnostics",
    "LoomDiagnosticError": "loom.diagnostics",
    "LoomBuilder": "loom.builders",
    "ModuleVerifier": "loom.verify",
    "OpCallable": "loom.builders",
    "SourceProvenance": "loom.diagnostics",
    "SourceRange": "loom.diagnostics",
    "VerifierRegistry": "loom.verify",
    "default_ops": "loom.builders",
    "default_types": "loom.builders",
    "module_builder": "loom.builders",
    "verify_module": "loom.verify",
}


def __getattr__(name: str) -> object:
    """Lazily resolves the root authoring surface for lightweight tools."""
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Returns module attributes including unresolved lazy exports."""
    return sorted(set(globals()) | set(__all__))
