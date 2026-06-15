# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python support for constructing, inspecting, and serializing Loom IR."""

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
