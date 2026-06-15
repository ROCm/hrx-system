# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core infrastructure shared by Loom foreign IR importers."""

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
from loom.importers.core.kernel import (
    KernelArgumentSpec,
    KernelConfigArgumentSpec,
    KernelLaunchConfigSpec,
    KernelModuleShell,
    KernelModuleSpec,
    build_static_launch_config,
    create_kernel_module,
    kernel_module_ops,
    normalize_launch_tuple,
    normalize_workgroup_size,
    target_preset_amdgpu_kind,
)
from loom.importers.core.names import (
    NameAllocator,
    sanitize_identifier,
    sanitize_symbol,
    source_name,
)
from loom.importers.core.options import ImportOptions, ImportResult
from loom.importers.core.printing import print_loom_module
from loom.importers.core.session import (
    ConversionRecord,
    ImportBodyReport,
    SourceImportSession,
    source_key,
)

__all__ = [
    "ConversionRecord",
    "Diagnostic",
    "DiagnosticEngine",
    "DiagnosticFieldRef",
    "DiagnosticHighlightRange",
    "DiagnosticParam",
    "DiagnosticRelatedLocation",
    "DiagnosticSeverity",
    "ImportBodyReport",
    "ImportOptions",
    "ImportResult",
    "KernelArgumentSpec",
    "KernelConfigArgumentSpec",
    "KernelLaunchConfigSpec",
    "KernelModuleShell",
    "KernelModuleSpec",
    "LoomDiagnosticError",
    "NameAllocator",
    "SourceImportSession",
    "SourceProvenance",
    "SourceRange",
    "build_static_launch_config",
    "create_kernel_module",
    "kernel_module_ops",
    "normalize_launch_tuple",
    "normalize_workgroup_size",
    "print_loom_module",
    "sanitize_identifier",
    "sanitize_symbol",
    "source_name",
    "source_key",
    "target_preset_amdgpu_kind",
]
