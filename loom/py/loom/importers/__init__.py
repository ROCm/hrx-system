# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Library entry points for importing foreign kernel IR into Loom."""

from loom.importers.core import (
    Diagnostic,
    DiagnosticEngine,
    ImportOptions,
    ImportResult,
    KernelArgumentSpec,
    KernelConfigArgumentSpec,
    KernelLaunchConfigSpec,
    KernelModuleShell,
    KernelModuleSpec,
    LoomDiagnosticError,
    NameAllocator,
    SourceImportSession,
    build_static_launch_config,
    create_kernel_module,
    kernel_module_ops,
    print_loom_module,
)

__all__ = [
    "Diagnostic",
    "DiagnosticEngine",
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
    "build_static_launch_config",
    "create_kernel_module",
    "kernel_module_ops",
    "print_loom_module",
]
