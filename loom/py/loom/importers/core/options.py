# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Importer options and result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loom.diagnostics import DiagnosticEngine
from loom.ir import Module


@dataclass(frozen=True, slots=True)
class ImportOptions:
    """Common options for library-form importers."""

    verify_structure: bool = True
    include_report: bool = False


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Result of a successful whole-module import."""

    module: Module
    diagnostics: DiagnosticEngine
    report: Any | None = None
