# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deterministic projections of the declarative VM bytecode specification."""

from .c_execution import render_execution_tables
from .c_header import (
    render_isa_assertions,
    render_isa_family_header,
    render_isa_opcodes_header,
    render_isa_shared_selectors_header,
    render_module_assertions,
    render_module_header,
    shared_selector_table_ids,
)
from .c_module_validation import render_module_validation_obligations
from .c_tooling import render_tooling_isa_tables, render_tooling_module_tables
from .json_projection import render_projection_json
from .manifest import render_completeness_manifest, render_release_manifest
from .markdown import (
    isa_family_path,
    render_isa_family_markdown,
    render_isa_index_markdown,
    render_module_markdown,
    render_specification_index_markdown,
)

__all__ = (
    "render_completeness_manifest",
    "render_execution_tables",
    "isa_family_path",
    "render_isa_assertions",
    "render_isa_family_header",
    "render_isa_family_markdown",
    "render_isa_index_markdown",
    "render_isa_opcodes_header",
    "render_isa_shared_selectors_header",
    "render_tooling_isa_tables",
    "render_tooling_module_tables",
    "render_module_assertions",
    "render_module_header",
    "render_module_markdown",
    "render_module_validation_obligations",
    "render_specification_index_markdown",
    "render_projection_json",
    "render_release_manifest",
    "shared_selector_table_ids",
)
