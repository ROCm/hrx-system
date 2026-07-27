# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C builder declaration and source generation for Loom ops."""

from __future__ import annotations

from loom.gen.ops.c_builder_declarations import generate_builder_header_lines
from loom.gen.ops.c_builder_source import generate_builders_c

__all__ = [
    "generate_builder_header_lines",
    "generate_builders_c",
]
