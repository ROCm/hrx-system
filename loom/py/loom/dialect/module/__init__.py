# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Module dialect: compile-time source composition metadata."""

from loom.dialect.module.defs import ALL_MODULE_OPS, module_import, module_ops

__all__ = [
    "ALL_MODULE_OPS",
    "module_import",
    "module_ops",
]
