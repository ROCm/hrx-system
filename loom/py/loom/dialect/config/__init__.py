# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Config dialect: compile/link-time configuration symbols."""

from loom.dialect.config.defs import (
    ALL_CONFIG_OPS,
    config_decl,
    config_def,
    config_get,
    config_ops,
)

__all__ = [
    "config_ops",
    "config_decl",
    "config_def",
    "config_get",
    "ALL_CONFIG_OPS",
]
