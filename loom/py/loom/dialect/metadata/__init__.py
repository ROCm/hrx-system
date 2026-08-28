# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Metadata dialect: stable keyed program metadata records."""

from loom.dialect.metadata.defs import (
    ALL_METADATA_OPS,
    metadata_module,
    metadata_ops,
)

__all__ = [
    "metadata_ops",
    "metadata_module",
    "ALL_METADATA_OPS",
]
