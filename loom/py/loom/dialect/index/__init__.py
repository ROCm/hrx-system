# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Index dialect: logical coordinate and address-boundary scalar ops."""

from loom.dialect.index.defs import (
    ALL_INDEX_OPS,
    IndexPredicate,
    index_ops,
)

__all__ = [
    "index_ops",
    "IndexPredicate",
    "ALL_INDEX_OPS",
]
