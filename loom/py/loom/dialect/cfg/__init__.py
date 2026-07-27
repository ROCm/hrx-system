# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CFG dialect: unstructured control-flow terminators.

Provides explicit branch terminators for regions that have been lowered out of
structured control flow. Successor edges are semantic block references; labels
remain text syntax and debug metadata.
"""

from loom.dialect.cfg.defs import (
    ALL_CFG_OPS,
    cfg_br,
    cfg_cond_br,
    cfg_ops,
)

__all__ = [
    "cfg_ops",
    "cfg_br",
    "cfg_cond_br",
    "ALL_CFG_OPS",
]
