# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Template dialect: compile-time callable-family specialization."""

from loom.dialect.template.defs import (
    ALL_TEMPLATE_OPS,
    template_apply,
    template_call,
    template_decl,
    template_def,
    template_ops,
    template_return,
    template_ukernel,
)

__all__ = [
    "ALL_TEMPLATE_OPS",
    "template_apply",
    "template_call",
    "template_decl",
    "template_def",
    "template_ops",
    "template_return",
    "template_ukernel",
]
