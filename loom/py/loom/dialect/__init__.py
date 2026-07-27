# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom dialect package.

This package contains all loom dialects. Each dialect is a submodule
that defines its operations and exports them for use in IR construction.
"""

from loom.dialect import (
    buffer,
    cfg,
    check,
    config,
    encoding,
    func,
    globals,
    hal,
    index,
    kernel,
    llvmir,
    low,
    pass_,
    pool,
    sanitizer,
    scalar,
    scf,
    target,
    tensor,
    test,
    tile,
    vector,
    view,
    vm,
)

__all__ = [
    "buffer",
    "cfg",
    "check",
    "config",
    "encoding",
    "func",
    "hal",
    "globals",
    "index",
    "kernel",
    "llvmir",
    "low",
    "pass_",
    "pool",
    "sanitizer",
    "scalar",
    "scf",
    "target",
    "tensor",
    "test",
    "tile",
    "vector",
    "view",
    "vm",
]
