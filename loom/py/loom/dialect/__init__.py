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
    channel,
    check,
    command,
    config,
    encoding,
    func,
    globals,
    group,
    hal,
    index,
    kernel,
    low,
    pass_,
    pipeline,
    pool,
    sanitizer,
    scalar,
    scf,
    target,
    template,
    tensor,
    test,
    tile,
    type,
    vector,
    view,
)

__all__ = [
    "buffer",
    "cfg",
    "channel",
    "check",
    "command",
    "config",
    "encoding",
    "func",
    "hal",
    "globals",
    "group",
    "index",
    "kernel",
    "low",
    "pass_",
    "pipeline",
    "pool",
    "sanitizer",
    "scalar",
    "scf",
    "target",
    "template",
    "tensor",
    "test",
    "tile",
    "type",
    "vector",
    "view",
]
