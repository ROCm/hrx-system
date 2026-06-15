# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pass pipeline control dialect."""

from loom.dialect.pass_.defs import (
    ALL_PASS_OPS,
    PassAnchor,
    PassRepeatMode,
    pass_call,
    pass_fail,
    pass_for,
    pass_halt,
    pass_ops,
    pass_pipeline,
    pass_repeat,
    pass_run,
    pass_where,
    pass_yield,
)

__all__ = [
    "ALL_PASS_OPS",
    "PassAnchor",
    "PassRepeatMode",
    "pass_call",
    "pass_fail",
    "pass_for",
    "pass_halt",
    "pass_ops",
    "pass_pipeline",
    "pass_repeat",
    "pass_run",
    "pass_where",
    "pass_yield",
]
