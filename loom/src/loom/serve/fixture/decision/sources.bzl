# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source closure for the decision serving fixture."""

SOURCE_FILES = [
    "card.json",
    "config.loom",
    "control.loom",
    "decide.loom",
    "model.json",
    "prepare.loom",
    "service.loom",
]

SOURCES = ["//loom/src/loom/serve/fixture/decision:" + path for path in SOURCE_FILES]
