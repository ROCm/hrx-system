# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source closure for the pixel-observing finite projector fixture."""

SOURCE_FILES = [
    "control.loom",
    "prepare.loom",
    "project.loom",
]

SOURCES = ["//loom/src/loom/serve/fixture/projector:" + path for path in SOURCE_FILES]
