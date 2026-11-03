# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source closure for the text serving fixture."""

SOURCE_FILES = [
    "advance.loom",
    "control.loom",
    "model.json",
    "prepare.loom",
    "sequence.loom",
    "storage.loom",
    "tokenizer.json",
    "weights.loom",
]

SOURCES = ["//loom/src/loom/serve/fixture/text:" + path for path in SOURCE_FILES]
