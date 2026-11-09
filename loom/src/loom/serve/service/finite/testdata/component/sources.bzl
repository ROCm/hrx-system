# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source closures for resident and demand-read finite-component fixtures."""

SOURCE_FILES = [
    "control.loom",
    "copy.loom",
    "demand_control.loom",
    "demand_kernels.loom",
    "demand_prepare.loom",
    "kernels.loom",
    "policy.loom",
    "prepare.loom",
    "retained_control.loom",
    "retained_kernels.loom",
    "retained_prepare.loom",
]

SOURCES = ["//loom/src/loom/serve/service/finite/testdata/component:" + path for path in SOURCE_FILES]
