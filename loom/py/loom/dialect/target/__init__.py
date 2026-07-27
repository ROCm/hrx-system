# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target planning record dialect."""

from loom.dialect.target.defs import (
    ALL_TARGET_OPS,
    TARGET_COMMON_OVERRIDE_ATTRS,
    ArtifactFormatAttr,
    ExportAbiKind,
    ExportLinkage,
    GenericTargetKind,
    SnapshotCodegenFormat,
    target_generic,
    target_ops,
    target_record_attrs,
)

__all__ = [
    "ALL_TARGET_OPS",
    "ArtifactFormatAttr",
    "ExportAbiKind",
    "ExportLinkage",
    "GenericTargetKind",
    "SnapshotCodegenFormat",
    "TARGET_COMMON_OVERRIDE_ATTRS",
    "target_generic",
    "target_ops",
    "target_record_attrs",
]
