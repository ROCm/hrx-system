# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared module-symbol vocabulary."""

from loom.dsl import EnumCase, EnumDef

Linkage = EnumDef(
    "Linkage",
    [
        EnumCase(
            "linkonce",
            1,
            doc="Coalesce definitions with producer-promised equivalent semantics.",
        ),
    ],
    doc=("Definition linkage. Absent (0) is an ordinary strong definition. Linkers validate symbol contracts, not link-once bodies."),
    c_type="loom_symbol_linkage_t",
    c_const_prefix="LOOM_SYMBOL_LINKAGE",
    c_include="loom/ir/ir.h",
)
