# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete Core 0.0 module-container specification."""

from model.module.v0.numeric import ENTITIES as NUMERIC_ENTITIES
from model.module.v0.records import ENTITIES as RECORD_ENTITIES
from model.module.v0.structure import ENTITIES as STRUCTURE_ENTITIES
from model.module.validation import ENTITIES as VALIDATION_ENTITIES
from model.schema import U8, U16, U32, U64
from model.specification import CORE_DOMAIN, Specification

MODULE_ENTITIES = (
    U8,
    U16,
    U32,
    U64,
    *VALIDATION_ENTITIES,
    *NUMERIC_ENTITIES,
    *RECORD_ENTITIES,
    *STRUCTURE_ENTITIES,
)

MODULE_SPECIFICATION = Specification(
    "iree.vm.bytecode.module",
    (CORE_DOMAIN,),
    MODULE_ENTITIES,
)
