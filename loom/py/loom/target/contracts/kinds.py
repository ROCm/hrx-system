# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Contract fragment discriminants."""

from enum import Enum, unique


@unique
class ContractSystem(Enum):
    """Shared interpreter system used by a contract fragment case."""

    DESCRIPTOR_RULE = "descriptor_rule"
    VALUE_ALIAS = "value_alias"
    VALUE_ELIDE = "value_elide"
    SOURCE_MEMORY = "source_memory"
    ENVIRONMENT = "environment"
    DESCRIPTOR_MATRIX = "descriptor_matrix"


@unique
class SourceValueKind(Enum):
    """Source value namespace referenced by a contract row."""

    OPERAND = "operand"
    RESULT = "result"
    TEMPORARY = "temporary"
    SOURCE_MEMORY_DYNAMIC_TERM = "source_memory_dynamic_term"
    SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET = "source_memory_dynamic_byte_offset"
