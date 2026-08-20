# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete versioned Core and optional HAL instruction specification."""

from model.isa.contracts import CONTRACTS
from model.isa.core.abi import FAMILY as ABI_FAMILY
from model.isa.core.abi import INSTRUCTIONS as ABI_INSTRUCTIONS
from model.isa.core.buffer import FAMILY as BUFFER_FAMILY
from model.isa.core.buffer import INSTRUCTIONS as BUFFER_INSTRUCTIONS
from model.isa.core.constant import FAMILY as CONSTANT_FAMILY
from model.isa.core.constant import INSTRUCTIONS as CONSTANT_INSTRUCTIONS
from model.isa.core.control import FAMILY as CONTROL_FAMILY
from model.isa.core.control import INSTRUCTIONS as CONTROL_INSTRUCTIONS
from model.isa.core.conversion import FAMILY as CONVERSION_FAMILY
from model.isa.core.conversion import INSTRUCTIONS as CONVERSION_INSTRUCTIONS
from model.isa.core.float import FAMILY as FLOAT_FAMILY
from model.isa.core.float import INSTRUCTIONS as FLOAT_INSTRUCTIONS
from model.isa.core.function import FAMILY as FUNCTION_FAMILY
from model.isa.core.function import INSTRUCTIONS as FUNCTION_INSTRUCTIONS
from model.isa.core.globals import FAMILY as GLOBAL_FAMILY
from model.isa.core.globals import INSTRUCTIONS as GLOBAL_INSTRUCTIONS
from model.isa.core.integer import FAMILY as INTEGER_FAMILY
from model.isa.core.integer import INSTRUCTIONS as INTEGER_INSTRUCTIONS
from model.isa.core.ref import FAMILY as REF_FAMILY
from model.isa.core.ref import INSTRUCTIONS as REF_INSTRUCTIONS
from model.isa.core.stack import FAMILY as STACK_FAMILY
from model.isa.core.stack import INSTRUCTIONS as STACK_INSTRUCTIONS
from model.isa.core.value import FAMILY as VALUE_FAMILY
from model.isa.core.value import INSTRUCTIONS as VALUE_INSTRUCTIONS
from model.isa.hal.buffer import FAMILY as HAL_BUFFER_FAMILY
from model.isa.hal.buffer import INSTRUCTIONS as HAL_BUFFER_INSTRUCTIONS
from model.isa.hal.command_buffer import FAMILY as HAL_COMMAND_BUFFER_FAMILY
from model.isa.hal.command_buffer import (
    INSTRUCTIONS as HAL_COMMAND_BUFFER_INSTRUCTIONS,
)
from model.isa.hal.device import FAMILY as HAL_DEVICE_FAMILY
from model.isa.hal.device import INSTRUCTIONS as HAL_DEVICE_INSTRUCTIONS
from model.isa.hal.queue import FAMILY as HAL_QUEUE_FAMILY
from model.isa.hal.queue import INSTRUCTIONS as HAL_QUEUE_INSTRUCTIONS
from model.isa.hal.semaphore import FAMILY as HAL_SEMAPHORE_FAMILY
from model.isa.hal.semaphore import INSTRUCTIONS as HAL_SEMAPHORE_INSTRUCTIONS
from model.isa.selectors import SELECTOR_TABLES, SELECTOR_VALUES
from model.isa.validation import FIELD_RULES, RECORD_RULES
from model.schema import I16, I32, U8, U16, U32
from model.specification import CORE_DOMAIN, HAL_DOMAIN, Specification

ENCODINGS = (U8, I16, U16, I32, U32)

FAMILIES = (
    ABI_FAMILY,
    BUFFER_FAMILY,
    CONSTANT_FAMILY,
    CONTROL_FAMILY,
    CONVERSION_FAMILY,
    FLOAT_FAMILY,
    FUNCTION_FAMILY,
    GLOBAL_FAMILY,
    HAL_BUFFER_FAMILY,
    HAL_COMMAND_BUFFER_FAMILY,
    HAL_DEVICE_FAMILY,
    HAL_QUEUE_FAMILY,
    HAL_SEMAPHORE_FAMILY,
    INTEGER_FAMILY,
    REF_FAMILY,
    STACK_FAMILY,
    VALUE_FAMILY,
)

INSTRUCTIONS = (
    *ABI_INSTRUCTIONS,
    *BUFFER_INSTRUCTIONS,
    *CONSTANT_INSTRUCTIONS,
    *CONTROL_INSTRUCTIONS,
    *CONVERSION_INSTRUCTIONS,
    *FLOAT_INSTRUCTIONS,
    *FUNCTION_INSTRUCTIONS,
    *GLOBAL_INSTRUCTIONS,
    *HAL_BUFFER_INSTRUCTIONS,
    *HAL_COMMAND_BUFFER_INSTRUCTIONS,
    *HAL_DEVICE_INSTRUCTIONS,
    *HAL_QUEUE_INSTRUCTIONS,
    *HAL_SEMAPHORE_INSTRUCTIONS,
    *INTEGER_INSTRUCTIONS,
    *REF_INSTRUCTIONS,
    *STACK_INSTRUCTIONS,
    *VALUE_INSTRUCTIONS,
)

ISA_ENTITIES = (
    *ENCODINGS,
    *FIELD_RULES,
    *RECORD_RULES,
    *CONTRACTS,
    *SELECTOR_TABLES,
    *SELECTOR_VALUES,
    *FAMILIES,
    *INSTRUCTIONS,
)

ISA_SPECIFICATION = Specification(
    "iree.vm.bytecode.isa",
    (CORE_DOMAIN, HAL_DOMAIN),
    ISA_ENTITIES,
)
