# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core VM instruction declarations."""

from iree.vm.bytecode.spec.isa.core.abi import ABI_FAMILY, ABI_INSTRUCTIONS
from iree.vm.bytecode.spec.isa.core.constant import (
    CONSTANT_FAMILY,
    CONSTANT_INSTRUCTIONS,
)
from iree.vm.bytecode.spec.isa.core.control import (
    CONTROL_CALL_TARGET_SELECTOR,
    CONTROL_FAMILY,
    CONTROL_INSTRUCTIONS,
    CONTROL_STATUS_SELECTOR,
)
from iree.vm.bytecode.spec.isa.core.conversion import (
    CONVERSION_FAMILY,
    CONVERSION_INSTRUCTIONS,
    CONVERSION_SELECTORS,
)
from iree.vm.bytecode.spec.isa.core.float import (
    FLOAT_FAMILY,
    FLOAT_INSTRUCTIONS,
    FLOAT_SELECTORS,
)
from iree.vm.bytecode.spec.isa.core.function import (
    FUNCTION_FAMILY,
    FUNCTION_INSTRUCTIONS,
)
from iree.vm.bytecode.spec.isa.core.globals import GLOBAL_FAMILY, GLOBAL_INSTRUCTIONS
from iree.vm.bytecode.spec.isa.core.integer import (
    INTEGER_COMPARE_SELECTOR,
    INTEGER_FAMILY,
    INTEGER_INSTRUCTIONS,
)
from iree.vm.bytecode.spec.isa.core.ref import REF_FAMILY, REF_INSTRUCTIONS
from iree.vm.bytecode.spec.isa.core.value import VALUE_FAMILY, VALUE_INSTRUCTIONS

FAMILIES = (
    CONTROL_FAMILY,
    VALUE_FAMILY,
    CONSTANT_FAMILY,
    FUNCTION_FAMILY,
    GLOBAL_FAMILY,
    INTEGER_FAMILY,
    FLOAT_FAMILY,
    CONVERSION_FAMILY,
    ABI_FAMILY,
    REF_FAMILY,
)
SELECTORS = (
    CONTROL_CALL_TARGET_SELECTOR,
    CONTROL_STATUS_SELECTOR,
    INTEGER_COMPARE_SELECTOR,
    *FLOAT_SELECTORS,
    *CONVERSION_SELECTORS,
)
INSTRUCTIONS = (
    *CONTROL_INSTRUCTIONS,
    *VALUE_INSTRUCTIONS,
    *CONSTANT_INSTRUCTIONS,
    *FUNCTION_INSTRUCTIONS,
    *GLOBAL_INSTRUCTIONS,
    *INTEGER_INSTRUCTIONS,
    *FLOAT_INSTRUCTIONS,
    *CONVERSION_INSTRUCTIONS,
    *ABI_INSTRUCTIONS,
    *REF_INSTRUCTIONS,
)
