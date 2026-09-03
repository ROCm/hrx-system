# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core VM instruction declarations."""

from iree.vm.bytecode.spec.isa.core.control import CONTROL_FAMILY, CONTROL_INSTRUCTIONS
from iree.vm.bytecode.spec.isa.core.integer import INTEGER_FAMILY, INTEGER_INSTRUCTIONS

FAMILIES = (CONTROL_FAMILY, INTEGER_FAMILY)
INSTRUCTIONS = (*CONTROL_INSTRUCTIONS, *INTEGER_INSTRUCTIONS)
