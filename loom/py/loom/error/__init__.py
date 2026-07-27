# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom structured error catalog.

Re-exports all error definitions from domain modules and collects them
into ALL_ERRORS for generator consumption.
"""

from loom.error.amdgpu import *  # noqa: F403
from loom.error.amdgpu import ALL_AMDGPU_ERRORS
from loom.error.backend import *  # noqa: F403
from loom.error.backend import ALL_BACKEND_ERRORS
from loom.error.bytecode import *  # noqa: F403
from loom.error.bytecode import ALL_BYTECODE_ERRORS
from loom.error.dominance import *  # noqa: F403
from loom.error.dominance import ALL_DOMINANCE_ERRORS
from loom.error.encoding import *  # noqa: F403
from loom.error.encoding import ALL_ENCODING_ERRORS
from loom.error.fold import *  # noqa: F403
from loom.error.fold import ALL_FOLD_ERRORS
from loom.error.lowering import *  # noqa: F403
from loom.error.lowering import ALL_LOWERING_ERRORS
from loom.error.parse import *  # noqa: F403
from loom.error.parse import ALL_PARSE_ERRORS
from loom.error.shape import *  # noqa: F403
from loom.error.shape import ALL_SHAPE_ERRORS
from loom.error.spirv import *  # noqa: F403
from loom.error.spirv import ALL_SPIRV_ERRORS
from loom.error.structure import *  # noqa: F403
from loom.error.structure import ALL_STRUCTURE_ERRORS
from loom.error.subrange import *  # noqa: F403
from loom.error.subrange import ALL_SUBRANGE_ERRORS
from loom.error.symbol import *  # noqa: F403
from loom.error.symbol import ALL_SYMBOL_ERRORS
from loom.error.target import *  # noqa: F403
from loom.error.target import ALL_TARGET_ERRORS
from loom.error.type import *  # noqa: F403
from loom.error.type import ALL_TYPE_ERRORS
from loom.error.wasm import *  # noqa: F403
from loom.error.wasm import ALL_WASM_ERRORS
from loom.error.x86 import *  # noqa: F403
from loom.error.x86 import ALL_X86_ERRORS
from loom.errors import ErrorDef

ALL_ERRORS: tuple[ErrorDef, ...] = (
    *ALL_TYPE_ERRORS,
    *ALL_SHAPE_ERRORS,
    *ALL_SUBRANGE_ERRORS,
    *ALL_ENCODING_ERRORS,
    *ALL_STRUCTURE_ERRORS,
    *ALL_DOMINANCE_ERRORS,
    *ALL_SYMBOL_ERRORS,
    *ALL_PARSE_ERRORS,
    *ALL_BYTECODE_ERRORS,
    *ALL_FOLD_ERRORS,
    *ALL_LOWERING_ERRORS,
    *ALL_BACKEND_ERRORS,
    *ALL_TARGET_ERRORS,
    *ALL_AMDGPU_ERRORS,
    *ALL_X86_ERRORS,
    *ALL_WASM_ERRORS,
    *ALL_SPIRV_ERRORS,
)
