# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dsl import FACT_IDENTITY, POISON, OpPhase


def test_scalar_operation_phases_are_complete() -> None:
    operation_by_name = {operation.name: operation for operation in ALL_SCALAR_OPS}
    non_executable_names = {operation.name for operation in ALL_SCALAR_OPS if operation.effective_phase is not OpPhase.EXECUTABLE}

    assert non_executable_names == {"scalar.assume", "scalar.poison"}
    assert operation_by_name["scalar.assume"].has_trait(FACT_IDENTITY.name)
    assert operation_by_name["scalar.poison"].has_trait(POISON.name)
