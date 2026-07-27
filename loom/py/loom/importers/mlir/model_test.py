# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.mlir.model import mlir_source_key


class _MlirValue:
    def __init__(self, name: str) -> None:
        self._name = name
        self.type = "index"

    def get_name(self) -> str:
        return self._name


def test_mlir_source_key_matches_value_wrappers_by_ssa_name() -> None:
    assert mlir_source_key(_MlirValue("%c0")) == mlir_source_key(_MlirValue("%c0"))
    assert mlir_source_key(_MlirValue("%c0")) != mlir_source_key(_MlirValue("%c1"))
