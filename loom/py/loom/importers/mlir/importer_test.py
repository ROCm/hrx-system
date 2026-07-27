# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.mlir.importer import (
    choose_chunk,
    split_input_file,
)


def test_split_input_chunk_selection_by_export() -> None:
    chunks = split_input_file(
        """
module {
  func.func @a()
}
// -----
module {
  hal.executable.export public @b
  func.func @impl()
}
"""
    )

    assert choose_chunk(chunks, "b") == chunks[1]
