# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact output and serialization coverage for the Python importer example."""

from pathlib import Path

from importer import MODULE_OPS, build_copy_module
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.importers import print_loom_module


def _expected_text() -> str:
    return Path(__file__).with_name("expected.loom").read_text()


def test_constructed_module_has_exact_canonical_text() -> None:
    module = build_copy_module()

    assert print_loom_module(module, ops=MODULE_OPS) == _expected_text()


def test_constructed_module_round_trips_through_bytecode() -> None:
    module = build_copy_module()
    bytecode = write_module(module, op_decls=MODULE_OPS)

    loaded = read_module(bytecode, op_decls=MODULE_OPS, verify=True)

    assert print_loom_module(loaded, ops=MODULE_OPS) == _expected_text()


def main() -> None:
    test_constructed_module_has_exact_canonical_text()
    test_constructed_module_round_trips_through_bytecode()


if __name__ == "__main__":
    main()
