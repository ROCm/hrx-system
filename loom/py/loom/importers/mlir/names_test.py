# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.mlir.names import (
    operation_block_arg_names,
    operation_result_names,
    parse_result_name_list,
    parse_source_name_records,
)


def test_result_name_list_expands_multi_result_groups() -> None:
    names, end_index = parse_result_name_list("%a, %b:3, %c", 0)

    assert names == ("%a", "%b", "%b", "%b", "%c")
    assert end_index == len("%a, %b:3, %c")


def test_operation_result_names_slices_lhs_before_operation_column() -> None:
    line = "    %a, %b:2 = test.multi %input : i32"
    column = line.index("test.multi") + 1

    assert operation_result_names(line, column) == ("%a", "%b", "%b")


def test_scf_for_block_arg_names_come_from_operation_line() -> None:
    line = "    %loop_sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %sum) -> (f32) {"
    column = line.index("scf.for") + 1

    assert operation_block_arg_names("scf.for", line, column) == ("%i", "%acc")


def test_source_records_capture_result_ops_and_scf_for_args() -> None:
    records = parse_source_name_records(
        """
      %input = hal.interface.binding.subspan layout(#layout) binding(0) : memref<4xf32>
      %loop_sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %sum) -> (f32) {
        %next = arith.addf %acc, %sum : f32
      }
"""
    )

    assert [record.op_name for record in records] == [
        "hal.interface.binding.subspan",
        "scf.for",
        "arith.addf",
    ]
    assert records[0].result_names == ("%input",)
    assert records[1].result_names == ("%loop_sum",)
    assert records[1].block_arg_names == ("%i", "%acc")
