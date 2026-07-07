#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest

import dispatch_profile_compare
from dispatch_profile_join import DispatchProfileJoinError


def _aggregate(
    name: str,
    module_path: str,
    function_name: str,
    total_duration_ns: int,
    count: int = 1,
) -> dict[str, object]:
    return {
        "name": name,
        "module_path": module_path,
        "function_name": function_name,
        "count": count,
        "valid_count": count,
        "invalid_count": 0,
        "duration_ns_available_count": count,
        "total_duration_ns": total_duration_ns,
        "p50_duration_ns": total_duration_ns // count,
    }


def _report(
    total_duration_ns: int,
    operation_rows: list[dict[str, object]],
    kernel_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "summary": {
            "dispatch_count": sum(int(row["count"]) for row in kernel_rows),
            "valid_duration_count": sum(int(row["count"]) for row in kernel_rows),
            "invalid_duration_count": 0,
            "duration_ns_available_count": sum(
                int(row["count"]) for row in kernel_rows
            ),
            "total_duration_ns": total_duration_ns,
            "duration_ticks_available_count": 0,
            "total_duration_ticks": None,
            "profile_dispatch_count": sum(int(row["count"]) for row in kernel_rows),
            "unmatched_dispatch_count": 0,
        },
        "by_operation": operation_rows,
        "by_kernel": kernel_rows,
        "unmatched_by_kernel": [],
        "dispatches": [],
        "unmatched_dispatches": [],
    }


class DispatchProfileCompareTest(unittest.TestCase):
    def test_format_report_compares_totals_and_hot_rows(self):
        base = _report(
            1000,
            [
                _aggregate("qwen.mlp", "qwen3_vl/mlp", "id4_qwen_mlp", 600, 3),
                _aggregate("qwen.o", "qwen3_vl/o", "id4_qwen_o", 400, 2),
            ],
            [
                _aggregate("mlp", "qwen3_vl/mlp", "id4_qwen_mlp", 600, 3),
                _aggregate("o", "qwen3_vl/o", "id4_qwen_o", 400, 2),
            ],
        )
        candidate = _report(
            800,
            [
                _aggregate("qwen.mlp", "qwen3_vl/mlp", "id4_qwen_mlp", 300, 3),
                _aggregate("qwen.o", "qwen3_vl/o", "id4_qwen_o", 500, 2),
            ],
            [
                _aggregate("mlp", "qwen3_vl/mlp", "id4_qwen_mlp", 300, 3),
                _aggregate("o", "qwen3_vl/o", "id4_qwen_o", 500, 2),
            ],
        )

        report = dispatch_profile_compare.format_comparison_report(base, candidate)

        self.assertIn("ratio=0.800x", report)
        self.assertIn("change=-20.00%", report)
        self.assertIn("qwen.mlp", report)
        self.assertIn("-300", report)
        self.assertIn("+100", report)
        self.assertIn("3/3", report)

    def test_rows_removed_from_candidate_remain_visible(self):
        base = _report(
            900,
            [_aggregate("qwen.hot", "qwen3_vl/hot", "id4_qwen_hot", 900, 9)],
            [_aggregate("hot", "qwen3_vl/hot", "id4_qwen_hot", 900, 9)],
        )
        candidate = _report(
            100,
            [_aggregate("qwen.small", "qwen3_vl/small", "id4_qwen_small", 100)],
            [_aggregate("small", "qwen3_vl/small", "id4_qwen_small", 100)],
        )

        report = dispatch_profile_compare.format_comparison_report(
            base, candidate, limit=1
        )

        self.assertIn("qwen.hot", report)
        self.assertIn("0/9", report)

    def test_duration_unit_mismatch_fails(self):
        base = _report(
            100,
            [_aggregate("qwen.op", "qwen3_vl/op", "id4_qwen_op", 100)],
            [_aggregate("op", "qwen3_vl/op", "id4_qwen_op", 100)],
        )
        candidate = _report(
            100,
            [_aggregate("qwen.op", "qwen3_vl/op", "id4_qwen_op", 100)],
            [_aggregate("op", "qwen3_vl/op", "id4_qwen_op", 100)],
        )
        candidate["summary"]["total_duration_ns"] = None
        candidate["summary"]["total_duration_ticks"] = 1000

        with self.assertRaisesRegex(DispatchProfileJoinError, "duration units differ"):
            dispatch_profile_compare.format_comparison_report(base, candidate)


if __name__ == "__main__":
    unittest.main()
