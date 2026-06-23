#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest

import dispatch_profile_join


def _plan() -> dict[str, object]:
    return {
        "stage": "id4.test",
        "regions": [
            {
                "id": 0,
                "name": "id4.test.forward",
            }
        ],
        "program": {
            "name": "id4.test.forward",
            "operation_count": 3,
            "dispatch_count": 2,
            "dispatches": [
                {
                    "dispatch_ordinal": 0,
                    "operation_ordinal": 4,
                    "region_id": 0,
                    "region_operation_ordinal": 0,
                    "name": "test.first",
                    "module_path": "test/first",
                    "function_name": "id4_test_first",
                    "workgroup_count": [4, 1, 1],
                    "workgroup_size": [64, 1, 1],
                    "config_bindings": [{"key": "@m", "value": "128"}],
                    "bindings": [
                        {
                            "index": 0,
                            "tensor_ordinal": 0,
                            "name": "input",
                            "access": "read",
                            "dtype": "bf16",
                            "byte_length": 256,
                            "shape": [128],
                        },
                        {
                            "index": 1,
                            "tensor_ordinal": 1,
                            "name": "hidden",
                            "access": "write",
                            "dtype": "bf16",
                            "byte_length": 256,
                            "shape": [128],
                        },
                    ],
                },
                {
                    "dispatch_ordinal": 1,
                    "operation_ordinal": 6,
                    "region_id": 0,
                    "region_operation_ordinal": 2,
                    "name": "test.second",
                    "module_path": "test/second",
                    "function_name": "id4_test_second",
                    "workgroup_count": [2, 1, 1],
                    "workgroup_size": [128, 1, 1],
                    "config_bindings": [],
                    "bindings": [
                        {
                            "index": 0,
                            "tensor_ordinal": 1,
                            "name": "hidden",
                            "access": "read",
                            "dtype": "bf16",
                            "byte_length": 256,
                            "shape": [128],
                        },
                        {
                            "index": 1,
                            "tensor_ordinal": 2,
                            "name": "output",
                            "access": "write",
                            "dtype": "bf16",
                            "byte_length": 256,
                            "shape": [128],
                        },
                    ],
                },
            ],
        },
    }


def _profile() -> list[dict[str, object]]:
    return [
        {
            "type": "dispatch_event",
            "event_id": 10,
            "submission_id": 20,
            "command_buffer_id": 30,
            "command_index": 0,
            "key": "id4_test_first",
            "workgroup_count": [4, 1, 1],
            "workgroup_size": [64, 1, 1],
            "valid": True,
            "duration_ns": 1000,
        },
        {
            "type": "dispatch_event",
            "event_id": 11,
            "submission_id": 20,
            "command_buffer_id": 30,
            "command_index": 3,
            "key": "id4_test_second",
            "workgroup_count": [2, 1, 1],
            "workgroup_size": [128, 1, 1],
            "valid": True,
            "duration_ns": 500,
        },
        {
            "type": "dispatch_summary",
            "total_duration_ns": 1500,
        },
    ]


class DispatchProfileJoinTest(unittest.TestCase):
    def test_joins_plan_dispatch_rows_to_profile_events_by_order(self) -> None:
        report = dispatch_profile_join.join_dispatch_profile(_plan(), _profile())

        self.assertEqual(report["summary"]["dispatch_count"], 2)
        self.assertEqual(report["summary"]["total_duration_ns"], 1500)
        self.assertEqual(len(report["dispatches"]), 2)
        first = report["dispatches"][0]
        self.assertEqual(first["dispatch_ordinal"], 0)
        self.assertEqual(first["profile_event_id"], 10)
        self.assertEqual(first["command_index"], 0)
        self.assertEqual(first["region_operation_ordinal"], 0)
        self.assertEqual(first["name"], "test.first")
        self.assertEqual(first["bindings"][0]["shape"], [128])
        second = report["dispatches"][1]
        self.assertEqual(second["dispatch_ordinal"], 1)
        self.assertEqual(second["profile_event_id"], 11)
        self.assertEqual(second["command_index"], 3)
        self.assertEqual(second["region_operation_ordinal"], 2)
        self.assertEqual(report["by_kernel"][0]["function_name"], "id4_test_first")
        self.assertEqual(report["by_kernel"][0]["min_duration_ns"], 1000)
        self.assertEqual(report["by_kernel"][0]["p50_duration_ns"], 1000)
        self.assertEqual(report["by_kernel"][0]["p90_duration_ns"], 1000)
        self.assertEqual(report["by_kernel"][0]["p99_duration_ns"], 1000)
        self.assertEqual(report["by_kernel"][0]["max_duration_ns"], 1000)

    def test_rejects_dispatch_count_mismatch(self) -> None:
        profile = _profile()[:1]

        with self.assertRaisesRegex(
            dispatch_profile_join.DispatchProfileJoinError,
            "dispatch count mismatch",
        ):
            dispatch_profile_join.join_dispatch_profile(_plan(), profile)

    def test_rejects_function_mismatch(self) -> None:
        profile = _profile()
        profile[1] = dict(profile[1])
        profile[1]["key"] = "id4_test_other"

        with self.assertRaisesRegex(
            dispatch_profile_join.DispatchProfileJoinError,
            "function mismatch",
        ):
            dispatch_profile_join.join_dispatch_profile(_plan(), profile)

    def test_rejects_workgroup_mismatch(self) -> None:
        profile = _profile()
        profile[0] = dict(profile[0])
        profile[0]["workgroup_count"] = [5, 1, 1]

        with self.assertRaisesRegex(
            dispatch_profile_join.DispatchProfileJoinError,
            "workgroup_count mismatch",
        ):
            dispatch_profile_join.join_dispatch_profile(_plan(), profile)


if __name__ == "__main__":
    unittest.main()
