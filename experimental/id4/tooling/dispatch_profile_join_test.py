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


def _dispatch(
    dispatch_ordinal: int,
    function_name: str,
    name: str,
    module_path: str,
) -> dict[str, object]:
    return {
        "dispatch_ordinal": dispatch_ordinal,
        "operation_ordinal": dispatch_ordinal * 2,
        "region_id": 0,
        "region_operation_ordinal": dispatch_ordinal,
        "name": name,
        "module_path": module_path,
        "function_name": function_name,
        "workgroup_count": [1, 1, 1],
        "workgroup_size": [64, 1, 1],
        "config_bindings": [],
        "bindings": [],
    }


def _stage_plan(stage_name: str, dispatch_names: list[str]) -> dict[str, object]:
    return {
        "stage": stage_name,
        "regions": [{"id": 0, "name": f"{stage_name}.forward"}],
        "program": {
            "name": f"{stage_name}.forward",
            "operation_count": len(dispatch_names),
            "dispatch_count": len(dispatch_names),
            "dispatches": [
                _dispatch(
                    i,
                    dispatch_name,
                    f"{stage_name}.{i}",
                    f"{stage_name.replace('.', '/')}/{i}",
                )
                for i, dispatch_name in enumerate(dispatch_names)
            ],
        },
    }


def _generation_plan() -> dict[str, object]:
    return {
        "kind": "ideogram4_generation",
        "summary": {
            "qwen_token_count": 19,
            "denoise_step_count": 2,
        },
        "stages": {
            "qwen": _stage_plan("qwen", ["qwen.forward"]),
            "sampler_noise": _stage_plan("sampler.noise", ["noise.forward"]),
            "dit_conditioned": _stage_plan("dit.cond", ["dit.prelude", "dit.forward"]),
            "dit_unconditioned": _stage_plan("dit.uncond", ["dit.forward"]),
            "sampler_denoise": _stage_plan("sampler.denoise", ["sampler.step"]),
            "decode": _stage_plan("decode", ["decode.forward"]),
        },
    }


def _profile_event(
    event_id: int,
    function_name: str,
    duration_ns: int,
    command_index: int,
) -> dict[str, object]:
    return {
        "type": "dispatch_event",
        "event_id": event_id,
        "submission_id": 100 + event_id,
        "command_buffer_id": 200 + event_id,
        "command_index": command_index,
        "key": function_name,
        "workgroup_count": [1, 1, 1],
        "workgroup_size": [64, 1, 1],
        "valid": True,
        "duration_ns": duration_ns,
    }


def _generation_profile() -> list[dict[str, object]]:
    functions = [
        "qwen.forward",
        "noise.forward",
        "dit.prelude",
        "dit.forward",
        "dit.forward",
        "sampler.step",
        "dit.prelude",
        "dit.forward",
        "dit.forward",
        "sampler.step",
        "decode.forward",
    ]
    return [
        _profile_event(i + 1, function_name, (i + 1) * 100, i)
        for i, function_name in enumerate(functions)
    ]


def _find_row(
    rows: list[dict[str, object]], key: str, value: object
) -> dict[str, object]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise AssertionError(f"row with {key}={value!r} not found")


def _find_kernel_row(
    rows: list[dict[str, object]], stage_key: str, function_name: str
) -> dict[str, object]:
    for row in rows:
        if (
            row.get("stage_key") == stage_key
            and row.get("function_name") == function_name
        ):
            return row
    raise AssertionError(
        f"kernel row with stage_key={stage_key!r} "
        f"and function_name={function_name!r} not found"
    )


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
        self.assertEqual(first["launch_geometry_source"], "plan+profile")
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

    def test_joins_generation_plan_by_production_issue_sequence(self) -> None:
        report = dispatch_profile_join.join_profile(
            _generation_plan(), _generation_profile()
        )

        self.assertEqual(report["kind"], "ideogram4_generation_dispatch_profile")
        self.assertEqual(report["summary"]["denoise_step_count"], 2)
        self.assertEqual(report["summary"]["stage_invocation_count"], 9)
        self.assertEqual(report["summary"]["dispatch_count"], 11)
        self.assertEqual(report["summary"]["total_duration_ns"], 6600)
        dit_conditioned = _find_row(report["by_stage"], "stage_key", "dit_conditioned")
        self.assertEqual(dit_conditioned["invocation_count"], 2)
        self.assertEqual(dit_conditioned["count"], 4)
        self.assertEqual(dit_conditioned["total_duration_ns"], 2200)
        sampler_denoise = _find_kernel_row(
            report["by_kernel"], "sampler_denoise", "sampler.step"
        )
        self.assertEqual(sampler_denoise["total_duration_ns"], 1600)
        self.assertEqual(report["stage_invocations"][2]["stage_key"], "dit_conditioned")
        self.assertEqual(report["stage_invocations"][2]["denoise_step_index"], 0)
        self.assertEqual(report["stage_invocations"][5]["stage_key"], "dit_conditioned")
        self.assertEqual(report["stage_invocations"][5]["denoise_step_index"], 1)
        self.assertEqual(report["dispatches"][0]["generation_dispatch_ordinal"], 0)
        self.assertEqual(report["dispatches"][3]["stage_key"], "dit_conditioned")
        self.assertEqual(report["dispatches"][4]["stage_key"], "dit_unconditioned")
        dit_forward = _find_kernel_row(
            report["by_kernel"], "dit_conditioned", "dit.forward"
        )
        self.assertEqual(dit_forward["total_duration_ns"], 1200)

    def test_rejects_generation_profile_with_extra_dispatch_rows(self) -> None:
        profile = _generation_profile()
        profile.insert(2, _profile_event(99, "prepare.encode", 50, 99))

        with self.assertRaisesRegex(
            dispatch_profile_join.DispatchProfileJoinError,
            "function mismatch",
        ):
            dispatch_profile_join.join_profile(_generation_plan(), profile)

    def test_uses_profile_launch_geometry_when_plan_omits_it(self) -> None:
        plan = _plan()
        for dispatch in plan["program"]["dispatches"]:
            del dispatch["workgroup_count"]
            del dispatch["workgroup_size"]

        report = dispatch_profile_join.join_dispatch_profile(plan, _profile())

        first = report["dispatches"][0]
        self.assertEqual(first["workgroup_count"], [4, 1, 1])
        self.assertEqual(first["workgroup_size"], [64, 1, 1])
        self.assertEqual(first["launch_geometry_source"], "profile")
        second = report["dispatches"][1]
        self.assertEqual(second["workgroup_count"], [2, 1, 1])
        self.assertEqual(second["workgroup_size"], [128, 1, 1])
        self.assertEqual(second["launch_geometry_source"], "profile")

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

    def test_rejects_partial_plan_launch_geometry(self) -> None:
        plan = _plan()
        del plan["program"]["dispatches"][0]["workgroup_size"]

        with self.assertRaisesRegex(
            dispatch_profile_join.DispatchProfileJoinError,
            "must contain both workgroup_count and workgroup_size, or neither",
        ):
            dispatch_profile_join.join_dispatch_profile(plan, _profile())


if __name__ == "__main__":
    unittest.main()
