# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CLI behavior against a controlled public libamdf dependency, without hardware."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

FIXTURE = Path(sys.argv.pop(1)).resolve()
TOOL = Path(sys.argv.pop(1)).resolve()
FIRST_ID = "00000000000000020000000000000009"
FIRST_URI = "rdna://id/" + FIRST_ID


def unique_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


class InvocationTest(unittest.TestCase):
    def invoke(self, *arguments, scenario="normal", expected=0, **environment):
        result = subprocess.run(
            [str(FIXTURE), *arguments],
            env={**os.environ, "AMDF_INFO_TEST_SCENARIO": scenario, **environment},
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def report(self, *arguments, scenario="normal", expected=0):
        result = self.invoke(
            *arguments, "--format=json", scenario=scenario, expected=expected
        )
        self.assertNotIn("\x1b", result.stdout)
        report = json.loads(result.stdout, object_pairs_hook=unique_members)
        self.assertEqual(report["result"]["exit_code"], expected)
        return report, result.stderr.splitlines()

    def test_help_and_parse_failures_do_not_load_library(self):
        self.assertIn("Usage:", self.invoke("--help").stdout)
        self.assertEqual(self.invoke("--help").stderr, "")
        actual = subprocess.run([str(TOOL), "--help"], capture_output=True, text=True)
        self.assertEqual(actual.returncode, 0, actual.stderr)
        self.assertIn("--activate", actual.stdout)
        for arguments in [
            ["--missing"],
            ["--device"],
            ["--device="],
            ["--format=yaml"],
            ["--engine=cpu"],
            ["--color=perhaps"],
            ["--target=x", "--target=y"],
            ["--format=json", "--format=text"],
            ["--device=rdna", "--device=xdna"],
            ["--device=rdna://4294967296"],
            ["--device=rdna://-1"],
            ["--device=rdna://id/12"],
            ["--device=xdna://GPU-ffffffffffffffff"],
            ["--device=amdgpu://GPU-0000000000000000"],
            ["--device=rdna://0?queue=pm4"],
        ]:
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments, expected=2)
                self.assertEqual(result.stdout, "")
                self.assertNotIn("query_api", result.stderr)

    def test_inventory_is_passive_and_preserves_source_identity(self):
        report, calls = self.report()
        self.assertTrue(report["result"]["complete"])
        self.assertFalse(any(call.startswith("device_") for call in calls))
        self.assertEqual(calls.count("enumerate_count"), 1)
        self.assertEqual(calls.count("enumerate_fill"), 1)
        self.assertEqual(calls[-1], "instance_destroy")
        self.assertEqual(len(report["memory_pools"]), 1)
        self.assertEqual(
            report["memory_pools"][0]["capacity_bytes"], "9007199254740993"
        )
        first, second, xdna = report["endpoints"]
        # Presentation sorts identities, but ordinals retain native view order.
        self.assertEqual(first["family_ordinal"], 1)
        self.assertEqual(second["family_ordinal"], 0)
        self.assertEqual(second["uri"], FIRST_URI)
        self.assertEqual(first["native_identity"]["luid"], "0xffffffffabcdef01")
        self.assertEqual(first["native_identity"]["physical_adapter_index"], 7)
        self.assertEqual(second["native_identity"]["minor"], 130)
        self.assertIsNone(first["pci_address"])
        self.assertEqual(second["pci_address"]["bus"], 0xAB)
        self.assertEqual(second["host_numa_node"], 4)
        self.assertIsNone(xdna["host_numa_node"])
        self.assertEqual(second["gpu"]["unique_id"], "0xffffffffffffffff")
        self.assertEqual(
            int(second["type_flags"]["value"], 16) & 0x80000000, 0x80000000
        )
        self.assertIn("compute_only", second["type_flags"]["names"])
        self.assertIn('"\\\n\x1b[31m π \x9b', second["name"])
        self.assertEqual(
            second["queue_families"][0]["maximum_ring_byte_length"], str(2**64 - 1)
        )
        self.assertEqual(
            second["queue_families"][0]["format_features"], "0xfedcba9876543210"
        )

    def test_selection_is_exact_and_never_first_target_wins(self):
        report, calls = self.report(
            "select", "--target=gfx1151", "--activate", expected=4
        )
        self.assertEqual(len(report["endpoints"]), 2)
        self.assertFalse(any(call.startswith("device_create") for call in calls))
        report, _ = self.report("select", "--target=not-present", expected=3)
        self.assertEqual(report["endpoints"], [])
        report, _ = self.report("select", "--engine=xdna")
        self.assertEqual(len(report["endpoints"]), 1)
        self.assertEqual(report["endpoints"][0]["xdna"]["target"], "amd.xdna.test")
        for selector in [
            "rdna",
            "rdna:0",
            "rdna://0",
            FIRST_URI,
            "amdgpu://GPU-ffffffffffffffff",
        ]:
            with self.subTest(selector=selector):
                report, _ = self.report("--device", selector)
                self.assertEqual(
                    [item["id"] for item in report["endpoints"]], [FIRST_ID]
                )

    def test_empty_machine_is_inventory_but_not_successful_preflight(self):
        report, _ = self.report(scenario="empty")
        self.assertEqual(report["endpoints"], [])
        self.report("select", scenario="empty", expected=3)
        self.report("--activate", scenario="empty", expected=3)

    def test_denied_optional_queries_keep_other_facts_visible(self):
        report, _ = self.report(scenario="location_denied")
        self.assertFalse(report["result"]["complete"])
        endpoint = next(item for item in report["endpoints"] if item["id"] == FIRST_ID)
        error = endpoint["pci_address"]["error"]
        self.assertEqual(error["domain"]["name"], "ntstatus")
        self.assertEqual(error["code"], 0xC0000022)
        self.assertEqual(endpoint["host_numa_node"], 4)
        self.assertIn("queue_families", endpoint)

    def test_unqualified_endpoint_cannot_silently_change_selection(self):
        report, _ = self.report(scenario="endpoint_denied")
        self.assertFalse(report["result"]["complete"])
        self.assertEqual(len(report["endpoints"]), 3)
        self.assertIsNone(report["endpoints"][0]["family_ordinal"])
        self.report("select", "--target=gfx1151", scenario="target_denied", expected=1)
        self.report("--device=rdna:0", scenario="target_denied", expected=1)
        self.report(
            "--device=amdgpu://id/" + FIRST_ID, scenario="endpoint_denied", expected=1
        )
        # An exact other endpoint remains usable despite unrelated query denial.
        report, _ = self.report(
            "--device=rdna://id/00000000000000010000000000000007",
            scenario="target_denied",
        )
        self.assertTrue(report["result"]["complete"])

    def test_activation_queries_and_checked_teardown(self):
        for selector, index in [(FIRST_URI, 0), ("xdna:0", 2)]:
            with self.subTest(selector=selector):
                report, calls = self.report("--device=" + selector, "--activate")
                activation = report["endpoints"][0]["activation"]
                self.assertTrue(activation["device_destroyed"])
                self.assertEqual(len(activation["queue_placements"]), 1)
                operations = [call for call in calls if call.endswith(f":{index}")]
                self.assertEqual(
                    operations,
                    [
                        f"{op}:{index}"
                        for op in [
                            "endpoint_open",
                            "device_create",
                            "device_query_info",
                            "placement_query_info",
                            "device_destroy",
                            "endpoint_close",
                        ]
                    ],
                )
                self.assertEqual(calls[-1], "instance_destroy")

    def test_activation_failure_still_releases_owned_resources(self):
        for scenario in [
            "device_create_failure",
            "device_query_failure",
            "placement_failure",
        ]:
            with self.subTest(scenario=scenario):
                report, calls = self.report(
                    "--device=" + FIRST_URI, "--activate", scenario=scenario, expected=1
                )
                self.assertFalse(report["result"]["complete"])
                self.assertEqual(calls[-1], "instance_destroy")
                self.assertEqual(
                    calls.count("device_destroy:0"),
                    0 if scenario == "device_create_failure" else 1,
                )

    def test_failed_child_teardown_retains_parents_without_retry(self):
        report, calls = self.report(
            "--device=" + FIRST_URI,
            "--activate",
            scenario="device_destroy_failure",
            expected=1,
        )
        self.assertFalse(report["endpoints"][0]["activation"]["device_destroyed"])
        self.assertEqual(calls.count("device_destroy:0"), 1)
        self.assertNotIn("endpoint_close:0", calls)
        self.assertNotIn("instance_destroy", calls)
        self.assertIn("endpoint_close:1", calls)
        report, calls = self.report(scenario="endpoint_close_failure", expected=1)
        self.assertEqual(report["cleanup_errors"][0]["uri"], FIRST_URI)
        self.assertNotIn("instance_destroy", calls)
        report, calls = self.report(scenario="instance_destroy_failure", expected=1)
        self.assertEqual(calls.count("instance_destroy"), 1)
        self.assertEqual(
            report["cleanup_errors"][0]["error"]["operation"], "instance_destroy"
        )

    def test_failure_documents_remain_valid_and_growth_does_not_retry(self):
        for scenario in [
            "abi_failure",
            "instance_failure",
            "enumeration_failure",
            "enumeration_growth",
            "extension_failure",
            "queue_failure",
        ]:
            with self.subTest(scenario=scenario):
                report, calls = self.report(scenario=scenario, expected=1)
                self.assertFalse(report["result"]["complete"])
                self.assertLessEqual(calls.count("enumerate_fill"), 1)
        report, _ = self.report(scenario="unknown_domain", expected=1)
        endpoint = next(item for item in report["endpoints"] if item["id"] == FIRST_ID)
        error = endpoint["queue_families"][0]["error"]
        self.assertEqual(error["domain"], {"value": 123, "name": None})
        self.assertEqual(error["code"], 2**32 - 1)
        report, _ = self.report(scenario="no_gpu_extension")
        self.assertEqual(len(report["endpoints"]), 3)
        self.assertFalse(report["library"]["extensions"]["gpu"]["available"])

    def test_text_is_pipe_friendly_and_escapes_native_control_sequences(self):
        plain = self.invoke().stdout
        self.assertIn("Memory pools", plain)
        self.assertIn("rdna://0", plain)
        self.assertIn("xdna://0", plain)
        self.assertNotIn("\x1b", plain)
        self.assertNotIn("\x9b", plain)
        self.assertIn("\\x1b[31m", plain)
        self.assertIn("π", plain)
        self.assertNotIn("Activation:", plain)
        self.assertIn(FIRST_URI, self.invoke("--details").stdout)
        self.assertIn("\x1b[1m", self.invoke("--color=always").stdout)
        self.assertNotIn("\x1b", self.invoke("--color=never").stdout)
        self.assertNotIn("\x1b", self.invoke(NO_COLOR="1").stdout)
        self.assertIn(
            "Activation: OK", self.invoke("--device=xdna", "--activate").stdout
        )
        failed = self.invoke(
            "--device=rdna", "--activate", scenario="device_destroy_failure", expected=1
        ).stdout
        self.assertIn("Activation: FAILED", failed)
        self.assertNotIn("Activation: OK", failed)
        failed = self.invoke(scenario="extension_failure", expected=1).stdout
        self.assertIn("query_extension(GPU)", failed)
        self.assertIn(
            "domain 123", self.invoke(scenario="unknown_domain", expected=1).stdout
        )

    def test_output_failure_is_not_success(self):
        for arguments in [[], ["--help"]]:
            with self.subTest(arguments=arguments):
                with open(__file__, "rb") as read_only:
                    result = subprocess.run(
                        [str(FIXTURE), *arguments],
                        stdout=read_only,
                        stderr=subprocess.PIPE,
                    )
                self.assertEqual(result.returncode, 1, result.stderr)
                if not arguments:
                    self.assertIn(b"failed to write the report", result.stderr)

    def test_invalid_native_path_does_not_create_an_instance(self):
        # A directory is not a character device on Linux; Windows rejects the
        # entire Linux pathname namespace. Neither case can activate anything.
        report, calls = self.report("--device=/", "--activate", expected=1)
        self.assertIn("error", report["device_path"])
        self.assertNotIn("instance_create", calls)


if __name__ == "__main__":
    unittest.main()
