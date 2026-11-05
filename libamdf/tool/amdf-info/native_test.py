# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public executable smoke on a runner-selected native GPU or NPU."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

TOOL = Path(sys.argv.pop(1)).resolve()
ENGINE = sys.argv.pop(1)


class NativeTest(unittest.TestCase):
    def report(self, *arguments):
        result = subprocess.run(
            [str(TOOL), *arguments, "--format=json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_exact_selection_and_activation(self):
        inventory = self.report("--engine=" + ENGINE)
        candidates = inventory["endpoints"]
        requested = (
            os.environ.get("AMDF_CTS_GPU_NATIVE_IDENTITY") if ENGINE == "gpu" else None
        )
        if requested:
            namespace, first, second = requested.split(":")
            if namespace == "linux_device":
                expected = (namespace, int(first), int(second))
            elif namespace == "windows_adapter":
                expected = (namespace, int(first, 16), int(second))
            else:
                self.fail(f"unknown runner identity namespace: {namespace}")
            matches = []
            for endpoint in candidates:
                identity = endpoint.get("native_identity", {})
                name = identity.get("type", {}).get("name")
                if name == "linux_device":
                    actual = (name, identity["major"], identity["minor"])
                elif name == "windows_adapter":
                    actual = (
                        name,
                        int(identity["luid"], 16),
                        identity["physical_adapter_index"],
                    )
                else:
                    continue
                if actual == expected:
                    matches.append(endpoint)
            self.assertEqual(
                len(matches), 1, f"runner selection {requested}: {inventory}"
            )
            endpoint = matches[0]
        else:
            if not candidates:
                self.skipTest(f"no {ENGINE} endpoint available")
            endpoint = candidates[0]
        self.assertNotIn("qualification", endpoint, endpoint)
        self.assertNotIn("activation", endpoint)
        self.assertTrue(endpoint["uri"])
        selected = self.report("select", "--device=" + endpoint["uri"])
        self.assertEqual(len(selected["endpoints"]), 1)
        self.assertEqual(
            selected["endpoints"][0]["native_identity"], endpoint["native_identity"]
        )
        activated = self.report("--device=" + endpoint["uri"], "--activate")
        self.assertEqual(activated["cleanup_errors"], [])
        live = activated["endpoints"][0]
        self.assertEqual(live["native_identity"], endpoint["native_identity"])
        self.assertTrue(live["activation"]["device_destroyed"])
        self.assertEqual(
            len(live["activation"]["queue_placements"]), len(endpoint["queue_families"])
        )


if __name__ == "__main__":
    unittest.main()
