# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs authored comparisons, checks their files, and replays a saved result."""

import ast
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(sys.argv.pop(1)).resolve()
SOURCE = Path(sys.argv.pop(1)).resolve()
REPLAY = Path(sys.argv.pop(1)).resolve()


class ScenarioCaptureTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.outputs = self.root / "captures"

    def run_tool(self, source, *arguments, succeeds=True):
        result = subprocess.run(
            [str(TOOL), str(source), *arguments],
            capture_output=True,
            text=True,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def run_scenario(self, name, *arguments, succeeds=True):
        return self.run_tool(
            SOURCE,
            f"--case=@{name}",
            f"--file-output-root={self.outputs}",
            *arguments,
            succeeds=succeeds,
        )

    def assert_array(self, path, shape, values):
        # Decode the public NPY header and exact integer payload independently of
        # Loom. The replay invocation below also tests Loom's typed reader.
        with path.open("rb") as stream:
            self.assertEqual(stream.read(8), b"\x93NUMPY\x01\x00")
            length = struct.unpack("<H", stream.read(2))[0]
            header = ast.literal_eval(stream.read(length).decode("latin1"))
            self.assertEqual(
                header, {"descr": "<i4", "fortran_order": False, "shape": shape}
            )
            self.assertEqual(stream.read(), struct.pack(f"<{len(values)}i", *values))

    def test_coordinates_aliases_dynamic_results_and_replay(self):
        result = self.run_scenario("coordinates")
        self.assertEqual(json.loads(result.stdout)["failed_trial_count"], 0)
        expected_paths = set()
        for configuration in range(2):
            base = (
                self.outputs / "scenario-coordinates" / f"configuration-{configuration}"
            )
            for trial in range(65):
                values = list(
                    range(100 * configuration + trial, 100 * configuration + trial + 8)
                )
                for side in ("actual", "expected"):
                    directory = base / f"trial-0-{trial}" / side
                    arrays = {
                        "storage.npy": ((8,), values),
                        "nested/matrix.npy": ((2, 3), values[1:7]),
                        "result.npy": ((trial % 3 + 1,), values[2 : 3 + trial % 3]),
                    }
                    for name, (shape, contents) in arrays.items():
                        path = directory / name
                        expected_paths.add(path)
                        self.assert_array(path, shape, contents)
            path = base / "trial-1-0" / "actual" / "result.npy"
            expected_paths.add(path)
            self.assert_array(path, (1,), [configuration * 100 + 2])
        self.assertEqual(set(self.outputs.rglob("*.npy")), expected_paths)
        replay = self.root / "replay.loom"
        shutil.copyfile(REPLAY, replay)
        self.run_tool(replay)

    def test_failure_policy_saves_distinct_completed_realizations(self):
        result = self.run_scenario("failure", succeeds=False)
        self.assertEqual(json.loads(result.stdout)["failed_trial_count"], 1)
        base = self.outputs / "scenario-failure" / "configuration-0" / "trial-0-1"
        self.assert_array(base / "actual/state.npy", (1,), [1])
        self.assert_array(base / "expected/state.npy", (1,), [42])
        self.assertEqual(
            set(self.outputs.rglob("*.npy")),
            {base / "actual/state.npy", base / "expected/state.npy"},
        )

    def test_passing_failure_only_capture_needs_no_output_provider(self):
        self.run_tool(SOURCE, "--case=@passing")
        self.assertFalse(self.outputs.exists())

    def test_required_output_provider_and_io_failure(self):
        result = self.run_tool(SOURCE, "--case=@a_2Eb", succeeds=False)
        self.assertIn("check.file.write requires --file-output-root", result.stderr)
        self.outputs.write_text("not a directory")
        result = self.run_scenario("a_2Eb", succeeds=False)
        self.assertTrue(result.stderr)
        self.assertEqual(self.outputs.read_text(), "not a directory")

    def test_symbol_paths_do_not_collide_or_escape(self):
        self.run_scenario("a.b")
        self.run_scenario("a_2Eb")
        self.assertEqual(
            {path.name for path in self.outputs.iterdir()},
            {"scenario-a%2Eb", "scenario-a_2%45b"},
        )
        for directory in self.outputs.iterdir():
            for side in ("actual", "expected"):
                self.assert_array(
                    directory / "configuration-0/trial-0-0" / side / "state.npy",
                    (1,),
                    [0],
                )


if __name__ == "__main__":
    unittest.main()
