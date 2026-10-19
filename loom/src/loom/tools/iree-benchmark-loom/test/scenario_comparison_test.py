# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises controlled scenario comparison through the public CLI."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(sys.argv.pop(1)).resolve()
SOURCE = Path(sys.argv.pop(1)).resolve()


class ScenarioComparisonTest(unittest.TestCase):
    def run_comparison(
        self, candidate: str
    ) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "results.jsonl"
            result = subprocess.run(
                [
                    str(TOOL),
                    str(SOURCE),
                    "--device=task",
                    f"--compare=@scenario_benchmark_baseline_throughput,@{candidate}",
                    "--sample=2",
                    "--repetitions=1",
                    "--interleave=ABABA",
                    "--batch-size=1",
                    "--input-ring-count=2",
                    "--iterations=1",
                    "--warmup-iterations=0",
                    "--min-time-ms=0",
                    "--warmup-time-ms=0",
                    "--max-batches=1",
                    "--output-format=jsonl",
                    f"--output={output_path}",
                ],
                capture_output=True,
                text=True,
            )
            rows = []
            if output_path.exists():
                rows = [
                    json.loads(line) for line in output_path.read_text().splitlines()
                ]
            return result, rows

    def test_equivalent_scenarios_share_one_interleaved_schedule(self):
        result, rows = self.run_comparison("scenario_benchmark_candidate_throughput")
        self.assertEqual(result.returncode, 0, result.stderr)

        repetitions = [row for row in rows if row["row"] == "benchmark.repetition"]
        self.assertEqual(
            [(row["schedule_token"], row["candidate_id"]) for row in repetitions],
            [("A", "c0"), ("B", "c1"), ("A", "c0")],
        )
        for row in repetitions:
            self.assertEqual(row["benchmark_sample_index"], 2)
            self.assertEqual(row["configuration_ordinal"], 0)
            self.assertEqual(row["trial_index"], 0)
            self.assertEqual(row["trial_ordinal"], 2)
            cache = row["benchmark_result"]["data_cache"]
            self.assertEqual(cache["binding_count"], 2)
            self.assertEqual(cache["binding_ring_count"], 2)
            self.assertEqual(cache["binding_ring_bytes"], 64)

        comparisons = [row for row in rows if row["row"] == "comparison"]
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["method"], "ABABA")
        self.assertEqual(comparisons[0]["baseline_repetition_count"], 2)
        self.assertEqual(comparisons[0]["candidate_repetition_count"], 1)

    def test_different_materialized_values_are_rejected_before_timing(self):
        result, rows = self.run_comparison("scenario_benchmark_mismatch_throughput")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("value 0 buffer contents differ", result.stderr)
        self.assertIn(
            "comparison candidate `scenario_benchmark_mismatch_throughput` "
            "materialized trial does not match baseline "
            "`scenario_benchmark_baseline_throughput`",
            result.stderr,
        )
        self.assertFalse(any(row["row"] == "benchmark.repetition" for row in rows))


if __name__ == "__main__":
    unittest.main()
