# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises benchmark execution contracts through the public CLI."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(sys.argv.pop(1)).resolve()
SCENARIO_SOURCE = Path(sys.argv.pop(1)).resolve()
COMMAND_SOURCE = Path(sys.argv.pop(1)).resolve()
SEQUENCE_SOURCE = Path(sys.argv.pop(1)).resolve()
HOST_VALUES_SOURCE = Path(sys.argv.pop(1)).resolve()


class BenchmarkExecutionTest(unittest.TestCase):
    def run_tool(
        self, source: Path, *arguments: str
    ) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "results.jsonl"
            result = subprocess.run(
                [
                    str(TOOL),
                    str(source),
                    "--device=task",
                    *arguments,
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

    def run_comparison(
        self, candidate: str
    ) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        return self.run_tool(
            SCENARIO_SOURCE,
            f"--compare=@scenario_benchmark_baseline_throughput,@{candidate}",
            "--sample=2",
            "--repetitions=1",
            "--interleave=ABABA",
            "--batch-size=1",
            "--input-ring-count=2",
        )

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
            self.assertEqual(cache["binding_range_count"], 1)
            self.assertEqual(cache["binding_range_set_bytes"], 20)
            self.assertEqual(cache["binding_range_ring_bytes"], 40)
            self.assertEqual(cache["materialized_storage_count"], 1)
            self.assertEqual(cache["materialized_storage_set_bytes"], 32)
            self.assertEqual(cache["materialized_storage_ring_bytes"], 64)

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

    def test_sequence_ring_counts_unique_reachable_ranges(self):
        result, rows = self.run_tool(
            SEQUENCE_SOURCE,
            "--benchmark=@shared_sequence_throughput",
            "--measure=dispatch_complete",
            "--input-ring-min-bytes=25",
            "--batch-size=4",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        benchmark_rows = [row for row in rows if row["row"] == "benchmark"]
        self.assertEqual(len(benchmark_rows), 1)
        benchmark_result = benchmark_rows[0]["benchmark_result"]
        self.assertEqual(benchmark_result["state"], "ok")
        self.assertEqual(benchmark_result["correctness"]["sample_count"], 1)
        self.assertEqual(benchmark_result["correctness"]["failed_sample_count"], 0)

        cache = benchmark_result["data_cache"]
        self.assertEqual(cache["binding_count"], 4)
        self.assertEqual(cache["binding_ring_count"], 3)
        self.assertEqual(cache["binding_set_bytes"], 16)
        self.assertEqual(cache["binding_ring_bytes"], 48)
        self.assertEqual(cache["binding_range_count"], 3)
        self.assertEqual(cache["binding_range_set_bytes"], 12)
        self.assertEqual(cache["binding_range_ring_bytes"], 36)
        self.assertEqual(cache["materialized_storage_count"], 3)
        self.assertEqual(cache["materialized_storage_set_bytes"], 32)
        self.assertEqual(cache["materialized_storage_ring_bytes"], 96)
        self.assertTrue(cache["ring_byte_target_met"])

    def test_dispatch_materialization_executes_host_value_prefixes(self):
        result, rows = self.run_tool(
            HOST_VALUES_SOURCE,
            "--measure=dispatch_complete",
            "--input-ring-count=2",
            "--batch-size=1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        benchmark_results = {
            row["benchmark_result"]["benchmark"]: row["benchmark_result"]
            for row in rows
            if row["row"] == "benchmark"
        }
        self.assertEqual(
            set(benchmark_results),
            {
                "single_host_value_throughput",
                "sequence_host_value_throughput",
            },
        )
        for benchmark_result in benchmark_results.values():
            self.assertEqual(benchmark_result["state"], "ok")
            self.assertEqual(benchmark_result["correctness"]["sample_count"], 1)
            self.assertEqual(benchmark_result["correctness"]["failed_sample_count"], 0)
            self.assertEqual(benchmark_result["data_cache"]["binding_ring_count"], 2)

        single_result = benchmark_results["single_host_value_throughput"]
        self.assertEqual(single_result["data_cache"]["binding_count"], 1)
        self.assertEqual(
            single_result["measurement"]["physical_dispatches_per_batch"], 1
        )

        sequence_result = benchmark_results["sequence_host_value_throughput"]
        self.assertEqual(sequence_result["data_cache"]["binding_count"], 2)
        self.assertEqual(
            sequence_result["measurement"]["physical_dispatches_per_batch"], 2
        )

    def test_scenario_result_binding_contributes_to_reachable_ranges(self):
        result, rows = self.run_tool(
            SCENARIO_SOURCE,
            "--benchmark=@scenario_argument_free_results_throughput",
            "--sample=1",
            "--input-ring-min-bytes=25",
            "--batch-size=2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        benchmark_rows = [row for row in rows if row["row"] == "benchmark"]
        self.assertEqual(len(benchmark_rows), 1)
        benchmark_result = benchmark_rows[0]["benchmark_result"]
        self.assertEqual(benchmark_result["state"], "ok")

        cache = benchmark_result["data_cache"]
        self.assertEqual(cache["binding_count"], 1)
        self.assertEqual(cache["binding_ring_count"], 2)
        self.assertEqual(cache["binding_set_bytes"], 16)
        self.assertEqual(cache["binding_ring_bytes"], 32)
        self.assertEqual(cache["binding_range_count"], 1)
        self.assertEqual(cache["binding_range_set_bytes"], 16)
        self.assertEqual(cache["binding_range_ring_bytes"], 32)
        self.assertEqual(cache["materialized_storage_count"], 0)
        self.assertEqual(cache["materialized_storage_set_bytes"], 0)
        self.assertEqual(cache["materialized_storage_ring_bytes"], 0)
        self.assertTrue(cache["ring_byte_target_met"])

    def test_command_scenario_replays_complete_programs(self):
        result, rows = self.run_tool(
            COMMAND_SOURCE,
            "--benchmark=@dependent_stage_values_throughput",
            "--sample=5",
            "--input-ring-count=2",
            "--batch-size=4",
            "--profile-final-batch=true",
            "--profile-data=dispatch-events",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        benchmark_rows = [row for row in rows if row["row"] == "benchmark"]
        self.assertEqual(len(benchmark_rows), 1)
        benchmark_result = benchmark_rows[0]["benchmark_result"]
        self.assertEqual(benchmark_result["state"], "ok")
        self.assertEqual(benchmark_result["configuration_ordinal"], 2)
        self.assertEqual(benchmark_result["trial_ordinal"], 1)

        measurement = benchmark_result["measurement"]
        self.assertEqual(measurement["logical_operations_per_batch"], 4)
        self.assertEqual(measurement["physical_dispatches_per_batch"], 20)
        self.assertEqual(measurement["physical_dispatches_per_logical_operation"], 5)

        cache = benchmark_result["data_cache"]
        self.assertEqual(cache["binding_count"], 3)
        self.assertEqual(cache["binding_ring_count"], 2)
        self.assertEqual(cache["command_buffer_ring_count"], 2)
        self.assertEqual(cache["binding_set_bytes"], 384)
        self.assertEqual(cache["binding_ring_bytes"], 768)
        self.assertEqual(cache["materialized_storage_count"], 3)
        self.assertEqual(cache["materialized_storage_ring_bytes"], 1504)


if __name__ == "__main__":
    unittest.main()
