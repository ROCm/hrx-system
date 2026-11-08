# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import fuzz_campaign


class FuzzCampaignTest(unittest.TestCase):
    def test_telemetry_arguments_force_final_statistics(self):
        self.assertEqual(
            fuzz_campaign.telemetry_arguments(["-runs=20", "-print_final_stats=0"]),
            ["-runs=20", "-print_final_stats=1"],
        )

    def test_observer_parses_progress_and_final_statistics(self):
        observer = fuzz_campaign.FuzzerOutputObserver()
        observer.consume(
            b"#10 INITED cov: 100 ft: 200 corp: 4/1Kb exec/s: 0 rss: 27Mb\n"
            b"#41 NEW cov: 103 ft: 217 corp: 7/1.5Kb lim: 64 "
            b"exec/s: 512 rss: 29Mb\n"
            b"stat::number_of_executed_units: 45\n"
            b"stat::average_exec_per_sec: 450\n"
        )
        observer.consume(
            b"stat::new_units_added: 6\n"
            b"stat::slowest_unit_time_sec: 0.125\n"
            b"stat::peak_rss_mb: 31"
        )
        observer.finalize()

        self.assertEqual(
            observer.statistics.json(),
            {
                "executed_units": 45,
                "average_executions_per_second": 450,
                "new_units_added": 6,
                "slowest_unit_seconds": 0.125,
                "peak_rss_mb": 31,
                "initialization_duration_ns": None,
                "initial_coverage_points": 100,
                "coverage_points": 103,
                "coverage_points_added": 3,
                "initial_features": 200,
                "features": 217,
                "features_added": 17,
                "initial_corpus_units": 4,
                "corpus_units": 7,
                "initial_corpus_bytes": 1024,
                "corpus_bytes": 1536,
            },
        )

    def test_observer_records_initialization_duration(self):
        observer = fuzz_campaign.FuzzerOutputObserver(start_ns=100)
        with mock.patch.object(fuzz_campaign.time, "monotonic_ns", return_value=375):
            observer.consume(b"#1 INITED cov: 2 ft: 3 corp: 1/1b\n")

        self.assertTrue(observer.initialized.is_set())
        self.assertEqual(observer.initialized_ns, 375)
        self.assertEqual(observer.statistics.initialization_duration_ns, 275)

    def test_observer_retains_only_bounded_tail(self):
        observer = fuzz_campaign.FuzzerOutputObserver(tail_limit=16)
        observer.consume(b"first output line\n")
        observer.consume(b"last output line\n")
        observer.finalize()

        self.assertEqual(observer.tail, b"ast output line\n")
        self.assertTrue(observer.tail_truncated)
        self.assertEqual(observer.total_bytes, 35)

    def test_observe_stream_preserves_direct_output(self):
        observer = fuzz_campaign.FuzzerOutputObserver()
        destination = io.StringIO()
        fuzz_campaign.observe_stream(
            io.BytesIO(b"fuzzer output\n"), observer, destination
        )

        self.assertEqual(destination.getvalue(), "fuzzer output\n")
        self.assertEqual(observer.tail, b"fuzzer output\n")

    def test_observe_stream_can_suppress_unattended_output(self):
        observer = fuzz_campaign.FuzzerOutputObserver()
        fuzz_campaign.observe_stream(io.BytesIO(b"fuzzer output\n"), observer, None)

        self.assertEqual(observer.tail, b"fuzzer output\n")

    def test_observe_stream_reads_available_pipe_bytes(self):
        class AvailableStream:
            def __init__(self):
                self.chunks = [b"#1 INITED cov: 2 ft: 3 corp: 1/1b\n", b""]
                self.closed = False

            def read(self, _size):
                raise AssertionError("buffered read must not gate live progress")

            def read1(self, _size):
                return self.chunks.pop(0)

            def close(self):
                self.closed = True

        stream = AvailableStream()
        observer = fuzz_campaign.FuzzerOutputObserver()
        fuzz_campaign.observe_stream(stream, observer, None)

        self.assertTrue(observer.initialized.is_set())
        self.assertTrue(stream.closed)

    def test_failure_output_is_bounded_and_marks_truncation(self):
        observer = fuzz_campaign.FuzzerOutputObserver(tail_limit=8)
        observer.consume(b"0123456789")
        observer.finalize()
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "last-failure.log"
            fuzz_campaign.write_failure_output(output_path, observer)
            output = output_path.read_bytes()

        self.assertEqual(output, b"... earlier output truncated ...\n23456789")

    def test_stop_request_is_published_nonempty(self):
        with tempfile.TemporaryDirectory() as temporary:
            stop_path = Path(temporary) / "stop"
            fuzz_campaign.write_stop_request(stop_path)

            self.assertEqual(stop_path.read_bytes(), b"stop\n")
            self.assertEqual(list(stop_path.parent.glob(".stop.*.tmp")), [])

    def test_directory_snapshot_and_change_are_recursive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nested"
            nested.mkdir()
            (root / "first").write_bytes(b"abc")
            before = fuzz_campaign.snapshot_directory(root)
            (nested / "second").write_bytes(b"12345")
            after = fuzz_campaign.snapshot_directory(root)

        self.assertEqual(before, fuzz_campaign.DirectorySnapshot(1, 3))
        self.assertEqual(
            fuzz_campaign.directory_change(before, after),
            {
                "before": {"files": 1, "bytes": 3},
                "after": {"files": 2, "bytes": 8},
                "delta": {"files": 1, "bytes": 5},
            },
        )

    def test_executable_identity_hashes_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable_path = Path(temporary) / "fuzzer"
            executable_path.write_bytes(b"exact executable")
            identity = fuzz_campaign.executable_identity(executable_path)

        self.assertEqual(identity.byte_count, 16)
        self.assertEqual(
            identity.sha256, hashlib.sha256(b"exact executable").hexdigest()
        )

    def test_source_identity_distinguishes_dirty_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            subprocess.run(["git", "init", "-q", repository], check=True)
            subprocess.run(
                ["git", "-C", repository, "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", repository, "config", "user.email", "test@example.com"],
                check=True,
            )
            source_path = repository / "source.c"
            source_path.write_text("int value;\n", encoding="utf-8")
            subprocess.run(["git", "-C", repository, "add", "source.c"], check=True)
            subprocess.run(
                ["git", "-C", repository, "commit", "-q", "-m", "initial"],
                check=True,
            )
            clean = fuzz_campaign.source_identity(repository)
            source_path.write_text("int changed;\n", encoding="utf-8")
            dirty = fuzz_campaign.source_identity(repository)
            source_path.write_text("int changed_again;\n", encoding="utf-8")
            changed_again = fuzz_campaign.source_identity(repository)
            untracked_path = repository / "untracked.c"
            untracked_path.write_text("int untracked;\n", encoding="utf-8")
            with_untracked = fuzz_campaign.source_identity(repository)
            untracked_path.write_text("int changed_untracked;\n", encoding="utf-8")
            changed_untracked = fuzz_campaign.source_identity(repository)

        self.assertFalse(clean.dirty)
        self.assertIsNone(clean.worktree_state_sha256)
        self.assertTrue(dirty.dirty)
        self.assertNotEqual(
            dirty.worktree_state_sha256, changed_again.worktree_state_sha256
        )
        self.assertNotEqual(
            with_untracked.worktree_state_sha256,
            changed_untracked.worktree_state_sha256,
        )


if __name__ == "__main__":
    unittest.main()
