# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from build_tools.ci import loom_release_recovery
from loom.packaging.python import release


def job(name: str, conclusion: str) -> dict:
    return {"name": name, "status": "completed", "conclusion": conclusion}


class Client:
    def __init__(self, run: dict, jobs: list[dict]):
        self.run = run
        self.jobs = jobs

    def json(self, path: str) -> dict:
        self.requested_run = path
        return self.run

    def pages(self, path: str, field: str) -> list[dict]:
        self.requested_jobs = (path, field)
        return self.jobs


class LoomReleaseRecoveryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = "fixture/sdk"
        self.run_id = 123
        self.revision = "a" * 40
        self.tag = "v1.2.3rc1"
        self.run = {
            "id": self.run_id,
            "path": loom_release_recovery.SOURCE_WORKFLOW,
            "event": "push",
            "status": "completed",
            "conclusion": "failure",
            "head_sha": self.revision,
            "head_branch": self.tag,
            "repository": {"full_name": self.repository},
            "head_repository": {"full_name": self.repository},
        }
        self.jobs = [
            job("Validate / Release identity", "success"),
            job("Release / Build / Linux + Windows x86-64 / Clang", "success"),
            job("Security / Native production / Analyze", "success"),
            job("Bundle / Verify artifact set", "failure"),
            job("Publish / GitHub assets", "skipped"),
            job("Publish / PyPI wheels", "skipped"),
            job("Complete / Advance main development", "skipped"),
        ]

    def verify(self, run=None, jobs=None):
        loom_release_recovery.verify_source_run(
            Client(run or self.run, jobs or self.jobs),
            self.run_id,
            self.repository,
            self.revision,
            self.tag,
        )

    def test_source_run_requires_exact_identity_and_successful_prerequisites(self):
        self.verify()

        wrong_revision = copy.deepcopy(self.run)
        wrong_revision["head_sha"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "exact tag"):
            self.verify(run=wrong_revision)

        failed_jobs = copy.deepcopy(self.jobs)
        failed_jobs[1]["conclusion"] = "failure"
        with self.assertRaisesRegex(ValueError, "unusable conclusion"):
            self.verify(jobs=failed_jobs)

        missing_security = [
            record
            for record in self.jobs
            if record["name"] != "Security / Native production / Analyze"
        ]
        with self.assertRaisesRegex(ValueError, "missing required jobs"):
            self.verify(jobs=missing_security)

        invalid_terminal_state = copy.deepcopy(self.jobs)
        invalid_terminal_state[3]["conclusion"] = "success"
        with self.assertRaisesRegex(ValueError, "did not fail"):
            self.verify(jobs=invalid_terminal_state)

    def release_inputs(self) -> tuple[Path, set[str]]:
        inputs = self.root / "inputs"
        wheels = inputs / "wheels"
        standalones = inputs / "standalones"
        wheels.mkdir(parents=True)
        standalones.mkdir()
        identity = release.Identity("1.2.3rc1", "github:fixture/catalog@v1.2.3rc1")
        names = set()
        for profile in release.PROFILES:
            for platform in release.PLATFORMS:
                version = identity.wheel_version(profile)
                name = f"hrx_loom-{version}-py3-none-{platform}.whl"
                (wheels / name).write_bytes(name.encode())
                names.add(name)
        for tool in release.STANDALONE_TOOLS:
            for platform in release.PLATFORMS:
                name = release.standalone_archive_name(identity, tool, platform)
                directory = (
                    standalones / "server-artifact"
                    if tool == "loom-serve"
                    else standalones
                )
                directory.mkdir(exist_ok=True)
                (directory / name).write_bytes(name.encode())
                names.add(name)
        return inputs, names

    def test_staging_flattens_legacy_layout_without_replacing_or_hiding_files(self):
        inputs, names = self.release_inputs()
        output = self.root / "output"

        staged = loom_release_recovery.stage_release_assets(inputs, output)

        self.assertEqual({path.name for path in staged}, names)
        self.assertEqual({path.name for path in output.iterdir()}, names)
        for path in staged:
            self.assertEqual(path.read_bytes(), path.name.encode())

        duplicate = (
            inputs
            / "standalones"
            / next(name for name in names if name.startswith("loom-serve-"))
        )
        duplicate.write_bytes(b"duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate filename"):
            loom_release_recovery.stage_release_assets(
                inputs, self.root / "duplicate-output"
            )
        duplicate.unlink()
        (inputs / "standalones" / "unexpected.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "Unexpected recovered artifact path"):
            loom_release_recovery.stage_release_assets(
                inputs, self.root / "unexpected-output"
            )


if __name__ == "__main__":
    unittest.main()
