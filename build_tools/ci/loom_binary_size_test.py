# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import tempfile
import unittest
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

from build_tools.ci import loom_binary_size

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
COMMIT_C = "c" * 40


def witness(
    witness_id: str = "loomc-amdgpu-all",
    *,
    fingerprint: str = "sha256:stable",
    stripped_bytes: int = 100,
) -> dict:
    executable_bytes = stripped_bytes // 2
    return {
        "id": witness_id,
        "title": witness_id,
        "fingerprint": fingerprint,
        "unstripped_bytes": stripped_bytes + 20,
        "stripped_bytes": stripped_bytes,
        "sections": {
            "executable": executable_bytes,
            "read_only": 0,
            "writable": 0,
            "unwind": 0,
            "other": stripped_bytes - executable_bytes,
        },
        "sha256": "0" * 64,
    }


def observation(commit: str, *witnesses: dict) -> dict:
    return {
        "commit": commit,
        "repository": "ROCm/hrx-system",
        "ref": "main",
        "run_id": "123",
        "measured_at": "2026-08-31T00:00:00+00:00",
        "toolchain": {"cc": "clang"},
        "witnesses": list(witnesses or (witness(),)),
    }


def current(commit: str, *witnesses: dict) -> dict:
    return {
        "schema": loom_binary_size.SCHEMA,
        "observation": observation(commit, *witnesses),
    }


def history(*observations: dict) -> dict:
    return {
        "schema": loom_binary_size.SCHEMA,
        "history": list(observations),
    }


class LoomBinarySizeTest(unittest.TestCase):
    def test_build_plan_coalesces_witnesses_into_two_actions(self):
        commands = loom_binary_size.build_commands(Path("/repo"))

        self.assertEqual(len(commands), 2)
        self.assertIn("//loom/binding/c/example:emit_amdgpu_offline", commands[0])
        self.assertIn("//loom/src/loom/tools/loom-compile:loom-compile", commands[0])
        self.assertIn("//loom/src/loom/tools/loom-check:loom-check", commands[0])
        self.assertEqual(
            [
                arg
                for arg in commands[0]
                if arg.startswith("--//runtime/config/hal:drivers=")
            ][-1],
            "--//runtime/config/hal:drivers=task,vulkan",
        )
        self.assertEqual(
            commands[1].count("//loom/binding/c/example:emit_amdgpu_offline"), 1
        )
        self.assertEqual(
            [
                arg
                for arg in commands[1]
                if arg.startswith("--//runtime/config/hal:drivers=")
            ][-1],
            "--//runtime/config/hal:drivers=task",
        )
        self.assertIn("--//loom/config/target/amdgpu:targets=gfx1151", commands[1])

    def test_tool_identity_ignores_versioned_rocm_install_root(self):
        identity = loom_binary_size.normalize_tool_identity(
            "InstalledDir: /opt/rocm-nightly-123/lib/llvm/bin",
            Path("/opt/rocm-nightly-123"),
        )

        self.assertEqual(identity, "InstalledDir: $ROCM_ROOT/lib/llvm/bin")

    def test_section_classes_reconcile_file_bytes(self):
        file_record = {
            "Sections": [
                {
                    "Section": {
                        "Name": {"Name": ".text"},
                        "Type": {"Name": "SHT_PROGBITS"},
                        "Size": 40,
                        "Flags": {
                            "Flags": [
                                {"Name": "SHF_ALLOC"},
                                {"Name": "SHF_EXECINSTR"},
                            ]
                        },
                    }
                },
                {
                    "Section": {
                        "Name": {"Name": ".rodata"},
                        "Type": {"Name": "SHT_PROGBITS"},
                        "Size": 20,
                        "Flags": {"Flags": [{"Name": "SHF_ALLOC"}]},
                    }
                },
                {
                    "Section": {
                        "Name": {"Name": ".eh_frame"},
                        "Type": {"Name": "SHT_PROGBITS"},
                        "Size": 10,
                        "Flags": {"Flags": [{"Name": "SHF_ALLOC"}]},
                    }
                },
                {
                    "Section": {
                        "Name": {"Name": ".data"},
                        "Type": {"Name": "SHT_PROGBITS"},
                        "Size": 5,
                        "Flags": {
                            "Flags": [
                                {"Name": "SHF_ALLOC"},
                                {"Name": "SHF_WRITE"},
                            ]
                        },
                    }
                },
                {
                    "Section": {
                        "Name": {"Name": ".bss"},
                        "Type": {"Name": "SHT_NOBITS"},
                        "Size": 1000,
                        "Flags": {
                            "Flags": [
                                {"Name": "SHF_ALLOC"},
                                {"Name": "SHF_WRITE"},
                            ]
                        },
                    }
                },
            ]
        }

        classes = loom_binary_size.classify_elf_sections(file_record, 100)

        self.assertEqual(
            classes,
            {
                "executable": 40,
                "read_only": 20,
                "writable": 5,
                "unwind": 10,
                "other": 25,
            },
        )

    def test_section_payload_larger_than_file_fails(self):
        file_record = {
            "Sections": [
                {
                    "Section": {
                        "Name": {"Name": ".text"},
                        "Type": {"Name": "SHT_PROGBITS"},
                        "Size": 101,
                        "Flags": {"Flags": [{"Name": "SHF_ALLOC"}]},
                    }
                }
            ]
        }

        with self.assertRaisesRegex(
            loom_binary_size.BinarySizeError, "exceeds file size"
        ):
            loom_binary_size.classify_elf_sections(file_record, 100)

    def test_history_replaces_rerun_and_stays_bounded(self):
        observations = [
            observation(f"{index:040x}", witness(stripped_bytes=index + 1))
            for index in range(loom_binary_size.MAX_MAIN_OBSERVATIONS)
        ]
        rerun = current(
            f"{loom_binary_size.MAX_MAIN_OBSERVATIONS - 1:040x}",
            witness(stripped_bytes=999),
        )

        merged = loom_binary_size.merge_history(
            history(*observations), rerun, update_main=True
        )

        self.assertEqual(len(merged["history"]), loom_binary_size.MAX_MAIN_OBSERVATIONS)
        self.assertEqual(merged["history"][-1]["witnesses"][0]["stripped_bytes"], 999)
        self.assertEqual(
            len({item["commit"] for item in merged["history"]}),
            loom_binary_size.MAX_MAIN_OBSERVATIONS,
        )

    def test_pull_request_does_not_enter_main_history(self):
        baseline = history(observation(COMMIT_A))

        merged = loom_binary_size.merge_history(
            baseline, current(COMMIT_B), update_main=False
        )

        self.assertEqual(merged, baseline)

    def test_summary_uses_only_compatible_fingerprint(self):
        baseline = history(
            observation(COMMIT_A, witness(fingerprint="sha256:old", stripped_bytes=90)),
            observation(
                COMMIT_B, witness(fingerprint="sha256:new", stripped_bytes=110)
            ),
        )

        summary = loom_binary_size.render_summary(
            baseline,
            current(COMMIT_C, witness(fingerprint="sha256:old", stripped_bytes=100)),
        )

        self.assertIn("100 B", summary)
        self.assertIn("90 B", summary)
        self.assertIn("+10 B (+11.11%)", summary)
        self.assertNotIn("110 B", summary)

    def test_summary_marks_new_fingerprint_series(self):
        summary = loom_binary_size.render_summary(
            history(observation(COMMIT_A)),
            current(COMMIT_B, witness(fingerprint="sha256:different")),
        )

        self.assertIn("new series", summary)

    def test_nearest_ancestor_artifact_wins_over_creation_time(self):
        artifacts = [
            {
                "id": 1,
                "created_at": "2026-08-31T03:00:00Z",
                "workflow_run": {"head_sha": COMMIT_A},
            },
            {
                "id": 2,
                "created_at": "2026-08-31T01:00:00Z",
                "workflow_run": {"head_sha": COMMIT_B},
            },
        ]
        distances = {COMMIT_A: 5, COMMIT_B: 1}

        selected = loom_binary_size.select_nearest_artifact(
            artifacts,
            anchor=COMMIT_C,
            exclude_commit=None,
            distance=lambda commit, _anchor: distances[commit],
        )

        self.assertEqual(selected["id"], 2)

    def test_current_main_commit_is_excluded_from_baseline(self):
        artifacts = [
            {
                "id": 1,
                "created_at": "2026-08-31T03:00:00Z",
                "workflow_run": {"head_sha": COMMIT_C},
            },
            {
                "id": 2,
                "created_at": "2026-08-31T01:00:00Z",
                "workflow_run": {"head_sha": COMMIT_B},
            },
        ]

        selected = loom_binary_size.select_nearest_artifact(
            artifacts,
            anchor=COMMIT_C,
            exclude_commit=COMMIT_C,
            distance=lambda commit, _anchor: 1 if commit == COMMIT_B else 0,
        )

        self.assertEqual(selected["id"], 2)

    def test_history_zip_must_match_workflow_head(self):
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(
                "history.json",
                json.dumps(history(observation(COMMIT_A))),
            )

        with self.assertRaisesRegex(
            loom_binary_size.BinarySizeError, "head does not match"
        ):
            loom_binary_size.history_from_zip(payload.getvalue(), COMMIT_B)

    def test_github_authorization_does_not_cross_redirect_origins(self):
        handler = loom_binary_size._GitHubRedirectHandler()
        request = urllib.request.Request(
            "https://api.github.com/repos/ROCm/hrx-system/actions/artifacts/1/zip",
            headers={"Authorization": "Bearer secret"},
        )

        cross_origin_request = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://results.example.com/artifact.zip",
        )
        same_origin_request = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://API.GITHUB.COM:443/artifact.zip",
        )

        self.assertIsNotNone(cross_origin_request)
        self.assertFalse(cross_origin_request.has_header("Authorization"))
        self.assertIsNotNone(same_origin_request)
        self.assertTrue(same_origin_request.has_header("Authorization"))

    def test_report_writes_rolling_state_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_path = root / "current.json"
            baseline_path = root / "baseline.json"
            output_dir = root / "output"
            current_path.write_text(json.dumps(current(COMMIT_B)), encoding="utf-8")
            baseline_path.write_text(
                json.dumps(history(observation(COMMIT_A))), encoding="utf-8"
            )
            args = type(
                "Args",
                (),
                {
                    "current": current_path,
                    "baseline": baseline_path,
                    "output_dir": output_dir,
                    "update_main": True,
                },
            )()

            loom_binary_size.report(args)

            reported_current = json.loads(
                (output_dir / "current.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reported_current, current(COMMIT_B))
            result = json.loads(
                (output_dir / "history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["commit"] for item in result["history"]],
                [COMMIT_A, COMMIT_B],
            )
            self.assertIn(
                "Loom compiler product size",
                (output_dir / "summary.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
