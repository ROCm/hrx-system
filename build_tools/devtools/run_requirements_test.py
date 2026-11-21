# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests resource admission and validation of build-system test metadata."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import run_requirements


class RunRequirementsTest(unittest.TestCase):
    def test_loads_run_declarations_through_shared_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shared.bzl").write_text("""
load("//tools:requirements.bzl", "build_requirement", "run_requirement")
BUILD = build_requirement(id="compile.api")
RUN = run_requirement(id="device.api")
""")
            (root / "owner.bzl").write_text("""
load("//:shared.bzl", "BUILD", "RUN")
REQUIREMENTS = [BUILD, RUN]
""")
            with mock.patch.object(
                run_requirements, "REQUIREMENT_MODULES", ("owner.bzl",)
            ):
                self.assertEqual(run_requirements.declared_ids(root), {"device.api"})

    def test_filters_require_every_resource_and_preserve_cpu_tests(self):
        known = frozenset(("native.gpu", "api.vulkan", "api.d3d12"))
        with mock.patch.object(run_requirements, "declared_ids", return_value=known):
            bazel_filters = run_requirements.bazel_exclusions(("api.vulkan",))
            ctest_filter = run_requirements.ctest_exclusion_regex(("api.vulkan",))
        self.assertEqual(
            set(bazel_filters),
            {
                "-iree-run-requirement=native.gpu",
                "-iree-run-requirement=api.d3d12",
            },
        )
        cases = (
            ([], True),
            (["api.vulkan"], True),
            (["api.vulkan", "native.gpu"], False),
            (["api.vulkan", "api.d3d12"], False),
        )
        for requirements, expected_admitted in cases:
            with self.subTest(requirements=requirements):
                excluded = any(
                    re.search(ctest_filter, run_requirements.TAG_PREFIX + value)
                    for value in requirements
                )
                self.assertEqual(not excluded, expected_admitted)
        # An anchored resource label cannot accidentally match a longer tag.
        self.assertIsNone(
            re.search(ctest_filter, "iree-run-requirement=native.gpu-extra")
        )

    def test_all_resources_available_needs_no_exclusion(self):
        with mock.patch.object(
            run_requirements, "declared_ids", return_value={"device"}
        ):
            self.assertEqual(run_requirements.bazel_exclusions(("device",)), ())
            self.assertEqual(run_requirements.ctest_exclusion_regex(("device",)), "")

    def test_unknown_profile_resource_is_an_error(self):
        with mock.patch.object(
            run_requirements, "declared_ids", return_value={"device"}
        ):
            with self.assertRaisesRegex(ValueError, "Unknown run requirements: typo"):
                run_requirements.bazel_exclusions(("typo",))

    def test_audits_actual_tag_fields_in_both_build_system_formats(self):
        for requirement in ("device", "typo"):
            tag = "iree-run-requirement=" + requirement
            other = {
                "name": "other",
                "stringListValue": ["iree-run-requirement=unrelated"],
            }
            records = []
            for name, value in (
                ("//suite:first", "iree-run-requirement=device"),
                ("//suite:test", tag),
            ):
                tags = {"name": "tags", "stringListValue": [value]}
                rule = {"name": name, "attribute": [other, tags]}
                records.append(json.dumps({"rule": rule}))
            ctest = json.dumps(
                {
                    "kind": "ctestInfo",
                    "version": {"major": 1},
                    "tests": [
                        {
                            "name": "suite/test",
                            "properties": [{"name": "LABELS", "value": [tag]}],
                        }
                    ],
                }
            )
            for audit, payload in (
                (run_requirements.audit_bazel_json, "\n".join(records)),
                (run_requirements.audit_ctest_json, ctest),
            ):
                with self.subTest(requirement=requirement, audit=audit.__name__):
                    with mock.patch.object(
                        run_requirements, "declared_ids", return_value={"device"}
                    ):
                        if requirement == "device":
                            audit(payload)
                        else:
                            with self.assertRaisesRegex(
                                ValueError, "unknown run requirements: typo"
                            ):
                                audit(payload)

    def test_query_preserves_exclusion_then_reinclusion(self):
        query = run_requirements.bazel_test_query(
            ("//suite/...", "-//suite/gpu/...", "//suite/gpu:cpu_test")
        )
        self.assertEqual(
            query,
            '(((set() union tests(set("//suite/..."))) except tests(set("//suite/gpu/..."))) union tests(set("//suite/gpu:cpu_test")))',
        )


if __name__ == "__main__":
    unittest.main()
