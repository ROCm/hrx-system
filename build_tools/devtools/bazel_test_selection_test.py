# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import bazel_test_selection as selection
from build_tools.devtools import project_presubmit


class BazelTestSelectionTest(unittest.TestCase):
    def setUp(self):
        self.suite = selection.TestSuite(
            name="consumer",
            targets=("//consumer/...",),
            configuration_args=("--config=presubmit", "--//consumer:feature=true"),
        )

    def test_source_label_uses_nearest_package_including_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "BUILD.bazel").touch()
            package = root / "producer/library"
            package.mkdir(parents=True)
            (package / "BUILD").touch()
            for path, expected in (
                ("producer/library/input.c", "//producer/library:input.c"),
                (
                    "producer/library/nested/input.c",
                    "//producer/library:nested/input.c",
                ),
                ("producer/tool.py", "//:producer/tool.py"),
                ("root.txt", "//:root.txt"),
            ):
                with self.subTest(path=path):
                    # Sources may have been removed from the current checkout.
                    self.assertEqual(selection.source_label(root, path), expected)

    def test_source_label_requires_a_canonical_repository_path(self):
        for path in (
            "",
            ".",
            "../a.c",
            "/a.c",
            "a/../b.c",
            "./a.c",
            "a//b.c",
            "a\\b.c",
        ):
            with self.subTest(path=path):
                self.assertIsNone(selection.source_label(Path.cwd(), path))

    def test_changed_files_reach_other_projects_without_a_directory_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "BUILD.bazel").touch()
            with mock.patch.object(
                selection.TestSuite,
                "query_affected_targets",
                return_value=["//consumer:test"],
            ) as query:
                self.assertEqual(
                    self.suite.selected_targets(root, ["producer/input.c", "tool.py"]),
                    ["//consumer:test"],
                )
            query.assert_called_once_with(
                root,
                {"//:producer/input.c", "//:tool.py"},
                set(),
                verbose=False,
            )

    def test_package_build_edits_seed_affected_query(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "producer/library"
            package.mkdir(parents=True)
            (package / "BUILD.bazel").touch()
            with mock.patch.object(
                selection.TestSuite,
                "query_affected_targets",
                return_value=["//consumer:test"],
            ) as query:
                self.assertEqual(
                    self.suite.selected_targets(root, ["producer/library/BUILD.bazel"]),
                    ["//consumer:test"],
                )
            query.assert_called_once_with(
                root, set(), {"//producer/library:"}, verbose=False
            )

    def test_unbounded_build_graph_edits_select_full_suite(self):
        for path in (
            "BUILD.bazel",
            "a/rules.bzl",
            "a/dep.BUILD",
            "MODULE.bazel.lock",
            "a/deps.MODULE.bazel",
            ".bazelrc.configured",
            "a/tool.bazelrc",
            ".bazelversion",
            "WORKSPACE.bazel",
            "requirements-dev.txt",
        ):
            with self.subTest(path=path):
                self.assertIsNone(self.suite.selected_targets(Path.cwd(), [path]))
        self.assertFalse(selection.requires_full_suite("a/BUILD"))
        self.assertFalse(selection.requires_full_suite("a/BUILD.bazel"))
        self.assertFalse(selection.requires_full_suite("build_tools/helper.py"))
        self.assertFalse(selection.requires_full_suite("build_tools/rules.cmake"))

    def test_removed_package_build_selects_full_suite(self):
        self.assertIsNone(
            self.suite.selected_targets(Path.cwd(), ["removed/BUILD.bazel"])
        )

    def test_query_matches_literal_labels_in_the_configured_closure(self):
        labels = {"//producer:input+[x].txt", "//:both'\".c"}
        query = selection.affected_query({"//consumer:test"}, labels, set())
        self.assertIn("let dependencies = deps($tests)", query)
        self.assertIn("$tests intersect rdeps($dependencies", query)
        pattern = query.split('filter("', 1)[1].split('", $dependencies)', 1)[0]
        self.assertNotIn('"', pattern)
        for label in labels:
            self.assertIsNotNone(re.fullmatch(pattern, label))
            self.assertIsNone(re.fullmatch(pattern, label + ".extra"))
        self.assertIsNone(re.fullmatch(pattern, "//producer:inputtxXtxt"))

    def test_query_matches_every_label_in_changed_packages(self):
        query = selection.affected_query(
            {"//consumer:test"}, set(), {"//producer/library:"}
        )
        pattern = query.split('filter("', 1)[1].split('", $dependencies)', 1)[0]
        self.assertIsNotNone(re.fullmatch(pattern, "//producer/library:library"))
        self.assertIsNotNone(re.fullmatch(pattern, "//producer/library:input.c"))
        self.assertIsNone(re.fullmatch(pattern, "//producer/library/nested:library"))

    def test_node_label_preserves_parentheses_in_target_names(self):
        self.assertEqual(
            selection.node_label("//consumer:test (name) (cfg)"),
            "//consumer:test (name)",
        )
        with self.assertRaisesRegex(ValueError, "malformed configured Bazel node"):
            selection.node_label("//consumer:test")

    def test_configuration_is_shared_by_query_and_execution(self):
        observed_query = None
        target_file = None

        def run(_name, command, _description, **kwargs):
            nonlocal observed_query, target_file
            self.assertEqual(kwargs["cwd"], Path.cwd())
            if command[1] == "test":
                target_file = Path(
                    next(
                        arg.split("=", 1)[1]
                        for arg in command
                        if arg.startswith("--target_pattern_file=")
                    )
                )
                self.assertEqual(target_file.read_text(), "//consumer:a_test\n")
                self.assertIn("--skip_incompatible_explicit_targets", command)
                self.assertEqual(kwargs["success_exit_codes"], (0, 4))
            else:
                output = Path(
                    next(
                        arg.split("=", 1)[1]
                        for arg in command
                        if arg.startswith("--output_file=")
                    )
                )
                if command[1] == "query":
                    output.write_text("//consumer:a_test\n//consumer:b_test\n")
                else:
                    query_path = Path(
                        next(
                            arg.split("=", 1)[1]
                            for arg in command
                            if arg.startswith("--query_file=")
                        )
                    )
                    observed_query = query_path.read_text()
                    output.write_text(
                        "//consumer:a_test (cfg)\n//consumer:a_test (exec)\n"
                    )
            return True

        with (
            mock.patch.dict(
                os.environ,
                {
                    project_presubmit.BAZEL_CONFIGS_ENV: json.dumps(
                        ["asan", "remote-execution"]
                    )
                },
                clear=True,
            ),
            mock.patch.object(
                selection, "source_label", return_value="//producer:input.c"
            ),
            mock.patch.object(
                project_presubmit, "run_command", side_effect=run
            ) as command,
        ):
            self.assertTrue(self.suite.run(Path.cwd(), paths=["producer/input.c"]))
        self.assertEqual(command.call_count, 3)
        for call in command.call_args_list[1:]:
            options = call.args[1]
            self.assertIn("--config=presubmit", options)
            self.assertIn("--//consumer:feature=true", options)
            self.assertIn("--config=asan", options)
            self.assertIn("--config=remote-execution", options)
        self.assertIn(
            "--experimental_explicit_aspects", command.call_args_list[1].args[1]
        )
        self.assertIn("//producer:input", observed_query)
        self.assertFalse(target_file.exists())

    def test_manual_policy_suites_are_explicit_without_enabling_other_manual_tests(
        self,
    ):
        suite = selection.TestSuite(
            name="policy",
            targets=("//tools/...", "-//tools:excluded_test"),
            manual_suites=("//tools:presubmit_tests",),
            excluded_tags=("requires-gpu",),
        )
        query = suite.discovery_query()
        self.assertIn("manual", query)
        self.assertIn("union tests(set(//tools:presubmit_tests))", query)
        self.assertIn("except tests(set(//tools:excluded_test))", query)
        command = suite.test_command()
        self.assertIn("--test_tag_filters=-requires-gpu", command)
        self.assertIn("//tools:presubmit_tests", command)
        self.assertIn("-//tools:excluded_test", command)
        self.assertIn("--test_tag_filters=-manual", self.suite.test_command())

    def test_full_policy_runs_use_the_discovered_members_and_require_tests(self):
        suite = selection.TestSuite(
            name="policy", targets=("//tools/...",), manual_suites=("//tools:policy",)
        )

        def run(_name, command, _description, **kwargs):
            self.assertEqual(command[1], "test")
            target_file = Path(
                next(
                    arg.split("=", 1)[1]
                    for arg in command
                    if arg.startswith("--target_pattern_file=")
                )
            )
            self.assertEqual(
                target_file.read_text(), "//tools:ordinary_test\n//tools:policy_test\n"
            )
            self.assertEqual(kwargs["success_exit_codes"], (0,))
            return True

        with (
            mock.patch.object(
                selection.TestSuite,
                "discover_targets",
                return_value={"//tools:ordinary_test", "//tools:policy_test"},
            ),
            mock.patch.object(project_presubmit, "run_command", side_effect=run),
        ):
            self.assertTrue(suite.run(Path.cwd()))

    def test_query_errors_are_not_empty_selections_or_full_suite_fallbacks(self):
        for failed_phase in ("query", "cquery"):
            with self.subTest(failed_phase=failed_phase):

                def run(_name, command, _description, **_kwargs):
                    if command[1] == failed_phase:
                        return False
                    self.assertEqual(command[1], "query")
                    output = Path(
                        next(
                            arg.split("=", 1)[1]
                            for arg in command
                            if arg.startswith("--output_file=")
                        )
                    )
                    output.write_text("//consumer:test\n")
                    return True

                with (
                    mock.patch.object(
                        selection, "source_label", return_value="//producer:input.c"
                    ),
                    mock.patch.object(
                        project_presubmit, "run_command", side_effect=run
                    ) as command,
                    contextlib.redirect_stderr(io.StringIO()) as error,
                ):
                    self.assertFalse(
                        self.suite.run(Path.cwd(), paths=["producer/input.c"])
                    )
                self.assertIn("failed to", error.getvalue())
                self.assertFalse(
                    any(call.args[1][1] == "test" for call in command.call_args_list)
                )

    def test_empty_selection_skips_execution_but_malformed_output_fails(self):
        for result in ("", "not a configured node\n"):
            with self.subTest(result=result):

                def run(_name, command, _description, **_kwargs):
                    self.assertIn(command[1], ("query", "cquery"))
                    output = Path(
                        next(
                            arg.split("=", 1)[1]
                            for arg in command
                            if arg.startswith("--output_file=")
                        )
                    )
                    output.write_text(
                        "//consumer:test\n" if command[1] == "query" else result
                    )
                    return True

                with (
                    mock.patch.object(
                        selection, "source_label", return_value="//producer:input.c"
                    ),
                    mock.patch.object(
                        project_presubmit, "run_command", side_effect=run
                    ),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(
                        self.suite.run(Path.cwd(), paths=["producer/input.c"]),
                        not result,
                    )

    def test_full_suite_requires_tests(self):
        for exit_code in (0, 1, 3, 4):
            with (
                self.subTest(exit_code=exit_code),
                mock.patch.object(
                    project_presubmit.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], exit_code),
                ) as run,
            ):
                self.assertEqual(self.suite.run(Path.cwd()), exit_code == 0)
                self.assertEqual(run.call_args.args[0][-1], "//consumer/...")


if __name__ == "__main__":
    unittest.main()
