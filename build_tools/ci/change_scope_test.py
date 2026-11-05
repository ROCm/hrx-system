# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from build_tools.ci import change_scope


class ChangeScopeTest(unittest.TestCase):
    def test_scope_requires_only_compiler_owned_changes(self):
        compiler = ["loom/src/loom/example.c", "loom/src/loom/BUILD.bazel"]
        self.assertTrue(change_scope.loom_only(compiler))
        self.assertFalse(change_scope.loom_only([]))
        for path in (
            "runtime/src/iree/example.c",
            "build_tools/cmake/rules.cmake",
            "loom/CMakeLists.txt",
            "loom/BUILD.bazel",
            "loom/build_tools/bazel/rules.bzl",
            "loom/config/arch/BUILD.bazel",
            "loom/requirements/defs.bzl",
        ):
            with self.subTest(path=path):
                self.assertFalse(change_scope.loom_only([*compiler, path]))

    def test_documentation_scope_matches_website_inputs_and_owners(self):
        for path in (
            ".github/workflows/build_loom_sdk_artifacts.yml",
            ".github/workflows/build_loom_sdk.yml",
            ".github/workflows/ci_iree.yml",
            "build_tools/ci/change_scope.py",
            "build_tools/ci/loom_wheel.py",
            "build_tools/devtools/setup.py",
            "dev.py",
            "requirements-analysis.lock.txt",
            "requirements-dev.lock.txt",
            "loom/binding/c/loomc.h",
            "loom/docs/mkdocs.yml",
            "loom/py/loom/gen/docs/reference.py",
            "loom/src/loom/editor/textmate/loom.tmLanguage.json",
        ):
            with self.subTest(path=path):
                self.assertTrue(change_scope.loom_docs_affected([path]))
        for path in (
            "README.md",
            "build_tools/ci/github_actions_watch.py",
            "loom/src/loom/ir/op.c",
            "runtime/src/iree/base/api.h",
        ):
            with self.subTest(path=path):
                self.assertFalse(change_scope.loom_docs_affected([path]))

    def test_revision_difference_includes_renames_deletions_and_all_commits(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)

            def git(*arguments: str) -> str:
                return subprocess.check_output(
                    ["git", *arguments],
                    cwd=repository,
                    text=True,
                    stderr=subprocess.PIPE,
                ).strip()

            def commit() -> str:
                git("add", ".")
                git(
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-qm",
                    "Update fixture",
                )
                return git("rev-parse", "HEAD")

            git("init", "-q")
            (repository / "runtime").mkdir()
            (repository / "runtime/original.c").write_text("int fixture;\n")
            base = commit()
            (repository / "loom").mkdir()
            name = "with space.c" if os.name == "nt" else "with space\nand newline.c"
            renamed = repository / "loom" / name
            (repository / "runtime/original.c").rename(renamed)
            renamed_revision = commit()
            (repository / "loom/second.c").write_text("int second;\n")
            commit()

            paths = change_scope.changed_paths(repository, base)
            self.assertEqual(
                set(paths),
                {
                    "runtime/original.c",
                    "loom/" + name,
                    "loom/second.c",
                },
            )
            self.assertFalse(change_scope.loom_only(paths))
            self.assertTrue(
                change_scope.loom_only(
                    change_scope.changed_paths(repository, renamed_revision)
                )
            )
            self.assertEqual(
                change_scope.changed_paths(repository, git("rev-parse", "HEAD")), []
            )
            with tempfile.TemporaryDirectory() as clone_directory:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--quiet",
                        "--depth=1",
                        repository.as_uri(),
                        clone_directory,
                    ],
                    check=True,
                )
                self.assertEqual(
                    change_scope.changed_paths(Path(clone_directory), base), paths
                )
            with self.assertRaises(subprocess.CalledProcessError):
                change_scope.changed_paths(repository, "1" * 40)

            self.assertFalse(change_scope.select_loom_only(repository, ""))
            self.assertFalse(change_scope.select_loom_only(repository, "0" * 40))
            self.assertTrue(change_scope.select_loom_only(repository, renamed_revision))
            self.assertTrue(change_scope.select_loom_docs(repository, ""))
            self.assertTrue(change_scope.select_loom_docs(repository, "0" * 40))
            self.assertFalse(
                change_scope.select_loom_docs(repository, renamed_revision)
            )


if __name__ == "__main__":
    unittest.main()
