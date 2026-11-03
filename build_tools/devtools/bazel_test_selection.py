# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs affected tests through their configured Bazel dependency graph."""

from __future__ import annotations

import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from build_tools.devtools import project_presubmit


def source_label(repo_root: Path, path: str) -> str | None:
    """Maps a canonical repository path to its nearest Bazel package."""
    source_path = PurePosixPath(path)
    if (
        not source_path.parts
        or source_path.is_absolute()
        or "\\" in path
        or source_path.as_posix() != path
        or ".." in source_path.parts
    ):
        return None
    directory = (repo_root / source_path).parent
    while True:
        if (directory / "BUILD.bazel").is_file() or (directory / "BUILD").is_file():
            package = (
                directory.relative_to(repo_root).as_posix()
                if directory != repo_root
                else ""
            )
            name = (repo_root / source_path).relative_to(directory).as_posix()
            return f"//{package}:{name}"
        if directory == repo_root:
            return None
        directory = directory.parent


def is_package_build_file(path: str) -> bool:
    """Returns whether path is a nested package-local BUILD definition."""
    source_path = PurePosixPath(path)
    return len(source_path.parts) > 1 and source_path.name in ("BUILD", "BUILD.bazel")


def package_label_prefix(repo_root: Path, path: str) -> str | None:
    """Returns the label prefix owned by an existing package BUILD file."""
    if not is_package_build_file(path):
        return None
    build_path = repo_root / path
    if not build_path.is_file():
        return None
    package = PurePosixPath(path).parent.as_posix()
    return f"//{package}:"


def requires_full_suite(path: str) -> bool:
    """Returns whether a build edit has no package-local dependency boundary."""
    if is_package_build_file(path):
        return False
    name = PurePosixPath(path).name
    return (
        name
        in (
            "BUILD",
            "BUILD.bazel",
            ".bazelversion",
            "WORKSPACE",
            "WORKSPACE.bazel",
            "WORKSPACE.bzlmod",
        )
        or path.endswith(
            (".bzl", ".BUILD", "MODULE.bazel", "MODULE.bazel.lock", ".bazelrc")
        )
        or name.startswith(".bazelrc")
        or (path.startswith("requirements") and path.endswith(".txt"))
    )


def node_label(node: str) -> str:
    separator = node.rfind(" (")
    if separator < 0 or not node.endswith(")"):
        raise ValueError(f"malformed configured Bazel node: {node}")
    return node[:separator]


def affected_query(
    test_targets: set[str],
    source_labels: set[str],
    package_label_prefixes: set[str],
) -> str:
    # Filter the analyzed closure instead of resolving changed paths as target
    # patterns: deleted, untracked, or configuration-excluded files may have no
    # node. Query strings preserve backslashes rather than decoding JSON escapes.
    source_pattern = "|".join(
        [
            *(re.escape(label) for label in sorted(source_labels)),
            *(re.escape(prefix) + ".*" for prefix in sorted(package_label_prefixes)),
        ]
    )
    source_pattern = source_pattern.replace('"', r"\x22")
    return (
        "let tests = set(\n  " + "\n  ".join(sorted(test_targets)) + ") in\n"
        "let dependencies = deps($tests) in\n"
        "$tests intersect rdeps($dependencies, "
        f'filter("^({source_pattern})$", $dependencies))\n'
    )


class DependencyQueryError(RuntimeError):
    pass


@dataclass(frozen=True)
class TestSuite:
    # Project or policy suite name used in command diagnostics.
    name: str
    # Bazel target patterns, including any negative target patterns.
    targets: tuple[str, ...]
    # Owner-selected configuration, shared by cquery and test execution.
    configuration_args: tuple[str, ...] = ("--config=presubmit",)
    # Explicit policy suites whose manual members participate in presubmit.
    manual_suites: tuple[str, ...] = ()
    # Additional resource tags excluded by this owner's test policy.
    excluded_tags: tuple[str, ...] = ()

    def discovery_query(self) -> str:
        included = " ".join(
            target for target in self.targets if not target.startswith("-")
        )
        excluded = " ".join(
            target[1:] for target in self.targets if target.startswith("-")
        )
        query = f"tests(set({included}))"
        query = f'({query} except attr(tags, "(^|, |\\[)manual(,|\\]|$)", {query}))'
        if self.manual_suites:
            query = f"({query} union tests(set({' '.join(self.manual_suites)})))"
        if excluded:
            query = f"({query} except tests(set({excluded})))"
        if self.excluded_tags:
            tags = "|".join(re.escape(tag) for tag in self.excluded_tags)
            query = f'let tests = {query} in $tests except attr(tags, "(^|, |\\[)({tags})(,|\\]|$)", $tests)'
        return query

    def test_command(self, target_pattern_file: Path | None = None) -> list[str]:
        command = [
            "bazel",
            "test",
            *self.configuration_args,
            *project_presubmit.bazel_config_args(),
        ]
        excluded_tags = self.excluded_tags + (() if self.manual_suites else ("manual",))
        if excluded_tags:
            command.append(
                "--test_tag_filters=" + ",".join(f"-{tag}" for tag in excluded_tags)
            )
        if target_pattern_file is None:
            command.extend(["--", *self.targets, *self.manual_suites])
        else:
            # Expanded tests retain wildcard compatibility semantics.
            command.extend(
                [
                    "--skip_incompatible_explicit_targets",
                    f"--target_pattern_file={target_pattern_file}",
                ]
            )
        return command

    def discover_targets(
        self, repo_root: Path, temp_path: Path, *, verbose: bool = False
    ) -> set[str]:
        # Discover concrete test rules without walking unconfigured deps:
        # inactive select branches may require unavailable SDKs, and other
        # support rules may intentionally be invalid outside analysis tests.
        test_targets_path = temp_path / "test_targets.txt"
        if not project_presubmit.run_command(
            self.name,
            [
                "bazel",
                "query",
                "--output=label",
                f"--output_file={test_targets_path}",
                self.discovery_query(),
            ],
            "Discover Bazel test targets",
            cwd=repo_root,
            verbose=verbose,
        ):
            raise DependencyQueryError("failed to discover Bazel test targets")
        test_targets = set(test_targets_path.read_text(encoding="utf-8").splitlines())
        if not test_targets:
            raise DependencyQueryError(f"Bazel reported no {self.name} test targets")
        return test_targets

    def query_affected_targets(
        self,
        repo_root: Path,
        source_labels: set[str],
        package_label_prefixes: set[str],
        *,
        verbose: bool = False,
    ) -> list[str]:
        with tempfile.TemporaryDirectory(prefix="iree-bazel-affected-") as temp_dir:
            temp_path = Path(temp_dir)
            test_targets = self.discover_targets(repo_root, temp_path, verbose=verbose)
            query_path = temp_path / "test_dependencies.query"
            query_path.write_text(
                affected_query(test_targets, source_labels, package_label_prefixes),
                encoding="utf-8",
            )
            affected_targets_path = temp_path / "affected_targets.txt"
            if not project_presubmit.run_command(
                self.name,
                [
                    "bazel",
                    "cquery",
                    *self.configuration_args,
                    *project_presubmit.bazel_config_args(),
                    # Flattened aspect edges can select unrelated siblings.
                    "--experimental_explicit_aspects",
                    "--output=label",
                    f"--query_file={query_path}",
                    f"--output_file={affected_targets_path}",
                ],
                "Resolve Bazel test dependencies",
                cwd=repo_root,
                verbose=verbose,
            ):
                raise DependencyQueryError(
                    "failed to resolve configured Bazel test dependencies"
                )
            try:
                targets = sorted(
                    {
                        node_label(line)
                        for line in affected_targets_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line
                    }
                )
            except (OSError, ValueError) as exc:
                raise DependencyQueryError(
                    f"failed to read configured Bazel test dependencies: {exc}"
                ) from exc
            print(
                f"{self.name} presubmit: Bazel dependency graph selected {len(targets)} of {len(test_targets)} tests"
            )
            return targets

    def selected_targets(
        self, repo_root: Path, paths: list[str], *, verbose: bool = False
    ) -> list[str] | None:
        if any(requires_full_suite(path) for path in paths):
            return None
        package_build_paths = [path for path in paths if is_package_build_file(path)]
        package_prefixes = [
            package_label_prefix(repo_root, path) for path in package_build_paths
        ]
        if any(prefix is None for prefix in package_prefixes):
            # A removed package boundary can rehome every file below it.
            return None
        package_label_prefixes = {
            prefix for prefix in package_prefixes if prefix is not None
        }
        source_labels = {
            label
            for path in paths
            if not is_package_build_file(path)
            if (label := source_label(repo_root, path)) is not None
        }
        if not source_labels and not package_label_prefixes:
            return []
        return self.query_affected_targets(
            repo_root,
            source_labels,
            package_label_prefixes,
            verbose=verbose,
        )

    def run(
        self, repo_root: Path, *, paths: list[str] | None = None, verbose: bool = False
    ) -> bool:
        try:
            targets = (
                self.selected_targets(repo_root, paths, verbose=verbose)
                if paths is not None
                else None
            )
            if targets is None and not self.manual_suites:
                return project_presubmit.run_command(
                    self.name,
                    self.test_command(),
                    "Bazel tests",
                    cwd=repo_root,
                    verbose=verbose,
                )
            if targets == []:
                print(
                    f"{self.name} presubmit: no Bazel tests depend on the selected files"
                )
                return True
            with tempfile.TemporaryDirectory(
                prefix="iree-bazel-test-targets-"
            ) as temp_dir:
                # Full policy runs use the same explicit manual membership as
                # affected runs. Wildcard suite expansion could admit other
                # manual members before we can apply this selection policy.
                success_exit_codes = (0, 4)
                if targets is None:
                    targets = sorted(
                        self.discover_targets(
                            repo_root, Path(temp_dir), verbose=verbose
                        )
                    )
                    success_exit_codes = (0,)
                target_file = Path(temp_dir) / "targets.txt"
                target_file.write_text(
                    "".join(f"{target}\n" for target in targets), encoding="utf-8"
                )
                # Exit 4 means every affected test was filtered by execution policy.
                return project_presubmit.run_command(
                    self.name,
                    self.test_command(target_file),
                    "Bazel tests",
                    cwd=repo_root,
                    verbose=verbose,
                    success_exit_codes=success_exit_codes,
                )
        except DependencyQueryError as exc:
            print(f"{self.name} presubmit: {exc}", file=sys.stderr)
            return False
