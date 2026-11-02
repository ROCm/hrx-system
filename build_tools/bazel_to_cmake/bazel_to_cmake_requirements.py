# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loads project requirement policy from declarative `.bzl` files."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import bazel_to_cmake_loads


@dataclass(frozen=True)
class Requirement:
    id: str
    label: str
    phase: str
    enabled_by: str | None = None
    cmake_label: str | None = None
    skip_contract: str | None = None


@dataclass(frozen=True)
class PackagePolicy:
    packages: list[str]
    build_requirements: list[Requirement]
    excluded_packages: list[str]
    run_requirements: list[Requirement]
    resource_group: str | None = None


@dataclass(frozen=True)
class CollectedPackagePolicy:
    build_requirements: list[Requirement]
    run_requirements: list[Requirement]
    resource_group: str | None

    def conditions(self) -> list[str]:
        return [requirement.enabled_by for requirement in self.build_requirements]

    def tags(self, include_run_requirements: bool) -> list[str]:
        tags = [
            f"iree-build-requirement={requirement.id}"
            for requirement in self.build_requirements
        ]
        if include_run_requirements:
            for requirement in self.run_requirements:
                tags.append(f"iree-run-requirement={requirement.id}")
                if requirement.cmake_label:
                    tags.append(requirement.cmake_label)
        return tags


@dataclass(frozen=True)
class ProjectRequirementPolicy:
    package_policies: list[PackagePolicy]

    def collect(self, package_name: str) -> CollectedPackagePolicy:
        build_requirements: list[Requirement] = []
        run_requirements: list[Requirement] = []
        resource_group = None
        resource_group_specificity = -1
        for policy in self.package_policies:
            matching_patterns = [
                pattern
                for pattern in policy.packages
                if _matches(pattern, package_name)
            ]
            if not matching_patterns:
                continue
            if any(
                _matches(pattern, package_name) for pattern in policy.excluded_packages
            ):
                continue
            _append_unique_requirements(build_requirements, policy.build_requirements)
            _append_unique_requirements(run_requirements, policy.run_requirements)
            if policy.resource_group:
                specificity = max(
                    _specificity(pattern) for pattern in matching_patterns
                )
                if specificity >= resource_group_specificity:
                    resource_group = policy.resource_group
                    resource_group_specificity = specificity
        return CollectedPackagePolicy(
            build_requirements=build_requirements,
            run_requirements=run_requirements,
            resource_group=resource_group,
        )


def build_requirement(id: str, label: str, enabled_by: str):
    return Requirement(
        id=id,
        label=label,
        phase="build",
        enabled_by=enabled_by,
    )


def run_requirement(id: str, label: str, cmake_label: str, skip_contract: str):
    return Requirement(
        id=id,
        label=label,
        phase="run",
        cmake_label=cmake_label,
        skip_contract=skip_contract,
    )


def package_policy(
    packages,
    build_requirements=None,
    excluded_packages=None,
    forbidden_deps=None,
    run_requirements=None,
    resource_group=None,
):
    del forbidden_deps
    return PackagePolicy(
        packages=list(packages),
        build_requirements=list(build_requirements or []),
        excluded_packages=list(excluded_packages or []),
        run_requirements=list(run_requirements or []),
        resource_group=resource_group,
    )


def append_conditions(target_compatible_with, conditions):
    if not conditions:
        return target_compatible_with
    if target_compatible_with is None:
        target_compatible_with = []
    return target_compatible_with + conditions


@lru_cache(maxsize=None)
def load_project_policy(repo_root: str, project: str) -> ProjectRequirementPolicy:
    loader = bazel_to_cmake_loads.ModuleLoader(
        repo_root,
        bindings={
            "build_requirement": build_requirement,
            "run_requirement": run_requirement,
            "package_policy": package_policy,
        },
    )
    env = loader.load_file(
        Path(repo_root) / project / "requirements/package_policy.bzl"
    )
    return ProjectRequirementPolicy(
        package_policies=list(env["PACKAGE_POLICIES"]),
    )


def _matches(pattern: str, package_name: str) -> bool:
    if pattern.endswith("/..."):
        prefix = pattern[:-4]
        return package_name == prefix or package_name.startswith(prefix + "/")
    return package_name == pattern


def _specificity(pattern: str) -> int:
    if pattern.endswith("/..."):
        return len(pattern) - 4
    return len(pattern)


def _append_unique_requirements(target, requirements):
    ids = {requirement.id for requirement in target}
    for requirement in requirements:
        if requirement.id not in ids:
            target.append(requirement)
            ids.add(requirement.id)
