# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Project configuration helpers for Bazel-to-CMake conversion."""

import hashlib
import importlib.util
import os
import sys

import bazel_to_cmake_targets


class ProjectConfig:
    """Configuration owned by a top-level project subtree."""

    def __init__(
        self,
        *,
        name,
        package_prefixes,
        build_file_functions=None,
        target_mappings=None,
        convert_unmatched_target=None,
    ):
        self.name = name
        self.package_prefixes = tuple(
            _normalize_package_prefix(prefix) for prefix in package_prefixes
        )
        self.build_file_functions = build_file_functions
        self.target_mappings = dict(target_mappings or {})
        self.convert_unmatched_target = convert_unmatched_target

    def match_length(self, repo_relative_path):
        repo_relative_path = _normalize_repo_path(repo_relative_path)
        match_length = -1
        for prefix in self.package_prefixes:
            package_root = prefix[:-1]
            if repo_relative_path == package_root or repo_relative_path.startswith(
                prefix
            ):
                match_length = max(match_length, len(prefix))
        return match_length


class ProjectTargetConverter(bazel_to_cmake_targets.TargetConverter):
    """Routes unmatched local labels through the project that owns the label."""

    def __init__(
        self,
        *,
        repo_map,
        projects,
        target_mappings=None,
        convert_unmatched_target=None,
    ):
        self._projects = tuple(projects)
        self._root_target_mappings = dict(target_mappings or {})
        self._root_convert_unmatched_target = convert_unmatched_target
        super().__init__(repo_map=repo_map)

    def _initialize(self):
        self._update_target_mappings(self._root_target_mappings)
        for project in self._projects:
            self._update_target_mappings(project.target_mappings)

    def _convert_unmatched_target(self, target):
        project = find_project_for_target(self._projects, target, self._repo_map)
        if project and project.convert_unmatched_target:
            return project.convert_unmatched_target(self, target)
        if self._root_convert_unmatched_target:
            return self._root_convert_unmatched_target(self, target)
        return super()._convert_unmatched_target(target)


def include_project(root_config_file, relative_path):
    """Loads a project config from a path relative to the root config file."""
    root_dir = os.path.dirname(os.path.abspath(root_config_file))
    config_path = os.path.join(root_dir, relative_path)
    module = _load_config_module(config_path)
    project_config = getattr(module, "PROJECT_CONFIG", None)
    if project_config is None:
        raise ValueError(f"{config_path} does not define PROJECT_CONFIG")
    return project_config


def include_projects(root_config_file, relative_paths):
    return [
        include_project(root_config_file, relative_path)
        for relative_path in relative_paths
    ]


def get_projects(repo_cfg):
    return tuple(getattr(repo_cfg, "PROJECTS", ()))


def select_build_file_functions(repo_cfg, build_dir, repo_root, default_class):
    projects = get_projects(repo_cfg)
    project = find_project_for_path(
        projects,
        os.path.relpath(build_dir, repo_root),
    )
    if project and project.build_file_functions:
        return project.build_file_functions
    return getattr(repo_cfg, "CustomBuildFileFunctions", default_class)


def create_target_converter(repo_cfg, repo_map, default_class):
    projects = get_projects(repo_cfg)
    if projects:
        return ProjectTargetConverter(
            repo_map=repo_map,
            projects=projects,
            target_mappings=getattr(repo_cfg, "TARGET_MAPPINGS", {}),
            convert_unmatched_target=getattr(
                repo_cfg, "convert_unmatched_target", None
            ),
        )
    return getattr(repo_cfg, "CustomTargetConverter", default_class)(repo_map=repo_map)


def find_project_for_path(projects, repo_relative_path):
    best_project = None
    best_match_length = -1
    for project in projects:
        match_length = project.match_length(repo_relative_path)
        if match_length > best_match_length:
            best_project = project
            best_match_length = match_length
    return best_project


def find_project_for_target(projects, target, repo_map):
    repo_relative_path = _repo_relative_target_path(target, repo_map)
    if not repo_relative_path:
        return None
    return find_project_for_path(projects, repo_relative_path)


def _load_config_module(config_path):
    module_digest = hashlib.sha1(config_path.encode("utf-8")).hexdigest()
    module_name = f"bazel_to_cmake_config_{module_digest}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if not spec or not spec.loader:
        raise ValueError(f"Could not evaluate {config_path} as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_repo_path(path):
    return path.replace("\\", "/").strip("/")


def _normalize_package_prefix(prefix):
    prefix = _normalize_repo_path(prefix)
    if not prefix:
        return ""
    return prefix + "/"


def _repo_relative_target_path(target, repo_map):
    local_label_prefixes = ["//"]
    for repo_name in ("@hrx",):
        repo_alias = repo_map.get(repo_name)
        if repo_alias:
            local_label_prefixes.append(repo_alias + "//")
        local_label_prefixes.append(repo_name + "//")
    for label_prefix in local_label_prefixes:
        if target.startswith(label_prefix):
            label_body = target[len(label_prefix) :]
            return _normalize_repo_path(label_body.split(":", 1)[0])
    return None
