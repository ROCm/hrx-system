#!/usr/bin/env python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates a build-local CMake package tree from Bazel declarations.

CMake invokes this converter during configuration and tracks its inputs for
native regeneration. Authored CMakeLists.txt files retain their directory
scopes; packages without one are generated under the owning build tree.

For manual inspection:
  python build_tools/bazel_to_cmake/bazel_to_cmake.py --preview runtime/src/iree/base

For generation (normally owned by CMake):
  python build_tools/bazel_to_cmake/bazel_to_cmake.py --output-dir build/cmake/.iree/cmake

Configuration
-------------
When invoked, bazel_to_cmake will traverse up from the current directory until
it finds a ".bazel_to_cmake.cfg.py" file with BAZEL_TO_CMAKE_REPO_ROOT = True.
This file serves as the repository root marker and provides repository specific
configuration. The root config can explicitly include project-local configs
that own rule handlers and target mappings for subtrees such as runtime/ or
libhrx/.

The file is evaluated as a module and can have the following customizations:

* DEFAULT_ROOT_DIRS: A list of root directory names that should be recursively
  processed (relative to the repository root) when generating a build tree.
* REPO_MAP: Mapping of canonical Bazel repo name (i.e. "@hrx") to what it
  is known as locally (most commonly the empty string). This is used in global
  target rules to make sure that they work either in the defining or referencing
  repository.
* PROJECTS: An explicit list of ProjectConfig values included by the root
  config. The converter selects a project's BuildFileFunctions by the BUILD
  file path and routes local target-label fallback conversion by the referenced
  label's owning project. Each project also publishes condition_mappings for
  its configuration labels; these are available to every BUILD file, including
  consumers in other projects.
* TARGET_MAPPINGS: Repository-owned target mappings applied before
  project-owned mappings.
* CONDITION_MAPPINGS: Repository-owned select-condition mappings applied before
  project-owned mappings.
* CustomBuildFileFunctions and CustomTargetConverter: Legacy single-config
  hooks used when PROJECTS is not provided.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import bazel_to_cmake_converter
import cc

repo_root = None
repo_cfg = None
REPO_CFG_FILE = ".bazel_to_cmake.cfg.py"
REPO_CFG_MODULE_NAME = "bazel_to_cmake_repo_config"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output-dir",
        type=Path,
        help="Owned build directory for the complete generated CMake package tree.",
    )
    output.add_argument(
        "--preview",
        type=Path,
        help="Print one package's generated CMake; accepts a directory or BUILD file. "
        "Resolved corpus products require --configuration from a native build.",
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        help="Effective native settings written by the CMake bootstrap.",
    )
    return parser.parse_args()


def setup_environment():
    """Sets up some environment globals."""
    global repo_root
    global repo_cfg

    # Scan up the directory tree for repo and project config files.
    candidate_config_dirs = []
    check_dir = os.getcwd()
    while True:
        if os.path.exists(os.path.join(check_dir, REPO_CFG_FILE)):
            candidate_config_dirs.append(check_dir)
        new_check_dir = os.path.dirname(check_dir)
        if not new_check_dir or new_check_dir == check_dir:
            break
        check_dir = new_check_dir

    if not candidate_config_dirs:
        print(
            f"ERROR: Could not find {REPO_CFG_FILE} in a parent directory "
            f"of {os.getcwd()}"
        )
        sys.exit(1)

    # Dynamically load the config file as a module.
    orig_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # Don't generate __pycache__ dir
    try:
        for candidate_dir in reversed(candidate_config_dirs):
            candidate_cfg_path = os.path.join(candidate_dir, REPO_CFG_FILE)
            spec = importlib.util.spec_from_file_location(
                REPO_CFG_MODULE_NAME,
                candidate_cfg_path,
            )
            if not spec or not spec.loader:
                print(
                    f"INTERNAL ERROR: Could not evaluate {candidate_cfg_path} as module"
                )
                sys.exit(1)
            candidate_cfg = importlib.util.module_from_spec(spec)
            sys.modules[REPO_CFG_MODULE_NAME] = candidate_cfg
            spec.loader.exec_module(candidate_cfg)
            if getattr(candidate_cfg, "BAZEL_TO_CMAKE_REPO_ROOT", False):
                repo_root = candidate_dir
                repo_cfg = candidate_cfg
                break
        if repo_cfg is None:
            repo_root = candidate_config_dirs[-1]
            repo_cfg_path = os.path.join(repo_root, REPO_CFG_FILE)
            spec = importlib.util.spec_from_file_location(
                REPO_CFG_MODULE_NAME,
                repo_cfg_path,
            )
            if not spec or not spec.loader:
                print(f"INTERNAL ERROR: Could not evaluate {repo_cfg_path} as module")
                sys.exit(1)
            repo_cfg = importlib.util.module_from_spec(spec)
            sys.modules[REPO_CFG_MODULE_NAME] = repo_cfg
            spec.loader.exec_module(repo_cfg)
    finally:
        sys.dont_write_bytecode = orig_dont_write_bytecode


@dataclass
class Package:
    # Repository-relative directory that owns the declarations and their outputs.
    directory: Path
    # Source declaration used for diagnostics and generated-file attribution.
    build_file: Path
    # Native fragments rendered after repository-wide declaration collection.
    converter: bazel_to_cmake_converter.Converter

    def convert(self, traversal="iree_add_all_subdirs()"):
        package = self.directory.as_posix()
        if not self.converter.requires_native_scope:
            # Defaults belong to this package's products, not its children.
            for kind in ("ARCHIVE", "LIBRARY", "RUNTIME"):
                traversal += (
                    f"\nif(NOT DEFINED CMAKE_{kind}_OUTPUT_DIRECTORY)\n"
                    f'  set(CMAKE_{kind}_OUTPUT_DIRECTORY "${{IREE_PACKAGE_BINARY_DIR}}")\n'
                    "endif()\n"
                )
            return source_package(self.directory, self.converter.convert(traversal))
        return (
            f"# Generated from {package}/{self.build_file.name}. Edit the BUILD declaration.\n"
            "set_property(DIRECTORY PROPERTY IREE_PACKAGE_SOURCE_DIR "
            f'"${{IREE_SOURCE_DIR}}/{package}")\n' + self.converter.convert(traversal)
        )


def package_function(directory):
    return (
        "_iree_source_package_"
        + hashlib.sha256(directory.as_posix().encode()).hexdigest()[:16]
    )


def source_package(directory, body):
    """Scopes BUILD variables without creating another native build directory."""
    package = directory.as_posix()
    function = package_function(directory)
    return (
        f"# Generated from {package} declarations.\n"
        f"function({function} IREE_PACKAGE_BINARY_DIR)\n"
        f'  set(IREE_PACKAGE_SOURCE_DIR "${{IREE_SOURCE_DIR}}/{package}")\n'
        '  file(MAKE_DIRECTORY "${IREE_PACKAGE_BINARY_DIR}")\n'
        + body
        + "\nendfunction()\n"
        + f"set(_IREE_PACKAGE_FUNCTION {function})\n"
    )


def evaluate_directory(
    directory,
    input_files,
    generated_files,
    configuration=None,
    libraries=None,
    declarations=None,
):
    """Collects a generated package; authored packages retain their native scope."""
    directory = Path(directory)
    if (directory / "CMakeLists.txt").is_file():
        return None
    build_file = next(
        (
            directory / name
            for name in ("BUILD.bazel", "BUILD")
            if (directory / name).is_file()
        ),
        None,
    )
    if build_file is None:
        return None
    input_files.add(build_file)
    source = build_file.read_text(encoding="utf-8")
    if "bazel-to-cmake: skip" in source:
        return None
    try:
        converter = bazel_to_cmake_converter.evaluate_build_file(
            compile(source, str(build_file), "exec"),
            repo_cfg=repo_cfg,
            build_dir=str(directory),
            repo_root=repo_root,
            input_files=input_files,
            generated_files=generated_files,
            configuration=configuration,
            libraries=libraries,
            declarations=declarations,
        )
    except (NameError, NotImplementedError, KeyError) as error:
        raise RuntimeError(f"Cannot convert {build_file}: {error}") from error
    return Package(directory.relative_to(repo_root), build_file, converter)


def evaluate_packages(root_dirs, input_files, configuration=None):
    """Establishes generated-file ownership before any consumer is emitted."""
    generated_files = {}
    declarations = {}
    libraries = cc.Graph(configuration) if configuration is not None else None
    packages = {}
    authored_directories = set()
    for root_dir in root_dirs:
        root_path = Path(repo_root) / root_dir
        if not root_path.is_dir():
            raise FileNotFoundError(f"Cannot find package root {root_path}")
        for parent in (root_path, *root_path.parents):
            if parent == Path(repo_root):
                break
            if (parent / "CMakeLists.txt").is_file():
                authored_directories.add(parent.relative_to(repo_root))
        for directory, children, _ in os.walk(root_path):
            if (Path(directory) / "CMakeLists.txt").is_file():
                authored_directories.add(Path(directory).relative_to(repo_root))
            # Native children may be linked into the source tree. os.walk does
            # not enter directory links; authored CMake still owns those scopes.
            for child in children:
                path = Path(directory) / child
                if path.is_symlink() and (path / "CMakeLists.txt").is_file():
                    authored_directories.add(path.relative_to(repo_root))
            package = evaluate_directory(
                directory,
                input_files,
                generated_files,
                configuration,
                libraries,
                declarations,
            )
            if package is not None:
                packages[package.directory] = package
    native_directories = authored_directories | {
        directory
        for directory, package in packages.items()
        if package.converter.requires_native_scope
    }
    if libraries:

        @cache
        def native_scope(directory):
            if directory == Path(".") or directory in native_directories:
                return directory.as_posix()
            return native_scope(directory.parent)

        libraries.native_scopes = {
            directory.as_posix(): native_scope(directory) for directory in packages
        }
    return packages, authored_directories, native_directories, libraries


def write_if_changed(path, content):
    """Preserves timestamps for unchanged native regeneration inputs."""
    data = content.encode("utf-8")
    if not path.exists() or path.read_bytes() != data:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def generate_build_tree(output_dir, root_dirs, configuration=None):
    """Emits packages and their regeneration inputs into one owned build tree."""
    output_dir = output_dir.resolve()
    source_root = Path(repo_root).resolve()
    if source_root == output_dir or source_root.is_relative_to(output_dir):
        raise ValueError("The generated package tree must not contain the source tree")
    if output_dir.exists() and not (output_dir / "inputs.cmake").is_file():
        # A failed first conversion or a deleted manifest leaves owned package
        # files behind. Recovery accepts those, without claiming authored files.
        for path in output_dir.rglob("*"):
            if path.is_file() and (
                path.name not in ("CMakeLists.txt", "package.cmake")
                or not path.read_bytes().startswith(b"# Generated from ")
            ):
                raise ValueError(
                    f"Generated package directory contains unowned file: {path}"
                )
    input_files = set()
    output_files = set()
    packages, authored_directories, native_directories, libraries = evaluate_packages(
        root_dirs, input_files, configuration
    )
    # Native directory traversal must reach every declared package, including
    # leaves beneath directories that have no declarations of their own.
    # Authored CMake retains control of its children and directory scope.
    entries = packages.keys() | authored_directories
    traversal_directories = set()
    for directory in entries:
        for parent in directory.parents:
            if (
                parent == Path(".")
                or parent in entries
                or parent in traversal_directories
                or (source_root / parent / "CMakeLists.txt").is_file()
            ):
                break
            traversal_directories.add(parent)
    # The source walk owns the complete child graph. Native scopes remain real
    # directories, but do not scan the filesystem again to discover their edges.
    children = {}
    for directory in sorted(entries | traversal_directories):
        if directory != directory.parent:
            children.setdefault(directory.parent, []).append(directory)

    def traversal(directory):
        lines = []
        for child in children.get(directory, ()):
            if child in native_directories:
                lines.append(f'iree_add_subdirectory("{child.name}")')
            else:
                binary = (
                    "CMAKE_CURRENT_BINARY_DIR"
                    if directory in native_directories
                    else "IREE_PACKAGE_BINARY_DIR"
                )
                lines.extend(
                    (
                        f'include("${{IREE_GENERATED_CMAKE_DIR}}/{child.as_posix()}/package.cmake")',
                        f'cmake_language(CALL {package_function(child)} "${{{binary}}}/{child.name}")',
                    )
                )
        return "\n".join(lines)

    def write_package(directory, content):
        path = output_dir / directory / "CMakeLists.txt"
        if directory not in native_directories:
            fragment = path.with_name("package.cmake")
            write_if_changed(fragment, content)
            output_files.add(fragment)
            # The entry point also supports a native caller explicitly asking
            # for a real directory, such as add_subdirectory(EXCLUDE_FROM_ALL).
            content = (
                f"# Generated from {directory.as_posix()} declarations.\n"
                'include("${CMAKE_CURRENT_LIST_DIR}/package.cmake")\n'
                'cmake_language(CALL ${_IREE_PACKAGE_FUNCTION} "${CMAKE_CURRENT_BINARY_DIR}")\n'
            )
        write_if_changed(path, content)
        output_files.add(path)

    for package in packages.values():
        write_package(package.directory, package.convert(traversal(package.directory)))

    for directory in sorted(traversal_directories):
        write_package(directory, source_package(directory, traversal(directory)))

    # Include configuration and handler modules actually imported by conversion.
    # The converter can live outside the source root in embedded fixture builds.
    converter_root = Path(__file__).resolve().parent
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(source_root) or path.is_relative_to(converter_root)
            ):
                input_files.add(path)
    for filename in ("CMakeLists.txt", "package.cmake"):
        for path in output_dir.rglob(filename):
            if path not in output_files:
                path.unlink()

    def quote(path):
        return '"' + Path(path).as_posix().replace('"', '\\"') + '"'

    # The native glob notices new/deleted packages and changes between authored
    # and generated packages. Source globs stay with their generated rules.
    # Entry points can be unused when their package is included directly. They
    # still belong to generation and must be repaired after deletion.
    input_files.update(output_files)
    patterns = [
        source_root / root / name
        for root in root_dirs
        for name in ("BUILD", "BUILD.bazel", "CMakeLists.txt")
    ]
    manifest = (
        "file(GLOB_RECURSE _IREE_CMAKE_PACKAGES CONFIGURE_DEPENDS\n  "
        + "\n  ".join(quote(path) for path in patterns)
        + ")\nset_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS\n  "
        + "\n  ".join(quote(path) for path in sorted(input_files))
        + ")\n"
    )
    if libraries:
        response_files = libraries.response_files()
        for path, content in response_files.items():
            write_if_changed(path, content)
        if response_files:
            manifest += (
                "set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS\n  "
                + "\n  ".join(quote(path) for path in response_files)
                + ")\n"
            )
    write_if_changed(output_dir / "inputs.cmake", manifest)


def main(args):
    configuration = (
        json.loads(args.configuration.read_text(encoding="utf-8"))
        if args.configuration
        else None
    )
    if args.preview:
        path = args.preview.resolve()
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            raise FileNotFoundError(f"Cannot find package {path}")
        packages, _, _, _ = evaluate_packages(
            repo_cfg.DEFAULT_ROOT_DIRS, set(), configuration
        )
        package = packages.get(path.relative_to(repo_root))
        if package is None:
            raise ValueError(f"{path} has no generated CMake package")
        print(package.convert(), end="")
    else:
        generate_build_tree(args.output_dir, repo_cfg.DEFAULT_ROOT_DIRS, configuration)


if __name__ == "__main__":
    setup_environment()
    main(parse_arguments())
