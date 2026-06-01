#!/usr/bin/env python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""This script assists with converting from Bazel BUILD files to CMakeLists.txt.

Bazel BUILD files should, where possible, be written to use simple features
that can be directly evaluated and avoid more advanced features like
variables, list comprehensions, etc.

Generated CMake files will be similar in structure to their source BUILD
files by using the functions in build_tools/cmake/ and product-local
build_tools/cmake/ directories that imitate corresponding Bazel rules
(e.g. cc_library -> iree_cc_library.cmake).

Common usage:
  Run across all default paths in the project (in .bazel_to_cmake.cfg.py):
      $ python build_tools/bazel_to_cmake/bazel_to_cmake.py

  Run on individual files or directories (most common):
      $ python build_tools/bazel_to_cmake/bazel_to_cmake.py runtime/src/iree/base/
      $ python build_tools/bazel_to_cmake/bazel_to_cmake.py runtime/src/iree/base/BUILD.bazel
      $ python build_tools/bazel_to_cmake/bazel_to_cmake.py runtime/src/iree/base/ runtime/src/iree/vm/

  Run on all files under a root directory (recursively - use sparingly):
      $ python build_tools/bazel_to_cmake/bazel_to_cmake.py --recursive_dir runtime/src/iree/

  Inspect whether generated files are stale without writing them:
      $ python build_tools/bazel_to_cmake/bazel_to_cmake.py --dry-run
      $ python build_tools/bazel_to_cmake/bazel_to_cmake.py --check

  Dump generated CMake contents to stdout without writing files:
      $ python build_tools/bazel_to_cmake/bazel_to_cmake.py --preview runtime/src/iree/base/

Configuration
-------------
When invoked, bazel_to_cmake will traverse up from the current directory until
it finds a ".bazel_to_cmake.cfg.py" file. This file both serves as a marker
for the repository root and provides repository specific configuration.

The file is evaluated as a module and can have the following customizations:

* DEFAULT_ROOT_DIRS: A list of root directory names that should be recursively
  processed (relative to the repository root) when invoked without arguments.
* REPO_MAP: Mapping of canonical Bazel repo name (i.e. "@iree_core") to what it
  is known as locally (most commonly the empty string). This is used in global
  target rules to make sure that they work either in the defining or referencing
  repository.
* CustomBuildFileFunctions: A class that extends
  `bazel_to_cmake_converter.BuildFileFunctions` and injects globals for
  processing the BUILD file. All symbols that do not start with "_" are
  available.
* CustomTargetConverter: A class that extends
  `bazel_to_cmake_targets.TargetConverter` and customizes target mapping.
  Typically, this is used for purely local targets in leaf projects (as global
  targets will be encoded in the main bazel_to_cmake_targets.py file).
"""
# pylint: disable=missing-docstring

import argparse
import importlib
import importlib.util
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import bazel_to_cmake_converter

repo_root = None
repo_cfg = None

EDIT_BLOCKING_PATTERN = re.compile(
    r"bazel[\s_]*to[\s_]*cmake[\s_]*:?[\s_]*do[\s_]*not[\s_]*edit", flags=re.IGNORECASE
)

PRESERVE_ABOVE_TAG = "### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_ABOVE_THIS_LINE ###"
PRESERVE_BELOW_TAG = "### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###"
REPO_CFG_FILE = ".bazel_to_cmake.cfg.py"
REPO_CFG_MODULE_NAME = "bazel_to_cmake_repo_config"


class Status(Enum):
    UPDATED = 1
    NOOP = 2
    FAILED = 3
    SKIPPED = 4
    NO_BUILD_FILE = 5


@dataclass
class ConversionSummary:
    updated_count: int = 0
    skip_count: int = 0
    noop_count: int = 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Bazel to CMake conversion helper.")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--preview",
        help="Print generated CMake contents instead of writing files.",
        action="store_true",
        default=False,
    )
    output_group.add_argument(
        "--dry-run",
        help="Report whether files are up to date without writing generated contents.",
        action="store_true",
        default=False,
    )
    output_group.add_argument(
        "--check",
        help="Verify generated CMake files are up to date without writing files. "
        "Exits with status 1 if any file would be updated.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        default=0,
        help="Specify verbosity level where higher verbosity emits more logging."
        " 0 (default): Only output errors and summary statistics."
        " 1: Also output the name of each directory as it's being processed and"
        " a status line for skipped or changed generated files."
        " 2: Also output when conversion required no update.",
    )

    # Specify only one of these (defaults to --recursive_dir=<main source dirs>).
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "paths",
        nargs="*",
        help="BUILD files or directories to convert (directories are treated as "
        "containing BUILD.bazel)",
        default=[],
    )
    group.add_argument(
        "--dir", help="Converts the BUILD file in the given directory", default=None
    )
    default_root_dirs = (
        repo_cfg.DEFAULT_ROOT_DIRS if hasattr(repo_cfg, "DEFAULT_ROOT_DIRS") else []
    )
    group.add_argument(
        "--recursive_dir",
        nargs="+",
        help="Recursively converts ALL BUILD files under the given directories, "
        "including third_party/. You almost never want this - prefer passing "
        "specific paths as positional arguments instead.",
        default=default_root_dirs,
    )

    args = parser.parse_args()

    # 'paths' and '--dir' take precedence over '--recursive_dir'.
    # They are mutually exclusive, but the default value is still set.
    if args.paths or args.dir:
        args.recursive_dir = None

    return args


def setup_environment():
    """Sets up some environment globals."""
    global repo_root
    global repo_cfg

    # Scan up the directory tree for a repo config file.
    check_dir = os.getcwd()
    while not os.path.exists(os.path.join(check_dir, REPO_CFG_FILE)):
        new_check_dir = os.path.dirname(check_dir)
        if not new_check_dir or new_check_dir == check_dir:
            print(
                f"ERROR: Could not find {REPO_CFG_FILE} in a parent directory "
                f"of {os.getcwd()}"
            )
            sys.exit(1)
        check_dir = new_check_dir
    repo_root = check_dir

    # Dynamically load the config file as a module.
    orig_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # Don't generate __pycache__ dir
    repo_cfg_path = os.path.join(repo_root, REPO_CFG_FILE)
    spec = importlib.util.spec_from_file_location(REPO_CFG_MODULE_NAME, repo_cfg_path)
    if spec and spec.loader:
        repo_cfg = importlib.util.module_from_spec(spec)
        sys.modules[REPO_CFG_MODULE_NAME] = repo_cfg
        spec.loader.exec_module(repo_cfg)
        sys.dont_write_bytecode = orig_dont_write_bytecode
    else:
        print(f"INTERNAL ERROR: Could not evaluate {repo_cfg_path} as module")
        sys.exit(1)


def repo_relpath(path):
    return os.path.relpath(path, repo_root).replace("\\", "/")


def log(string, *args, indent=0, **kwargs):
    print(
        textwrap.indent(string, prefix=(indent * " ")), *args, **kwargs, file=sys.stderr
    )


def convert_directories(directories, write_files, print_generated_content, verbosity):
    summary = ConversionSummary()
    failure_dirs = []
    for directory in directories:
        status = convert_directory(
            directory,
            write_files=write_files,
            print_generated_content=print_generated_content,
            verbosity=verbosity,
        )
        if status == Status.FAILED:
            failure_dirs.append(repo_relpath(directory))
        elif status == Status.SKIPPED:
            summary.skip_count += 1
        elif status == Status.UPDATED:
            summary.updated_count += 1
        elif status == Status.NOOP:
            summary.noop_count += 1

    update_phrase = "were updated" if write_files else "would be updated"
    log(
        f"{summary.updated_count} CMakeLists.txt files {update_phrase}, "
        f"{summary.skip_count} were skipped, and {summary.noop_count} required "
        f"no change."
    )
    if failure_dirs:
        log(
            f"ERROR: Encountered unexpected errors converting {len(failure_dirs)}"
            " directories:"
        )
        log("\n".join(failure_dirs), indent=2)
        sys.exit(1)
    return summary


def convert_directory(directory_path, write_files, print_generated_content, verbosity):
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Cannot find directory '{directory_path}'")

    rel_dir_path = repo_relpath(directory_path)
    if verbosity >= 1:
        log(f"Processing {rel_dir_path}")

    # Scan for a BUILD file.
    build_file_found = False
    build_file_basenames = ["BUILD", "BUILD.bazel"]
    for build_file_basename in build_file_basenames:
        build_file_path = os.path.join(directory_path, build_file_basename)

        rel_build_file_path = repo_relpath(build_file_path)
        if os.path.isfile(build_file_path):
            build_file_found = True
            break
    cmakelists_file_path = os.path.join(directory_path, "CMakeLists.txt")
    rel_cmakelists_file_path = repo_relpath(cmakelists_file_path)

    if not build_file_found:
        return Status.NO_BUILD_FILE

    autogeneration_tag = f"Autogenerated by {repo_relpath(os.path.abspath(__file__))}"

    header = "\n".join(
        ["#" * 80]
        + [
            l.ljust(79) + "#"
            for l in [
                f"# {autogeneration_tag} from",
                f"# {rel_build_file_path}",
                "#",
                "# Add CMake-only content below the preserve marker at the end of this file.",
                "#",
                f"# To disable autogeneration for this file entirely, delete this header.",
            ]
        ]
        + ["#" * 80]
    )

    old_lines = []
    possible_preserved_header_lines = []
    preserved_footer_lines = ["\n" + PRESERVE_BELOW_TAG + "\n"]

    # Read CMakeLists.txt and check if it has the auto-generated header.
    found_preserve_below_tag = False
    found_preserve_above_tag = False
    if os.path.isfile(cmakelists_file_path):
        found_autogeneration_tag = False
        with open(cmakelists_file_path) as f:
            old_lines = f.readlines()

        for line in old_lines:
            if not found_preserve_above_tag:
                possible_preserved_header_lines.append(line)
            if not found_autogeneration_tag and autogeneration_tag in line:
                found_autogeneration_tag = True
            if not found_preserve_below_tag and PRESERVE_BELOW_TAG in line:
                found_preserve_below_tag = True
            elif not found_preserve_above_tag and PRESERVE_ABOVE_TAG in line:
                found_preserve_above_tag = True
            elif found_preserve_below_tag:
                preserved_footer_lines.append(line)
        if not found_autogeneration_tag:
            if verbosity >= 1:
                log(f"Skipped. Did not find autogeneration line.", indent=2)
            return Status.SKIPPED
    preserved_header = (
        "".join(possible_preserved_header_lines) if found_preserve_above_tag else ""
    )
    preserved_footer = "".join(preserved_footer_lines)

    # Read the Bazel BUILD file and interpret it.
    with open(build_file_path, "rt") as build_file:
        build_file_contents = build_file.read()
    if "bazel-to-cmake: skip" in build_file_contents:
        if verbosity >= 1:
            log(f"Skipped. BUILD file has bazel-to-cmake: skip.", indent=2)
        return Status.SKIPPED
    build_file_code = compile(build_file_contents, build_file_path, "exec")
    try:
        converted_build_file = bazel_to_cmake_converter.convert_build_file(
            build_file_code,
            repo_cfg=repo_cfg,
            build_dir=directory_path,
            repo_root=repo_root,
        )
    except (NameError, NotImplementedError) as e:
        log(
            f"ERROR generating {rel_dir_path}.\n"
            f"Missing a rule handler in bazel_to_cmake_converter.py?\n"
            f"Reason: `{type(e).__name__}: {e}`",
            indent=2,
        )
        return Status.FAILED
    except KeyError as e:
        log(
            f"ERROR generating {rel_dir_path}.\n"
            f"Missing a conversion in bazel_to_cmake_targets.py?\n"
            f"Reason: `{type(e).__name__}: {e}`",
            indent=2,
        )
        return Status.FAILED
    converted_content = (
        preserved_header + header + converted_build_file + preserved_footer
    )
    if print_generated_content:
        print(converted_content, end="")

    if converted_content == "".join(old_lines):
        if verbosity >= 2:
            log(f"{rel_cmakelists_file_path} required no update", indent=2)
        return Status.NOOP

    if write_files:
        with open(cmakelists_file_path, "wt") as cmakelists_file:
            cmakelists_file.write(converted_content)

    if verbosity >= 1:
        status_label = "Updated" if write_files else "Would update"
        log(
            f"{status_label} {rel_cmakelists_file_path} from {rel_build_file_path}",
            indent=2,
        )
    return Status.UPDATED


def path_to_directory(path):
    """Converts a path (file or directory) to the directory containing the BUILD file."""
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        return str(Path(path).parent)
    raise FileNotFoundError(f"Cannot find BUILD file or directory '{path}'")


def main(args):
    """Runs Bazel to CMake conversion."""
    global repo_root

    if args.verbosity >= 1:
        log(f"Using repo root {repo_root}")

    write_files = not (args.preview or args.dry_run or args.check)
    print_generated_content = args.preview
    summaries = []

    if args.paths:
        try:
            directories = [path_to_directory(path) for path in args.paths]
        except FileNotFoundError as e:
            log(f"ERROR: {e}")
            sys.exit(1)
        summaries.append(
            convert_directories(
                directories,
                write_files=write_files,
                print_generated_content=print_generated_content,
                verbosity=args.verbosity,
            )
        )
    elif args.recursive_dir:
        for root_dir in args.recursive_dir:
            root_directory_path = os.path.join(repo_root, root_dir)
            if not os.path.isdir(root_directory_path):
                log(f"ERROR: Cannot find recursive directory '{root_dir}'")
                sys.exit(1)
            log(f"Converting directory tree rooted at: {root_directory_path}")
            summaries.append(
                convert_directories(
                    (root for root, _, _ in os.walk(root_directory_path)),
                    write_files=write_files,
                    print_generated_content=print_generated_content,
                    verbosity=args.verbosity,
                )
            )
    elif args.dir:
        summaries.append(
            convert_directories(
                [os.path.join(repo_root, args.dir)],
                write_files=write_files,
                print_generated_content=print_generated_content,
                verbosity=args.verbosity,
            )
        )
    else:
        log(
            f"ERROR: No paths provided and no DEFAULT_ROOT_DIRS in "
            f".bazel_to_cmake.cfg.py. Pass BUILD files or directories as "
            f"positional arguments, use --dir for a single directory, or "
            f"--recursive_dir to process an entire tree."
        )
        sys.exit(1)

    if args.check and any(summary.updated_count for summary in summaries):
        sys.exit(1)


if __name__ == "__main__":
    setup_environment()
    main(parse_arguments())
