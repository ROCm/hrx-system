# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Archives failed Linux Bazel test executables and their materialized runfiles."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path

from build_tools.ci import bazel_runfiles


def failed_test_programs(events: Path, workspace: Path) -> set[Path]:
    """Joins actual failed attempts to the default outputs of that configuration."""
    failed = set()
    target_outputs = {}
    named_sets = {}
    with events.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            identity = event["id"]
            if "namedSet" in identity:
                named_sets[identity["namedSet"]["id"]] = event["namedSetOfFiles"]
            elif "targetCompleted" in identity and "completed" in event:
                target = identity["targetCompleted"]
                if "aspect" in target:
                    continue
                target_outputs[target["label"], target["configuration"]["id"]] = [
                    item["id"]
                    for group in event["completed"].get("outputGroup", ())
                    if group["name"] == "default"
                    for item in group.get("fileSets", ())
                ]
            elif "testResult" in event and event["testResult"]["status"] != "PASSED":
                test = identity["testResult"]
                failed.add((test["label"], test["configuration"]["id"]))

    programs = set()
    for key in sorted(failed):
        pending = list(target_outputs[key])
        visited = set()
        matches = set()
        while pending:
            set_id = pending.pop()
            if set_id in visited:
                continue
            visited.add(set_id)
            file_set = named_sets[set_id]
            pending.extend(child["id"] for child in file_set.get("fileSets", ()))
            for item in file_set.get("files", ()):
                path = Path(*item.get("pathPrefix", ()), item["name"])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Invalid Bazel output path for {key}: {path}")
                executable = workspace / path
                if (
                    executable.is_file()
                    and Path(f"{executable}.runfiles_manifest").is_file()
                ):
                    matches.add(path)
        if not matches:
            raise ValueError(f"No test executable and runfiles manifest for {key}")
        programs.update(matches)
    return programs


def archive_test(archive: tarfile.TarFile, workspace: Path, program: Path) -> None:
    archive.add(workspace / program, arcname=str(program))
    runfiles = Path(f"{program}.runfiles")
    manifest = workspace / f"{program}.runfiles_manifest"
    directory = tarfile.TarInfo(str(runfiles))
    directory.type = tarfile.DIRTYPE
    directory.mode = 0o755
    archive.addfile(directory)
    # Remote execution can leave the manifest and its inputs on the client
    # without ever constructing a runfiles symlink tree there.
    with manifest.open(encoding="utf-8", newline="\n") as stream:
        for number, line in enumerate(stream, 1):
            name, target = bazel_runfiles.parse_manifest_line(line)
            logical = Path(name)
            if (
                target is None
                or not logical.parts
                or logical.is_absolute()
                or ".." in logical.parts
            ):
                raise ValueError(f"{manifest}:{number}: invalid runfile {name!r}")
            member = str(runfiles / logical)
            if Path(target).is_absolute():
                archive.add(target, arcname=member)
            else:
                entry = tarfile.TarInfo(member)
                if target:
                    # Relative targets are symlinks within the runfiles tree.
                    destination = Path(os.path.normpath(logical.parent / target))
                    if destination.parts[:1] == ("..",):
                        raise ValueError(
                            f"{manifest}:{number}: link escapes the runfiles tree"
                        )
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = target
                archive.addfile(entry)


def archive_failed_tests(events: Path, workspace: Path, output: Path) -> None:
    programs = failed_test_programs(events, workspace)
    if not programs:
        print("The build event stream contains no failed test attempts.")
        return
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        pending = Path(temporary) / output.name
        with tarfile.open(
            pending, "w:gz", compresslevel=1, dereference=True
        ) as archive:
            for program in sorted(programs):
                print(f"Archiving {program}", flush=True)
                archive_test(archive, workspace, program)
        pending.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="Bazel JSON build event stream")
    parser.add_argument("output", type=Path, help="Destination .tar.gz archive")
    args = parser.parse_args()
    archive_failed_tests(args.events, Path.cwd(), args.output)
