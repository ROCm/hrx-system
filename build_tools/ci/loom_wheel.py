#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds, installs, tests, and exports one final Loom SDK wheel."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from zipfile import BadZipFile

from loom.packaging.python import release

PROFILE_FLAGS = {
    "release": ["--config=opt"],
    "tracy": ["--config=opt", "--//runtime/config/base/tracing:provider=tracy"],
    "debug": ["--compilation_mode=dbg"],
    "asan": ["--config=asan"],
}
WHEEL = "//loom/packaging/python:release_wheel"
INSTALLED_TEST = "//loom/src/loom/tooling/target/cpu/serve:wheel_model_card_test"


def build_wheel(
    root: Path, output: Path, profile: str, bazel: str, options: list[str]
) -> release.Wheel:
    flags = ["--config=loom-sdk", *PROFILE_FLAGS[profile], *options]
    subprocess.run(
        [
            bazel,
            "test",
            *flags,
            "--test_arg=task,amdgpu2,rdna,cdna,xdna",
            INSTALLED_TEST,
        ],
        cwd=root,
        check=True,
    )
    # Query the target just exercised, in the same configuration. A guessed
    # bazel-bin filename could select an old artifact after a profile change.
    result = subprocess.run(
        [bazel, "cquery", *flags, WHEEL, "--output=files"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    paths = result.stdout.splitlines()
    if len(paths) != 1:
        raise ValueError(f"Expected one release wheel output, found {paths!r}")
    package = root / "loom" / "packaging" / "python"
    wheel = release.inspect_wheel(
        root / paths[0],
        release.read_identity(package / "version.json"),
        (package / "PYPI_README.md").read_text(encoding="utf-8"),
    )
    if wheel.profile != profile:
        raise ValueError(f"Requested {profile} wheel, built {wheel.profile}")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / wheel.path.name
    with wheel.path.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target)
    print(f"{wheel.digest}  {destination}")
    return release.Wheel(destination, profile, wheel.platform, wheel.digest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=release.PROFILES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bazel", default="bazelisk")
    parser.add_argument(
        "options", nargs=argparse.REMAINDER, help="Bazel flags after --"
    )
    args = parser.parse_args(argv)
    options = args.options[1:] if args.options[:1] == ["--"] else args.options
    try:
        # Windows shell launchers may be .cmd files; CreateProcess does not
        # perform the shell's PATHEXT search for a bare executable name.
        bazel = shutil.which(args.bazel)
        if bazel is None:
            raise ValueError(f"Bazel executable not found: {args.bazel}")
        build_wheel(Path.cwd(), args.output, args.profile, bazel, options)
    except (ValueError, OSError, BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SDK wheel build failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
