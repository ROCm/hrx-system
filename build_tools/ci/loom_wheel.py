#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build SDK artifacts once, then qualify them on their destination platform."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
ASSOCIATED_WHEEL = "//loom/packaging/python/test:associated_wheel"
MODEL = "//loom/src/loom/serve/fixture/text:card_model_mixed"
TEST_DIRECTORY = "loom/src/loom/tooling/target/cpu/serve"


def build_wheel(
    root: Path, output: Path, profile: str, bazel: str, options: list[str]
) -> release.Wheel:
    flags = ["--config=loom-sdk", *PROFILE_FLAGS[profile], *options]
    targets = [WHEEL, ASSOCIATED_WHEEL, MODEL]
    subprocess.run(
        [bazel, "build", *flags, *targets],
        cwd=root,
        check=True,
    )
    # Resolve this configuration's declared outputs, including the fixture's
    # runfiles. Neither bazel-bin guesses nor a directory copy can distinguish
    # current inputs from another profile or a removed fixture asset.
    result = subprocess.run(
        [
            bazel,
            "cquery",
            *flags,
            "set(" + " ".join(targets) + ")",
            "--output=starlark",
            '--starlark:expr=json.encode({"package": target.label.package, '
            '"files": [f.path for f in target.files.to_list()], '
            '"runfiles": [f.path for f in '
            'providers(target)["OutputGroupInfo"].model_files.to_list()] '
            'if target.label.package == "loom/src/loom/serve/fixture/text" else []})',
        ],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    outputs = {}
    for line in result.stdout.splitlines():
        record = json.loads(line)
        package = record["package"]
        if package in outputs or len(record["files"]) != 1:
            raise ValueError(f"Expected one output per SDK input, found {record!r}")
        outputs[package] = record
    expected = {label[2:].split(":")[0] for label in targets}
    if set(outputs) != expected:
        raise ValueError(f"Expected SDK inputs {expected}, found {set(outputs)}")
    package = root / "loom" / "packaging" / "python"
    wheel = release.inspect_wheel(
        root / outputs["loom/packaging/python"]["files"][0],
        release.read_identity(package / "version.json"),
        (package / "PYPI_README.md").read_text(encoding="utf-8"),
    )
    if wheel.profile != profile:
        raise ValueError(f"Requested {profile} wheel, built {wheel.profile}")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / wheel.path.name
    shutil.copyfile(wheel.path, destination)
    associated = root / outputs["loom/packaging/python/test"]["files"][0]
    (output / "associated").mkdir()
    shutil.copyfile(associated, output / "associated" / associated.name)
    model = outputs["loom/src/loom/serve/fixture/text"]
    model_root = (root / model["files"][0]).parent
    for filename in model["runfiles"]:
        source = root / filename
        target = output / "model" / source.relative_to(model_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    print(f"{wheel.digest}  {destination}")
    return release.Wheel(destination, profile, wheel.platform, wheel.digest)


def test_wheel(root: Path, inputs: Path, output: Path, elf_inspector: str) -> None:
    """Runs the installed-wheel oracle without invoking a build tool."""
    inputs = inputs.resolve()
    wheels = list(inputs.glob("*.whl"))
    associated = list((inputs / "associated").glob("*.whl"))
    if len(wheels) != 1 or len(associated) != 1:
        raise ValueError("SDK inputs require one wheel and one associated test wheel")
    package = root / "loom/packaging/python"
    wheel = release.inspect_wheel(
        wheels[0],
        release.read_identity(package / "version.json"),
        (package / "PYPI_README.md").read_text(encoding="utf-8"),
    )
    if (wheel.platform == "win_amd64") != (os.name == "nt"):
        raise ValueError(
            f"Run {wheel.platform} installed qualification on its native OS"
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        str(root / directory)
        for directory in ("loom/packaging/python", "loom/src/loom/serve/testing")
    )
    subprocess.run(
        [
            sys.executable,
            str(root / TEST_DIRECTORY / "wheel_model_card_test.py"),
            str(wheel.path),
            str(associated[0]),
            str(inputs / "model/serve.json"),
            elf_inspector,
            str(package / "version.json"),
            str(package / "PYPI_README.md"),
            "task,amdgpu,rdna,cdna,xdna",
        ],
        cwd=root,
        env=environment,
        check=True,
    )
    # Only the destination-OS qualifier can export a publication artifact.
    output.mkdir(parents=True, exist_ok=True)
    destination = output / wheel.path.name
    with wheel.path.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target)
    print(f"{wheel.digest}  {destination}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Build wheels and their small test fixture"
    )
    build.add_argument("--profile", required=True, choices=release.PROFILES)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--bazel", default="bazelisk")
    build.add_argument("options", nargs=argparse.REMAINDER, help="Bazel flags after --")
    test = commands.add_parser("test", help="Qualify and export an already-built wheel")
    test.add_argument("--inputs", required=True, type=Path)
    test.add_argument("--output", required=True, type=Path)
    test.add_argument("--elf-inspector", default=shutil.which("patchelf") or "unused")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            options = args.options[1:] if args.options[:1] == ["--"] else args.options
            # Resolve Windows .cmd launchers as well as native executables.
            bazel = shutil.which(args.bazel)
            if bazel is None:
                raise ValueError(f"Bazel executable not found: {args.bazel}")
            build_wheel(Path.cwd(), args.output, args.profile, bazel, options)
        else:
            test_wheel(Path.cwd(), args.inputs, args.output, args.elf_inspector)
    except (ValueError, OSError, BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SDK wheel {args.command} failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
