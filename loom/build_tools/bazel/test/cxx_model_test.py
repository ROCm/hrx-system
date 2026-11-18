# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rebuilds and executes packaged C++ components after source relocation."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command, directory):
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def main(importer, linker, runner, *packages):
    importer, linker, runner = [
        str(Path(path).resolve()) for path in (importer, linker, runner)
    ]
    for package in packages:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            relocated = directory / "published"
            shutil.copytree(Path(package).resolve().parent, relocated)
            card = json.loads((relocated / "model.json").read_text())
            program = card["programs"]["source"]
            providers = program["inputs"] + program["libraries"]
            preserved = providers[0]["format"] == "cxx"
            expected = {"model.json", "serve.json"}
            modules = []
            for ordinal, provider in enumerate(providers):
                source = relocated / provider["path"]
                expected.add(provider["path"])
                expected.update(provider.get("data", []))
                if preserved:
                    module = directory / f"unit{ordinal}.loombc"
                    run(
                        [
                            importer,
                            str(source),
                            "--to=bc",
                            f"--output={module}",
                            "--options=" + provider["options"][0].removeprefix("cxx:"),
                        ],
                        directory,
                    )
                else:
                    assert "options" not in provider
                    assert not any(
                        path.endswith(".h") for path in provider.get("data", [])
                    )
                    module = source
                modules.append(str(module))
            actual = {
                path.relative_to(relocated).as_posix()
                for path in relocated.rglob("*")
                if path.is_file()
            }
            assert actual == expected, (actual, expected)
            fixture = relocated / "loom/build_tools/bazel/test/model_fixture.txt"
            assert fixture.is_file()
            if not preserved:
                assert not list(relocated.rglob("*.h"))
            linked = directory / "checks.loombc"
            run(
                [
                    linker,
                    "--input-format=loom",
                    "--mode=link",
                    "--include-input-tests",
                    "--to=bc",
                    f"--output={linked}",
                    modules[0],
                    *[f"--library={module}" for module in modules[1:]],
                ],
                directory,
            )
            report = json.loads(run([runner, str(linked)], directory))
            assert report["case_count"] == 1, report
            assert report["sample_count"] == 1, report
            assert report["skipped_case_count"] == 0, report
            assert report["planning_issue_count"] == 0, report
            assert report["samples"][0]["expectations"]["expectation_count"] == 1, (
                report
            )
            print(f"{Path(package).parent.name}: relocated check passed")


if __name__ == "__main__":
    main(*sys.argv[1:])
