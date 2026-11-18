# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from build_tools.ci import bazel_test_reproducers


class BazelTestReproducersTest(unittest.TestCase):
    def test_failed_configuration_replays_with_its_own_runfiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []
            for config, status in (("one", "PASSED"), ("two", "TIMEOUT")):
                prefix = ["bazel-out", config, "bin"]
                name = "suite/launcher"
                program = root.joinpath(*prefix, name)
                program.parent.mkdir(parents=True)
                program.write_text(
                    '#!/bin/sh\nset -eu\ntest -f "$0.runfiles/empty"\ncat "$0.runfiles/subdir/fixture"\n'
                )
                program.chmod(0o755)
                fixture = root / f"input-{config}"
                fixture.write_text(config)
                Path(f"{program}.runfiles_manifest").write_text(
                    f" original\\sinput {fixture}\n"
                    " subdir/fixture ../original input\n"
                    "empty \n"
                )
                target = {"label": "//suite:example", "configuration": {"id": config}}
                file_set = {"id": config}
                parent = {"id": f"parent-{config}"}
                events.extend(
                    [
                        {
                            "id": {"testResult": target},
                            "testResult": {"status": status},
                        },
                        {
                            "id": {"targetCompleted": target},
                            "completed": {
                                "outputGroup": [
                                    {"name": "default", "fileSets": [parent]}
                                ]
                            },
                        },
                        {
                            "id": {"namedSet": parent},
                            "namedSetOfFiles": {"fileSets": [file_set]},
                        },
                        {
                            "id": {"namedSet": file_set},
                            "namedSetOfFiles": {
                                "files": [{"name": name, "pathPrefix": prefix}]
                            },
                        },
                    ]
                )
            stream = root / "events.json"
            stream.write_text("".join(json.dumps(event) + "\n" for event in events))
            output = root / "failed.tar.gz"
            bazel_test_reproducers.archive_failed_tests(stream, root, output)
            restored = root / "restored"
            with tarfile.open(output) as archive:
                archive.extractall(restored, filter="data")
            self.assertFalse((restored / "bazel-out" / "one").exists())
            program = restored / "bazel-out" / "two" / "bin" / "suite" / "launcher"
            self.assertEqual(subprocess.check_output([program], text=True), "two")


if __name__ == "__main__":
    unittest.main()
