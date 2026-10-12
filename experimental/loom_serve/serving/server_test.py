# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise the public executable and operator flow without a GPU or weights."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.testing.http import control, running_server

SERVER = Path(sys.argv.pop(1)).resolve()
PACKAGE = Path(sys.argv.pop(1)).resolve().parent


class ServerTest(unittest.TestCase):
    def test_positional_catalog_is_rejected_before_listening(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log = directory / "server.log"
            with running_server(
                [
                    str(SERVER),
                    "--device=task",
                    "--pool_backing=fixed",
                    "--workers=1",
                    "--listen=127.0.0.1:0",
                    "models.json",
                ],
                log,
                cwd=directory,
            ) as (process, address):
                self.assertIsNone(address, log.read_text())
                self.assertNotEqual(process.wait(), 0)
            self.assertIn(
                "Unexpected positional argument 'models.json'", log.read_text()
            )
            self.assertIn("--models=path", log.read_text())

    def test_empty_server_runs_external_source_and_retires(self):
        for workers in (1, 4):
            with (
                self.subTest(workers=workers),
                tempfile.TemporaryDirectory() as temporary,
            ):
                directory = Path(temporary)
                package = directory / "external package"
                package.mkdir()
                for name in (
                    "config.loom",
                    "embed.loom",
                    "prepare.loom",
                    "control.loom",
                    "sources.txt",
                    "tokenizer.json",
                ):
                    shutil.copyfile(PACKAGE / name, package / name)
                options = [
                    "--device=task",
                    "--pool_backing=fixed",
                    f"--workers={workers}",
                    "--listen=127.0.0.1:0",
                ]
                if workers == 4:
                    (directory / "empty.json").write_text('{"models":[]}')
                    options.append("--models=empty.json")
                    (directory / "server flags.txt").write_text(
                        "\n".join(options) + "\n"
                    )
                    options = ["--flagfile=server flags.txt"]
                log = directory / "server.log"
                with running_server([str(SERVER), *options], log, cwd=directory) as (
                    process,
                    bound_address,
                ):
                    self.assertIsNotNone(bound_address, log.read_text())
                    address = urlsplit(f"http://{bound_address}")
                    self.assertEqual(
                        control(address, "GET", "/readyz", 200), {"status": "ready"}
                    )
                    self.assertEqual(
                        control(address, "GET", "/v1/models", 200)["data"], []
                    )
                    registration = self.post(
                        address,
                        "/v1/models/register",
                        {
                            "kind": "embedding",
                            "name": "vectors",
                            "source": package.name,
                            "assets": package.name,
                            "options": {"batch_units": 64},
                            "admission": "drained",
                        },
                        expected=202,
                    )
                    self.assertEqual(registration["loom"]["registration"], "registered")
                    self.assertEqual(
                        registration["loom"]["status"]["admission"], "drained"
                    )
                    self.post(address, "/v1/models/undrain", {"model": "vectors"})
                    for inputs, expected in (
                        (["a", "bbb"], [[1, 1], [6, 3]]),
                        (["ab"], [[3, 2]]),
                    ):
                        result = self.post(
                            address,
                            "/v1/embeddings",
                            {"model": "vectors", "input": inputs, "dimensions": 2},
                        )
                        self.assertEqual(
                            [item["embedding"] for item in result["data"]], expected
                        )
                        self.assertEqual(
                            [item["index"] for item in result["data"]],
                            list(range(len(inputs))),
                        )
                    self.post(
                        address,
                        "/v1/models/unregister",
                        {"model": "vectors"},
                        expected=202,
                    )
                    while control(address, "GET", "/v1/models", 200)["data"]:
                        pass
                    self.assertEqual(
                        control(address, "GET", "/healthz", 200), {"status": "ready"}
                    )
                self.assertEqual(process.returncode, 0, log.read_text())

    @staticmethod
    def post(address, path, body, expected=200):
        return control(
            address,
            "POST",
            path,
            expected,
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )


if __name__ == "__main__":
    unittest.main()
