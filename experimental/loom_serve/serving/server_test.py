# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise the public executable and operator flow without a GPU or weights."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.testing.http import control, running_server

SERVER = Path(sys.argv.pop(1)).resolve()
PACKAGE = Path(sys.argv.pop(1)).resolve().parent
FORMAT = Path(sys.argv.pop(1)).resolve()


class ServerTest(unittest.TestCase):
    def test_positional_model_is_rejected_before_listening(self):
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
            self.assertIn("--model=path", log.read_text())

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
                    "tokenizer.json",
                ):
                    shutil.copyfile(PACKAGE / name, package / name)
                if workers == 4:
                    # Native providers can mix formats. VM role files exercise
                    # format detection through their separate VM loading path.
                    for name in ("embed", "prepare", "control"):
                        encoded = package / f"{name}.loombc"
                        subprocess.run(
                            [
                                str(FORMAT),
                                str(package / f"{name}.loom"),
                                "--from=text",
                                "--to=bc",
                                f"--output={encoded}",
                            ],
                            check=True,
                        )
                        (package / f"{name}.loom").unlink()
                options = [
                    "--device=task",
                    "--pool_backing=fixed",
                    f"--workers={workers}",
                    "--listen=127.0.0.1:0",
                ]
                if workers == 4:
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
                            "name": "vectors",
                            "base_directory": str(package),
                            "definition": {
                                "kind": "embedding",
                                "prepare": "prepare.loombc"
                                if workers == 4
                                else "prepare.loom",
                                "control": "control.loombc"
                                if workers == 4
                                else "control.loom",
                                "sources": [
                                    "config.loom",
                                    "embed.loombc" if workers == 4 else "embed.loom",
                                ],
                                "assets": ".",
                                "options": {"batch_units": 64},
                            },
                            "admission": "drained",
                        },
                        expected=202,
                    )
                    self.assertEqual(registration["loom"]["registration"], "registered")
                    self.assertEqual(
                        registration["loom"]["status"]["admission"], "drained"
                    )
                    self.assertEqual(
                        control(address, "GET", "/v1/models", 200)["data"], []
                    )
                    self.assertEqual(
                        control(address, "GET", "/v1/status", 200)["data"][0]["id"],
                        "vectors",
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
                    model = control(address, "GET", "/v1/models", 200)["data"][0]
                    self.assertEqual(model["id"], "vectors")
                    self.assertEqual(model["object"], "model")
                    self.assertGreater(model["created"], 0)
                    self.assertEqual(model["created"], registration["created"])
                    self.assertEqual(model["owned_by"], "local")
                    self.assertEqual(model["input_modalities"], ["text"])
                    self.assertEqual(model["output_modalities"], ["embedding"])
                    self.assertNotIn("context_length", model)
                    self.post(address, "/v1/models/drain", {"model": "vectors"})
                    self.assertEqual(
                        control(address, "GET", "/v1/models", 200)["data"][0]["id"],
                        "vectors",
                    )
                    self.post(
                        address,
                        "/v1/models/unregister",
                        {"model": "vectors"},
                        expected=202,
                    )
                    while control(address, "GET", "/v1/status", 200)["data"]:
                        pass
                    self.assertEqual(
                        control(address, "GET", "/healthz", 200), {"status": "ready"}
                    )
                self.assertEqual(process.returncode, 0, log.read_text())

    def test_repeated_documents_share_cli_and_http_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            documents = []
            for name in ("alpha", "beta"):
                package = directory / "models" / name
                shutil.copytree(PACKAGE, package, copy_function=shutil.copyfile)
                document = json.loads((package / "model.json").read_text())
                document["name"] = name
                (package / "model.json").write_text(json.dumps(document))
                documents.append((package, document))
            settings = directory / "settings"
            settings.mkdir()
            # Flagfile arguments keep ordinary process-CWD semantics; the model
            # document's own references instead use its containing directory.
            (settings / "server.flags").write_text("--model=models/alpha/model.json\n")
            log = directory / "server.log"
            with running_server(
                [
                    str(SERVER),
                    "--device=task",
                    "--pool_backing=fixed",
                    "--workers=2",
                    "--listen=127.0.0.1:0",
                    "--flagfile=settings/server.flags",
                    "--model=models/beta/model.json",
                ],
                log,
                cwd=directory,
            ) as (process, bound_address):
                self.assertIsNotNone(bound_address, log.read_text())
                address = urlsplit(f"http://{bound_address}")
                for name in ("alpha", "beta"):
                    result = self.post(
                        address,
                        "/v1/embeddings",
                        {"model": name, "input": "ab", "dimensions": 2},
                    )
                    self.assertEqual(result["data"][0]["embedding"], [3, 2])
                self.post(
                    address, "/v1/models/unregister", {"model": "alpha"}, expected=202
                )
                while any(
                    entry["id"] == "alpha"
                    for entry in control(address, "GET", "/v1/status", 200)["data"]
                ):
                    pass
                package, document = documents[0]
                # HTTP has no file containing the object to supply its base.
                self.post(address, "/v1/models/register", document, expected=400)
                document["base_directory"] = str(package)
                self.post(address, "/v1/models/register", document, expected=202)
                result = self.post(
                    address,
                    "/v1/embeddings",
                    {"model": "alpha", "input": "bbb", "dimensions": 2},
                )
                self.assertEqual(result["data"][0]["embedding"], [6, 3])
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
