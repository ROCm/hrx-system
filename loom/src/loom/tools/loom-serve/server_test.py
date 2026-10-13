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

from loom_serve_testing.http import control, running_server

SERVER = Path(sys.argv.pop(1)).resolve()
PACKAGE = Path(sys.argv.pop(1)).resolve().parent
FORMAT = Path(sys.argv.pop(1)).resolve()


class ServerTest(unittest.TestCase):
    def test_device_is_required_before_listening(self):
        for arguments in ([], ["--device="], ["--device=task", "--device="]):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [
                        str(SERVER),
                        *arguments,
                        "--pool_backing=fixed",
                        "--workers=1",
                        "--listen=127.0.0.1:0",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn("--device is required", result.stderr)
                self.assertNotIn('"event":"ready"', result.stderr)

    def test_shared_byte_limit_requires_one_device(self):
        result = subprocess.run(
            [
                str(SERVER),
                "--device=task",
                "--device=task",
                "--memory_bytes=4096",
                "--workers=1",
                "--listen=127.0.0.1:0",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("--memory_bytes requires one device", result.stderr)
        self.assertNotIn('"event":"ready"', result.stderr)

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
            for index, name in enumerate(("alpha", "beta")):
                package = directory / "models" / name
                shutil.copytree(PACKAGE, package, copy_function=shutil.copyfile)
                document = json.loads((package / "model.json").read_text())
                document["name"] = name
                document["device"] = index
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
                    "--device=task",
                ],
                log,
                cwd=directory,
            ) as (process, bound_address):
                self.assertIsNotNone(bound_address, log.read_text())
                address = urlsplit(f"http://{bound_address}")
                snapshot = control(address, "GET", "/v1/status", 200)
                devices = snapshot["server"]["resources"]["devices"]
                self.assertEqual([device["index"] for device in devices], [0, 1])
                self.assertEqual(
                    [device["driver"] for device in devices], ["task", "task"]
                )
                self.assertEqual(
                    {
                        entry["id"]: entry["loom"]["device"]
                        for entry in snapshot["data"]
                    },
                    {"alpha": 0, "beta": 1},
                )

                # Both admission modes reject ambiguous/out-of-group bindings
                # before publishing a slot, even when no model I/O is needed.
                for admission in ("open", "drained"):
                    for binding in ({}, {"device": 2}):
                        invalid = dict(documents[0][1])
                        invalid.pop("device")
                        invalid.update(
                            name="invalid",
                            admission=admission,
                            base_directory=str(documents[0][0]),
                            **binding,
                        )
                        self.post(address, "/v1/models/register", invalid, expected=400)
                self.assertEqual(
                    len(control(address, "GET", "/v1/status", 200)["data"]), 2
                )

                def reservations():
                    snapshot = control(address, "GET", "/v1/status", 200)
                    return [
                        device["memory"]["workspace"]["reserve_count"]
                        for device in snapshot["server"]["resources"]["devices"]
                    ]

                for index, name in enumerate(("alpha", "beta")):
                    before = reservations()
                    result = self.post(
                        address,
                        "/v1/embeddings",
                        {"model": name, "input": "ab", "dimensions": 2},
                    )
                    self.assertEqual(result["data"][0]["embedding"], [3, 2])
                    after = reservations()
                    self.assertGreater(after[index], before[index])
                    self.assertEqual(after[1 - index], before[1 - index])
                self.post(
                    address, "/v1/models/unregister", {"model": "alpha"}, expected=202
                )
                while any(
                    entry["id"] == "alpha"
                    for entry in control(address, "GET", "/v1/status", 200)["data"]
                ):
                    pass
                before = reservations()
                result = self.post(
                    address,
                    "/v1/embeddings",
                    {"model": "beta", "input": "bbb", "dimensions": 2},
                )
                self.assertEqual(result["data"][0]["embedding"], [6, 3])
                after = reservations()
                self.assertEqual(after[0], before[0])
                self.assertGreater(after[1], before[1])
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
