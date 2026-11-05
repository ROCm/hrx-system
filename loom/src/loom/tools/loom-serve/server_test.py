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

from loom_serve_testing.http_harness import control, running_server

SERVER = Path(sys.argv.pop(1)).resolve()
PACKAGE = Path(sys.argv.pop(1)).resolve().parent
FORMAT = Path(sys.argv.pop(1)).resolve()


class ServerTest(unittest.TestCase):
    def test_wait_mode_accepts_both_policies(self):
        for mode in ("active", "blocking"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                log = Path(temporary) / "server.log"
                with running_server(
                    [
                        str(SERVER),
                        "--device=task",
                        "--pool_backing=fixed",
                        f"--wait_mode={mode}",
                        "--workers=1",
                        "--listen=127.0.0.1:0",
                    ],
                    log,
                ) as (process, bound_address):
                    self.assertIsNotNone(bound_address, log.read_text())
                    address = urlsplit(f"http://{bound_address}")
                    control(address, "GET", "/v1/status", 200)
                self.assertEqual(process.returncode, 0, log.read_text())

    def test_invalid_wait_mode_fails_before_opening_a_device(self):
        result = subprocess.run(
            [str(SERVER), "--device=not-opened", "--wait_mode=busy"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("--wait_mode", result.stderr)
        self.assertIn("must be active or blocking", result.stderr)
        self.assertNotIn('"event":"ready"', result.stderr)

    def test_byte_size_flags_publish_exact_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "server.log"
            with running_server(
                [
                    str(SERVER),
                    "--device=task",
                    "--pool_backing=elastic",
                    "--weight_slab_bytes=1gib",
                    "--state_slab_bytes=64mib",
                    "--memory_bytes=64gib",
                    "--workers=1",
                    "--listen=127.0.0.1:0",
                ],
                log,
            ) as (process, bound_address):
                self.assertIsNotNone(bound_address, log.read_text())
                address = urlsplit(f"http://{bound_address}")
                snapshot = control(address, "GET", "/v1/status", 200)
                memory = snapshot["server"]["resources"]["pools"][0]["memory"]
                self.assertEqual(memory["retained"]["limit_bytes"], 64 << 30)
                self.assertEqual(memory["retained"]["committed_bytes"], 0)
            self.assertEqual(process.returncode, 0, log.read_text())

    def test_byte_size_errors_name_the_flag_before_opening_a_device(self):
        cases = (
            ("weight_slab_bytes", "123fb", "INVALID_ARGUMENT"),
            ("state_slab_bytes", "1.5gib", "INVALID_ARGUMENT"),
            ("memory_bytes", "17179869184gib", "OUT_OF_RANGE"),
        )
        for flag, value, status in cases:
            with self.subTest(flag=flag):
                result = subprocess.run(
                    [str(SERVER), "--device=not-opened", f"--{flag}={value}"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn(status, result.stderr)
                self.assertIn(f"--{flag}", result.stderr)
                self.assertNotIn('"event":"ready"', result.stderr)

    def test_idle_durations_are_nonnegative(self):
        for flag in ("workspace_idle_ms", "parameter_idle_ms"):
            with self.subTest(flag=flag):
                result = subprocess.run(
                    [str(SERVER), "--device=task", f"--{flag}=-1"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn("INVALID_ARGUMENT", result.stderr)
                self.assertNotIn('"event":"ready"', result.stderr)

    def test_device_is_required_before_listening(self):
        for arguments in ([], ["--device="]):
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

    def test_device_uri_runtime_services_fail_before_listening(self):
        cases = (
            ("task:?feedback&asan&tsan", "feature flags 0x7"),
            ("task://?feedback=enabled", "does not accept a value"),
        )
        for device, diagnostic in cases:
            with self.subTest(device=device):
                result = subprocess.run(
                    [
                        str(SERVER),
                        f"--device={device}",
                        "--pool_backing=fixed",
                        "--workers=1",
                        "--listen=127.0.0.1:0",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn(diagnostic, result.stderr)
                self.assertNotIn('"event":"ready"', result.stderr)

    def test_multiple_devices_require_named_configuration(self):
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
        self.assertIn("multiple devices require named entries", result.stderr)
        self.assertNotIn('"event":"ready"', result.stderr)

    def test_named_devices_do_not_merge_single_device_flags(self):
        for flag in (
            "--device=task",
            "--pool_backing=elastic",
            "--weight_slab_bytes=1073741824",
            "--state_slab_bytes=67108864",
            "--memory_bytes=0",
            "--workspace_idle_ms=0",
            "--parameter_idle_ms=0",
        ):
            with self.subTest(flag=flag):
                result = subprocess.run(
                    [str(SERVER), "--devices=not-opened.json", flag],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn("--devices cannot be combined", result.stderr)
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
                    "model.json",
                    "config.loom",
                    "embed.loom",
                    "prepare.loom",
                    "control.loom",
                    "tokenizer.json",
                ):
                    shutil.copyfile(PACKAGE / name, package / name)
                document = json.loads((package / "model.json").read_text())
                document.update(base_directory=str(package), admission="drained")
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
                    document["definition"].update(
                        prepare="prepare.loombc",
                        control="control.loombc",
                        sources=["config.loom", "embed.loombc"],
                    )
                options = [
                    "--device=task",
                    "--pool_backing=fixed",
                    "--weight_slab_bytes=1gib",
                    "--state_slab_bytes=64mib",
                    "--memory_bytes=0b",
                    f"--workers={workers}",
                    "--listen=127.0.0.1:0",
                    "--workspace_idle_ms=1",
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
                    self.assertEqual(
                        control(address, "GET", "/v1/status", 200)["server"][
                            "resources"
                        ]["pools"][0]["idle"],
                        {"workspace_ms": 1, "parameter_ms": 0},
                    )
                    registration = self.post(
                        address,
                        "/v1/models/register",
                        document,
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
                        token_count = sum(len(value) for value in inputs)
                        self.assertEqual(
                            result["usage"],
                            {
                                "prompt_tokens": token_count,
                                "total_tokens": token_count,
                            },
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
            device_keys = ("cpu.primary", "cpu.secondary")
            pool_keys = ("foreground", "background", "isolated")
            for index, name in enumerate(("alpha", "beta")):
                package = directory / "models" / name
                shutil.copytree(PACKAGE, package, copy_function=shutil.copyfile)
                document = json.loads((package / "model.json").read_text())
                document["name"] = name
                document["devices"] = [device_keys[index]]
                document["pools"] = [pool_keys[1 - index]]
                (package / "model.json").write_text(json.dumps(document))
                documents.append((package, document))
            settings = directory / "settings"
            settings.mkdir()
            (settings / "devices.json").write_text(
                json.dumps(
                    {
                        "wait_mode": "blocking",
                        "devices": {key: {"uri": "task"} for key in device_keys},
                        "pools": {
                            pool_keys[0]: {
                                "devices": list(device_keys),
                                "pool_backing": "fixed",
                            },
                            pool_keys[1]: {
                                "devices": list(device_keys),
                                "pool_backing": "fixed",
                                "weight_slab_bytes": "1gib",
                                "state_slab_bytes": "64mib",
                                "memory_bytes": "0b",
                                "workspace_idle_ms": 60000,
                                "parameter_idle_ms": 120000,
                            },
                            pool_keys[2]: {
                                "devices": [device_keys[0]],
                                "pool_backing": "fixed",
                            },
                        },
                    }
                )
            )
            # Flagfile arguments keep ordinary process-CWD semantics; the model
            # document's own references instead use its containing directory.
            (settings / "server.flags").write_text("--model=models/alpha/model.json\n")
            log = directory / "server.log"
            with running_server(
                [
                    str(SERVER),
                    "--devices=settings/devices.json",
                    "--wait_mode=active",
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
                snapshot = control(address, "GET", "/v1/status", 200)
                devices = snapshot["server"]["resources"]["devices"]
                pools = snapshot["server"]["resources"]["pools"]
                self.assertEqual(
                    [device["key"] for device in devices], list(device_keys)
                )
                self.assertEqual([pool["key"] for pool in pools], list(pool_keys))
                self.assertEqual(
                    [pool["devices"] for pool in pools], [[0, 1], [0, 1], [0]]
                )
                self.assertEqual(
                    [pool["idle"] for pool in pools],
                    [
                        {"workspace_ms": 0, "parameter_ms": 0},
                        {"workspace_ms": 60000, "parameter_ms": 120000},
                        {"workspace_ms": 0, "parameter_ms": 0},
                    ],
                )
                self.assertEqual(
                    [device["driver"] for device in devices], ["task", "task"]
                )
                self.assertEqual(
                    {entry["id"]: entry["loom"]["pools"] for entry in snapshot["data"]},
                    {"alpha": [pool_keys[1]], "beta": [pool_keys[0]]},
                )
                self.assertEqual(
                    {
                        entry["id"]: entry["loom"]["devices"]
                        for entry in snapshot["data"]
                    },
                    {"alpha": [device_keys[0]], "beta": [device_keys[1]]},
                )

                # Both admission modes reject ambiguous/out-of-group bindings
                # before publishing a slot, even when no model I/O is needed.
                for admission in ("open", "drained"):
                    for binding in ({}, {"devices": ["absent"]}, {"devices": 0}):
                        invalid = dict(documents[0][1])
                        invalid.pop("devices")
                        invalid.update(
                            name="invalid",
                            admission=admission,
                            base_directory=str(documents[0][0]),
                            **binding,
                        )
                        self.post(address, "/v1/models/register", invalid, expected=400)
                    for binding in (
                        {},
                        {"pools": ["absent"]},
                        {"pools": 0},
                        {"pools": [pool_keys[2]]},
                    ):
                        invalid = dict(documents[1][1])
                        invalid.pop("pools")
                        invalid.update(
                            name="invalid",
                            admission=admission,
                            base_directory=str(documents[1][0]),
                            **binding,
                        )
                        self.post(address, "/v1/models/register", invalid, expected=400)
                self.assertEqual(
                    len(control(address, "GET", "/v1/status", 200)["data"]), 2
                )
                self.post(
                    address,
                    "/v1/pools/trim",
                    {
                        "pool": 0,
                        "target_bytes": 0,
                    },
                    expected=400,
                )
                self.post(
                    address,
                    "/v1/pools/trim",
                    {
                        "pool": "absent",
                        "target_bytes": 0,
                    },
                    expected=404,
                )
                trimmed = self.post(
                    address,
                    "/v1/pools/trim",
                    {
                        "pool": pool_keys[1],
                        "target_bytes": 0,
                    },
                )
                self.assertEqual(trimmed["pool"], pools[1])
                programs_trimmed = self.post(
                    address, "/v1/devices/trim", {"device": device_keys[1]}
                )
                self.assertEqual(programs_trimmed, {"device": devices[1]})
                self.assertEqual(
                    control(address, "GET", "/v1/status", 200)["server"]["resources"][
                        "pools"
                    ],
                    pools,
                )
                self.assertEqual(
                    control(address, "GET", "/healthz", 200), {"status": "ready"}
                )

                def reservations():
                    snapshot = control(address, "GET", "/v1/status", 200)
                    return [
                        pool["memory"]["workspace"]["reserve_count"]
                        for pool in snapshot["server"]["resources"]["pools"]
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
                    self.assertGreater(after[1 - index], before[1 - index])
                    self.assertEqual(after[index], before[index])
                    self.assertEqual(after[2], before[2])
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
                self.assertGreater(after[0], before[0])
                self.assertEqual(after[1:], before[1:])
                package, document = documents[0]
                # HTTP has no file containing the object to supply its base.
                self.post(address, "/v1/models/register", document, expected=400)
                document["base_directory"] = str(package)
                # Two execution devices now use the same explicit owner. No
                # second pool or copied payload is introduced for the binding.
                document["pools"] = [pool_keys[0]]
                self.post(address, "/v1/models/register", document, expected=202)
                before = reservations()
                result = self.post(
                    address,
                    "/v1/embeddings",
                    {"model": "alpha", "input": "bbb", "dimensions": 2},
                )
                self.assertEqual(result["data"][0]["embedding"], [6, 3])
                after = reservations()
                self.assertGreater(after[0], before[0])
                self.assertEqual(after[1:], before[1:])
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
