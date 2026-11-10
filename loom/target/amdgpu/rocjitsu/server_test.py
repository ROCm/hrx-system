# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Serve canonical embeddings on two devices in one simulated topology."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http_harness import control, running_server

SERVER = Path(sys.argv.pop(1)).resolve()
PACKAGE = Path(sys.argv.pop(1)).resolve().parent


class ServerTest(unittest.TestCase):
    def test_two_devices_share_a_pool_and_retire_normally(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            device_keys = ("gpu.primary", "gpu.secondary")
            model_names = ("alpha", "beta")
            models = []
            for name, device in zip(model_names, device_keys):
                package = directory / name
                shutil.copytree(PACKAGE, package, copy_function=shutil.copyfile)
                document = json.loads((package / "model.json").read_text())
                document.update(name=name, device=device, pool="shared")
                model = package / "model.json"
                model.write_text(json.dumps(document))
                models.append(model)
            devices = directory / "devices.json"
            devices.write_text(
                json.dumps(
                    {
                        "wait_mode": "blocking",
                        "devices": {
                            key: {"uri": f"amdgpu://{index}"}
                            for index, key in enumerate(device_keys)
                        },
                        "pools": {
                            "shared": {
                                "devices": list(device_keys),
                                "pool_backing": "fixed",
                                "weight_slab_bytes": "16mib",
                                "state_slab_bytes": "16mib",
                                "memory_bytes": "0b",
                            }
                        },
                    }
                )
            )
            log = directory / "server.log"
            with running_server(
                [
                    str(SERVER),
                    f"--devices={devices}",
                    "--workers=2",
                    "--listen=127.0.0.1:0",
                    *[f"--model={model}" for model in models],
                ],
                log,
            ) as (process, bound_address):
                self.assertIsNotNone(bound_address, log.read_text())
                address = urlsplit(f"http://{bound_address}")
                snapshot = control(address, "GET", "/v1/status", 200)
                resources = snapshot["server"]["resources"]
                self.assertEqual(
                    [device["key"] for device in resources["devices"]],
                    list(device_keys),
                )
                self.assertEqual(
                    [device["driver"] for device in resources["devices"]],
                    ["amd", "amd"],
                )
                self.assertEqual(
                    [pool["devices"] for pool in resources["pools"]], [[0, 1]]
                )
                self.assertEqual(
                    {
                        model["id"]: (model["loom"]["device"], model["loom"]["pool"])
                        for model in snapshot["data"]
                    },
                    {
                        name: (device, "shared")
                        for name, device in zip(model_names, device_keys)
                    },
                )
                for name in model_names:
                    for inputs, expected in (
                        (["a", "bbb"], [[1, 1], [6, 3]]),
                        (["ab"], [[3, 2]]),
                    ):
                        with self.subTest(model=name, inputs=inputs):
                            result = control(
                                address,
                                "POST",
                                "/v1/embeddings",
                                200,
                                body=json.dumps(
                                    {"model": name, "input": inputs, "dimensions": 2}
                                ),
                                headers={"Content-Type": "application/json"},
                            )
                            self.assertEqual(
                                [item["embedding"] for item in result["data"]], expected
                            )
                            self.assertEqual(
                                [item["index"] for item in result["data"]],
                                list(range(len(inputs))),
                            )
                            token_count = sum(map(len, inputs))
                            self.assertEqual(
                                result["usage"],
                                {
                                    "prompt_tokens": token_count,
                                    "total_tokens": token_count,
                                },
                            )
            self.assertEqual(process.returncode, 0, log.read_text())


if __name__ == "__main__":
    unittest.main()
