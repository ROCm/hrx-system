# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_smoke_test_module():
    smoke_test_path = Path(__file__).with_name("smoke_test.py")
    spec = importlib.util.spec_from_file_location("id4_smoke_test", smoke_test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {smoke_test_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Id4SmokeTestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke_test = load_smoke_test_module()

    def test_default_request_has_generation_metadata(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=unused",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
            ]
        )

        request = self.smoke_test.load_request(args)

        self.assertIn("prompt", request)
        self.assertEqual(
            request["generation"],
            {
                "latent_width": 8,
                "latent_height": 8,
                "denoise_steps": 20,
                "seed": 20260625,
                "guidance_scale": 3.5,
            },
        )

    def test_request_json_must_be_full_generation_request(self):
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "prompt.json"
            request_path.write_text(json.dumps({"prompt": "hello"}), encoding="utf-8")
            args = self.smoke_test.parse_arguments(
                [
                    "--output_dir=unused",
                    "--device=amdgpu",
                    "--tokenizer=tokenizer.json",
                    "--parameters=qwen=qwen.safetensors",
                    f"--request_json={request_path}",
                ]
            )

            with self.assertRaisesRegex(
                self.smoke_test.SmokeTestError, "generation metadata"
            ):
                self.smoke_test.load_request(args)

    def test_request_json_must_contain_prompt_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "prompt.json"
            request_path.write_text(
                json.dumps(
                    {
                        "generation": {
                            "latent_width": 1,
                            "latent_height": 1,
                            "denoise_steps": 1,
                            "seed": 1,
                            "guidance_scale": 1.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = self.smoke_test.parse_arguments(
                [
                    "--output_dir=unused",
                    "--device=amdgpu",
                    "--tokenizer=tokenizer.json",
                    "--parameters=qwen=qwen.safetensors",
                    f"--request_json={request_path}",
                ]
            )

            with self.assertRaisesRegex(
                self.smoke_test.SmokeTestError, "prompt payload"
            ):
                self.smoke_test.load_request(args)

    def test_build_id4_command_routes_artifacts_to_output_dir(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=out",
                "--id4_binary=bin/id4",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
                "--parameters=vae=vae.safetensors",
                "--extra_id4_arg=--list_devices=false",
            ]
        )

        command = self.smoke_test.build_id4_command(args, Path("artifacts"))

        self.assertEqual(command[0], "bin/id4")
        self.assertIn("--prompt_json_file=artifacts/request.json", command)
        self.assertIn("--output=artifacts/image.ppm", command)
        self.assertIn("--dump_plan=artifacts/plan.json", command)
        self.assertIn("--dump_diagnostics=artifacts/diagnostics", command)
        self.assertIn("--profile_output=artifacts/profile.txt", command)
        self.assertIn("--dit_attention_implementation=materialized_wmma", command)
        self.assertIn("--device=amdgpu", command)
        self.assertIn("--parameters=qwen=qwen.safetensors", command)
        self.assertIn("--parameters=vae=vae.safetensors", command)
        self.assertEqual(command[-1], "--list_devices=false")

    def test_ppm_metrics_preserve_binary_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.ppm"
            payload = bytes(
                [
                    0,
                    0,
                    0,
                    255,
                    255,
                    255,
                    16,
                    32,
                    64,
                    128,
                    64,
                    32,
                ]
            )
            path.write_bytes(b"P6\n2 2\n255\n" + payload)

            metrics = self.smoke_test.read_ppm_metrics(path)

            self.assertEqual(metrics.width, 2)
            self.assertEqual(metrics.height, 2)
            self.assertEqual(metrics.payload_size, len(payload))
            self.assertEqual(metrics.minimum_channel_value, 0)
            self.assertEqual(metrics.maximum_channel_value, 255)

    def test_uniform_ppm_fails_validation(self):
        request = {
            "generation": {
                "latent_width": 1,
                "latent_height": 1,
                "denoise_steps": 1,
                "seed": 1,
                "guidance_scale": 1.0,
            }
        }
        metrics = self.smoke_test.ImageMetrics(
            width=16,
            height=16,
            max_value=255,
            payload_size=16 * 16 * 3,
            minimum_channel_value=128,
            maximum_channel_value=128,
            mean_channel_value=128.0,
            luma_stddev=0.0,
            sampled_unique_rgb_count=1,
        )

        with self.assertRaisesRegex(self.smoke_test.SmokeTestError, "dynamic range"):
            self.smoke_test.validate_image(metrics, request)

    def test_run_smoke_writes_summary_for_invalid_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_id4 = root / "fake_id4.py"
            fake_id4.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "from pathlib import Path",
                        "import sys",
                        "output = None",
                        "for arg in sys.argv[1:]:",
                        "    if arg.startswith('--output='):",
                        "        output = Path(arg.split('=', 1)[1])",
                        "if output is None:",
                        "    raise SystemExit(2)",
                        "output.write_bytes(b'P6\\n16 16\\n255\\n' + bytes([128]) * 16 * 16 * 3)",
                    ]
                ),
                encoding="utf-8",
            )
            fake_id4.chmod(0o755)
            artifact_dir = root / "artifacts"
            args = self.smoke_test.parse_arguments(
                [
                    f"--output_dir={artifact_dir}",
                    f"--id4_binary={fake_id4}",
                    "--device=amdgpu",
                    "--tokenizer=tokenizer.json",
                    "--parameters=qwen=qwen.safetensors",
                    "--latent_width=1",
                    "--latent_height=1",
                    "--denoise_steps=1",
                ]
            )

            with self.assertRaisesRegex(
                self.smoke_test.SmokeTestError, "dynamic range"
            ):
                self.smoke_test.run_smoke(args)

            with (artifact_dir / "summary.json").open(encoding="utf-8") as file:
                summary = json.load(file)
            self.assertEqual(summary["state"], "failed")
            self.assertIn("dynamic range", summary["validation_error"])
            self.assertEqual(summary["image_metrics"]["width"], 16)


if __name__ == "__main__":
    unittest.main()
