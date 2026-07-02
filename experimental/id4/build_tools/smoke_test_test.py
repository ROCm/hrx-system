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


def minimal_generation_plan() -> dict:
    return {
        "kind": "ideogram4_generation",
        "summary": {
            "qwen_token_count": 37,
            "qwen_token_capacity": 64,
            "image_token_count": 64,
            "conditioned_dit_token_count": 101,
            "conditioned_dit_token_capacity": 128,
            "unconditioned_dit_token_count": 64,
            "unconditioned_dit_token_capacity": 64,
            "denoise_step_count": 1,
            "diffusion_latent_shape": {
                "rank": 4,
                "dims": [8, 8, 128, 1],
            },
            "decoded_image_shape": {
                "rank": 4,
                "dims": [128, 128, 3, 1],
            },
            "dit_activation_format": 2,
            "dit_weight_execution_format": 1,
            "qwen_weight_execution_strategy": 3,
            "qwen_attention_implementation": 1,
            "dit_attention_implementation": 4,
            "dit_feed_forward_implementation": 2,
            "vae_tiling": {
                "mode": 1,
                "tile_size_x": 0,
                "tile_size_y": 0,
                "relative_size_x": 0,
                "relative_size_y": 0,
                "overlap": 0,
                "memory_budget": 0,
            },
        },
        "residency": {
            "total_stage_parameter_byte_length": 1234,
            "phase_parameter_high_water_mark": 1024,
            "largest_stage_parameter_byte_length": 1024,
            "total_stage_boundary_byte_length": 256,
            "phases": [
                {
                    "name": "conditioning",
                    "stage_keys": ["qwen"],
                    "repeated_per_denoise_step": False,
                    "parameter_byte_length": 1024,
                    "largest_stage_parameter_byte_length": 1024,
                    "constant_byte_length": 0,
                    "local_slab_byte_length": 512,
                    "local_high_water_mark": 384,
                    "stage_boundary_byte_length": 128,
                }
            ],
        },
        "stages": {
            "qwen": {
                "statistics": {
                    "parameter_slab_byte_length": 1024,
                    "largest_parameter_slab_byte_length": 1024,
                    "parameter_source_byte_length": 1024,
                    "parameter_direct_source_byte_length": 1024,
                    "parameter_encoded_source_byte_length": 0,
                    "parameter_gather_load_step_count": 1,
                    "parameter_encode_load_step_count": 0,
                    "parameter_load_group_count": 1,
                    "parameter_gather_load_group_count": 1,
                    "parameter_encode_load_group_count": 0,
                    "constant_slab_byte_length": 0,
                    "memory_slab_byte_length": 512,
                    "memory_slab_high_water_mark": 384,
                    "shared_tensor_byte_length": 64,
                    "boundary_tensor_byte_length": 128,
                    "diagnostic_tap_byte_length": 0,
                    "kernel_count": 35,
                    "region_count": 1,
                    "shared_tensor_count": 1,
                    "operation_count": 1571,
                    "dispatch_count": 1062,
                    "parameter_load_kind_statistics": {
                        "gather": {
                            "step_count": 1,
                            "source_byte_length": 1024,
                            "target_byte_length": 1024,
                        },
                        "encode_fp8_e4m3_scaled_to_bf16": {
                            "step_count": 0,
                            "source_byte_length": 0,
                            "target_byte_length": 0,
                        },
                        "encode_bf16_linear_rhs_tile": {
                            "step_count": 0,
                            "source_byte_length": 0,
                            "target_byte_length": 0,
                        },
                        "encode_fp8_e4m3_scaled_to_bf16_linear_rhs_tile": {
                            "step_count": 0,
                            "source_byte_length": 0,
                            "target_byte_length": 0,
                        },
                    },
                },
                "parameter_window_statistics": [
                    {
                        "region_window_size": 1,
                        "window_count": 1,
                        "full_slab_target_byte_length": 1024,
                        "peak_window_target_byte_length": 1024,
                        "peak_window_source_byte_length": 1024,
                        "total_window_target_byte_length": 1024,
                        "total_window_source_byte_length": 1024,
                        "peak_window_load_group_count": 1,
                        "total_window_load_group_count": 1,
                        "peak_window_encode_load_step_count": 0,
                        "total_window_encode_load_step_count": 0,
                        "largest_load_group_target_byte_length": 1024,
                        "largest_load_group_index": 0,
                        "largest_request_target_byte_length": 1024,
                        "largest_request_index": 0,
                        "largest_request_load_group_index": 0,
                    }
                ],
            }
        },
    }


def minimal_result_tensor(shape: dict) -> dict:
    return {
        "dtype": "f32",
        "shape": shape,
        "byte_length": 4,
        "element_count": 1,
        "finite_count": 1,
        "nan_count": 0,
        "infinity_count": 0,
        "first_nonfinite_index": None,
        "finite_min": 0.25,
        "finite_max": 0.25,
        "finite_mean": 0.25,
    }


def minimal_result_summary() -> dict:
    latent_shape = {
        "rank": 4,
        "dims": [8, 8, 128, 1],
    }
    image_shape = {
        "rank": 4,
        "dims": [128, 128, 3, 1],
    }
    return {
        "conditioned_velocity": minimal_result_tensor(latent_shape),
        "unconditioned_velocity": minimal_result_tensor(latent_shape),
        "denoised_latent": minimal_result_tensor(latent_shape),
        "final_latent": minimal_result_tensor(latent_shape),
        "decoded_image": minimal_result_tensor(image_shape),
    }


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
                    f"--request_json_file={request_path}",
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
                    f"--request_json_file={request_path}",
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
        self.assertIn("--dump_result_summary=artifacts/result_summary.json", command)
        self.assertIn("--profile_output=artifacts/profile.txt", command)
        self.assertIn("--dit_weight_execution_format=bf16_resident", command)
        self.assertIn("--qwen_weight_execution_strategy=hybrid_compact_rhs", command)
        self.assertIn("--qwen_attention_implementation=auto", command)
        self.assertIn("--dit_attention_implementation=online_wmma", command)
        self.assertIn("--dit_feed_forward_implementation=fused_product", command)
        self.assertIn("--generation_residency=issue_phases", command)
        self.assertIn("--generation_issue_mode=phases", command)
        self.assertIn("--parameter_load_prefetch_region_distance=0", command)
        self.assertIn("--vae_tiling_mode=memory_budget", command)
        self.assertIn("--vae_memory_budget=536870912", command)
        self.assertIn("--vae_overlap=0.5", command)
        self.assertIn("--device=amdgpu", command)
        self.assertIn("--parameters=qwen=qwen.safetensors", command)
        self.assertIn("--parameters=vae=vae.safetensors", command)
        self.assertEqual(command[-1], "--list_devices=false")

    def test_build_id4_command_routes_selected_policy(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=out",
                "--id4_binary=bin/id4",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
                "--qwen_weight_execution_strategy=compact_rhs",
                "--qwen_attention_implementation=materialized",
                "--dit_weight_execution_format=fp8_direct_feed_forward_bf16_resident",
                "--dit_attention_implementation=blocked_wmma",
                "--dit_feed_forward_implementation=fused_product",
            ]
        )

        command = self.smoke_test.build_id4_command(args, Path("artifacts"))

        self.assertIn(
            "--dit_weight_execution_format=fp8_direct_feed_forward_bf16_resident",
            command,
        )
        self.assertIn("--qwen_weight_execution_strategy=compact_rhs", command)
        self.assertIn("--qwen_attention_implementation=materialized", command)
        self.assertIn("--dit_attention_implementation=blocked_wmma", command)
        self.assertIn("--dit_feed_forward_implementation=fused_product", command)

    def test_build_id4_command_routes_selected_residency_policy(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=out",
                "--id4_binary=bin/id4",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
                "--generation_residency=selected_stage_bundles",
                "--generation_issue_mode=phases",
                "--generation_resident_stage_bundles=dit_conditioned",
            ]
        )

        command = self.smoke_test.build_id4_command(args, Path("artifacts"))

        self.assertIn("--generation_residency=selected_stage_bundles", command)
        self.assertIn("--generation_issue_mode=phases", command)
        self.assertIn("--generation_resident_stage_bundles=dit_conditioned", command)

    def test_build_id4_command_routes_phase_stage_residency_policy(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=out",
                "--id4_binary=bin/id4",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
                "--generation_residency=phase_stage_bundles",
                "--generation_issue_mode=phases",
            ]
        )

        command = self.smoke_test.build_id4_command(args, Path("artifacts"))

        self.assertIn("--generation_residency=phase_stage_bundles", command)
        self.assertIn("--generation_issue_mode=phases", command)

    def test_build_id4_command_routes_memory_budgeted_residency_policy(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=out",
                "--id4_binary=bin/id4",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
                "--generation_residency=memory_budgeted",
                "--generation_residency_budget=37580963840",
                "--generation_resident_stage_bundles=qwen,dit_conditioned",
            ]
        )

        command = self.smoke_test.build_id4_command(args, Path("artifacts"))

        self.assertIn("--generation_residency=memory_budgeted", command)
        self.assertIn("--generation_residency_budget=37580963840", command)
        self.assertIn(
            "--generation_resident_stage_bundles=qwen,dit_conditioned", command
        )

    def test_generation_residency_budget_requires_memory_budgeted_mode(self):
        with self.assertRaises(SystemExit):
            self.smoke_test.parse_arguments(
                [
                    "--output_dir=out",
                    "--device=amdgpu",
                    "--tokenizer=tokenizer.json",
                    "--parameters=qwen=qwen.safetensors",
                    "--generation_residency_budget=37580963840",
                ]
            )

    def test_memory_budgeted_residency_requires_candidate_stage_bundles(self):
        with self.assertRaises(SystemExit):
            self.smoke_test.parse_arguments(
                [
                    "--output_dir=out",
                    "--device=amdgpu",
                    "--tokenizer=tokenizer.json",
                    "--parameters=qwen=qwen.safetensors",
                    "--generation_residency=memory_budgeted",
                    "--generation_residency_budget=37580963840",
                ]
            )

    def test_build_id4_command_routes_stage_serial_issue_mode(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=out",
                "--id4_binary=bin/id4",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
                "--generation_issue_mode=stage_serial",
            ]
        )

        command = self.smoke_test.build_id4_command(args, Path("artifacts"))

        self.assertIn("--generation_issue_mode=stage_serial", command)

    def test_build_id4_command_routes_parameter_prefetch_distance(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=out",
                "--id4_binary=bin/id4",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
                "--parameter_load_prefetch_region_distance=2",
            ]
        )

        command = self.smoke_test.build_id4_command(args, Path("artifacts"))

        self.assertIn("--parameter_load_prefetch_region_distance=2", command)

    def test_parameter_prefetch_distance_must_be_non_negative(self):
        with self.assertRaises(SystemExit):
            self.smoke_test.parse_arguments(
                [
                    "--output_dir=out",
                    "--device=amdgpu",
                    "--tokenizer=tokenizer.json",
                    "--parameters=qwen=qwen.safetensors",
                    "--parameter_load_prefetch_region_distance=-1",
                ]
            )

    def test_build_id4_command_omits_disabled_vae_detail_flags(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=out",
                "--id4_binary=bin/id4",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
                "--vae_tiling_mode=disabled",
            ]
        )

        command = self.smoke_test.build_id4_command(args, Path("artifacts"))

        self.assertIn("--vae_tiling_mode=disabled", command)
        self.assertNotIn("--vae_memory_budget=536870912", command)
        self.assertNotIn("--vae_overlap=0.5", command)

    def test_build_id4_command_routes_explicit_vae_tiling_policy(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=out",
                "--id4_binary=bin/id4",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
                "--vae_tiling_mode=explicit_tile_size",
                "--vae_tile_size_x=32",
                "--vae_tile_size_y=24",
                "--vae_overlap=0.25",
            ]
        )

        command = self.smoke_test.build_id4_command(args, Path("artifacts"))

        self.assertIn("--vae_tiling_mode=explicit_tile_size", command)
        self.assertIn("--vae_tile_size_x=32", command)
        self.assertIn("--vae_tile_size_y=24", command)
        self.assertIn("--vae_overlap=0.25", command)

    def test_build_id4_command_routes_relative_vae_tiling_policy(self):
        args = self.smoke_test.parse_arguments(
            [
                "--output_dir=out",
                "--id4_binary=bin/id4",
                "--device=amdgpu",
                "--tokenizer=tokenizer.json",
                "--parameters=qwen=qwen.safetensors",
                "--vae_tiling_mode=relative_tile_size",
                "--vae_relative_size_x=0.5",
                "--vae_relative_size_y=0.25",
                "--vae_overlap=0.125",
            ]
        )

        command = self.smoke_test.build_id4_command(args, Path("artifacts"))

        self.assertIn("--vae_tiling_mode=relative_tile_size", command)
        self.assertIn("--vae_relative_size_x=0.5", command)
        self.assertIn("--vae_relative_size_y=0.25", command)
        self.assertIn("--vae_overlap=0.125", command)

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

    def test_generation_plan_metrics_capture_dynamic_shape_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(minimal_generation_plan()), encoding="utf-8")

            metrics = self.smoke_test.read_generation_plan_metrics(path)

            self.assertEqual(metrics["summary"]["qwen_token_count"], 37)
            self.assertEqual(metrics["summary"]["qwen_token_capacity"], 64)
            self.assertEqual(metrics["summary"]["conditioned_dit_token_count"], 101)
            self.assertEqual(metrics["summary"]["qwen_weight_execution_strategy"], 3)
            self.assertEqual(metrics["summary"]["qwen_attention_implementation"], 1)
            self.assertEqual(
                metrics["summary"]["diffusion_latent_shape"]["dims"],
                [8, 8, 128, 1],
            )
            self.assertEqual(
                metrics["residency"]["phase_parameter_high_water_mark"], 1024
            )
            self.assertEqual(metrics["residency"]["phases"][0]["stage_keys"], ["qwen"])
            self.assertEqual(
                metrics["stages"]["qwen"]["memory_slab_high_water_mark"], 384
            )
            self.assertEqual(metrics["stages"]["qwen"]["parameter_load_group_count"], 1)
            self.assertEqual(
                metrics["stages"]["qwen"]["parameter_load_kind_statistics"]["gather"][
                    "source_byte_length"
                ],
                1024,
            )
            self.assertEqual(
                metrics["stages"]["qwen"]["parameter_window_statistics"][0][
                    "largest_request_target_byte_length"
                ],
                1024,
            )
            self.assertEqual(metrics["stages"]["qwen"]["shared_tensor_count"], 1)
            self.assertEqual(metrics["stages"]["qwen"]["dispatch_count"], 1062)

    def test_generation_plan_shape_rank_must_match_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan = minimal_generation_plan()
            plan["summary"]["decoded_image_shape"]["rank"] = 3
            path.write_text(json.dumps(plan), encoding="utf-8")

            with self.assertRaisesRegex(self.smoke_test.SmokeTestError, "rank"):
                self.smoke_test.read_generation_plan_metrics(path)

    def test_profile_metrics_rank_dispatch_and_queue_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.txt"
            path.write_text(
                "\n".join(
                    [
                        "IREE HAL device statistics:",
                        (
                            "  host_queue p=0 q=0 copy  count=2 total=- "
                            "avg=- invalid=2 operations=2 payload=64B"
                        ),
                        (
                            "  device_queue p=0 q=0 copy  count=2 "
                            "total=100 ticks avg=50 ticks operations=2 "
                            "payload=64B"
                        ),
                        (
                            "  device_queue p=0 q=0 execute  count=1 "
                            "total=300 ticks avg=300 ticks operations=4"
                        ),
                        (
                            "  device_queue p=0 q=0 barrier  count=1 "
                            "total=5 ticks avg=5 ticks"
                        ),
                        "  dispatch kernel_b  count=3 total=900 ticks avg=300 ticks",
                        "  dispatch kernel_a  count=1 total=100 ticks avg=100 ticks",
                        "  dispatch kernel_b  count=2 total=600 ticks avg=300 ticks",
                    ]
                ),
                encoding="utf-8",
            )

            metrics = self.smoke_test.read_profile_metrics(path)

            self.assertEqual(metrics["dispatches"]["row_count"], 3)
            self.assertEqual(metrics["dispatches"]["function_count"], 2)
            self.assertEqual(metrics["dispatches"]["total_count"], 6)
            self.assertEqual(
                metrics["dispatches"]["top_by_total_ticks"][0],
                {
                    "name": "kernel_b",
                    "count": 5,
                    "total_ticks": 1500,
                    "average_ticks": 300,
                },
            )
            self.assertEqual(metrics["queue_operations"]["row_count"], 4)
            self.assertEqual(metrics["queue_operations"]["timed_row_count"], 3)
            self.assertEqual(metrics["queue_operations"]["invalid_row_count"], 1)
            self.assertEqual(
                metrics["queue_operations"]["top_by_total_ticks"][0]["operation"],
                "execute",
            )
            self.assertEqual(metrics["queue_operations"]["operation_count"], 7)
            self.assertEqual(metrics["unknown_timed_row_count"], 0)

    def test_result_summary_metrics_require_finite_required_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result_summary.json"
            path.write_text(json.dumps(minimal_result_summary()), encoding="utf-8")

            metrics = self.smoke_test.read_result_summary_metrics(path)
            self.smoke_test.validate_result_summary(metrics)

            self.assertEqual(metrics["decoded_image"]["dtype"], "f32")
            self.assertEqual(metrics["decoded_image"]["finite_count"], 1)
            self.assertEqual(metrics["decoded_image"]["nan_count"], 0)

    def test_result_summary_validation_rejects_nonfinite_tensor(self):
        result_summary = minimal_result_summary()
        decoded_image = result_summary["decoded_image"]
        decoded_image["finite_count"] = 0
        decoded_image["nan_count"] = 1
        decoded_image["first_nonfinite_index"] = 0
        decoded_image["finite_min"] = None
        decoded_image["finite_max"] = None
        decoded_image["finite_mean"] = None

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result_summary.json"
            path.write_text(json.dumps(result_summary), encoding="utf-8")

            metrics = self.smoke_test.read_result_summary_metrics(path)
            with self.assertRaisesRegex(
                self.smoke_test.SmokeTestError, "decoded_image contains nonfinite"
            ):
                self.smoke_test.validate_result_summary(metrics)

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
            plan_payload = json.dumps(minimal_generation_plan())
            result_summary_payload = json.dumps(minimal_result_summary())
            fake_id4.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "from pathlib import Path",
                        "import sys",
                        "output = None",
                        "plan = None",
                        "profile = None",
                        "result_summary = None",
                        "for arg in sys.argv[1:]:",
                        "    if arg.startswith('--output='):",
                        "        output = Path(arg.split('=', 1)[1])",
                        "    if arg.startswith('--dump_plan='):",
                        "        plan = Path(arg.split('=', 1)[1])",
                        "    if arg.startswith('--profile_output='):",
                        "        profile = Path(arg.split('=', 1)[1])",
                        "    if arg.startswith('--dump_result_summary='):",
                        "        result_summary = Path(arg.split('=', 1)[1])",
                        "if output is None:",
                        "    raise SystemExit(2)",
                        "if plan is None:",
                        "    raise SystemExit(2)",
                        "if profile is None:",
                        "    raise SystemExit(2)",
                        "if result_summary is None:",
                        "    raise SystemExit(2)",
                        f"plan.write_text({plan_payload!r}, encoding='utf-8')",
                        (
                            "result_summary.write_text("
                            f"{result_summary_payload!r}, encoding='utf-8')"
                        ),
                        (
                            "profile.write_text('  dispatch fake_kernel  count=1 "
                            "total=1 ticks avg=1 ticks\\n', encoding='utf-8')"
                        ),
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
            self.assertEqual(summary["plan_metrics"]["summary"]["qwen_token_count"], 37)
            self.assertEqual(summary["profile_metrics"]["dispatches"]["total_count"], 1)
            self.assertEqual(
                summary["result_summary_metrics"]["decoded_image"]["finite_count"],
                1,
            )
            self.assertEqual(summary["image_metrics"]["width"], 16)


if __name__ == "__main__":
    unittest.main()
