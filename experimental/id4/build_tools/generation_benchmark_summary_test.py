# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import generation_benchmark_summary


def minimal_generation_plan(qwen_token_count: int) -> dict:
    return {
        "kind": "ideogram4_generation",
        "summary": {
            "qwen_token_count": qwen_token_count,
            "qwen_token_capacity": 64,
            "image_token_count": 64,
            "conditioned_dit_token_count": qwen_token_count + 64,
            "conditioned_dit_token_capacity": 128,
            "unconditioned_dit_token_count": 64,
            "unconditioned_dit_token_capacity": 128,
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
            "total_stage_parameter_byte_length": 4096,
            "phase_parameter_high_water_mark": 2048,
            "largest_stage_parameter_byte_length": 2048,
            "total_stage_boundary_byte_length": 1024,
            "phases": [
                {
                    "name": "conditioning",
                    "stage_keys": ["qwen"],
                    "repeated_per_denoise_step": False,
                    "parameter_byte_length": 2048,
                    "largest_stage_parameter_byte_length": 2048,
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
                    "parameter_slab_byte_length": 2048,
                    "largest_parameter_slab_byte_length": 2048,
                    "parameter_source_byte_length": 2048,
                    "parameter_direct_source_byte_length": 2048,
                    "parameter_encoded_source_byte_length": 0,
                    "parameter_gather_load_step_count": 1,
                    "parameter_encode_load_step_count": 0,
                    "constant_slab_byte_length": 0,
                    "memory_slab_byte_length": 512,
                    "memory_slab_high_water_mark": 384,
                    "boundary_tensor_byte_length": 128,
                    "diagnostic_tap_byte_length": 0,
                    "kernel_count": 35,
                    "region_count": 1,
                    "operation_count": 1571,
                    "dispatch_count": 1062,
                }
            },
            "decode": {
                "statistics": {
                    "parameter_slab_byte_length": 95,
                    "largest_parameter_slab_byte_length": 95,
                    "parameter_source_byte_length": 95,
                    "parameter_direct_source_byte_length": 95,
                    "parameter_encoded_source_byte_length": 0,
                    "parameter_gather_load_step_count": 1,
                    "parameter_encode_load_step_count": 0,
                    "constant_slab_byte_length": 0,
                    "memory_slab_byte_length": 256,
                    "memory_slab_high_water_mark": 128,
                    "boundary_tensor_byte_length": 64,
                    "diagnostic_tap_byte_length": 0,
                    "kernel_count": 37,
                    "region_count": 1,
                    "operation_count": 210,
                    "dispatch_count": 106,
                }
            },
        },
    }


class GenerationBenchmarkSummaryTest(unittest.TestCase):
    def test_summarize_generation_benchmark_joins_rows_to_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = root / "plans"
            plan_dir.mkdir()
            (plan_dir / "short128.json").write_text(
                json.dumps(minimal_generation_plan(19)), encoding="utf-8"
            )
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "name": (
                                    "BM_Ideogram4SessionGenerationEndToEnd/"
                                    "short128/real_time"
                                ),
                                "real_time": 12.5,
                                "cpu_time": 2.5,
                                "iterations": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = generation_benchmark_summary.summarize_generation_benchmark(
                benchmark_path, plan_dir
            )

            self.assertEqual(summary["rows"][0]["bucket"], "short128")
            self.assertEqual(summary["rows"][0]["real_time_ms"], 12.5)
            self.assertEqual(summary["rows"][0]["qwen_token_count"], 19)
            self.assertEqual(summary["rows"][0]["conditioned_dit_token_count"], 83)
            self.assertEqual(summary["rows"][0]["dispatch_count"], 1168)
            self.assertEqual(summary["rows"][0]["local_high_water_mark"], 512)

    def test_summarize_generation_benchmark_requires_matching_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = root / "plans"
            plan_dir.mkdir()
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "name": (
                                    "BM_Ideogram4SessionGenerationEndToEnd/"
                                    "missing128/real_time"
                                ),
                                "real_time": 12.5,
                                "cpu_time": 2.5,
                                "iterations": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(OSError):
                generation_benchmark_summary.summarize_generation_benchmark(
                    benchmark_path, plan_dir
                )


if __name__ == "__main__":
    unittest.main()
