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

MIB = 1024 * 1024

TEST_BENCHMARK_LABEL = (
    "scope=prepared_issue prompt=short128 qwen_tokens=19 qwen_capacity=32 "
    "image_tokens=64 dit_cond_tokens=83 dit_cond_capacity=128 "
    "dit_uncond_tokens=64 dit_uncond_capacity=128 latent=8x8 steps=1 "
    "image=128x128 residency_request=memory_budgeted "
    "residency=selected_stage_bundles issue=stage_serial "
    "prefetch_regions=1 resident_stage_mask=0x00000003 "
    "residency_budget=35840MiB params=fp8_e4m3 "
    "activation=bf16_linear_input weights=bf16_resident "
    "qwen_weights=hybrid_compact_rhs attention=online_wmma "
    "ff=fused_product param_total=49460MiB "
    "param_largest=17699MiB param_source=32017MiB "
    "param_source_direct=4169MiB param_source_encoded=27848MiB "
    "param_load_steps[gather=322,encode=529] "
    "param_load_groups[total=851,gather=322,encode=529] "
    "local_hw_total=43MiB "
    "local_hw_largest=20MiB boundary=9MiB kernels=119 dispatches=1507 "
    "logical_live[boundary=5MiB,taps=0MiB,resident=0MiB,"
    "phase_peak=34951MiB,stage_serial_peak=17712MiB,"
    "selected_peak=17712MiB] "
    "prefetch_groups[count=142,avg_regions=1.00,max_regions=1] "
    "direct_gather_groups[count=322,requests=384,source=4169MiB,"
    "target=4169MiB,max=256MiB] "
    "issue_encode_window[count=2,staging=1081MiB,max=576MiB,"
    "source=18991MiB,target=27597MiB,chunks=72,batches=72,"
    "dispatches=318] "
    "stage.qwen[param=14436MiB,src=14436MiB,src_direct=4068MiB,"
    "src_encoded=10368MiB,load_steps=36/108,load_groups=36/108,"
    "local_hw=4MiB,boundary=4MiB,kernels=28,dispatches=485] "
    "stage.decode[param=95MiB,src=95MiB,src_direct=95MiB,"
    "src_encoded=0MiB,load_steps=1/0,load_groups=1/0,local_hw=20MiB,"
    "boundary=1MiB,kernels=37,dispatches=106] "
    "timing_ms[plan=0.000,prepare=0.000,issue=3477.614,begin=0.000,"
    "final_wait=150.049,total=3627.663]"
)


def stage_statistics(
    *,
    parameter_mib: int,
    source_mib: int,
    source_direct_mib: int,
    source_encoded_mib: int,
    gather_loads: int,
    encode_loads: int,
    local_high_water_mib: int,
    boundary_mib: int,
    kernels: int,
    dispatches: int,
) -> dict:
    return {
        "statistics": {
            "parameter_slab_byte_length": parameter_mib * MIB,
            "largest_parameter_slab_byte_length": parameter_mib * MIB,
            "parameter_source_byte_length": source_mib * MIB,
            "parameter_direct_source_byte_length": source_direct_mib * MIB,
            "parameter_encoded_source_byte_length": source_encoded_mib * MIB,
            "parameter_gather_load_step_count": gather_loads,
            "parameter_encode_load_step_count": encode_loads,
            "parameter_load_group_count": gather_loads + encode_loads,
            "parameter_gather_load_group_count": gather_loads,
            "parameter_encode_load_group_count": encode_loads,
            "constant_slab_byte_length": 0,
            "memory_slab_byte_length": local_high_water_mib * MIB,
            "memory_slab_high_water_mark": local_high_water_mib * MIB,
            "shared_tensor_byte_length": 0,
            "boundary_tensor_byte_length": boundary_mib * MIB,
            "diagnostic_tap_byte_length": 0,
            "kernel_count": kernels,
            "region_count": 1,
            "shared_tensor_count": 0,
            "operation_count": dispatches,
            "dispatch_count": dispatches,
        }
    }


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
            "qwen_weight_execution_strategy": 3,
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
                    "parameter_load_group_count": 1,
                    "parameter_gather_load_group_count": 1,
                    "parameter_encode_load_group_count": 0,
                    "constant_slab_byte_length": 0,
                    "memory_slab_byte_length": 256,
                    "memory_slab_high_water_mark": 128,
                    "shared_tensor_byte_length": 0,
                    "boundary_tensor_byte_length": 64,
                    "diagnostic_tap_byte_length": 0,
                    "kernel_count": 37,
                    "region_count": 1,
                    "shared_tensor_count": 0,
                    "operation_count": 210,
                    "dispatch_count": 106,
                }
            },
        },
    }


def live_label_generation_plan() -> dict:
    plan = minimal_generation_plan(19)
    plan["residency"]["total_stage_parameter_byte_length"] = 49460 * MIB
    plan["residency"]["phase_parameter_high_water_mark"] = 34951 * MIB
    plan["residency"]["largest_stage_parameter_byte_length"] = 17699 * MIB
    plan["residency"]["total_stage_boundary_byte_length"] = 9 * MIB
    plan["stages"] = {
        "sampler_noise": stage_statistics(
            parameter_mib=0,
            source_mib=0,
            source_direct_mib=0,
            source_encoded_mib=0,
            gather_loads=0,
            encode_loads=0,
            local_high_water_mib=0,
            boundary_mib=1,
            kernels=1,
            dispatches=1,
        ),
        "qwen": stage_statistics(
            parameter_mib=14436,
            source_mib=14436,
            source_direct_mib=4068,
            source_encoded_mib=10368,
            gather_loads=36,
            encode_loads=108,
            local_high_water_mib=4,
            boundary_mib=4,
            kernels=28,
            dispatches=485,
        ),
        "dit_conditioned": stage_statistics(
            parameter_mib=17699,
            source_mib=8860,
            source_direct_mib=3,
            source_encoded_mib=8857,
            gather_loads=143,
            encode_loads=211,
            local_high_water_mib=9,
            boundary_mib=5,
            kernels=27,
            dispatches=458,
        ),
        "dit_unconditioned": stage_statistics(
            parameter_mib=17231,
            source_mib=8626,
            source_direct_mib=3,
            source_encoded_mib=8623,
            gather_loads=142,
            encode_loads=210,
            local_high_water_mib=9,
            boundary_mib=1,
            kernels=24,
            dispatches=455,
        ),
        "sampler_denoise": stage_statistics(
            parameter_mib=0,
            source_mib=0,
            source_direct_mib=0,
            source_encoded_mib=0,
            gather_loads=0,
            encode_loads=0,
            local_high_water_mib=1,
            boundary_mib=1,
            kernels=2,
            dispatches=2,
        ),
        "decode": stage_statistics(
            parameter_mib=95,
            source_mib=95,
            source_direct_mib=95,
            source_encoded_mib=0,
            gather_loads=1,
            encode_loads=0,
            local_high_water_mib=20,
            boundary_mib=1,
            kernels=37,
            dispatches=106,
        ),
    }
    return plan


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
            self.assertEqual(summary["rows"][0]["dit_activation_format"], 2)
            self.assertEqual(summary["rows"][0]["dit_weight_execution_format"], 1)
            self.assertEqual(summary["rows"][0]["qwen_weight_execution_strategy"], 3)
            self.assertEqual(summary["rows"][0]["dit_attention_implementation"], 4)
            self.assertEqual(summary["rows"][0]["dit_feed_forward_implementation"], 2)
            self.assertEqual(summary["rows"][0]["dispatch_count"], 1168)
            self.assertEqual(summary["rows"][0]["local_high_water_mark"], 512)

    def test_summarize_generation_benchmark_parses_live_label_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = root / "plans"
            plan_dir.mkdir()
            (plan_dir / "short128.json").write_text(
                json.dumps(live_label_generation_plan()), encoding="utf-8"
            )
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "name": (
                                    "BM_Ideogram4SessionGenerationIssuePrepared/"
                                    "short128/real_time"
                                ),
                                "real_time": 3627.692699432373,
                                "cpu_time": 10.391671999999907,
                                "iterations": 1,
                                "label": TEST_BENCHMARK_LABEL,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = generation_benchmark_summary.summarize_generation_benchmark(
                benchmark_path, plan_dir
            )

            row = summary["rows"][0]
            self.assertEqual(row["benchmark_scope"], "prepared_issue")
            self.assertEqual(row["prompt_label"], "short128")
            self.assertEqual(row["generation_issue_mode"], "stage_serial")
            self.assertEqual(row["generation_residency_request"], "memory_budgeted")
            self.assertEqual(row["generation_residency"], "selected_stage_bundles")
            self.assertEqual(row["parameter_load_prefetch_region_distance"], 1)
            self.assertEqual(row["generation_residency_budget_mib"], 35840)
            self.assertEqual(
                row["runtime_qwen_weight_execution_strategy"], "hybrid_compact_rhs"
            )
            self.assertEqual(row["runtime_parameter_total_mib"], 49460)
            self.assertEqual(row["runtime_parameter_source_encoded_mib"], 27848)
            self.assertEqual(row["runtime_parameter_gather_load_step_count"], 322)
            self.assertEqual(row["runtime_parameter_load_group_count"], 851)
            self.assertEqual(row["runtime_parameter_gather_load_group_count"], 322)
            self.assertEqual(row["runtime_parameter_encode_load_group_count"], 529)
            self.assertEqual(row["runtime_dispatch_count"], 1507)
            self.assertEqual(row["logical_live_phase_peak_mib"], 34951)
            self.assertEqual(row["logical_live_stage_serial_peak_mib"], 17712)
            self.assertEqual(row["logical_live_selected_peak_mib"], 17712)
            self.assertEqual(row["prefetch_group_submit_count"], 142)
            self.assertAlmostEqual(row["prefetch_group_average_region_distance"], 1.0)
            self.assertEqual(row["prefetch_group_max_region_distance"], 1)
            self.assertEqual(row["direct_gather_group_count"], 322)
            self.assertEqual(row["direct_gather_request_count"], 384)
            self.assertEqual(row["direct_gather_source_mib"], 4169)
            self.assertEqual(row["direct_gather_target_mib"], 4169)
            self.assertEqual(row["direct_gather_max_mib"], 256)
            self.assertEqual(row["issue_encode_window_count"], 2)
            self.assertEqual(row["issue_encode_window_staging_mib"], 1081)
            self.assertEqual(row["issue_encode_window_dispatch_count"], 318)
            self.assertEqual(row["runtime_stages"]["qwen"]["source_encoded_mib"], 10368)
            self.assertEqual(
                row["runtime_stages"]["qwen"]["parameter_gather_load_group_count"], 36
            )
            self.assertEqual(row["runtime_stages"]["decode"]["dispatch_count"], 106)
            self.assertAlmostEqual(row["timing_issue_ms"], 3477.614)
            self.assertAlmostEqual(row["timing_final_wait_ms"], 150.049)

    def test_summarize_generation_benchmark_rejects_malformed_live_label(self):
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
                                    "BM_Ideogram4SessionGenerationIssuePrepared/"
                                    "short128/real_time"
                                ),
                                "real_time": 1.0,
                                "cpu_time": 1.0,
                                "iterations": 1,
                                "label": "scope=prepared_issue",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(
                generation_benchmark_summary.GenerationBenchmarkSummaryError
            ):
                generation_benchmark_summary.summarize_generation_benchmark(
                    benchmark_path, plan_dir
                )

    def test_summarize_generation_benchmark_rejects_stale_plan_join(self):
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
                                    "BM_Ideogram4SessionGenerationIssuePrepared/"
                                    "short128/real_time"
                                ),
                                "real_time": 1.0,
                                "cpu_time": 1.0,
                                "iterations": 1,
                                "label": TEST_BENCHMARK_LABEL,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(
                generation_benchmark_summary.GenerationBenchmarkSummaryError
            ):
                generation_benchmark_summary.summarize_generation_benchmark(
                    benchmark_path, plan_dir
                )

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
