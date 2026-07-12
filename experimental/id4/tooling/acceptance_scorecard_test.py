#!/usr/bin/env python3
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

import acceptance_scorecard


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _parameter_statistics(
    direct_byte_length: int, transformed_byte_length: int
) -> dict[str, object]:
    return {
        "gather": {
            "step_count": 1 if direct_byte_length else 0,
            "source_byte_length": direct_byte_length,
            "target_byte_length": direct_byte_length,
        },
        "encode_fp8_e4m3_linear_rhs_tile": {
            "step_count": 1 if transformed_byte_length else 0,
            "source_byte_length": transformed_byte_length,
            "target_byte_length": transformed_byte_length,
        },
    }


def _stage(
    dispatch_count: int,
    direct_byte_length: int,
    transformed_byte_length: int,
    local_byte_length: int,
) -> dict[str, object]:
    regions = [
        {
            "statistics": {
                "local_slab_byte_length": local_byte_length,
            }
        }
    ]
    memory_slabs = []
    if local_byte_length:
        memory_slabs.append(
            {
                "scope": "region_local",
                "byte_length": local_byte_length,
            }
        )
    return {
        "statistics": {
            "dispatch_count": dispatch_count,
            "memory_slab_high_water_mark": local_byte_length,
            "parameter_load_kind_statistics": _parameter_statistics(
                direct_byte_length, transformed_byte_length
            ),
        },
        "regions": regions,
        "memory_slabs": memory_slabs,
    }


class AcceptanceScorecardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self._write_fixture()

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def _write_fixture(self) -> None:
        label = " ".join(
            [
                "scope=prepared_issue",
                "qwen_tokens=3",
                "image_tokens=4",
                "dit_cond_tokens=7",
                "dit_uncond_tokens=4",
                "latent=2x2",
                "steps=1",
                "image=32x32",
                "residency=all_stage_bundles",
                "issue=stage_serial",
                "parameter_source=execution_layout",
                "params=fp8_e4m3",
                "activation=bf16_linear_input",
                "weights=fp8_compact_rhs",
                "qwen_params=fp8_e4m3_block_scaled",
                "qwen_weights=compact_rhs",
                "qwen_attention=auto",
                "attention=materialized_wmma",
                "ff=pytorch_parity",
                "param_total=1MiB",
                "local_hw_largest=1MiB",
                "dispatches=3",
                "logical_live[selected_peak=1MiB]",
            ]
        )
        _write_json(
            self.root / "id4.json",
            {
                "benchmarks": [
                    {
                        "name": "BM_Request/short/real_time",
                        "real_time": 8.0,
                        "time_unit": "ms",
                        "label": label,
                    }
                ]
            },
        )
        _write_json(
            self.root / "plan.json",
            {
                "kind": "ideogram4_generation",
                "summary": {
                    "qwen_token_count": 3,
                    "image_token_count": 4,
                    "conditioned_dit_token_count": 7,
                    "unconditioned_dit_token_count": 4,
                    "denoise_step_count": 1,
                    "diffusion_latent_shape": {
                        "rank": 4,
                        "dims": [2, 2, 128, 1],
                    },
                    "decoded_image_shape": {"rank": 4, "dims": [32, 32, 3, 1]},
                },
                "execution_resources": {
                    "parameter_source_kind": "execution_layout",
                    "resident_stage_parameter_byte_length": 120,
                    "boundary_buffer_byte_length": 32,
                    "stage_serial_local_peak_byte_length": 64,
                    "stage_serial_total_peak_byte_length": 216,
                },
                "residency": {
                    "phases": [
                        {
                            "stage_keys": ["qwen"],
                            "repeated_per_denoise_step": False,
                        },
                        {
                            "stage_keys": ["decode"],
                            "repeated_per_denoise_step": False,
                        },
                    ]
                },
                "stages": {
                    "qwen": _stage(2, 20, 100, 64),
                    "decode": _stage(1, 0, 0, 0),
                },
            },
        )
        _write_json(
            self.root / "qwen.json",
            {
                "token_count": 3,
                "attention_mask_shape": [3, 3],
                "condition_shape": [16, 3],
                "median_ms": 2.0,
                "dtype_policy": {
                    "model_dtype": "torch.bfloat16",
                    "condition_output_dtype": "torch.float32",
                },
            },
        )
        _write_json(
            self.root / "dit.json",
            {
                "dtype": "bf16",
                "branches": {
                    "cond": {
                        "median_ms": 3.0,
                        "parameters": {
                            "policy": "weights",
                            "state_dict_kind": "fp8",
                        },
                        "shape": {
                            "image_height": 32,
                            "image_width": 32,
                            "latent_height": 2,
                            "latent_width": 2,
                            "image_token_count": 4,
                            "text_token_count": 3,
                            "total_token_count": 7,
                        },
                    },
                    "uncond": {
                        "median_ms": 4.0,
                        "parameters": {
                            "policy": "weights",
                            "state_dict_kind": "fp8",
                        },
                        "shape": {
                            "image_height": 32,
                            "image_width": 32,
                            "latent_height": 2,
                            "latent_width": 2,
                            "image_token_count": 4,
                            "text_token_count": 0,
                            "total_token_count": 4,
                        },
                    },
                },
            },
        )
        _write_json(
            self.root / "vae.json",
            {
                "latent_shape": [2, 2, 128, 1],
                "decoded_shape_bchw": [1, 3, 32, 32],
                "median_ms": 1.0,
                "dtype_policy": {
                    "decoder_activations": "torch.bfloat16",
                    "decoder_weights": "torch.bfloat16",
                },
            },
        )
        _write_json(
            self.root / "fixture_compare.json",
            {
                "comparisons": [
                    {
                        "dtype": "f32",
                        "name": "condition",
                        "shape": [16, 3],
                        "stage": "qwen.encoder",
                        "status": "pass",
                    }
                ],
                "summary": {
                    "compared_count": 1,
                    "expected_count": 1,
                    "passed_count": 1,
                    "failed_count": 0,
                },
            },
        )
        _write_json(
            self.root / "metrics.json",
            {
                "missing_count": 0,
                "records": [
                    {
                        "name": "velocity",
                        "shape": [2, 2, 128, 1],
                        "mean_abs": 0.01,
                        "p99_abs": 0.02,
                        "max_abs": 0.04,
                    }
                ],
            },
        )
        _write_json(
            self.root / "manifest.json",
            {
                "schema_version": 1,
                "maximum_ratio": 1.0,
                "policy": {
                    "id4_label": {
                        "scope": "prepared_issue",
                        "residency": "all_stage_bundles",
                        "issue": "stage_serial",
                        "parameter_source": "execution_layout",
                        "params": "fp8_e4m3",
                        "activation": "bf16_linear_input",
                        "weights": "fp8_compact_rhs",
                        "qwen_params": "fp8_e4m3_block_scaled",
                        "qwen_weights": "compact_rhs",
                        "qwen_attention": "auto",
                        "attention": "materialized_wmma",
                        "ff": "pytorch_parity",
                    },
                    "pytorch": {
                        "qwen": {
                            "dtype_policy": {
                                "model_dtype": "torch.bfloat16",
                                "condition_output_dtype": "torch.float32",
                            }
                        },
                        "dit": {
                            "dtype": "bf16",
                            "branch_parameters": {
                                "policy": "weights",
                                "state_dict_kind": "fp8",
                            },
                        },
                        "vae": {
                            "dtype_policy": {
                                "decoder_activations": "torch.bfloat16",
                                "decoder_weights": "torch.bfloat16",
                            }
                        },
                    },
                },
                "rows": [
                    {
                        "name": "short32",
                        "request": {
                            "qwen_token_count": 3,
                            "latent_width": 2,
                            "latent_height": 2,
                            "image_width": 32,
                            "image_height": 32,
                            "denoise_step_count": 1,
                        },
                        "id4": {
                            "benchmark": {
                                "file": "id4.json",
                                "name": "BM_Request/short/real_time",
                            },
                            "plan": "plan.json",
                        },
                        "pytorch": {
                            "qwen": "qwen.json",
                            "dit_conditioned": {
                                "file": "dit.json",
                                "branch": "cond",
                            },
                            "dit_unconditioned": {
                                "file": "dit.json",
                                "branch": "uncond",
                            },
                            "vae": "vae.json",
                        },
                        "correctness": [
                            {
                                "name": "qwen",
                                "kind": "fixture_compare",
                                "file": "fixture_compare.json",
                                "comparison": "condition",
                                "expect": {
                                    "dtype": "f32",
                                    "shape": [16, 3],
                                    "stage": "qwen.encoder",
                                },
                            },
                            {
                                "name": "dit_conditioned",
                                "kind": "aggregate_metrics",
                                "file": "metrics.json",
                                "record": "velocity",
                                "expect": {
                                    "shape": [2, 2, 128, 1],
                                },
                                "limits": {
                                    "mean_abs": 0.015625,
                                    "p99_abs": 0.0625,
                                    "max_abs": 0.125,
                                },
                            },
                        ],
                    }
                ],
            },
        )

    def test_builds_strict_scorecard(self) -> None:
        scorecard = acceptance_scorecard.build_scorecard(self.root / "manifest.json")

        self.assertEqual(scorecard["status"], "pass")
        row = scorecard["rows"][0]
        self.assertEqual(row["timing"]["id4_ms"], 8.0)
        self.assertEqual(row["timing"]["pytorch_ms"], 10.0)
        self.assertAlmostEqual(row["timing"]["ratio"], 0.8)
        self.assertEqual(row["memory"]["resident_parameter_byte_length"], 120)
        self.assertEqual(row["memory"]["transformed_persistent_byte_length"], 100)
        self.assertEqual(row["topology"]["stage_command_buffer_count"], 2)
        self.assertEqual(row["topology"]["stage_queue_operation_count"], 4)
        markdown = acceptance_scorecard.format_markdown(scorecard)
        self.assertIn("| short32 | 3 | 32x32 | 8.000 ms |", markdown)

    def test_rejects_pytorch_request_mismatch(self) -> None:
        qwen = json.loads((self.root / "qwen.json").read_text(encoding="utf-8"))
        qwen["token_count"] = 4
        _write_json(self.root / "qwen.json", qwen)

        with self.assertRaisesRegex(
            acceptance_scorecard.AcceptanceScorecardError,
            "PyTorch Qwen token_count mismatch",
        ):
            acceptance_scorecard.build_scorecard(self.root / "manifest.json")

    def test_rejects_id4_policy_mismatch(self) -> None:
        benchmark = json.loads((self.root / "id4.json").read_text(encoding="utf-8"))
        benchmark["benchmarks"][0]["label"] = benchmark["benchmarks"][0][
            "label"
        ].replace("weights=fp8_compact_rhs", "weights=bf16_resident")
        _write_json(self.root / "id4.json", benchmark)

        with self.assertRaisesRegex(
            acceptance_scorecard.AcceptanceScorecardError,
            "benchmark label weights mismatch",
        ):
            acceptance_scorecard.build_scorecard(self.root / "manifest.json")

    def test_reports_performance_and_correctness_failures(self) -> None:
        benchmark = json.loads((self.root / "id4.json").read_text(encoding="utf-8"))
        benchmark["benchmarks"][0]["real_time"] = 12.0
        _write_json(self.root / "id4.json", benchmark)
        metrics = json.loads((self.root / "metrics.json").read_text(encoding="utf-8"))
        metrics["records"][0]["mean_abs"] = 0.25
        _write_json(self.root / "metrics.json", metrics)

        scorecard = acceptance_scorecard.build_scorecard(self.root / "manifest.json")

        row = scorecard["rows"][0]
        self.assertEqual(scorecard["status"], "fail")
        self.assertEqual(row["timing"]["status"], "fail")
        self.assertEqual(row["correctness"]["status"], "fail")

    def test_scales_repeated_phase_for_multiple_steps(self) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        manifest["rows"][0]["request"]["denoise_step_count"] = 2
        _write_json(self.root / "manifest.json", manifest)
        benchmark = json.loads((self.root / "id4.json").read_text(encoding="utf-8"))
        benchmark["benchmarks"][0]["label"] = benchmark["benchmarks"][0][
            "label"
        ].replace("steps=1", "steps=2")
        _write_json(self.root / "id4.json", benchmark)
        plan = json.loads((self.root / "plan.json").read_text(encoding="utf-8"))
        plan["summary"]["denoise_step_count"] = 2
        plan["residency"]["phases"][0]["repeated_per_denoise_step"] = True
        _write_json(self.root / "plan.json", plan)

        scorecard = acceptance_scorecard.build_scorecard(self.root / "manifest.json")

        row = scorecard["rows"][0]
        self.assertEqual(row["timing"]["pytorch_ms"], 17.0)
        self.assertEqual(row["topology"]["stage_invocation_count"], 3)
        self.assertEqual(row["topology"]["dispatch_count"], 5)
        self.assertEqual(row["topology"]["stage_queue_operation_count"], 7)

    def test_rejects_correctness_shape_mismatch(self) -> None:
        report = json.loads(
            (self.root / "fixture_compare.json").read_text(encoding="utf-8")
        )
        report["comparisons"][0]["shape"] = [16, 4]
        _write_json(self.root / "fixture_compare.json", report)

        with self.assertRaisesRegex(
            acceptance_scorecard.AcceptanceScorecardError,
            "correctness report qwen.condition.shape mismatch",
        ):
            acceptance_scorecard.build_scorecard(self.root / "manifest.json")

    def test_rejects_negative_plan_accounting(self) -> None:
        plan = json.loads((self.root / "plan.json").read_text(encoding="utf-8"))
        plan["execution_resources"]["boundary_buffer_byte_length"] = -1
        _write_json(self.root / "plan.json", plan)

        with self.assertRaisesRegex(
            acceptance_scorecard.AcceptanceScorecardError,
            "boundary_buffer_byte_length must be non-negative",
        ):
            acceptance_scorecard.build_scorecard(self.root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
