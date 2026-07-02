#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Summarizes ID4 generation benchmark rows with their plan metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import smoke_test


class GenerationBenchmarkSummaryError(ValueError):
    """Raised when benchmark summary inputs are missing or malformed."""


_LABEL_LOAD_STEPS_PATTERN = re.compile(
    r"\bparam_load_steps\[gather=(?P<gather>[0-9]+),"
    r"encode=(?P<encode>[0-9]+)\]"
)
_LABEL_LOAD_GROUPS_PATTERN = re.compile(
    r"\bparam_load_groups\[total=(?P<total>[0-9]+),"
    r"gather=(?P<gather>[0-9]+),encode=(?P<encode>[0-9]+)\]"
)
_LABEL_LOAD_SUBMIT_PATTERN = re.compile(
    r"\bparam_load_submit\[count=(?P<count>[0-9]+),"
    r"gather=(?P<gather>[0-9]+),encode=(?P<encode>[0-9]+),"
    r"total_ms=(?P<total_ms>[0-9]+(?:\.[0-9]+)?),"
    r"gather_ms=(?P<gather_ms>[0-9]+(?:\.[0-9]+)?),"
    r"encode_ms=(?P<encode_ms>[0-9]+(?:\.[0-9]+)?),"
    r"max_ms=(?P<max_ms>[0-9]+(?:\.[0-9]+)?)\]"
)
_LABEL_LOAD_KIND_PATTERN = re.compile(
    r"\bparam_load_kind\["
    r"gather_steps=(?P<gather_steps>[0-9]+),"
    r"gather_source=(?P<gather_source>[0-9]+)MiB,"
    r"gather_target=(?P<gather_target>[0-9]+)MiB,"
    r"fp8_bf16_steps=(?P<fp8_bf16_steps>[0-9]+),"
    r"fp8_bf16_source=(?P<fp8_bf16_source>[0-9]+)MiB,"
    r"fp8_bf16_target=(?P<fp8_bf16_target>[0-9]+)MiB,"
    r"bf16_rhs_steps=(?P<bf16_rhs_steps>[0-9]+),"
    r"bf16_rhs_source=(?P<bf16_rhs_source>[0-9]+)MiB,"
    r"bf16_rhs_target=(?P<bf16_rhs_target>[0-9]+)MiB,"
    r"fp8_bf16_rhs_steps=(?P<fp8_bf16_rhs_steps>[0-9]+),"
    r"fp8_bf16_rhs_source=(?P<fp8_bf16_rhs_source>[0-9]+)MiB,"
    r"fp8_bf16_rhs_target=(?P<fp8_bf16_rhs_target>[0-9]+)MiB\]"
)
_LABEL_LOGICAL_LIVE_PATTERN = re.compile(
    r"\blogical_live\[boundary=(?P<boundary>[0-9]+)MiB,"
    r"taps=(?P<taps>[0-9]+)MiB,"
    r"resident=(?P<resident>[0-9]+)MiB,"
    r"phase_peak=(?P<phase_peak>[0-9]+)MiB,"
    r"stage_serial_peak=(?P<stage_serial_peak>[0-9]+)MiB,"
    r"selected_peak=(?P<selected_peak>[0-9]+)MiB\]"
)
_LABEL_STAGE_PATTERN = re.compile(
    r"\bstage\.(?P<name>[A-Za-z0-9_]+)\["
    r"param=(?P<parameter>[0-9]+)MiB,"
    r"src=(?P<source>[0-9]+)MiB,"
    r"src_direct=(?P<source_direct>[0-9]+)MiB,"
    r"src_encoded=(?P<source_encoded>[0-9]+)MiB,"
    r"load_steps=(?P<gather_load_steps>[0-9]+)/"
    r"(?P<encode_load_steps>[0-9]+),"
    r"load_groups=(?P<gather_load_groups>[0-9]+)/"
    r"(?P<encode_load_groups>[0-9]+),"
    r"local_hw=(?P<local_high_water>[0-9]+)MiB,"
    r"boundary=(?P<boundary>[0-9]+)MiB,"
    r"kernels=(?P<kernels>[0-9]+),"
    r"dispatches=(?P<dispatches>[0-9]+)\]"
)
_LABEL_TIMING_PATTERN = re.compile(
    r"\btiming_ms\[plan=(?P<plan>[0-9]+(?:\.[0-9]+)?),"
    r"prepare=(?P<prepare>[0-9]+(?:\.[0-9]+)?),"
    r"issue=(?P<issue>[0-9]+(?:\.[0-9]+)?),"
    r"begin=(?P<begin>[0-9]+(?:\.[0-9]+)?),"
    r"final_wait=(?P<final_wait>[0-9]+(?:\.[0-9]+)?),"
    r"total=(?P<total>[0-9]+(?:\.[0-9]+)?)\]"
)
_LABEL_PHASE_TIMING_PATTERN = re.compile(
    r"\bphase\.(?P<name>[A-Za-z0-9_]+)_ms\["
    r"prepare=(?P<prepare>[0-9]+(?:\.[0-9]+)?),"
    r"issue=(?P<issue>[0-9]+(?:\.[0-9]+)?),"
    r"wait=(?P<wait>[0-9]+(?:\.[0-9]+)?),"
    r"release=(?P<release>[0-9]+(?:\.[0-9]+)?)\]"
)
_LABEL_PREFETCH_GROUPS_PATTERN = re.compile(
    r"\bprefetch_groups\[count=(?P<count>[0-9]+),"
    r"avg_regions=(?P<average>[0-9]+(?:\.[0-9]+)?),"
    r"max_regions=(?P<maximum>[0-9]+)\]"
)
_LABEL_DIRECT_GATHER_GROUPS_PATTERN = re.compile(
    r"\bdirect_gather_groups\[count=(?P<count>[0-9]+),"
    r"requests=(?P<requests>[0-9]+),"
    r"source=(?P<source>[0-9]+)MiB,target=(?P<target>[0-9]+)MiB,"
    r"max=(?P<maximum>[0-9]+)MiB\]"
)
_LABEL_ENCODE_WINDOWS_PATTERN = re.compile(
    r"\bencode_windows\[count=(?P<count>[0-9]+),"
    r"staging=(?P<staging>[0-9]+)MiB,max=(?P<maximum>[0-9]+)MiB,"
    r"source=(?P<source>[0-9]+)MiB,target=(?P<target>[0-9]+)MiB,"
    r"chunks=(?P<chunks>[0-9]+),batches=(?P<batches>[0-9]+),"
    r"dispatches=(?P<dispatches>[0-9]+)\]"
)
_LABEL_PREPARE_ENCODE_WINDOW_PATTERN = re.compile(
    r"\bprepare_encode_window\[count=(?P<count>[0-9]+),"
    r"staging=(?P<staging>[0-9]+)MiB,max=(?P<maximum>[0-9]+)MiB,"
    r"source=(?P<source>[0-9]+)MiB,target=(?P<target>[0-9]+)MiB,"
    r"chunks=(?P<chunks>[0-9]+),batches=(?P<batches>[0-9]+),"
    r"dispatches=(?P<dispatches>[0-9]+)\]"
)
_LABEL_ISSUE_ENCODE_WINDOW_PATTERN = re.compile(
    r"\bissue_encode_window\[count=(?P<count>[0-9]+),"
    r"staging=(?P<staging>[0-9]+)MiB,max=(?P<maximum>[0-9]+)MiB,"
    r"source=(?P<source>[0-9]+)MiB,target=(?P<target>[0-9]+)MiB,"
    r"chunks=(?P<chunks>[0-9]+),batches=(?P<batches>[0-9]+),"
    r"dispatches=(?P<dispatches>[0-9]+)\]"
)


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationBenchmarkSummaryError(f"{context} must be a JSON object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise GenerationBenchmarkSummaryError(f"{context} must be a JSON array")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise GenerationBenchmarkSummaryError(f"{context} must be a non-empty string")
    return value


def _require_number(value: Any, context: str) -> int | float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise GenerationBenchmarkSummaryError(f"{context} must be a number")
    return value


def _require_match(pattern: re.Pattern[str], text: str, context: str) -> re.Match[str]:
    match = pattern.search(text)
    if not match:
        raise GenerationBenchmarkSummaryError(f"{context} missing {pattern.pattern}")
    return match


def _label_token(label: str, key: str, context: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(key)}=([^ ]+)", label)
    if not match:
        raise GenerationBenchmarkSummaryError(f"{context} missing {key}=...")
    return match.group(1)


def _label_unsigned(label: str, key: str, context: str) -> int:
    value = _label_token(label, key, context)
    if not value.isdecimal():
        raise GenerationBenchmarkSummaryError(
            f"{context} {key} must be an unsigned integer"
        )
    return int(value)


def _label_hex(label: str, key: str, context: str) -> int:
    value = _label_token(label, key, context)
    if not re.fullmatch(r"0x[0-9A-Fa-f]+", value):
        raise GenerationBenchmarkSummaryError(f"{context} {key} must be hex")
    return int(value, 16)


def _label_mib(label: str, key: str, context: str) -> int:
    value = _label_token(label, key, context)
    if not value.endswith("MiB") or not value[:-3].isdecimal():
        raise GenerationBenchmarkSummaryError(f"{context} {key} must be MiB")
    return int(value[:-3])


def _match_unsigned_group(match: re.Match[str], group: str, context: str) -> int:
    value = match.group(group)
    if not value.isdecimal():
        raise GenerationBenchmarkSummaryError(
            f"{context} {group} must be an unsigned integer"
        )
    return int(value)


def _match_float_group(match: re.Match[str], group: str, context: str) -> float:
    value = match.group(group)
    try:
        return float(value)
    except ValueError as exc:
        raise GenerationBenchmarkSummaryError(
            f"{context} {group} must be a number"
        ) from exc


def _ceil_mib(byte_length: int) -> int:
    return (byte_length + 1024 * 1024 - 1) // (1024 * 1024)


_LOAD_KIND_ROW_PREFIXES = {
    "gather": "parameter_load_gather",
    "encode_fp8_e4m3_scaled_to_bf16": "parameter_load_fp8_bf16",
    "encode_bf16_linear_rhs_tile": "parameter_load_bf16_rhs",
    "encode_fp8_e4m3_scaled_to_bf16_linear_rhs_tile": ("parameter_load_fp8_bf16_rhs"),
}


def _sum_parameter_load_kind_statistics(
    stages: dict[str, dict[str, Any]],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for kind_name in smoke_test.PLAN_PARAMETER_LOAD_KIND_NAMES:
        row_prefix = _LOAD_KIND_ROW_PREFIXES[kind_name]
        totals[f"{row_prefix}_steps"] = 0
        totals[f"{row_prefix}_source_byte_length"] = 0
        totals[f"{row_prefix}_target_byte_length"] = 0
        for stage_name, stage in stages.items():
            load_kind_statistics = _require_object(
                stage.get("parameter_load_kind_statistics"),
                f"stages.{stage_name}.parameter_load_kind_statistics",
            )
            kind_statistics = _require_object(
                load_kind_statistics.get(kind_name),
                f"stages.{stage_name}.parameter_load_kind_statistics.{kind_name}",
            )
            totals[f"{row_prefix}_steps"] += int(
                _require_number(
                    kind_statistics.get("step_count"),
                    f"stages.{stage_name}.parameter_load_kind_statistics."
                    f"{kind_name}.step_count",
                )
            )
            totals[f"{row_prefix}_source_byte_length"] += int(
                _require_number(
                    kind_statistics.get("source_byte_length"),
                    f"stages.{stage_name}.parameter_load_kind_statistics."
                    f"{kind_name}.source_byte_length",
                )
            )
            totals[f"{row_prefix}_target_byte_length"] += int(
                _require_number(
                    kind_statistics.get("target_byte_length"),
                    f"stages.{stage_name}.parameter_load_kind_statistics."
                    f"{kind_name}.target_byte_length",
                )
            )
    return totals


def _require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise GenerationBenchmarkSummaryError(
            f"{context} mismatch: got {actual}, expected {expected}"
        )


def _parse_generation_benchmark_label(label: str, context: str) -> dict[str, Any]:
    load_steps_match = _require_match(
        _LABEL_LOAD_STEPS_PATTERN, label, f"{context}.label"
    )
    load_groups_match = _require_match(
        _LABEL_LOAD_GROUPS_PATTERN, label, f"{context}.label"
    )
    load_submit_match = _require_match(
        _LABEL_LOAD_SUBMIT_PATTERN, label, f"{context}.label"
    )
    load_kind_match = _require_match(
        _LABEL_LOAD_KIND_PATTERN, label, f"{context}.label"
    )
    logical_live_match = _require_match(
        _LABEL_LOGICAL_LIVE_PATTERN, label, f"{context}.label"
    )
    prefetch_groups_match = _require_match(
        _LABEL_PREFETCH_GROUPS_PATTERN, label, f"{context}.label"
    )
    direct_gather_groups_match = _require_match(
        _LABEL_DIRECT_GATHER_GROUPS_PATTERN, label, f"{context}.label"
    )
    encode_windows_match = _require_match(
        _LABEL_ENCODE_WINDOWS_PATTERN, label, f"{context}.label"
    )
    prepare_encode_window_match = _require_match(
        _LABEL_PREPARE_ENCODE_WINDOW_PATTERN, label, f"{context}.label"
    )
    issue_encode_window_match = _require_match(
        _LABEL_ISSUE_ENCODE_WINDOW_PATTERN, label, f"{context}.label"
    )
    timing_match = _require_match(_LABEL_TIMING_PATTERN, label, f"{context}.label")

    runtime_stages: dict[str, Any] = {}
    for stage_match in _LABEL_STAGE_PATTERN.finditer(label):
        stage_name = stage_match.group("name")
        runtime_stages[stage_name] = {
            "parameter_mib": _match_unsigned_group(
                stage_match, "parameter", f"{context}.label.stage.{stage_name}"
            ),
            "source_mib": _match_unsigned_group(
                stage_match, "source", f"{context}.label.stage.{stage_name}"
            ),
            "source_direct_mib": _match_unsigned_group(
                stage_match, "source_direct", f"{context}.label.stage.{stage_name}"
            ),
            "source_encoded_mib": _match_unsigned_group(
                stage_match, "source_encoded", f"{context}.label.stage.{stage_name}"
            ),
            "parameter_gather_load_step_count": _match_unsigned_group(
                stage_match,
                "gather_load_steps",
                f"{context}.label.stage.{stage_name}",
            ),
            "parameter_encode_load_step_count": _match_unsigned_group(
                stage_match,
                "encode_load_steps",
                f"{context}.label.stage.{stage_name}",
            ),
            "parameter_gather_load_group_count": _match_unsigned_group(
                stage_match,
                "gather_load_groups",
                f"{context}.label.stage.{stage_name}",
            ),
            "parameter_encode_load_group_count": _match_unsigned_group(
                stage_match,
                "encode_load_groups",
                f"{context}.label.stage.{stage_name}",
            ),
            "local_high_water_mib": _match_unsigned_group(
                stage_match, "local_high_water", f"{context}.label.stage.{stage_name}"
            ),
            "boundary_mib": _match_unsigned_group(
                stage_match, "boundary", f"{context}.label.stage.{stage_name}"
            ),
            "kernel_count": _match_unsigned_group(
                stage_match, "kernels", f"{context}.label.stage.{stage_name}"
            ),
            "dispatch_count": _match_unsigned_group(
                stage_match, "dispatches", f"{context}.label.stage.{stage_name}"
            ),
        }
    if not runtime_stages:
        raise GenerationBenchmarkSummaryError(
            f"{context}.label must contain at least one stage.*[...] group"
        )

    phase_timings: dict[str, dict[str, float]] = {}
    phase_prepare_ms = 0.0
    phase_issue_ms = 0.0
    phase_wait_ms = 0.0
    phase_release_ms = 0.0
    for phase_match in _LABEL_PHASE_TIMING_PATTERN.finditer(label):
        phase_name = phase_match.group("name")
        phase_timing = {
            "prepare_ms": _match_float_group(
                phase_match, "prepare", f"{context}.label.phase.{phase_name}"
            ),
            "issue_ms": _match_float_group(
                phase_match, "issue", f"{context}.label.phase.{phase_name}"
            ),
            "wait_ms": _match_float_group(
                phase_match, "wait", f"{context}.label.phase.{phase_name}"
            ),
            "release_ms": _match_float_group(
                phase_match, "release", f"{context}.label.phase.{phase_name}"
            ),
        }
        phase_timings[phase_name] = phase_timing
        phase_prepare_ms += phase_timing["prepare_ms"]
        phase_issue_ms += phase_timing["issue_ms"]
        phase_wait_ms += phase_timing["wait_ms"]
        phase_release_ms += phase_timing["release_ms"]

    return {
        "benchmark_scope": _label_token(label, "scope", f"{context}.label"),
        "prompt_label": _label_token(label, "prompt", f"{context}.label"),
        "generation_residency_request": _label_token(
            label, "residency_request", f"{context}.label"
        ),
        "generation_residency": _label_token(label, "residency", f"{context}.label"),
        "generation_issue_mode": _label_token(label, "issue", f"{context}.label"),
        "parameter_load_prefetch_region_distance": _label_unsigned(
            label, "prefetch_regions", f"{context}.label"
        ),
        "generation_resident_stage_mask": _label_hex(
            label, "resident_stage_mask", f"{context}.label"
        ),
        "generation_residency_budget_mib": _label_mib(
            label, "residency_budget", f"{context}.label"
        ),
        "runtime_qwen_token_count": _label_unsigned(
            label, "qwen_tokens", f"{context}.label"
        ),
        "runtime_qwen_token_capacity": _label_unsigned(
            label, "qwen_capacity", f"{context}.label"
        ),
        "runtime_image_token_count": _label_unsigned(
            label, "image_tokens", f"{context}.label"
        ),
        "runtime_conditioned_dit_token_count": _label_unsigned(
            label, "dit_cond_tokens", f"{context}.label"
        ),
        "runtime_conditioned_dit_token_capacity": _label_unsigned(
            label, "dit_cond_capacity", f"{context}.label"
        ),
        "runtime_unconditioned_dit_token_count": _label_unsigned(
            label, "dit_uncond_tokens", f"{context}.label"
        ),
        "runtime_unconditioned_dit_token_capacity": _label_unsigned(
            label, "dit_uncond_capacity", f"{context}.label"
        ),
        "runtime_denoise_step_count": _label_unsigned(
            label, "steps", f"{context}.label"
        ),
        "runtime_qwen_weight_execution_strategy": _label_token(
            label, "qwen_weights", f"{context}.label"
        ),
        "runtime_parameter_total_mib": _label_mib(
            label, "param_total", f"{context}.label"
        ),
        "runtime_largest_parameter_mib": _label_mib(
            label, "param_largest", f"{context}.label"
        ),
        "runtime_parameter_source_mib": _label_mib(
            label, "param_source", f"{context}.label"
        ),
        "runtime_parameter_source_direct_mib": _label_mib(
            label, "param_source_direct", f"{context}.label"
        ),
        "runtime_parameter_source_encoded_mib": _label_mib(
            label, "param_source_encoded", f"{context}.label"
        ),
        "runtime_parameter_gather_load_step_count": _match_unsigned_group(
            load_steps_match, "gather", f"{context}.label"
        ),
        "runtime_parameter_encode_load_step_count": _match_unsigned_group(
            load_steps_match, "encode", f"{context}.label"
        ),
        "runtime_parameter_load_group_count": _match_unsigned_group(
            load_groups_match, "total", f"{context}.label"
        ),
        "runtime_parameter_gather_load_group_count": _match_unsigned_group(
            load_groups_match, "gather", f"{context}.label"
        ),
        "runtime_parameter_encode_load_group_count": _match_unsigned_group(
            load_groups_match, "encode", f"{context}.label"
        ),
        "parameter_load_submit_count": _match_unsigned_group(
            load_submit_match, "count", f"{context}.label"
        ),
        "parameter_load_submit_gather_count": _match_unsigned_group(
            load_submit_match, "gather", f"{context}.label"
        ),
        "parameter_load_submit_encode_count": _match_unsigned_group(
            load_submit_match, "encode", f"{context}.label"
        ),
        "parameter_load_submit_total_ms": _match_float_group(
            load_submit_match, "total_ms", f"{context}.label"
        ),
        "parameter_load_submit_gather_ms": _match_float_group(
            load_submit_match, "gather_ms", f"{context}.label"
        ),
        "parameter_load_submit_encode_ms": _match_float_group(
            load_submit_match, "encode_ms", f"{context}.label"
        ),
        "parameter_load_submit_max_ms": _match_float_group(
            load_submit_match, "max_ms", f"{context}.label"
        ),
        "runtime_parameter_load_gather_steps": _match_unsigned_group(
            load_kind_match, "gather_steps", f"{context}.label"
        ),
        "runtime_parameter_load_gather_source_mib": _match_unsigned_group(
            load_kind_match, "gather_source", f"{context}.label"
        ),
        "runtime_parameter_load_gather_target_mib": _match_unsigned_group(
            load_kind_match, "gather_target", f"{context}.label"
        ),
        "runtime_parameter_load_fp8_bf16_steps": _match_unsigned_group(
            load_kind_match, "fp8_bf16_steps", f"{context}.label"
        ),
        "runtime_parameter_load_fp8_bf16_source_mib": _match_unsigned_group(
            load_kind_match, "fp8_bf16_source", f"{context}.label"
        ),
        "runtime_parameter_load_fp8_bf16_target_mib": _match_unsigned_group(
            load_kind_match, "fp8_bf16_target", f"{context}.label"
        ),
        "runtime_parameter_load_bf16_rhs_steps": _match_unsigned_group(
            load_kind_match, "bf16_rhs_steps", f"{context}.label"
        ),
        "runtime_parameter_load_bf16_rhs_source_mib": _match_unsigned_group(
            load_kind_match, "bf16_rhs_source", f"{context}.label"
        ),
        "runtime_parameter_load_bf16_rhs_target_mib": _match_unsigned_group(
            load_kind_match, "bf16_rhs_target", f"{context}.label"
        ),
        "runtime_parameter_load_fp8_bf16_rhs_steps": _match_unsigned_group(
            load_kind_match, "fp8_bf16_rhs_steps", f"{context}.label"
        ),
        "runtime_parameter_load_fp8_bf16_rhs_source_mib": _match_unsigned_group(
            load_kind_match, "fp8_bf16_rhs_source", f"{context}.label"
        ),
        "runtime_parameter_load_fp8_bf16_rhs_target_mib": _match_unsigned_group(
            load_kind_match, "fp8_bf16_rhs_target", f"{context}.label"
        ),
        "runtime_local_high_water_total_mib": _label_mib(
            label, "local_hw_total", f"{context}.label"
        ),
        "runtime_local_high_water_largest_mib": _label_mib(
            label, "local_hw_largest", f"{context}.label"
        ),
        "runtime_boundary_mib": _label_mib(label, "boundary", f"{context}.label"),
        "runtime_kernel_count": _label_unsigned(label, "kernels", f"{context}.label"),
        "runtime_dispatch_count": _label_unsigned(
            label, "dispatches", f"{context}.label"
        ),
        "logical_live_boundary_mib": _match_unsigned_group(
            logical_live_match, "boundary", f"{context}.label"
        ),
        "logical_live_tap_mib": _match_unsigned_group(
            logical_live_match, "taps", f"{context}.label"
        ),
        "logical_live_resident_mib": _match_unsigned_group(
            logical_live_match, "resident", f"{context}.label"
        ),
        "logical_live_phase_peak_mib": _match_unsigned_group(
            logical_live_match, "phase_peak", f"{context}.label"
        ),
        "logical_live_stage_serial_peak_mib": _match_unsigned_group(
            logical_live_match, "stage_serial_peak", f"{context}.label"
        ),
        "logical_live_selected_peak_mib": _match_unsigned_group(
            logical_live_match, "selected_peak", f"{context}.label"
        ),
        "prefetch_group_submit_count": _match_unsigned_group(
            prefetch_groups_match, "count", f"{context}.label"
        ),
        "prefetch_group_average_region_distance": _match_float_group(
            prefetch_groups_match, "average", f"{context}.label"
        ),
        "prefetch_group_max_region_distance": _match_unsigned_group(
            prefetch_groups_match, "maximum", f"{context}.label"
        ),
        "direct_gather_group_count": _match_unsigned_group(
            direct_gather_groups_match, "count", f"{context}.label"
        ),
        "direct_gather_request_count": _match_unsigned_group(
            direct_gather_groups_match, "requests", f"{context}.label"
        ),
        "direct_gather_source_mib": _match_unsigned_group(
            direct_gather_groups_match, "source", f"{context}.label"
        ),
        "direct_gather_target_mib": _match_unsigned_group(
            direct_gather_groups_match, "target", f"{context}.label"
        ),
        "direct_gather_max_mib": _match_unsigned_group(
            direct_gather_groups_match, "maximum", f"{context}.label"
        ),
        "encode_window_count": _match_unsigned_group(
            encode_windows_match, "count", f"{context}.label"
        ),
        "encode_window_staging_mib": _match_unsigned_group(
            encode_windows_match, "staging", f"{context}.label"
        ),
        "encode_window_staging_max_mib": _match_unsigned_group(
            encode_windows_match, "maximum", f"{context}.label"
        ),
        "encode_window_source_mib": _match_unsigned_group(
            encode_windows_match, "source", f"{context}.label"
        ),
        "encode_window_target_mib": _match_unsigned_group(
            encode_windows_match, "target", f"{context}.label"
        ),
        "encode_window_chunk_count": _match_unsigned_group(
            encode_windows_match, "chunks", f"{context}.label"
        ),
        "encode_window_batch_count": _match_unsigned_group(
            encode_windows_match, "batches", f"{context}.label"
        ),
        "encode_window_dispatch_count": _match_unsigned_group(
            encode_windows_match, "dispatches", f"{context}.label"
        ),
        "prepare_encode_window_count": _match_unsigned_group(
            prepare_encode_window_match, "count", f"{context}.label"
        ),
        "prepare_encode_window_staging_mib": _match_unsigned_group(
            prepare_encode_window_match, "staging", f"{context}.label"
        ),
        "prepare_encode_window_staging_max_mib": _match_unsigned_group(
            prepare_encode_window_match, "maximum", f"{context}.label"
        ),
        "prepare_encode_window_source_mib": _match_unsigned_group(
            prepare_encode_window_match, "source", f"{context}.label"
        ),
        "prepare_encode_window_target_mib": _match_unsigned_group(
            prepare_encode_window_match, "target", f"{context}.label"
        ),
        "prepare_encode_window_chunk_count": _match_unsigned_group(
            prepare_encode_window_match, "chunks", f"{context}.label"
        ),
        "prepare_encode_window_batch_count": _match_unsigned_group(
            prepare_encode_window_match, "batches", f"{context}.label"
        ),
        "prepare_encode_window_dispatch_count": _match_unsigned_group(
            prepare_encode_window_match, "dispatches", f"{context}.label"
        ),
        "issue_encode_window_count": _match_unsigned_group(
            issue_encode_window_match, "count", f"{context}.label"
        ),
        "issue_encode_window_staging_mib": _match_unsigned_group(
            issue_encode_window_match, "staging", f"{context}.label"
        ),
        "issue_encode_window_staging_max_mib": _match_unsigned_group(
            issue_encode_window_match, "maximum", f"{context}.label"
        ),
        "issue_encode_window_source_mib": _match_unsigned_group(
            issue_encode_window_match, "source", f"{context}.label"
        ),
        "issue_encode_window_target_mib": _match_unsigned_group(
            issue_encode_window_match, "target", f"{context}.label"
        ),
        "issue_encode_window_chunk_count": _match_unsigned_group(
            issue_encode_window_match, "chunks", f"{context}.label"
        ),
        "issue_encode_window_batch_count": _match_unsigned_group(
            issue_encode_window_match, "batches", f"{context}.label"
        ),
        "issue_encode_window_dispatch_count": _match_unsigned_group(
            issue_encode_window_match, "dispatches", f"{context}.label"
        ),
        "timing_plan_ms": _match_float_group(timing_match, "plan", f"{context}.label"),
        "timing_prepare_ms": _match_float_group(
            timing_match, "prepare", f"{context}.label"
        ),
        "timing_issue_ms": _match_float_group(
            timing_match, "issue", f"{context}.label"
        ),
        "timing_begin_ms": _match_float_group(
            timing_match, "begin", f"{context}.label"
        ),
        "timing_final_wait_ms": _match_float_group(
            timing_match, "final_wait", f"{context}.label"
        ),
        "timing_total_ms": _match_float_group(
            timing_match, "total", f"{context}.label"
        ),
        "phase_timings": phase_timings,
        "phase_prepare_ms": phase_prepare_ms if phase_timings else None,
        "phase_issue_ms": phase_issue_ms if phase_timings else None,
        "phase_wait_ms": phase_wait_ms if phase_timings else None,
        "phase_release_ms": phase_release_ms if phase_timings else None,
        "runtime_stages": runtime_stages,
    }


def _validate_label_plan_join(
    row: dict[str, Any], label_metrics: dict[str, Any], context: str
) -> None:
    _require_equal(
        label_metrics["prompt_label"],
        row["bucket"],
        f"{context}.label.prompt",
    )
    _require_equal(
        label_metrics["runtime_qwen_token_count"],
        row["qwen_token_count"],
        f"{context}.label.qwen_tokens",
    )
    _require_equal(
        label_metrics["runtime_qwen_token_capacity"],
        row["qwen_token_capacity"],
        f"{context}.label.qwen_capacity",
    )
    _require_equal(
        label_metrics["runtime_image_token_count"],
        row["image_token_count"],
        f"{context}.label.image_tokens",
    )
    _require_equal(
        label_metrics["runtime_conditioned_dit_token_count"],
        row["conditioned_dit_token_count"],
        f"{context}.label.dit_cond_tokens",
    )
    _require_equal(
        label_metrics["runtime_conditioned_dit_token_capacity"],
        row["conditioned_dit_token_capacity"],
        f"{context}.label.dit_cond_capacity",
    )
    _require_equal(
        label_metrics["runtime_unconditioned_dit_token_count"],
        row["unconditioned_dit_token_count"],
        f"{context}.label.dit_uncond_tokens",
    )
    _require_equal(
        label_metrics["runtime_unconditioned_dit_token_capacity"],
        row["unconditioned_dit_token_capacity"],
        f"{context}.label.dit_uncond_capacity",
    )
    _require_equal(
        label_metrics["runtime_denoise_step_count"],
        row["denoise_step_count"],
        f"{context}.label.steps",
    )
    _require_equal(
        label_metrics["runtime_parameter_total_mib"],
        _ceil_mib(row["parameter_byte_length"]),
        f"{context}.label.param_total",
    )
    _require_equal(
        label_metrics["runtime_parameter_source_mib"],
        _ceil_mib(row["parameter_source_byte_length"]),
        f"{context}.label.param_source",
    )
    _require_equal(
        label_metrics["runtime_parameter_source_direct_mib"],
        _ceil_mib(row["parameter_direct_source_byte_length"]),
        f"{context}.label.param_source_direct",
    )
    _require_equal(
        label_metrics["runtime_parameter_source_encoded_mib"],
        _ceil_mib(row["parameter_encoded_source_byte_length"]),
        f"{context}.label.param_source_encoded",
    )
    _require_equal(
        label_metrics["runtime_parameter_gather_load_step_count"],
        row["parameter_gather_load_step_count"],
        f"{context}.label.param_load_steps.gather",
    )
    _require_equal(
        label_metrics["runtime_parameter_encode_load_step_count"],
        row["parameter_encode_load_step_count"],
        f"{context}.label.param_load_steps.encode",
    )
    _require_equal(
        label_metrics["runtime_parameter_load_group_count"],
        row["parameter_load_group_count"],
        f"{context}.label.param_load_groups.total",
    )
    _require_equal(
        label_metrics["runtime_parameter_gather_load_group_count"],
        row["parameter_gather_load_group_count"],
        f"{context}.label.param_load_groups.gather",
    )
    _require_equal(
        label_metrics["runtime_parameter_encode_load_group_count"],
        row["parameter_encode_load_group_count"],
        f"{context}.label.param_load_groups.encode",
    )
    for row_prefix in _LOAD_KIND_ROW_PREFIXES.values():
        runtime_prefix = f"runtime_{row_prefix}"
        _require_equal(
            label_metrics[f"{runtime_prefix}_steps"],
            row[f"{row_prefix}_steps"],
            f"{context}.label.param_load_kind.{row_prefix}.steps",
        )
        _require_equal(
            label_metrics[f"{runtime_prefix}_source_mib"],
            _ceil_mib(row[f"{row_prefix}_source_byte_length"]),
            f"{context}.label.param_load_kind.{row_prefix}.source",
        )
        _require_equal(
            label_metrics[f"{runtime_prefix}_target_mib"],
            _ceil_mib(row[f"{row_prefix}_target_byte_length"]),
            f"{context}.label.param_load_kind.{row_prefix}.target",
        )
    _require_equal(
        label_metrics["runtime_boundary_mib"],
        _ceil_mib(row["stage_boundary_byte_length"]),
        f"{context}.label.boundary",
    )
    _require_equal(
        label_metrics["runtime_kernel_count"],
        sum(stage["kernel_count"] for stage in row["stages"].values()),
        f"{context}.label.kernels",
    )
    _require_equal(
        label_metrics["runtime_dispatch_count"],
        row["dispatch_count"],
        f"{context}.label.dispatches",
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise GenerationBenchmarkSummaryError(f"invalid JSON in {path}: {exc}") from exc
    return _require_object(payload, str(path))


def _benchmark_bucket(name: str) -> str:
    parts = name.split("/")
    if len(parts) < 2:
        raise GenerationBenchmarkSummaryError(
            f"benchmark name does not contain a prompt bucket: {name}"
        )
    if parts[-1] == "real_time" and len(parts) >= 3:
        return parts[-2]
    return parts[-1]


def summarize_generation_benchmark(
    benchmark_json_path: Path, plan_directory: Path
) -> dict[str, Any]:
    benchmark_json = _load_json(benchmark_json_path)
    rows: list[dict[str, Any]] = []
    for index, benchmark in enumerate(
        _require_list(benchmark_json.get("benchmarks"), "benchmarks")
    ):
        context = f"benchmarks[{index}]"
        benchmark_object = _require_object(benchmark, context)
        name = _require_string(benchmark_object.get("name"), f"{context}.name")
        bucket = _benchmark_bucket(name)
        plan_metrics = smoke_test.read_generation_plan_metrics(
            plan_directory / f"{bucket}.json"
        )
        plan_summary = plan_metrics["summary"]
        residency = plan_metrics["residency"]
        stages = plan_metrics["stages"]
        load_kind_statistics = _sum_parameter_load_kind_statistics(stages)
        row = {
            "bucket": bucket,
            "benchmark": name,
            "real_time_ms": _require_number(
                benchmark_object.get("real_time"), f"{context}.real_time"
            ),
            "cpu_time_ms": _require_number(
                benchmark_object.get("cpu_time"), f"{context}.cpu_time"
            ),
            "iterations": _require_number(
                benchmark_object.get("iterations"), f"{context}.iterations"
            ),
            "qwen_token_count": plan_summary["qwen_token_count"],
            "qwen_token_capacity": plan_summary["qwen_token_capacity"],
            "image_token_count": plan_summary["image_token_count"],
            "conditioned_dit_token_count": plan_summary["conditioned_dit_token_count"],
            "conditioned_dit_token_capacity": plan_summary[
                "conditioned_dit_token_capacity"
            ],
            "unconditioned_dit_token_count": plan_summary[
                "unconditioned_dit_token_count"
            ],
            "unconditioned_dit_token_capacity": plan_summary[
                "unconditioned_dit_token_capacity"
            ],
            "denoise_step_count": plan_summary["denoise_step_count"],
            "dit_activation_format": plan_summary["dit_activation_format"],
            "dit_weight_execution_format": plan_summary["dit_weight_execution_format"],
            "qwen_weight_execution_strategy": plan_summary[
                "qwen_weight_execution_strategy"
            ],
            "dit_attention_implementation": plan_summary[
                "dit_attention_implementation"
            ],
            "dit_feed_forward_implementation": plan_summary[
                "dit_feed_forward_implementation"
            ],
            "parameter_byte_length": residency["total_stage_parameter_byte_length"],
            "parameter_source_byte_length": sum(
                stage["parameter_source_byte_length"] for stage in stages.values()
            ),
            "parameter_direct_source_byte_length": sum(
                stage["parameter_direct_source_byte_length"]
                for stage in stages.values()
            ),
            "parameter_encoded_source_byte_length": sum(
                stage["parameter_encoded_source_byte_length"]
                for stage in stages.values()
            ),
            "parameter_gather_load_step_count": sum(
                stage["parameter_gather_load_step_count"] for stage in stages.values()
            ),
            "parameter_encode_load_step_count": sum(
                stage["parameter_encode_load_step_count"] for stage in stages.values()
            ),
            "parameter_load_group_count": sum(
                stage["parameter_load_group_count"] for stage in stages.values()
            ),
            "parameter_gather_load_group_count": sum(
                stage["parameter_gather_load_group_count"] for stage in stages.values()
            ),
            "parameter_encode_load_group_count": sum(
                stage["parameter_encode_load_group_count"] for stage in stages.values()
            ),
            "phase_parameter_high_water_mark": residency[
                "phase_parameter_high_water_mark"
            ],
            "stage_boundary_byte_length": residency["total_stage_boundary_byte_length"],
            "shared_tensor_byte_length": sum(
                stage["shared_tensor_byte_length"] for stage in stages.values()
            ),
            "shared_tensor_count": sum(
                stage["shared_tensor_count"] for stage in stages.values()
            ),
            "stage_count": len(stages),
            "dispatch_count": sum(stage["dispatch_count"] for stage in stages.values()),
            "local_high_water_mark": sum(
                stage["memory_slab_high_water_mark"] for stage in stages.values()
            ),
            "stages": stages,
        }
        row.update(load_kind_statistics)
        label = benchmark_object.get("label")
        if label is not None:
            label_metrics = _parse_generation_benchmark_label(
                _require_string(label, f"{context}.label"), context
            )
            _validate_label_plan_join(row, label_metrics, context)
            row.update(label_metrics)
        rows.append(row)
    return {
        "source": {
            "benchmark_json": str(benchmark_json_path),
            "plan_directory": str(plan_directory),
        },
        "rows": rows,
    }


_MARKDOWN_COLUMNS = (
    ("bucket", "bucket"),
    ("residency", "generation_residency"),
    ("resident mask", "generation_resident_stage_mask"),
    ("selected peak MiB", "logical_live_selected_peak_mib"),
    ("qwen tokens", "qwen_token_count"),
    ("qwen cap", "qwen_token_capacity"),
    ("cond tokens", "conditioned_dit_token_count"),
    ("cond cap", "conditioned_dit_token_capacity"),
    ("real ms", "real_time_ms"),
    ("issue ms", "timing_issue_ms"),
    ("final wait ms", "timing_final_wait_ms"),
    ("phase prepare ms", "phase_prepare_ms"),
    ("phase issue ms", "phase_issue_ms"),
    ("phase wait ms", "phase_wait_ms"),
    ("phase release ms", "phase_release_ms"),
    ("load submit ms", "parameter_load_submit_total_ms"),
    ("gather submit ms", "parameter_load_submit_gather_ms"),
    ("encode submit ms", "parameter_load_submit_encode_ms"),
    ("max submit ms", "parameter_load_submit_max_ms"),
    ("direct MiB", "runtime_parameter_source_direct_mib"),
    ("encoded MiB", "runtime_parameter_source_encoded_mib"),
    ("fp8 bf16 src MiB", "runtime_parameter_load_fp8_bf16_source_mib"),
    ("fp8 bf16 target MiB", "runtime_parameter_load_fp8_bf16_target_mib"),
    ("bf16 rhs src MiB", "runtime_parameter_load_bf16_rhs_source_mib"),
    ("fp8 rhs src MiB", "runtime_parameter_load_fp8_bf16_rhs_source_mib"),
    ("issue windows", "issue_encode_window_count"),
    ("issue encodes", "issue_encode_window_dispatch_count"),
    ("staging MiB", "issue_encode_window_staging_mib"),
    ("max staging MiB", "issue_encode_window_staging_max_mib"),
    ("dispatches", "runtime_dispatch_count"),
)

_MARKDOWN_TEXT_COLUMNS = frozenset(("bucket", "generation_residency"))


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_generation_benchmark_markdown(summary: dict[str, Any]) -> str:
    rows = _require_list(summary.get("rows"), "rows")
    lines = [
        "| " + " | ".join(header for header, _ in _MARKDOWN_COLUMNS) + " |",
        "| "
        + " | ".join(
            "---" if column in _MARKDOWN_TEXT_COLUMNS else "---:"
            for _, column in _MARKDOWN_COLUMNS
        )
        + " |",
    ]
    for row_index, row_value in enumerate(rows):
        row = _require_object(row_value, f"rows[{row_index}]")
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row.get(column)) for _, column in _MARKDOWN_COLUMNS
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark_json",
        type=Path,
        required=True,
        help="Google Benchmark JSON emitted by session_live_benchmark.",
    )
    parser.add_argument(
        "--plan_dir",
        type=Path,
        required=True,
        help="Directory containing generation plan JSON files named by bucket.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help=(
            "Output format. JSON preserves all fields; markdown emits a compact table."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        summary = summarize_generation_benchmark(args.benchmark_json, args.plan_dir)
    except (OSError, GenerationBenchmarkSummaryError, smoke_test.SmokeTestError) as exc:
        print(f"generation_benchmark_summary: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        text = json.dumps(summary, indent=2) + "\n"
    else:
        text = format_generation_benchmark_markdown(summary)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
