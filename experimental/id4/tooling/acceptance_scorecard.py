#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds an ID4/PyTorch warm-serving acceptance scorecard."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

MEBIBYTE = 1024 * 1024


class AcceptanceScorecardError(ValueError):
    """Raised when acceptance artifacts are missing or semantically mismatched."""


def _round_up_mebibytes(byte_length: int) -> int:
    return (byte_length + MEBIBYTE - 1) // MEBIBYTE


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceScorecardError(f"{field_name} must be an object")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AcceptanceScorecardError(f"{field_name} must be a list")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AcceptanceScorecardError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise AcceptanceScorecardError(f"{field_name} must be an integer")
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    result = _require_int(value, field_name)
    if result < 0:
        raise AcceptanceScorecardError(f"{field_name} must be non-negative")
    return result


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceScorecardError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise AcceptanceScorecardError(f"{field_name} must be finite")
    return result


def _load_json(path: Path, description: str) -> Any:
    if not path.is_file():
        raise AcceptanceScorecardError(f"{description} not found: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise AcceptanceScorecardError(
            f"invalid {description} JSON at {path}: {exc}"
        ) from exc


def _resolve_path(root: Path, value: Any, field_name: str) -> Path:
    path = Path(_require_string(value, field_name))
    return path if path.is_absolute() else root / path


def _require_equal(actual: Any, expected: Any, field_name: str) -> None:
    if actual != expected:
        raise AcceptanceScorecardError(
            f"{field_name} mismatch: expected {expected!r}, found {actual!r}"
        )


def _require_subset(actual: Any, expected: Any, field_name: str) -> None:
    expected_dict = _require_dict(expected, field_name)
    actual_dict = _require_dict(actual, field_name)
    for key, expected_value in expected_dict.items():
        child_name = f"{field_name}.{key}"
        if key not in actual_dict:
            raise AcceptanceScorecardError(f"{child_name} is missing")
        actual_value = actual_dict[key]
        if isinstance(expected_value, dict):
            _require_subset(actual_value, expected_value, child_name)
        else:
            _require_equal(actual_value, expected_value, child_name)


def _parse_label(label: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    values: dict[str, str] = {}
    groups: dict[str, dict[str, str]] = {}
    for token in label.split():
        bracket_index = token.find("[")
        if (
            bracket_index > 0
            and token.endswith("]")
            and "=" not in token[:bracket_index]
        ):
            name = token[:bracket_index]
            if name in groups or name in values:
                raise AcceptanceScorecardError(f"duplicate benchmark label key: {name}")
            group: dict[str, str] = {}
            body = token[bracket_index + 1 : -1]
            for item in body.split(",") if body else []:
                if "=" not in item:
                    raise AcceptanceScorecardError(
                        f"malformed benchmark label group item: {token}"
                    )
                key, value = item.split("=", 1)
                if not key or not value or key in group:
                    raise AcceptanceScorecardError(
                        f"malformed benchmark label group item: {token}"
                    )
                group[key] = value
            groups[name] = group
            continue
        if "=" not in token:
            raise AcceptanceScorecardError(f"malformed benchmark label token: {token}")
        key, value = token.split("=", 1)
        if not key or not value or key in values or key in groups:
            raise AcceptanceScorecardError(f"malformed benchmark label token: {token}")
        values[key] = value
    return values, groups


def _label_value(values: dict[str, str], key: str) -> str:
    value = values.get(key)
    if value is None:
        raise AcceptanceScorecardError(f"benchmark label is missing {key}")
    return value


def _label_int(values: dict[str, str], key: str) -> int:
    value = _label_value(values, key)
    try:
        return int(value, 10)
    except ValueError as exc:
        raise AcceptanceScorecardError(
            f"benchmark label {key} must be an integer, found {value!r}"
        ) from exc


def _label_mebibytes(values: dict[str, str], key: str) -> int:
    value = _label_value(values, key)
    return _parse_mebibytes(value, f"benchmark label {key}")


def _parse_mebibytes(value: str, field_name: str) -> int:
    if not value.endswith("MiB"):
        raise AcceptanceScorecardError(
            f"{field_name} must use MiB units, found {value!r}"
        )
    try:
        return int(value[:-3], 10)
    except ValueError as exc:
        raise AcceptanceScorecardError(
            f"{field_name} must be an integer MiB value, found {value!r}"
        ) from exc


def _benchmark_duration_ms(benchmark: dict[str, Any]) -> float:
    duration = _require_number(benchmark.get("real_time"), "benchmark.real_time")
    unit = _require_string(benchmark.get("time_unit"), "benchmark.time_unit")
    scale = {"ns": 1.0e-6, "us": 1.0e-3, "ms": 1.0, "s": 1.0e3}.get(unit)
    if scale is None:
        raise AcceptanceScorecardError(f"unsupported benchmark time unit: {unit}")
    duration_ms = duration * scale
    if duration_ms <= 0:
        raise AcceptanceScorecardError("benchmark.real_time must be positive")
    return duration_ms


def _load_benchmark(
    root: Path, specification: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, str]]]:
    path = _resolve_path(root, specification.get("file"), "id4.benchmark.file")
    payload = _require_dict(_load_json(path, "ID4 benchmark"), "ID4 benchmark")
    expected_name = _require_string(specification.get("name"), "id4.benchmark.name")
    matches = [
        _require_dict(row, "ID4 benchmark row")
        for row in _require_list(payload.get("benchmarks"), "ID4 benchmark.benchmarks")
        if isinstance(row, dict) and row.get("name") == expected_name
    ]
    if len(matches) != 1:
        raise AcceptanceScorecardError(
            f"ID4 benchmark {path} contains {len(matches)} rows named {expected_name!r}"
        )
    benchmark = matches[0]
    label = _require_string(benchmark.get("label"), "ID4 benchmark.label")
    values, groups = _parse_label(label)
    return benchmark, values, groups


def _request(row: dict[str, Any]) -> dict[str, int]:
    source = _require_dict(row.get("request"), "row.request")
    request = {
        key: _require_int(source.get(key), f"row.request.{key}")
        for key in (
            "qwen_token_count",
            "latent_width",
            "latent_height",
            "image_width",
            "image_height",
            "denoise_step_count",
        )
    }
    if request["qwen_token_count"] <= 0:
        raise AcceptanceScorecardError("row.request.qwen_token_count must be positive")
    for key in (
        "latent_width",
        "latent_height",
        "image_width",
        "image_height",
        "denoise_step_count",
    ):
        if request[key] <= 0:
            raise AcceptanceScorecardError(f"row.request.{key} must be positive")
    return request


def _validate_id4_label(
    request: dict[str, int],
    values: dict[str, str],
    expected_policy: dict[str, Any],
) -> None:
    image_token_count = request["latent_width"] * request["latent_height"]
    expected_values: dict[str, Any] = {
        "qwen_tokens": request["qwen_token_count"],
        "image_tokens": image_token_count,
        "dit_cond_tokens": request["qwen_token_count"] + image_token_count,
        "dit_uncond_tokens": image_token_count,
        "steps": request["denoise_step_count"],
        "latent": f"{request['latent_width']}x{request['latent_height']}",
        "image": f"{request['image_width']}x{request['image_height']}",
    }
    expected_values.update(expected_policy)
    for key, expected in expected_values.items():
        actual = _label_value(values, key)
        if isinstance(expected, int):
            try:
                actual = int(actual, 10)
            except ValueError as exc:
                raise AcceptanceScorecardError(
                    f"benchmark label {key} must be an integer, found {actual!r}"
                ) from exc
        _require_equal(actual, expected, f"benchmark label {key}")


def _plan_statistics(
    plan: dict[str, Any],
    request: dict[str, int],
    values: dict[str, str],
    groups: dict[str, dict[str, str]],
) -> dict[str, int]:
    _require_equal(plan.get("kind"), "ideogram4_generation", "ID4 plan.kind")
    summary = _require_dict(plan.get("summary"), "ID4 plan.summary")
    image_token_count = request["latent_width"] * request["latent_height"]
    summary_expectations = {
        "qwen_token_count": request["qwen_token_count"],
        "image_token_count": image_token_count,
        "conditioned_dit_token_count": request["qwen_token_count"] + image_token_count,
        "unconditioned_dit_token_count": image_token_count,
        "denoise_step_count": request["denoise_step_count"],
    }
    for key, expected in summary_expectations.items():
        _require_equal(summary.get(key), expected, f"ID4 plan.summary.{key}")
    decoded_shape = _require_dict(
        summary.get("decoded_image_shape"), "ID4 plan.summary.decoded_image_shape"
    )
    _require_equal(
        decoded_shape.get("dims"),
        [request["image_width"], request["image_height"], 3, 1],
        "ID4 plan.summary.decoded_image_shape.dims",
    )
    latent_shape = _require_dict(
        summary.get("diffusion_latent_shape"),
        "ID4 plan.summary.diffusion_latent_shape",
    )
    _require_equal(
        latent_shape.get("dims"),
        [request["latent_width"], request["latent_height"], 128, 1],
        "ID4 plan.summary.diffusion_latent_shape.dims",
    )

    resources = _require_dict(
        plan.get("execution_resources"), "ID4 plan.execution_resources"
    )
    _require_equal(
        resources.get("parameter_source_kind"),
        "execution_layout",
        "ID4 plan.execution_resources.parameter_source_kind",
    )
    stages = _require_dict(plan.get("stages"), "ID4 plan.stages")
    if not stages:
        raise AcceptanceScorecardError("ID4 plan.stages must not be empty")

    stage_invocation_counts: dict[str, int] = {}
    residency = _require_dict(plan.get("residency"), "ID4 plan.residency")
    phases = _require_list(residency.get("phases"), "ID4 plan.residency.phases")
    for phase_index, phase_value in enumerate(phases):
        phase = _require_dict(phase_value, f"ID4 plan.residency.phases[{phase_index}]")
        repeated = phase.get("repeated_per_denoise_step")
        if type(repeated) is not bool:
            raise AcceptanceScorecardError(
                f"ID4 plan.residency.phases[{phase_index}]."
                "repeated_per_denoise_step must be a boolean"
            )
        invocation_count = request["denoise_step_count"] if repeated else 1
        for stage_key_value in _require_list(
            phase.get("stage_keys"),
            f"ID4 plan.residency.phases[{phase_index}].stage_keys",
        ):
            stage_key = _require_string(
                stage_key_value,
                f"ID4 plan.residency.phases[{phase_index}].stage_keys",
            )
            if stage_key in stage_invocation_counts:
                raise AcceptanceScorecardError(
                    f"ID4 plan stage {stage_key!r} occurs in multiple phases"
                )
            stage_invocation_counts[stage_key] = invocation_count
    _require_equal(
        set(stage_invocation_counts), set(stages), "ID4 plan phase stage set"
    )

    prepared_dispatch_count = 0
    dispatch_count = 0
    command_buffer_count = 0
    local_allocation_count = 0
    shared_allocation_count = 0
    transformed_source_byte_length = 0
    transformed_persistent_byte_length = 0
    direct_parameter_byte_length = 0
    for stage_key, stage_value in stages.items():
        invocation_count = stage_invocation_counts[stage_key]
        stage = _require_dict(stage_value, f"ID4 plan.stages.{stage_key}")
        statistics = _require_dict(
            stage.get("statistics"), f"ID4 plan.stages.{stage_key}.statistics"
        )
        stage_dispatch_count = _require_nonnegative_int(
            statistics.get("dispatch_count"),
            f"ID4 plan.stages.{stage_key}.statistics.dispatch_count",
        )
        prepared_dispatch_count += stage_dispatch_count
        dispatch_count += stage_dispatch_count * invocation_count
        kind_statistics = _require_dict(
            statistics.get("parameter_load_kind_statistics"),
            f"ID4 plan.stages.{stage_key}.statistics.parameter_load_kind_statistics",
        )
        for kind, kind_value in kind_statistics.items():
            kind_record = _require_dict(
                kind_value,
                f"ID4 plan.stages.{stage_key}.parameter_load_kind_statistics.{kind}",
            )
            source_bytes = _require_nonnegative_int(
                kind_record.get("source_byte_length"), f"{kind}.source_byte_length"
            )
            target_bytes = _require_nonnegative_int(
                kind_record.get("target_byte_length"), f"{kind}.target_byte_length"
            )
            if kind == "gather":
                direct_parameter_byte_length += target_bytes
            else:
                transformed_source_byte_length += source_bytes
                transformed_persistent_byte_length += target_bytes

        regions = _require_list(
            stage.get("regions"), f"ID4 plan.stages.{stage_key}.regions"
        )
        if not regions:
            raise AcceptanceScorecardError(
                f"ID4 plan.stages.{stage_key}.regions must not be empty"
            )
        command_buffer_count += len(regions)
        for region_index, region_value in enumerate(regions):
            region = _require_dict(
                region_value, f"ID4 plan.stages.{stage_key}.regions[{region_index}]"
            )
            region_statistics = _require_dict(
                region.get("statistics"),
                f"ID4 plan.stages.{stage_key}.regions[{region_index}].statistics",
            )
            local_bytes = _require_nonnegative_int(
                region_statistics.get("local_slab_byte_length"),
                f"ID4 plan.stages.{stage_key}.regions[{region_index}].local_slab_byte_length",
            )
            if local_bytes > 0:
                local_allocation_count += invocation_count
        for slab_index, slab_value in enumerate(
            _require_list(
                stage.get("memory_slabs"), f"ID4 plan.stages.{stage_key}.memory_slabs"
            )
        ):
            slab = _require_dict(
                slab_value,
                f"ID4 plan.stages.{stage_key}.memory_slabs[{slab_index}]",
            )
            if slab.get("scope") != "region_local":
                shared_allocation_count += invocation_count

    _require_equal(
        prepared_dispatch_count,
        _label_int(values, "dispatches"),
        "prepared dispatch count",
    )
    resident_parameter_bytes = _require_nonnegative_int(
        resources.get("resident_stage_parameter_byte_length"),
        "ID4 plan.execution_resources.resident_stage_parameter_byte_length",
    )
    local_transient_bytes = _require_nonnegative_int(
        resources.get("stage_serial_local_peak_byte_length"),
        "ID4 plan.execution_resources.stage_serial_local_peak_byte_length",
    )
    cross_stage_bytes = _require_nonnegative_int(
        resources.get("boundary_buffer_byte_length"),
        "ID4 plan.execution_resources.boundary_buffer_byte_length",
    )
    selected_peak_bytes = _require_nonnegative_int(
        resources.get("stage_serial_total_peak_byte_length"),
        "ID4 plan.execution_resources.stage_serial_total_peak_byte_length",
    )
    if (
        direct_parameter_byte_length + transformed_persistent_byte_length
        != resident_parameter_bytes
    ):
        raise AcceptanceScorecardError(
            "ID4 plan resident parameter bytes do not equal direct plus "
            "transformed persistent bytes"
        )
    rounded_checks = {
        "param_total": resident_parameter_bytes,
        "local_hw_largest": max(
            _require_nonnegative_int(
                _require_dict(stage, "stage")
                .get("statistics", {})
                .get("memory_slab_high_water_mark"),
                "stage.statistics.memory_slab_high_water_mark",
            )
            for stage in stages.values()
        ),
    }
    for key, byte_length in rounded_checks.items():
        _require_equal(
            _label_mebibytes(values, key),
            _round_up_mebibytes(byte_length),
            f"benchmark label {key}",
        )
    logical_live = groups.get("logical_live")
    if logical_live is None:
        raise AcceptanceScorecardError(
            "benchmark label is missing logical_live statistics"
        )
    _require_equal(
        _parse_mebibytes(
            _require_string(
                logical_live.get("selected_peak"),
                "benchmark label logical_live.selected_peak",
            ),
            "benchmark label logical_live.selected_peak",
        ),
        _round_up_mebibytes(selected_peak_bytes),
        "benchmark label logical_live.selected_peak",
    )

    queue_alloca_count = local_allocation_count + shared_allocation_count
    queue_execute_count = sum(
        len(
            _require_list(
                _require_dict(stage, f"ID4 plan.stages.{stage_key}").get("regions"),
                f"ID4 plan.stages.{stage_key}.regions",
            )
        )
        * stage_invocation_counts[stage_key]
        for stage_key, stage in stages.items()
    )
    queue_dealloca_count = queue_alloca_count
    return {
        "resident_parameter_byte_length": resident_parameter_bytes,
        "direct_parameter_byte_length": direct_parameter_byte_length,
        "transformed_source_byte_length": transformed_source_byte_length,
        "transformed_persistent_byte_length": transformed_persistent_byte_length,
        "local_transient_high_water_mark": local_transient_bytes,
        "cross_stage_byte_length": cross_stage_bytes,
        "selected_peak_byte_length": selected_peak_bytes,
        "stage_command_buffer_count": command_buffer_count,
        "stage_invocation_count": sum(stage_invocation_counts.values()),
        "stage_queue_alloca_count": queue_alloca_count,
        "stage_queue_execute_count": queue_execute_count,
        "stage_queue_dealloca_count": queue_dealloca_count,
        "stage_queue_operation_count": (
            queue_alloca_count + queue_execute_count + queue_dealloca_count
        ),
        "dispatch_count": dispatch_count,
        "prepared_dispatch_count": prepared_dispatch_count,
    }


def _load_plan(
    root: Path,
    specification: dict[str, Any],
    request: dict[str, int],
    values: dict[str, str],
    groups: dict[str, dict[str, str]],
) -> dict[str, int]:
    path = _resolve_path(root, specification.get("plan"), "id4.plan")
    plan = _require_dict(_load_json(path, "ID4 generation plan"), "ID4 plan")
    return _plan_statistics(plan, request, values, groups)


def _load_pytorch_qwen(
    root: Path, value: Any, request: dict[str, int], policy: dict[str, Any]
) -> float:
    path = _resolve_path(root, value, "pytorch.qwen")
    payload = _require_dict(_load_json(path, "PyTorch Qwen timing"), "PyTorch Qwen")
    _require_equal(
        payload.get("token_count"),
        request["qwen_token_count"],
        "PyTorch Qwen token_count",
    )
    _require_equal(
        payload.get("attention_mask_shape"),
        [request["qwen_token_count"], request["qwen_token_count"]],
        "PyTorch Qwen attention_mask_shape",
    )
    condition_shape = _require_list(
        payload.get("condition_shape"), "PyTorch Qwen condition_shape"
    )
    if len(condition_shape) != 2:
        raise AcceptanceScorecardError("PyTorch Qwen condition_shape must have rank 2")
    _require_equal(
        condition_shape[1],
        request["qwen_token_count"],
        "PyTorch Qwen condition token count",
    )
    _require_subset(payload, policy, "PyTorch Qwen policy")
    median_ms = _require_number(payload.get("median_ms"), "PyTorch Qwen median_ms")
    if median_ms <= 0:
        raise AcceptanceScorecardError("PyTorch Qwen median_ms must be positive")
    return median_ms


def _load_pytorch_dit(
    root: Path,
    value: Any,
    request: dict[str, int],
    conditioned: bool,
    policy: dict[str, Any],
) -> float:
    specification = _require_dict(value, "PyTorch DiT specification")
    path = _resolve_path(root, specification.get("file"), "PyTorch DiT file")
    branch_name = _require_string(specification.get("branch"), "PyTorch DiT branch")
    expected_branch = "cond" if conditioned else "uncond"
    _require_equal(branch_name, expected_branch, "PyTorch DiT branch")
    payload = _require_dict(_load_json(path, "PyTorch DiT timing"), "PyTorch DiT")
    top_level_policy = {
        key: expected for key, expected in policy.items() if key != "branch_parameters"
    }
    _require_subset(payload, top_level_policy, "PyTorch DiT policy")
    branches = _require_dict(payload.get("branches"), "PyTorch DiT branches")
    branch = _require_dict(
        branches.get(branch_name), f"PyTorch DiT branches.{branch_name}"
    )
    shape = _require_dict(branch.get("shape"), f"PyTorch DiT {branch_name} shape")
    image_token_count = request["latent_width"] * request["latent_height"]
    text_token_count = request["qwen_token_count"] if conditioned else 0
    expectations = {
        "image_height": request["image_height"],
        "image_width": request["image_width"],
        "latent_height": request["latent_height"],
        "latent_width": request["latent_width"],
        "image_token_count": image_token_count,
        "text_token_count": text_token_count,
        "total_token_count": image_token_count + text_token_count,
    }
    for key, expected in expectations.items():
        _require_equal(
            shape.get(key), expected, f"PyTorch DiT {branch_name} shape.{key}"
        )
    branch_parameters = _require_dict(
        policy.get("branch_parameters"), "PyTorch DiT branch_parameters policy"
    )
    _require_subset(
        branch,
        {"parameters": branch_parameters},
        "PyTorch DiT branch policy",
    )
    median_ms = _require_number(
        branch.get("median_ms"), f"PyTorch DiT {branch_name} median_ms"
    )
    if median_ms <= 0:
        raise AcceptanceScorecardError(
            f"PyTorch DiT {branch_name} median_ms must be positive"
        )
    return median_ms


def _load_pytorch_vae(
    root: Path, value: Any, request: dict[str, int], policy: dict[str, Any]
) -> float:
    path = _resolve_path(root, value, "pytorch.vae")
    payload = _require_dict(_load_json(path, "PyTorch VAE timing"), "PyTorch VAE")
    _require_equal(
        payload.get("latent_shape"),
        [request["latent_width"], request["latent_height"], 128, 1],
        "PyTorch VAE latent_shape",
    )
    _require_equal(
        payload.get("decoded_shape_bchw"),
        [1, 3, request["image_height"], request["image_width"]],
        "PyTorch VAE decoded_shape_bchw",
    )
    _require_subset(payload, policy, "PyTorch VAE policy")
    median_ms = _require_number(payload.get("median_ms"), "PyTorch VAE median_ms")
    if median_ms <= 0:
        raise AcceptanceScorecardError("PyTorch VAE median_ms must be positive")
    return median_ms


def _validate_correctness_expectation(
    specification: dict[str, Any], record: dict[str, Any], field_name: str
) -> None:
    expected = _require_dict(specification.get("expect"), f"{field_name}.expect")
    if not expected:
        raise AcceptanceScorecardError(f"{field_name}.expect must not be empty")
    _require_subset(record, expected, field_name)


def _correctness_check(root: Path, value: Any) -> dict[str, Any]:
    specification = _require_dict(value, "correctness check")
    name = _require_string(specification.get("name"), "correctness check.name")
    kind = _require_string(specification.get("kind"), f"correctness check {name}.kind")
    path = _resolve_path(
        root, specification.get("file"), f"correctness check {name}.file"
    )
    payload = _load_json(path, f"correctness report {name}")
    if kind == "fixture_compare":
        report = _require_dict(payload, f"correctness report {name}")
        summary = _require_dict(
            report.get("summary"), f"correctness report {name}.summary"
        )
        compared_count = _require_nonnegative_int(
            summary.get("compared_count"), f"correctness report {name}.compared_count"
        )
        failed_count = _require_nonnegative_int(
            summary.get("failed_count"), f"correctness report {name}.failed_count"
        )
        if compared_count <= 0:
            raise AcceptanceScorecardError(
                f"correctness report {name} compared no tensors"
            )
        if "expected_count" in summary:
            _require_equal(
                compared_count,
                _require_nonnegative_int(
                    summary.get("expected_count"),
                    f"correctness report {name}.expected_count",
                ),
                f"correctness report {name} compared_count",
            )
        comparison_name = _require_string(
            specification.get("comparison"),
            f"correctness check {name}.comparison",
        )
        comparisons = _require_list(
            report.get("comparisons"), f"correctness report {name}.comparisons"
        )
        matches = [
            _require_dict(comparison, f"correctness report {name} comparison")
            for comparison in comparisons
            if isinstance(comparison, dict)
            and comparison.get("name") == comparison_name
        ]
        if len(matches) != 1:
            raise AcceptanceScorecardError(
                f"correctness report {name} contains {len(matches)} comparisons "
                f"named {comparison_name!r}"
            )
        comparison = matches[0]
        _validate_correctness_expectation(
            specification, comparison, f"correctness report {name}.{comparison_name}"
        )
        comparison_passed = comparison.get("status") == "pass"
        return {
            "name": name,
            "status": "pass" if failed_count == 0 and comparison_passed else "fail",
            "comparison": comparison_name,
            "compared_count": compared_count,
            "failed_count": failed_count,
        }
    if kind == "aggregate_metrics":
        if isinstance(payload, list):
            records = payload
        else:
            report = _require_dict(payload, f"correctness report {name}")
            records = _require_list(
                report.get("records"), f"correctness report {name}.records"
            )
        record_name = _require_string(
            specification.get("record"), f"correctness check {name}.record"
        )
        matches = [
            _require_dict(record, f"correctness report {name} record")
            for record in records
            if isinstance(record, dict) and record.get("name") == record_name
        ]
        if len(matches) != 1:
            raise AcceptanceScorecardError(
                f"correctness report {name} contains {len(matches)} records named {record_name!r}"
            )
        record = matches[0]
        _validate_correctness_expectation(
            specification, record, f"correctness report {name}.{record_name}"
        )
        if "status" in record:
            _require_equal(
                record.get("status"),
                "compared",
                f"correctness report {name}.{record_name}.status",
            )
        limits = _require_dict(
            specification.get("limits"), f"correctness check {name}.limits"
        )
        if not limits:
            raise AcceptanceScorecardError(
                f"correctness check {name}.limits must not be empty"
            )
        metrics: dict[str, float] = {}
        passed = True
        for metric, limit_value in limits.items():
            limit = _require_number(
                limit_value, f"correctness check {name}.limits.{metric}"
            )
            if limit < 0:
                raise AcceptanceScorecardError(
                    f"correctness check {name}.limits.{metric} must be non-negative"
                )
            actual = _require_number(
                record.get(metric), f"correctness report {name}.{record_name}.{metric}"
            )
            metrics[metric] = actual
            if actual > limit:
                passed = False
        return {
            "name": name,
            "status": "pass" if passed else "fail",
            "record": record_name,
            "metrics": metrics,
            "limits": {key: float(value) for key, value in limits.items()},
        }
    raise AcceptanceScorecardError(
        f"correctness check {name} has unsupported kind {kind!r}"
    )


def build_scorecard(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _require_dict(
        _load_json(manifest_path, "acceptance manifest"), "acceptance manifest"
    )
    _require_equal(manifest.get("schema_version"), 1, "acceptance schema_version")
    maximum_ratio = _require_number(
        manifest.get("maximum_ratio"), "acceptance maximum_ratio"
    )
    if maximum_ratio <= 0:
        raise AcceptanceScorecardError("acceptance maximum_ratio must be positive")
    policy = _require_dict(manifest.get("policy"), "acceptance policy")
    id4_policy = _require_dict(policy.get("id4_label"), "acceptance policy.id4_label")
    pytorch_policy = _require_dict(policy.get("pytorch"), "acceptance policy.pytorch")
    root = manifest_path.parent
    output_rows = []
    seen_names = set()
    for row_index, row_value in enumerate(
        _require_list(manifest.get("rows"), "acceptance rows")
    ):
        row = _require_dict(row_value, f"acceptance rows[{row_index}]")
        name = _require_string(row.get("name"), f"acceptance rows[{row_index}].name")
        if name in seen_names:
            raise AcceptanceScorecardError(f"duplicate acceptance row name: {name}")
        seen_names.add(name)
        request = _request(row)
        id4 = _require_dict(row.get("id4"), f"acceptance row {name}.id4")
        benchmark, label_values, label_groups = _load_benchmark(
            root,
            _require_dict(id4.get("benchmark"), f"acceptance row {name}.id4.benchmark"),
        )
        _validate_id4_label(request, label_values, id4_policy)
        plan_statistics = _load_plan(root, id4, request, label_values, label_groups)

        pytorch = _require_dict(row.get("pytorch"), f"acceptance row {name}.pytorch")
        qwen_ms = _load_pytorch_qwen(
            root,
            pytorch.get("qwen"),
            request,
            _require_dict(pytorch_policy.get("qwen"), "acceptance policy.pytorch.qwen"),
        )
        dit_policy = _require_dict(
            pytorch_policy.get("dit"), "acceptance policy.pytorch.dit"
        )
        conditioned_dit_ms = _load_pytorch_dit(
            root, pytorch.get("dit_conditioned"), request, True, dit_policy
        )
        unconditioned_dit_ms = _load_pytorch_dit(
            root, pytorch.get("dit_unconditioned"), request, False, dit_policy
        )
        vae_ms = _load_pytorch_vae(
            root,
            pytorch.get("vae"),
            request,
            _require_dict(pytorch_policy.get("vae"), "acceptance policy.pytorch.vae"),
        )
        pytorch_ms = (
            qwen_ms
            + request["denoise_step_count"]
            * (conditioned_dit_ms + unconditioned_dit_ms)
            + vae_ms
        )
        id4_ms = _benchmark_duration_ms(benchmark)
        ratio = id4_ms / pytorch_ms

        checks = [
            _correctness_check(root, check)
            for check in _require_list(
                row.get("correctness"), f"acceptance row {name}.correctness"
            )
        ]
        if not checks:
            raise AcceptanceScorecardError(
                f"acceptance row {name}.correctness must not be empty"
            )
        correctness_passed = all(check["status"] == "pass" for check in checks)
        performance_passed = ratio <= maximum_ratio
        output_rows.append(
            {
                "name": name,
                "status": "pass"
                if correctness_passed and performance_passed
                else "fail",
                "request": request,
                "timing": {
                    "id4_ms": id4_ms,
                    "pytorch_ms": pytorch_ms,
                    "ratio": ratio,
                    "maximum_ratio": maximum_ratio,
                    "status": "pass" if performance_passed else "fail",
                    "pytorch_stages_ms": {
                        "qwen": qwen_ms,
                        "dit_conditioned": conditioned_dit_ms,
                        "dit_unconditioned": unconditioned_dit_ms,
                        "vae": vae_ms,
                    },
                },
                "memory": {
                    key: value
                    for key, value in plan_statistics.items()
                    if key.endswith("byte_length") or key.endswith("high_water_mark")
                },
                "topology": {
                    key: value
                    for key, value in plan_statistics.items()
                    if key.endswith("count")
                },
                "correctness": {
                    "status": "pass" if correctness_passed else "fail",
                    "checks": checks,
                },
                "artifacts": {
                    "id4": id4,
                    "pytorch": pytorch,
                    "correctness_reports": [
                        _require_string(
                            _require_dict(check, "correctness check").get("file"),
                            "correctness check.file",
                        )
                        for check in _require_list(
                            row.get("correctness"),
                            f"acceptance row {name}.correctness",
                        )
                    ],
                },
            }
        )
    if not output_rows:
        raise AcceptanceScorecardError("acceptance rows must not be empty")
    return {
        "schema_version": 1,
        "status": "pass"
        if all(row["status"] == "pass" for row in output_rows)
        else "fail",
        "maximum_ratio": maximum_ratio,
        "policy": policy,
        "rows": output_rows,
    }


def format_markdown(scorecard: dict[str, Any]) -> str:
    lines = [
        "| Request | Tokens | Image | ID4 | PyTorch | Ratio | Resident | Transformed | Transient | Command buffers | Stage queue ops | Dispatches | Correctness | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in _require_list(scorecard.get("rows"), "scorecard.rows"):
        request = row["request"]
        timing = row["timing"]
        memory = row["memory"]
        topology = row["topology"]
        lines.append(
            "| {name} | {tokens} | {width}x{height} | {id4:.3f} ms | "
            "{pytorch:.3f} ms | {ratio:.3f}x | {resident:.1f} MiB | "
            "{transformed:.1f} MiB | {transient:.1f} MiB | {command_buffers} | "
            "{queue_ops} | {dispatches} | {correctness} | {status} |".format(
                name=row["name"],
                tokens=request["qwen_token_count"],
                width=request["image_width"],
                height=request["image_height"],
                id4=timing["id4_ms"],
                pytorch=timing["pytorch_ms"],
                ratio=timing["ratio"],
                resident=memory["resident_parameter_byte_length"] / MEBIBYTE,
                transformed=memory["transformed_persistent_byte_length"] / MEBIBYTE,
                transient=memory["local_transient_high_water_mark"] / MEBIBYTE,
                command_buffers=topology["stage_command_buffer_count"],
                queue_ops=topology["stage_queue_operation_count"],
                dispatches=topology["dispatch_count"],
                correctness=row["correctness"]["status"],
                status=row["status"],
            )
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict ID4/PyTorch warm-serving acceptance scorecard."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown table path.")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        scorecard = build_scorecard(args.manifest)
        _write_json(args.output, scorecard)
        if args.markdown is not None:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(format_markdown(scorecard), encoding="utf-8")
    except AcceptanceScorecardError as exc:
        print(f"acceptance_scorecard: {exc}", file=sys.stderr)
        return 2
    return 0 if scorecard["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
