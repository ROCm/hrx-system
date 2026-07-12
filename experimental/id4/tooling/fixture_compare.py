#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compares ID4 actual run captures against reference fixtures."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trace_reduce

NPY_MAGIC = b"\x93NUMPY"
ID4_TENSOR_MAGIC = b"ID4TENSR"
SUPPORTED_ACTUAL_ROLES = frozenset(("actual", "output"))
SUPPORTED_EXPECTED_ROLES = frozenset(("expected",))
RESULT_TENSOR_STAGE = "result"


class FixtureCompareError(ValueError):
    """Raised when fixture comparison inputs are malformed."""


@dataclass(frozen=True)
class TensorPayload:
    dtype: str
    shape: tuple[int, ...]
    values: tuple[Any, ...]


@dataclass(frozen=True)
class SliceSpec:
    start: int
    length: int


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FixtureCompareError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise FixtureCompareError(f"{field_name} must be an integer")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise FixtureCompareError(f"{field_name} must be a list")
    return value


def _relative_path(value: Any, field_name: str) -> Path:
    path = Path(_require_string(value, field_name))
    if path.is_absolute():
        raise FixtureCompareError(f"{field_name} must be a relative path")
    if ".." in path.parts:
        raise FixtureCompareError(f"{field_name} must not contain '..'")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FixtureCompareError(f"{description} not found: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise FixtureCompareError(f"invalid {description}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FixtureCompareError(f"{description} must be a JSON object: {path}")
    return payload


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        _require_string(record.get("stage"), "record.stage"),
        _require_string(record.get("name"), "record.name"),
    )


def _record_role(record: dict[str, Any]) -> str:
    return _require_string(record.get("role"), "record.role")


def _record_kind(record: dict[str, Any]) -> str:
    return _require_string(record.get("kind"), "record.kind")


def _records_by_key(
    manifest: dict[str, Any],
    accepted_roles: frozenset[str],
    description: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    records = {}
    for index, record in enumerate(_require_list(manifest.get("records"), "records")):
        if not isinstance(record, dict):
            raise FixtureCompareError(
                f"{description}.records[{index}] must be an object"
            )
        role = _record_role(record)
        if role not in accepted_roles:
            continue
        key = _record_key(record)
        if key in records:
            stage, name = key
            raise FixtureCompareError(
                f"duplicate {description} record for {stage}/{name}"
            )
        records[key] = record
    return records


def _result_records_by_key(
    manifest: dict[str, Any],
    description: str,
) -> dict[str, dict[str, Any]]:
    if (
        _require_string(manifest.get("format"), f"{description}.format")
        != "id4tensor-v1"
    ):
        raise FixtureCompareError(
            f"{description}.format must be id4tensor-v1 for result tensor compare"
        )
    records = {}
    for index, record in enumerate(_require_list(manifest.get("records"), "records")):
        if not isinstance(record, dict):
            raise FixtureCompareError(
                f"{description}.records[{index}] must be an object"
            )
        key = _require_string(record.get("key"), f"{description}.records[{index}].key")
        if key in records:
            raise FixtureCompareError(f"duplicate {description} result record: {key}")
        records[key] = record
    return records


def _result_shape(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, dict):
        rank = _require_int(value.get("rank"), f"{field_name}.rank")
        dims = _require_list(value.get("dims"), f"{field_name}.dims")
        if rank != len(dims):
            raise FixtureCompareError(
                f"{field_name}.rank {rank} does not match dims length {len(dims)}"
            )
        return dims
    return _require_list(value, field_name)


def _result_tensor_record(
    record: dict[str, Any],
    role: str,
    tolerance: dict[str, float] | None = None,
) -> dict[str, Any]:
    converted = {
        "dtype": _require_string(record.get("dtype"), "result.dtype"),
        "file": _require_string(record.get("file"), "result.file"),
        "kind": "tensor",
        "name": _require_string(record.get("key"), "result.key"),
        "role": role,
        "shape": _result_shape(record.get("shape"), "result.shape"),
        "stage": RESULT_TENSOR_STAGE,
    }
    if tolerance is not None:
        converted["tolerance"] = tolerance
    return converted


def _dtype_from_npy_descr(descr: str) -> str:
    for dtype, info in trace_reduce.DTYPE_TABLE.items():
        if info["npy_descr"] == descr:
            return dtype
    raise FixtureCompareError(f"unsupported NPY dtype descriptor: {descr}")


def _read_npy_tensor(path: Path) -> TensorPayload:
    payload = path.read_bytes()
    if len(payload) < 10 or payload[: len(NPY_MAGIC)] != NPY_MAGIC:
        raise FixtureCompareError(f"NPY payload missing magic: {path}")
    if payload[6:8] != b"\x01\x00":
        raise FixtureCompareError(f"unsupported NPY version for {path}: {payload[6:8]}")
    header_length = struct.unpack("<H", payload[8:10])[0]
    header_start = 10
    header_end = header_start + header_length
    if header_end > len(payload):
        raise FixtureCompareError(f"truncated NPY header: {path}")
    try:
        header = ast.literal_eval(payload[header_start:header_end].decode("ascii"))
    except (SyntaxError, ValueError) as exc:
        raise FixtureCompareError(f"invalid NPY header for {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise FixtureCompareError(f"NPY header must be a dictionary: {path}")
    if header.get("fortran_order") is not False:
        raise FixtureCompareError(f"fortran-order NPY arrays are unsupported: {path}")
    dtype = _dtype_from_npy_descr(_require_string(header.get("descr"), "npy.descr"))
    shape_value = header.get("shape")
    if not isinstance(shape_value, tuple):
        raise FixtureCompareError(f"NPY shape must be a tuple: {path}")
    shape = tuple(_require_int(dim, "npy.shape") for dim in shape_value)
    info = trace_reduce.DTYPE_TABLE[dtype]
    element_count = math.prod(shape)
    expected_byte_count = element_count * int(info["byte_count"])
    data = payload[header_end:]
    if len(data) != expected_byte_count:
        raise FixtureCompareError(
            f"NPY payload byte length mismatch for {path}: expected "
            f"{expected_byte_count}, found {len(data)}"
        )
    values = trace_reduce.unpack_numeric_payload(dtype, data)
    return TensorPayload(dtype=dtype, shape=shape, values=values)


def _read_id4_tensor(path: Path) -> TensorPayload:
    payload = path.read_bytes()
    if len(payload) < len(ID4_TENSOR_MAGIC) + 4:
        raise FixtureCompareError(f"ID4 tensor payload is truncated: {path}")
    if payload[: len(ID4_TENSOR_MAGIC)] != ID4_TENSOR_MAGIC:
        raise FixtureCompareError(f"ID4 tensor payload missing magic: {path}")
    header_length = struct.unpack("<I", payload[len(ID4_TENSOR_MAGIC) : 12])[0]
    header_start = len(ID4_TENSOR_MAGIC) + 4
    header_end = header_start + header_length
    if header_end > len(payload):
        raise FixtureCompareError(f"ID4 tensor header overruns file: {path}")
    try:
        header = json.loads(payload[header_start:header_end].decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureCompareError(f"invalid ID4 tensor header: {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise FixtureCompareError(f"ID4 tensor header must be an object: {path}")
    if _require_int(header.get("version"), "id4tensor.version") != 1:
        raise FixtureCompareError(f"unsupported ID4 tensor version in {path}")
    if _require_string(header.get("kind"), "id4tensor.kind") != "tensor":
        raise FixtureCompareError(f"ID4 payload is not a tensor: {path}")
    if _require_string(header.get("layout"), "id4tensor.layout") != "dense-row-major":
        raise FixtureCompareError(f"unsupported ID4 tensor layout: {path}")
    dtype = _require_string(header.get("dtype"), "id4tensor.dtype")
    storage_dtype = _require_string(
        header.get("storage_dtype"), "id4tensor.storage_dtype"
    )
    if dtype != storage_dtype:
        raise FixtureCompareError(
            f"ID4 tensor dtype/storage_dtype mismatch in {path}: "
            f"{dtype} vs {storage_dtype}"
        )
    if dtype not in trace_reduce.DTYPE_TABLE:
        raise FixtureCompareError(f"unsupported ID4 tensor dtype in {path}: {dtype}")
    shape = tuple(
        _require_int(dim, "id4tensor.shape")
        for dim in _require_list(header.get("shape"), "id4tensor.shape")
    )
    storage_shape = tuple(
        _require_int(dim, "id4tensor.storage_shape")
        for dim in _require_list(header.get("storage_shape"), "id4tensor.storage_shape")
    )
    if shape != storage_shape:
        raise FixtureCompareError(
            f"ID4 tensor shape/storage_shape mismatch in {path}: "
            f"{list(shape)} vs {list(storage_shape)}"
        )
    data = payload[header_end:]
    byte_length = _require_int(header.get("byte_length"), "id4tensor.byte_length")
    if byte_length != len(data):
        raise FixtureCompareError(
            f"ID4 tensor payload byte length mismatch for {path}: header "
            f"{byte_length}, found {len(data)}"
        )
    info = trace_reduce.DTYPE_TABLE[dtype]
    expected_byte_count = math.prod(shape) * int(info["byte_count"])
    if len(data) != expected_byte_count:
        raise FixtureCompareError(
            f"ID4 tensor dense byte length mismatch for {path}: expected "
            f"{expected_byte_count}, found {len(data)}"
        )
    return TensorPayload(
        dtype=dtype,
        shape=shape,
        values=trace_reduce.unpack_numeric_payload(dtype, data),
    )


def _read_tensor_payload(path: Path) -> TensorPayload:
    prefix = path.read_bytes()[: max(len(NPY_MAGIC), len(ID4_TENSOR_MAGIC))]
    if prefix.startswith(NPY_MAGIC):
        return _read_npy_tensor(path)
    if prefix.startswith(ID4_TENSOR_MAGIC):
        return _read_id4_tensor(path)
    raise FixtureCompareError(f"unsupported tensor payload format: {path}")


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    running = 1
    for dim_index in range(len(shape) - 1, -1, -1):
        strides[dim_index] = running
        running *= shape[dim_index]
    return tuple(strides)


def _normalize_slice(
    record: dict[str, Any],
    actual_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> tuple[SliceSpec, ...] | None:
    slice_value = record.get("slice")
    if slice_value is None:
        return None
    slice_entries = _require_list(slice_value, "expected.slice")
    if len(slice_entries) != len(actual_shape):
        raise FixtureCompareError(
            f"{record['stage']}/{record['name']} slice rank {len(slice_entries)} "
            f"does not match actual rank {len(actual_shape)}"
        )
    normalized = []
    for dim_index, (entry, dim_size) in enumerate(zip(slice_entries, actual_shape)):
        if not isinstance(entry, dict):
            raise FixtureCompareError(
                f"{record['stage']}/{record['name']} slice[{dim_index}] "
                "must be an object"
            )
        start = _require_int(entry.get("start"), f"slice[{dim_index}].start")
        length = _require_int(entry.get("length"), f"slice[{dim_index}].length")
        if start < 0:
            raise FixtureCompareError(
                f"{record['stage']}/{record['name']} slice[{dim_index}].start "
                "must be non-negative"
            )
        if length <= 0:
            raise FixtureCompareError(
                f"{record['stage']}/{record['name']} slice[{dim_index}].length "
                "must be positive"
            )
        if start + length > dim_size:
            raise FixtureCompareError(
                f"{record['stage']}/{record['name']} slice[{dim_index}] "
                f"[{start}, {start + length}) exceeds actual dimension {dim_size}"
            )
        normalized.append(SliceSpec(start=start, length=length))
    slice_shape = tuple(spec.length for spec in normalized)
    if slice_shape != expected_shape:
        raise FixtureCompareError(
            f"{record['stage']}/{record['name']} slice shape {list(slice_shape)} "
            f"does not match expected shape {list(expected_shape)}"
        )
    return tuple(normalized)


def _slice_tensor(
    tensor: TensorPayload,
    slices: tuple[SliceSpec, ...],
) -> TensorPayload:
    strides = _row_major_strides(tensor.shape)
    values = []

    def copy_from_dimension(dim_index: int, base_index: int) -> None:
        spec = slices[dim_index]
        if dim_index == len(tensor.shape) - 1:
            start_index = base_index + spec.start * strides[dim_index]
            end_index = start_index + spec.length
            values.extend(tensor.values[start_index:end_index])
            return
        for index in range(spec.start, spec.start + spec.length):
            copy_from_dimension(dim_index + 1, base_index + index * strides[dim_index])

    copy_from_dimension(0, 0)
    return TensorPayload(
        dtype=tensor.dtype,
        shape=tuple(spec.length for spec in slices),
        values=tuple(values),
    )


def _record_tensor(
    root: Path, record: dict[str, Any], description: str
) -> TensorPayload:
    dtype = _require_string(record.get("dtype"), f"{description}.dtype")
    if dtype not in trace_reduce.DTYPE_TABLE:
        raise FixtureCompareError(f"unsupported {description}.dtype: {dtype}")
    shape = tuple(
        _require_int(dim, f"{description}.shape")
        for dim in _require_list(record.get("shape"), f"{description}.shape")
    )
    path = root / _relative_path(record.get("file"), f"{description}.file")
    if not path.is_file():
        raise FixtureCompareError(f"{description} payload not found: {path}")
    tensor = _read_tensor_payload(path)
    if tensor.dtype != dtype:
        raise FixtureCompareError(
            f"{description} dtype mismatch: manifest {dtype}, payload {tensor.dtype}"
        )
    if tensor.shape != shape:
        raise FixtureCompareError(
            f"{description} shape mismatch: manifest {list(shape)}, "
            f"payload {list(tensor.shape)}"
        )
    return tensor


def _record_text(root: Path, record: dict[str, Any], description: str) -> bytes:
    path = root / _relative_path(record.get("file"), f"{description}.file")
    if not path.is_file():
        raise FixtureCompareError(f"{description} payload not found: {path}")
    return path.read_bytes()


def _tolerance(record: dict[str, Any]) -> dict[str, float | str]:
    tolerance = record.get("tolerance")
    if not isinstance(tolerance, dict):
        raise FixtureCompareError(
            f"{record['stage']}/{record['name']} expected tensor requires tolerance"
        )
    atol = tolerance.get("atol")
    rtol = tolerance.get("rtol")
    has_elementwise = atol is not None or rtol is not None
    mean_abs = tolerance.get("mean_abs")
    p99_abs = tolerance.get("p99_abs")
    max_abs = tolerance.get("max_abs")
    has_aggregate = mean_abs is not None or p99_abs is not None or max_abs is not None
    if has_elementwise and has_aggregate:
        raise FixtureCompareError(
            f"{record['stage']}/{record['name']} tolerance must declare exactly one mode"
        )
    if has_elementwise:
        if type(atol) not in (int, float) or type(rtol) not in (int, float):
            raise FixtureCompareError(
                f"{record['stage']}/{record['name']} tolerance requires numeric atol/rtol"
            )
        if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
            raise FixtureCompareError(
                f"{record['stage']}/{record['name']} tolerance must be finite and "
                "non-negative"
            )
        return {"mode": "elementwise", "atol": float(atol), "rtol": float(rtol)}
    if not has_aggregate:
        raise FixtureCompareError(
            f"{record['stage']}/{record['name']} tolerance must declare a comparison mode"
        )
    if any(type(value) not in (int, float) for value in (mean_abs, p99_abs, max_abs)):
        raise FixtureCompareError(
            f"{record['stage']}/{record['name']} aggregate tolerance requires numeric "
            "mean_abs/p99_abs/max_abs"
        )
    if (
        not math.isfinite(mean_abs)
        or not math.isfinite(p99_abs)
        or not math.isfinite(max_abs)
        or mean_abs < 0
        or p99_abs < 0
        or max_abs < 0
    ):
        raise FixtureCompareError(
            f"{record['stage']}/{record['name']} tolerance must be finite and "
            "non-negative"
        )
    return {
        "mode": "aggregate",
        "mean_abs": float(mean_abs),
        "p99_abs": float(p99_abs),
        "max_abs": float(max_abs),
    }


def _compare_float_values(
    expected: TensorPayload,
    actual: TensorPayload,
    expected_record: dict[str, Any],
) -> dict[str, Any]:
    tolerance = _tolerance(expected_record)
    tolerance_mode = _require_string(tolerance.get("mode"), "tolerance.mode")
    atol = float(tolerance.get("atol", 0.0))
    rtol = float(tolerance.get("rtol", 0.0))
    mismatch_count = 0
    first_mismatch = None
    abs_error_sum = 0.0
    squared_error_sum = 0.0
    max_abs_error = 0.0
    max_abs_error_index = None
    max_rel_error = 0.0
    max_rel_error_index = None
    absolute_errors = []
    for index, (expected_value, actual_value) in enumerate(
        zip(expected.values, actual.values, strict=True)
    ):
        if math.isnan(expected_value) or math.isnan(actual_value):
            passed = math.isnan(expected_value) and math.isnan(actual_value)
            abs_error = 0.0 if passed else math.inf
            rel_error = 0.0 if passed else math.inf
        elif math.isinf(expected_value) or math.isinf(actual_value):
            passed = expected_value == actual_value
            abs_error = 0.0 if passed else math.inf
            rel_error = 0.0 if passed else math.inf
        else:
            abs_error = abs(float(actual_value) - float(expected_value))
            if expected_value == 0:
                rel_error = 0.0 if abs_error == 0 else math.inf
            else:
                rel_error = abs_error / abs(float(expected_value))
            passed = abs_error <= atol + rtol * abs(float(expected_value))
        if tolerance_mode == "aggregate":
            if not math.isfinite(expected_value) or not math.isfinite(actual_value):
                abs_error = math.inf
                rel_error = math.inf
            passed = True
        abs_error_sum += abs_error
        squared_error_sum += abs_error * abs_error
        if tolerance_mode == "aggregate":
            absolute_errors.append(abs_error)
        if abs_error >= max_abs_error:
            max_abs_error = abs_error
            max_abs_error_index = index
        if rel_error >= max_rel_error:
            max_rel_error = rel_error
            max_rel_error_index = index
        if not passed:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "index": index,
                    "expected": expected_value,
                    "actual": actual_value,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                }
    element_count = len(expected.values)
    mean_abs_error = abs_error_sum / element_count if element_count != 0 else 0.0
    rmse = math.sqrt(squared_error_sum / element_count) if element_count != 0 else 0.0
    metrics = {
        "tolerance_mode": tolerance_mode,
        "element_count": element_count,
        "mismatch_count": mismatch_count,
        "mean_abs_error": mean_abs_error,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "max_abs_error_index": max_abs_error_index,
        "max_rel_error": max_rel_error,
        "max_rel_error_index": max_rel_error_index,
        "first_mismatch": first_mismatch,
    }
    if tolerance_mode == "elementwise":
        metrics.update({"atol": atol, "rtol": rtol})
        return metrics
    if not absolute_errors:
        raise FixtureCompareError(
            f"{expected_record['stage']}/{expected_record['name']} aggregate "
            "comparison requires at least one tensor element"
        )
    absolute_errors.sort()
    p99_index = (element_count * 99 + 99) // 100 - 1
    p99_abs_error = absolute_errors[p99_index]
    mean_abs_limit = float(tolerance["mean_abs"])
    p99_abs_limit = float(tolerance["p99_abs"])
    max_abs_limit = float(tolerance["max_abs"])
    aggregate_passed = (
        mean_abs_error <= mean_abs_limit
        and p99_abs_error <= p99_abs_limit
        and max_abs_error <= max_abs_limit
    )
    metrics.update(
        {
            "mean_abs_limit": mean_abs_limit,
            "p99_abs_error": p99_abs_error,
            "p99_abs_limit": p99_abs_limit,
            "max_abs_limit": max_abs_limit,
            "mismatch_count": 0 if aggregate_passed else 1,
        }
    )
    return metrics


def _compare_exact_values(
    expected: TensorPayload,
    actual: TensorPayload,
) -> dict[str, Any]:
    mismatch_count = 0
    first_mismatch = None
    for index, (expected_value, actual_value) in enumerate(
        zip(expected.values, actual.values, strict=True)
    ):
        if expected_value == actual_value:
            continue
        mismatch_count += 1
        if first_mismatch is None:
            first_mismatch = {
                "index": index,
                "expected": expected_value,
                "actual": actual_value,
            }
    return {
        "element_count": len(expected.values),
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
    }


def _compare_tensors(
    fixture_root: Path,
    actual_root: Path,
    expected_record: dict[str, Any],
    actual_record: dict[str, Any],
) -> dict[str, Any]:
    stage, name = _record_key(expected_record)
    expected = _record_tensor(
        fixture_root, expected_record, f"expected tensor {stage}/{name}"
    )
    actual = _record_tensor(actual_root, actual_record, f"actual tensor {stage}/{name}")
    comparison = {
        "kind": "tensor",
        "stage": stage,
        "name": name,
        "dtype": expected.dtype,
        "shape": list(expected.shape),
    }
    if actual.dtype != expected.dtype:
        comparison.update(
            {
                "status": "fail",
                "reason": "dtype_mismatch",
                "actual_dtype": actual.dtype,
            }
        )
        return comparison
    actual_for_comparison = actual
    if actual.shape != expected.shape:
        actual_slice = _normalize_slice(expected_record, actual.shape, expected.shape)
        if actual_slice is None:
            comparison.update(
                {
                    "status": "fail",
                    "reason": "shape_mismatch",
                    "actual_shape": list(actual.shape),
                }
            )
            return comparison
        actual_for_comparison = _slice_tensor(actual, actual_slice)
        comparison["actual_shape"] = list(actual.shape)
        comparison["slice"] = [
            {"start": spec.start, "length": spec.length} for spec in actual_slice
        ]
    if trace_reduce.DTYPE_TABLE[expected.dtype]["floating"]:
        metrics = _compare_float_values(
            expected, actual_for_comparison, expected_record
        )
    else:
        metrics = _compare_exact_values(expected, actual_for_comparison)
    comparison.update(metrics)
    comparison["status"] = "pass" if metrics["mismatch_count"] == 0 else "fail"
    return comparison


def _compare_texts(
    fixture_root: Path,
    actual_root: Path,
    expected_record: dict[str, Any],
    actual_record: dict[str, Any],
) -> dict[str, Any]:
    stage, name = _record_key(expected_record)
    expected = _record_text(
        fixture_root, expected_record, f"expected text {stage}/{name}"
    )
    actual = _record_text(actual_root, actual_record, f"actual text {stage}/{name}")
    return {
        "kind": "text",
        "stage": stage,
        "name": name,
        "expected_bytes": len(expected),
        "actual_bytes": len(actual),
        "status": "pass" if expected == actual else "fail",
    }


def _manifest_summary(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "manifest_sha256": _file_sha256(path),
    }
    fixture_id = manifest.get("fixture_id")
    if isinstance(fixture_id, str) and fixture_id:
        summary["fixture_id"] = fixture_id
    run_id = manifest.get("run_id")
    if isinstance(run_id, str) and run_id:
        summary["run_id"] = run_id
    source_trace = manifest.get("source_trace")
    if isinstance(source_trace, dict):
        manifest_sha256 = source_trace.get("manifest_sha256")
        if isinstance(manifest_sha256, str) and manifest_sha256:
            summary["source_trace_manifest_sha256"] = manifest_sha256
    return summary


def compare_fixtures(fixture_root: Path, actual_root: Path) -> dict[str, Any]:
    fixture_manifest_path = _manifest_path(fixture_root)
    actual_manifest_path = _manifest_path(actual_root)
    fixture_manifest = _load_json(fixture_manifest_path, "fixture manifest")
    actual_manifest = _load_json(actual_manifest_path, "actual manifest")
    expected_records = _records_by_key(
        fixture_manifest, SUPPORTED_EXPECTED_ROLES, "fixture"
    )
    if not expected_records:
        raise FixtureCompareError("fixture manifest contains no expected records")
    actual_records = _records_by_key(actual_manifest, SUPPORTED_ACTUAL_ROLES, "actual")
    comparisons = []
    for key, expected_record in sorted(expected_records.items()):
        stage, name = key
        actual_record = actual_records.get(key)
        if not actual_record:
            comparisons.append(
                {
                    "stage": stage,
                    "name": name,
                    "kind": _record_kind(expected_record),
                    "status": "fail",
                    "reason": "missing_actual_record",
                }
            )
            continue
        expected_kind = _record_kind(expected_record)
        actual_kind = _record_kind(actual_record)
        if expected_kind != actual_kind:
            comparisons.append(
                {
                    "stage": stage,
                    "name": name,
                    "kind": expected_kind,
                    "status": "fail",
                    "reason": "kind_mismatch",
                    "actual_kind": actual_kind,
                }
            )
            continue
        if expected_kind == "tensor":
            comparisons.append(
                _compare_tensors(
                    fixture_root, actual_root, expected_record, actual_record
                )
            )
        elif expected_kind == "text":
            comparisons.append(
                _compare_texts(
                    fixture_root, actual_root, expected_record, actual_record
                )
            )
        else:
            raise FixtureCompareError(
                f"unsupported expected record kind for {stage}/{name}: {expected_kind}"
            )
    passed_count = sum(
        1 for comparison in comparisons if comparison["status"] == "pass"
    )
    failed_count = len(comparisons) - passed_count
    return {
        "schema_version": 1,
        "fixture": _manifest_summary(fixture_manifest_path, fixture_manifest),
        "actual": _manifest_summary(actual_manifest_path, actual_manifest),
        "summary": {
            "expected_count": len(expected_records),
            "compared_count": len(comparisons),
            "passed_count": passed_count,
            "failed_count": failed_count,
        },
        "comparisons": comparisons,
    }


def compare_result_tensors(
    expected_root: Path,
    actual_root: Path,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected_manifest_path = _manifest_path(expected_root)
    actual_manifest_path = _manifest_path(actual_root)
    expected_manifest = _load_json(expected_manifest_path, "expected result manifest")
    actual_manifest = _load_json(actual_manifest_path, "actual result manifest")
    expected_records = _result_records_by_key(expected_manifest, "expected result")
    if not expected_records:
        raise FixtureCompareError("expected result manifest contains no records")
    actual_records = _result_records_by_key(actual_manifest, "actual result")
    tolerance = {"atol": atol, "rtol": rtol}
    comparisons = []
    for key, expected_record in sorted(expected_records.items()):
        actual_record = actual_records.get(key)
        if not actual_record:
            comparisons.append(
                {
                    "stage": RESULT_TENSOR_STAGE,
                    "name": key,
                    "kind": "tensor",
                    "status": "fail",
                    "reason": "missing_actual_record",
                }
            )
            continue
        comparisons.append(
            _compare_tensors(
                expected_root,
                actual_root,
                _result_tensor_record(expected_record, "expected", tolerance),
                _result_tensor_record(actual_record, "actual"),
            )
        )
    for key in sorted(set(actual_records) - set(expected_records)):
        comparisons.append(
            {
                "stage": RESULT_TENSOR_STAGE,
                "name": key,
                "kind": "tensor",
                "status": "fail",
                "reason": "extra_actual_record",
            }
        )
    passed_count = sum(
        1 for comparison in comparisons if comparison["status"] == "pass"
    )
    failed_count = len(comparisons) - passed_count
    return {
        "schema_version": 1,
        "expected": _manifest_summary(expected_manifest_path, expected_manifest),
        "actual": _manifest_summary(actual_manifest_path, actual_manifest),
        "summary": {
            "expected_count": len(expected_records),
            "actual_count": len(actual_records),
            "compared_count": len(comparisons),
            "passed_count": passed_count,
            "failed_count": failed_count,
        },
        "comparisons": comparisons,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ID4 actual run captures against reference fixtures or "
            "compare two CLI result tensor directories."
        )
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Generated fixture directory containing manifest.json.",
    )
    parser.add_argument(
        "--actual-dir",
        type=Path,
        help="Actual run capture directory containing manifest.json.",
    )
    parser.add_argument(
        "--expected-result-dir",
        type=Path,
        help="CLI result tensor directory to use as the expected payload set.",
    )
    parser.add_argument(
        "--actual-result-dir",
        type=Path,
        help="CLI result tensor directory to compare against the expected set.",
    )
    parser.add_argument(
        "--atol",
        default=0.0,
        type=float,
        help="Absolute tolerance for result tensor floating-point comparisons.",
    )
    parser.add_argument(
        "--rtol",
        default=0.0,
        type=float,
        help="Relative tolerance for result tensor floating-point comparisons.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. Reports are printed to stdout by default.",
    )
    args = parser.parse_args()
    fixture_mode = args.fixture_dir is not None or args.actual_dir is not None
    result_mode = (
        args.expected_result_dir is not None or args.actual_result_dir is not None
    )
    if fixture_mode == result_mode:
        parser.error(
            "select exactly one input mode: --fixture-dir/--actual-dir or "
            "--expected-result-dir/--actual-result-dir"
        )
    if fixture_mode and (args.fixture_dir is None or args.actual_dir is None):
        parser.error("--fixture-dir and --actual-dir must be provided together")
    if result_mode and (
        args.expected_result_dir is None or args.actual_result_dir is None
    ):
        parser.error(
            "--expected-result-dir and --actual-result-dir must be provided together"
        )
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    return args


def write_report(report: dict[str, Any], output: Path | None) -> None:
    if output is None:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")


def main() -> int:
    args = parse_arguments()
    try:
        if args.fixture_dir is not None:
            report = compare_fixtures(args.fixture_dir, args.actual_dir)
        else:
            report = compare_result_tensors(
                args.expected_result_dir,
                args.actual_result_dir,
                args.atol,
                args.rtol,
            )
        write_report(report, args.output)
    except (FixtureCompareError, trace_reduce.TraceReduceError) as exc:
        print(f"fixture_compare: {exc}", file=sys.stderr)
        return 2
    return 0 if report["summary"]["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
