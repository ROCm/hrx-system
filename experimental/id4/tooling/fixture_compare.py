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
SUPPORTED_ACTUAL_ROLES = frozenset(("actual", "output"))
SUPPORTED_EXPECTED_ROLES = frozenset(("expected",))


class FixtureCompareError(ValueError):
    """Raised when fixture comparison inputs are malformed."""


@dataclass(frozen=True)
class TensorPayload:
    dtype: str
    shape: tuple[int, ...]
    values: tuple[Any, ...]


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
    values = tuple(
        value[0] for value in struct.iter_unpack(str(info["struct_format"]), data)
    )
    return TensorPayload(dtype=dtype, shape=shape, values=values)


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
    tensor = _read_npy_tensor(path)
    if tensor.dtype != dtype:
        raise FixtureCompareError(
            f"{description} dtype mismatch: manifest {dtype}, NPY {tensor.dtype}"
        )
    if tensor.shape != shape:
        raise FixtureCompareError(
            f"{description} shape mismatch: manifest {list(shape)}, "
            f"NPY {list(tensor.shape)}"
        )
    return tensor


def _record_text(root: Path, record: dict[str, Any], description: str) -> bytes:
    path = root / _relative_path(record.get("file"), f"{description}.file")
    if not path.is_file():
        raise FixtureCompareError(f"{description} payload not found: {path}")
    return path.read_bytes()


def _tolerance(record: dict[str, Any]) -> tuple[float, float]:
    tolerance = record.get("tolerance")
    if not isinstance(tolerance, dict):
        raise FixtureCompareError(
            f"{record['stage']}/{record['name']} expected tensor requires tolerance"
        )
    atol = tolerance.get("atol")
    rtol = tolerance.get("rtol")
    if type(atol) not in (int, float) or type(rtol) not in (int, float):
        raise FixtureCompareError(
            f"{record['stage']}/{record['name']} tolerance requires numeric atol/rtol"
        )
    if atol < 0 or rtol < 0:
        raise FixtureCompareError(
            f"{record['stage']}/{record['name']} tolerance must be non-negative"
        )
    return float(atol), float(rtol)


def _compare_float_values(
    expected: TensorPayload,
    actual: TensorPayload,
    expected_record: dict[str, Any],
) -> dict[str, Any]:
    atol, rtol = _tolerance(expected_record)
    mismatch_count = 0
    first_mismatch = None
    max_abs_error = 0.0
    max_rel_error = 0.0
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
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)
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
    return {
        "atol": atol,
        "rtol": rtol,
        "element_count": len(expected.values),
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "first_mismatch": first_mismatch,
    }


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
    if actual.shape != expected.shape:
        comparison.update(
            {
                "status": "fail",
                "reason": "shape_mismatch",
                "actual_shape": list(actual.shape),
            }
        )
        return comparison
    if trace_reduce.DTYPE_TABLE[expected.dtype]["floating"]:
        metrics = _compare_float_values(expected, actual, expected_record)
    else:
        metrics = _compare_exact_values(expected, actual)
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an ID4 actual run capture against a reference fixture."
    )
    parser.add_argument(
        "--fixture-dir",
        required=True,
        type=Path,
        help="Generated fixture directory containing manifest.json.",
    )
    parser.add_argument(
        "--actual-dir",
        required=True,
        type=Path,
        help="Actual run capture directory containing manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. Reports are printed to stdout by default.",
    )
    return parser.parse_args()


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
        report = compare_fixtures(args.fixture_dir, args.actual_dir)
        write_report(report, args.output)
    except (FixtureCompareError, trace_reduce.TraceReduceError) as exc:
        print(f"fixture_compare: {exc}", file=sys.stderr)
        return 2
    return 0 if report["summary"]["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
