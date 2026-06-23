#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reduces raw ID4 reference traces into compact fixture files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NPY_MAGIC = b"\x93NUMPY"
NPY_VERSION = b"\x01\x00"
NPY_PREFIX_LENGTH = len(NPY_MAGIC) + len(NPY_VERSION) + 2

DTYPE_TABLE = {
    "f32": {
        "byte_count": 4,
        "npy_descr": "<f4",
        "struct_format": "<f",
        "floating": True,
    },
    "f16": {
        "byte_count": 2,
        "npy_descr": "<f2",
        "struct_format": "<e",
        "floating": True,
    },
    "bf16": {
        "byte_count": 2,
        "npy_descr": "<u2",
        "struct_format": "<H",
        "floating": True,
    },
    "i32": {
        "byte_count": 4,
        "npy_descr": "<i4",
        "struct_format": "<i",
        "floating": False,
    },
    "u32": {
        "byte_count": 4,
        "npy_descr": "<u4",
        "struct_format": "<I",
        "floating": False,
    },
}

VALID_ROLES = frozenset(("input", "expected", "metadata"))


class TraceReduceError(ValueError):
    """Raised when a trace, plan, or tensor payload violates the reducer API."""


@dataclass(frozen=True)
class TraceRecord:
    kind: str
    stage: str
    name: str
    ordinal: int
    payload: dict[str, Any]


def _bf16_bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def unpack_numeric_payload(dtype: str, payload: bytes) -> tuple[Any, ...]:
    dtype_info = DTYPE_TABLE[dtype]
    element_byte_count = int(dtype_info["byte_count"])
    if len(payload) % element_byte_count != 0:
        raise TraceReduceError(f"{dtype} payload length is not element-aligned")
    values = tuple(
        value[0]
        for value in struct.iter_unpack(str(dtype_info["struct_format"]), payload)
    )
    if dtype == "bf16":
        return tuple(_bf16_bits_to_f32(value) for value in values)
    return values


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TraceReduceError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise TraceReduceError(f"{field_name} must be an integer")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TraceReduceError(f"{field_name} must be a list")
    return value


def _record_key(stage: str, name: str) -> tuple[str, str]:
    return (stage, name)


def _relative_path(value: Any, field_name: str) -> Path:
    path = Path(_require_string(value, field_name))
    if path.is_absolute():
        raise TraceReduceError(f"{field_name} must be a relative path")
    if ".." in path.parts:
        raise TraceReduceError(f"{field_name} must not contain '..'")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace_manifest(trace_dir: Path) -> dict[tuple[str, str], TraceRecord]:
    manifest_path = trace_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise TraceReduceError(f"trace manifest not found: {manifest_path}")
    records = {}
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceReduceError(
                    f"invalid JSON in manifest line {line_number}: {exc}"
                ) from exc
            kind = _require_string(payload.get("kind"), "manifest record kind")
            stage = _require_string(payload.get("stage"), "manifest record stage")
            name = _require_string(payload.get("name"), "manifest record name")
            ordinal = _require_int(payload.get("ordinal"), "manifest record ordinal")
            key = _record_key(stage, name)
            if key in records:
                raise TraceReduceError(f"duplicate trace record: {stage}/{name}")
            records[key] = TraceRecord(
                kind=kind,
                stage=stage,
                name=name,
                ordinal=ordinal,
                payload=payload,
            )
    if not records:
        raise TraceReduceError(f"trace manifest is empty: {manifest_path}")
    return records


def load_plan(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as plan_file:
            plan = json.load(plan_file)
    except json.JSONDecodeError as exc:
        raise TraceReduceError(f"invalid reduction plan JSON: {exc}") from exc
    if not isinstance(plan, dict):
        raise TraceReduceError("reduction plan must be a JSON object")
    _require_string(plan.get("fixture_id"), "fixture_id")
    tensors = plan.get("tensors")
    texts = plan.get("texts")
    _require_list(tensors, "tensors")
    _require_list(texts, "texts")
    if not tensors and not texts:
        raise TraceReduceError("reduction plan must select at least one record")
    return plan


def _normalize_slice(shape: list[int], slice_specs: list[Any]) -> list[tuple[int, int]]:
    if len(slice_specs) != len(shape):
        raise TraceReduceError(
            f"slice rank {len(slice_specs)} does not match tensor rank {len(shape)}"
        )
    normalized = []
    for dim_index, (dim_size, spec) in enumerate(zip(shape, slice_specs)):
        if not isinstance(spec, dict):
            raise TraceReduceError(f"slice[{dim_index}] must be an object")
        start = _require_int(spec.get("start"), f"slice[{dim_index}].start")
        length = _require_int(spec.get("length"), f"slice[{dim_index}].length")
        if start < 0:
            raise TraceReduceError(f"slice[{dim_index}].start must be non-negative")
        if length <= 0:
            raise TraceReduceError(f"slice[{dim_index}].length must be positive")
        if start + length > dim_size:
            raise TraceReduceError(
                f"slice[{dim_index}] [{start}, {start + length}) exceeds "
                f"dimension size {dim_size}"
            )
        normalized.append((start, length))
    return normalized


def _trace_payload_strides(shape: list[int]) -> list[int]:
    # Reference traces are emitted from sd::Tensor/ggml buffers, where
    # dimension 0 is contiguous. The reduced NPY payload is written in row-major
    # order so fixture consumers do not inherit that storage convention.
    strides = [1] * len(shape)
    running = 1
    for dim_index, dim_size in enumerate(shape):
        strides[dim_index] = running
        running *= dim_size
    return strides


def _read_tensor_slice(
    tensor_path: Path,
    shape: list[int],
    dtype: str,
    slices: list[tuple[int, int]],
) -> bytes:
    dtype_info = DTYPE_TABLE[dtype]
    element_byte_count = int(dtype_info["byte_count"])
    strides = _trace_payload_strides(shape)
    output = bytearray()

    with tensor_path.open("rb") as tensor_file:
        expected_byte_count = math.prod(shape) * element_byte_count
        actual_byte_count = tensor_path.stat().st_size
        if actual_byte_count != expected_byte_count:
            raise TraceReduceError(
                f"tensor payload size mismatch for {tensor_path}: expected "
                f"{expected_byte_count} bytes, found {actual_byte_count}"
            )
        source_payload = tensor_file.read()

    def copy_from_dimension(dim_index: int, base_offset: int) -> None:
        if dim_index == len(shape):
            byte_offset = base_offset * element_byte_count
            output.extend(
                source_payload[byte_offset : byte_offset + element_byte_count]
            )
            return
        start, length = slices[dim_index]
        for index in range(start, start + length):
            copy_from_dimension(dim_index + 1, base_offset + index * strides[dim_index])

    copy_from_dimension(0, 0)
    return bytes(output)


def _shape_literal(shape: list[int]) -> str:
    if not shape:
        return "()"
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def write_npy(path: Path, dtype: str, shape: list[int], payload: bytes) -> None:
    dtype_info = DTYPE_TABLE[dtype]
    expected_byte_count = math.prod(shape) * int(dtype_info["byte_count"])
    if len(payload) != expected_byte_count:
        raise TraceReduceError(
            f"NPY payload for {path} has {len(payload)} bytes, expected "
            f"{expected_byte_count}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "{'descr': '"
        + str(dtype_info["npy_descr"])
        + "', 'fortran_order': False, 'shape': "
        + _shape_literal(shape)
        + ", }"
    )
    padding = 16 - ((NPY_PREFIX_LENGTH + len(header) + 1) % 16)
    if padding == 16:
        padding = 0
    header_bytes = (header + (" " * padding) + "\n").encode("ascii")
    if len(header_bytes) > 65535:
        raise TraceReduceError(f"NPY header is too large for {path}")
    with path.open("wb") as file:
        file.write(NPY_MAGIC)
        file.write(NPY_VERSION)
        file.write(struct.pack("<H", len(header_bytes)))
        file.write(header_bytes)
        file.write(payload)


def _summarize_numeric_payload(dtype: str, payload: bytes) -> dict[str, Any]:
    dtype_info = DTYPE_TABLE[dtype]
    finite_count = 0
    nan_count = 0
    inf_count = 0
    minimum = None
    maximum = None
    total = 0.0
    for value in unpack_numeric_payload(dtype, payload):
        if dtype_info["floating"] and math.isnan(value):
            nan_count += 1
            continue
        if dtype_info["floating"] and math.isinf(value):
            inf_count += 1
            continue
        finite_count += 1
        total += float(value)
        minimum = value if minimum is None else min(minimum, value)
        maximum = value if maximum is None else max(maximum, value)
    mean = None if finite_count == 0 else total / finite_count
    return {
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min": minimum,
        "max": maximum,
        "mean": mean,
    }


def _require_role(entry: dict[str, Any], entry_name: str) -> str:
    role = _require_string(entry.get("role"), f"{entry_name}.role")
    if role not in VALID_ROLES:
        raise TraceReduceError(
            f"{entry_name}.role must be one of {sorted(VALID_ROLES)}"
        )
    return role


def _validate_expected_tolerance(
    entry: dict[str, Any], role: str, dtype: str, entry_name: str
) -> dict[str, Any] | None:
    tolerance = entry.get("tolerance")
    if tolerance is None:
        if role == "expected" and DTYPE_TABLE[dtype]["floating"]:
            raise TraceReduceError(
                f"{entry_name}.tolerance is required for floating expected tensors"
            )
        return None
    if not isinstance(tolerance, dict):
        raise TraceReduceError(
            f"{entry_name}.tolerance must be an object with atol and rtol"
        )
    for field in ("atol", "rtol"):
        value = tolerance.get(field)
        if type(value) not in (int, float):
            raise TraceReduceError(f"{entry_name}.tolerance.{field} must be numeric")
        if value < 0:
            raise TraceReduceError(
                f"{entry_name}.tolerance.{field} must be non-negative"
            )
    return tolerance


def _reduce_tensor(
    trace_dir: Path,
    output_dir: Path,
    records: dict[tuple[str, str], TraceRecord],
    entry: dict[str, Any],
    entry_index: int,
) -> dict[str, Any]:
    entry_name = f"tensors[{entry_index}]"
    role = _require_role(entry, entry_name)
    stage = _require_string(entry.get("stage"), f"{entry_name}.stage")
    name = _require_string(entry.get("name"), f"{entry_name}.name")
    output_path = _relative_path(entry.get("output"), f"{entry_name}.output")
    record = records.get(_record_key(stage, name))
    if not record:
        raise TraceReduceError(f"trace tensor record not found: {stage}/{name}")
    if record.kind != "tensor":
        raise TraceReduceError(f"trace record is not a tensor: {stage}/{name}")
    dtype = _require_string(record.payload.get("dtype"), f"{stage}/{name}.dtype")
    if dtype not in DTYPE_TABLE:
        raise TraceReduceError(f"unsupported tensor dtype for {stage}/{name}: {dtype}")
    shape = [
        _require_int(dim, f"{stage}/{name}.shape")
        for dim in _require_list(record.payload.get("shape"), f"{stage}/{name}.shape")
    ]
    slices = _normalize_slice(
        shape, _require_list(entry.get("slice"), f"{entry_name}.slice")
    )
    output_shape = [length for _, length in slices]
    tolerance = _validate_expected_tolerance(entry, role, dtype, entry_name)
    source_file = _relative_path(record.payload.get("file"), f"{stage}/{name}.file")
    tensor_path = trace_dir / source_file
    if not tensor_path.is_file():
        raise TraceReduceError(f"tensor payload not found: {tensor_path}")
    payload = _read_tensor_slice(tensor_path, shape, dtype, slices)
    write_npy(output_dir / output_path, dtype, output_shape, payload)
    reduced_record = {
        "kind": "tensor",
        "role": role,
        "stage": stage,
        "name": name,
        "source_ordinal": record.ordinal,
        "source_dtype": dtype,
        "source_shape": shape,
        "source_file_sha256": _file_sha256(tensor_path),
        "slice": [{"start": start, "length": length} for start, length in slices],
        "dtype": dtype,
        "shape": output_shape,
        "file": output_path.as_posix(),
        "summary": _summarize_numeric_payload(dtype, payload),
    }
    if tolerance is not None:
        reduced_record["tolerance"] = tolerance
    return reduced_record


def _reduce_text(
    trace_dir: Path,
    output_dir: Path,
    records: dict[tuple[str, str], TraceRecord],
    entry: dict[str, Any],
    entry_index: int,
) -> dict[str, Any]:
    entry_name = f"texts[{entry_index}]"
    role = _require_role(entry, entry_name)
    stage = _require_string(entry.get("stage"), f"{entry_name}.stage")
    name = _require_string(entry.get("name"), f"{entry_name}.name")
    output_path = _relative_path(entry.get("output"), f"{entry_name}.output")
    record = records.get(_record_key(stage, name))
    if not record:
        raise TraceReduceError(f"trace text record not found: {stage}/{name}")
    if record.kind != "text":
        raise TraceReduceError(f"trace record is not text: {stage}/{name}")
    source_file = _relative_path(record.payload.get("file"), f"{stage}/{name}.file")
    text_path = trace_dir / source_file
    if not text_path.is_file():
        raise TraceReduceError(f"text payload not found: {text_path}")
    text_payload = text_path.read_bytes()
    target_path = output_dir / output_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(text_payload)
    return {
        "kind": "text",
        "role": role,
        "stage": stage,
        "name": name,
        "source_ordinal": record.ordinal,
        "source_file_sha256": _file_sha256(text_path),
        "bytes": len(text_payload),
        "file": output_path.as_posix(),
    }


def _empty_role_counts() -> dict[str, int]:
    return {role: 0 for role in sorted(VALID_ROLES)}


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _inventory_record(record: dict[str, Any]) -> dict[str, Any]:
    kind = _require_string(record.get("kind"), "inventory record kind")
    if kind not in ("tensor", "text"):
        raise TraceReduceError(f"unsupported inventory record kind: {kind}")
    inventory = {
        "kind": kind,
        "role": record["role"],
        "stage": record["stage"],
        "name": record["name"],
        "file": record["file"],
    }
    if kind == "text":
        inventory["bytes"] = record["bytes"]
        return inventory
    inventory.update(
        {
            "dtype": record["dtype"],
            "shape": record["shape"],
            "source_dtype": record["source_dtype"],
            "source_shape": record["source_shape"],
        }
    )
    if "tolerance" in record:
        inventory["tolerance"] = record["tolerance"]
    return inventory


def build_fixture_inventory(manifest: dict[str, Any]) -> dict[str, Any]:
    stage_map: dict[str, dict[str, Any]] = {}
    role_counts = _empty_role_counts()
    kind_counts: dict[str, int] = {}
    for record in _require_list(manifest.get("records"), "manifest.records"):
        if not isinstance(record, dict):
            raise TraceReduceError("manifest.records entries must be objects")
        stage = _require_string(record.get("stage"), "manifest record stage")
        role = _require_role(record, "manifest record")
        kind = _require_string(record.get("kind"), "manifest record kind")
        _increment_count(role_counts, role)
        _increment_count(kind_counts, kind)
        stage_entry = stage_map.get(stage)
        if stage_entry is None:
            stage_entry = {
                "stage": stage,
                "role_counts": _empty_role_counts(),
                "records": [],
            }
            stage_map[stage] = stage_entry
        _increment_count(stage_entry["role_counts"], role)
        stage_entry["records"].append(_inventory_record(record))
    return {
        "schema_version": 1,
        "fixture_id": _require_string(manifest.get("fixture_id"), "fixture_id"),
        "record_count": len(manifest["records"]),
        "kind_counts": dict(sorted(kind_counts.items())),
        "role_counts": role_counts,
        "stages": [
            {
                "stage": stage_entry["stage"],
                "role_counts": stage_entry["role_counts"],
                "records": stage_entry["records"],
            }
            for stage_entry in sorted(
                stage_map.values(), key=lambda item: item["stage"]
            )
        ],
    }


def reduce_trace(
    trace_dir: Path, output_dir: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    records = load_trace_manifest(trace_dir)
    manifest_path = trace_dir / "manifest.jsonl"
    reduced_records = []
    for index, entry in enumerate(_require_list(plan.get("texts"), "texts")):
        if not isinstance(entry, dict):
            raise TraceReduceError(f"texts[{index}] must be an object")
        reduced_records.append(
            _reduce_text(trace_dir, output_dir, records, entry, index)
        )
    for index, entry in enumerate(_require_list(plan.get("tensors"), "tensors")):
        if not isinstance(entry, dict):
            raise TraceReduceError(f"tensors[{index}] must be an object")
        reduced_records.append(
            _reduce_tensor(trace_dir, output_dir, records, entry, index)
        )
    output_manifest = {
        "schema_version": 1,
        "fixture_id": _require_string(plan.get("fixture_id"), "fixture_id"),
        "source_trace": {
            "manifest_sha256": _file_sha256(manifest_path),
        },
        "records": reduced_records,
    }
    source_label = plan.get("source_label")
    if source_label is not None:
        output_manifest["source_trace"]["source_label"] = _require_string(
            source_label, "source_label"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(output_manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    inventory = build_fixture_inventory(output_manifest)
    with (output_dir / "inventory.json").open("w", encoding="utf-8") as inventory_file:
        json.dump(inventory, inventory_file, indent=2, sort_keys=True)
        inventory_file.write("\n")
    return output_manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce an ID4 raw trace into compact fixture files."
    )
    parser.add_argument(
        "--trace-dir",
        required=True,
        type=Path,
        help="Raw trace directory containing manifest.jsonl.",
    )
    parser.add_argument(
        "--plan",
        required=True,
        type=Path,
        help="JSON reduction plan.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory that will receive manifest.json and fixture files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        plan = load_plan(args.plan)
        reduce_trace(args.trace_dir, args.output_dir, plan)
    except TraceReduceError as exc:
        print(f"trace_reduce: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
