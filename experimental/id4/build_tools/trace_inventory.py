#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds compact inventories from raw ID4 reference traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import trace_reduce

LAYER_BOUNDARY_PATTERN = re.compile(
    r"^ideogram4\.(?P<branch>cond|uncond)\.layers\.(?P<layer>[0-9]+)\."
)


class TraceInventoryError(ValueError):
    """Raised when a trace inventory cannot be produced."""


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TraceInventoryError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise TraceInventoryError(f"{field_name} must be an integer")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TraceInventoryError(f"{field_name} must be a list")
    return value


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_payload_path(record: trace_reduce.TraceRecord) -> Path | None:
    value = record.payload.get("file")
    if value is None:
        return None
    path = Path(_require_string(value, f"{record.stage}/{record.name}.file"))
    if path.is_absolute():
        raise TraceInventoryError(f"{record.stage}/{record.name}.file must be relative")
    if ".." in path.parts:
        raise TraceInventoryError(
            f"{record.stage}/{record.name}.file must not contain '..'"
        )
    return path


def _payload_hash(
    trace_dir: Path, record: trace_reduce.TraceRecord, hash_payloads: bool
) -> str | None:
    if not hash_payloads:
        return None
    payload_path = _relative_payload_path(record)
    if payload_path is None:
        return None
    absolute_path = trace_dir / payload_path
    if not absolute_path.is_file():
        raise TraceInventoryError(
            f"payload file for {record.stage}/{record.name} not found: "
            f"{payload_path.as_posix()}"
        )
    return _file_sha256(absolute_path)


def _record_family(record: trace_reduce.TraceRecord) -> str:
    stage = record.stage
    name = record.name
    if stage == "qwen.prompt":
        return "qwen.prompt"
    if stage == "qwen.encoder":
        return "qwen.encoder.boundary"
    if stage == "graph.debug":
        if ".prelude." in name:
            return "ideogram4.dit.prelude.boundary"
        if ".layers." in name:
            return "ideogram4.dit.layer.boundary"
        if ".output." in name:
            return "ideogram4.dit.output.boundary"
        return "graph.debug"
    if stage.startswith("ideogram4."):
        return "ideogram4.dit.input"
    if stage.startswith("sampler.step."):
        if name in ("guided_pred", "denoised"):
            return "sampler.guidance"
        if name.endswith("dit_output") or name.endswith("_out"):
            return "sampler.dit.output"
        return "sampler.step.input"
    if stage == "sampler.final":
        return "sampler.final"
    return "unclassified"


def _json_value_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value)}
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    if isinstance(value, str):
        return {"type": "string", "bytes": len(value.encode("utf-8"))}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if type(value) in (int, float):
        return {"type": "number"}
    if value is None:
        return {"type": "null"}
    raise TraceInventoryError(f"unsupported JSON trace value type: {type(value)}")


def _inventory_record(
    trace_dir: Path, record: trace_reduce.TraceRecord, hash_payloads: bool
) -> dict[str, Any]:
    kind = _require_string(record.kind, "record kind")
    inventory = {
        "ordinal": _require_int(record.ordinal, "record ordinal"),
        "kind": kind,
        "stage": record.stage,
        "name": record.name,
        "family": _record_family(record),
    }
    producer = record.payload.get("producer")
    if producer is not None:
        inventory["producer"] = _require_string(
            producer, f"{record.stage}/{record.name}.producer"
        )
    payload_path = _relative_payload_path(record)
    if payload_path is not None:
        inventory["file"] = payload_path.as_posix()
    file_sha256 = _payload_hash(trace_dir, record, hash_payloads)
    if file_sha256 is not None:
        inventory["file_sha256"] = file_sha256

    if kind == "tensor":
        shape = [
            _require_int(dim, f"{record.stage}/{record.name}.shape[{index}]")
            for index, dim in enumerate(
                _require_list(
                    record.payload.get("shape"), f"{record.stage}/{record.name}.shape"
                )
            )
        ]
        inventory.update(
            {
                "dtype": _require_string(
                    record.payload.get("dtype"), f"{record.stage}/{record.name}.dtype"
                ),
                "shape": shape,
                "elements": _require_int(
                    record.payload.get("elements"),
                    f"{record.stage}/{record.name}.elements",
                ),
                "bytes": _require_int(
                    record.payload.get("bytes"), f"{record.stage}/{record.name}.bytes"
                ),
            }
        )
        summary = record.payload.get("summary")
        if summary is not None:
            if not isinstance(summary, dict):
                raise TraceInventoryError(
                    f"{record.stage}/{record.name}.summary must be an object"
                )
            inventory["summary"] = summary
        return inventory

    if kind == "text":
        inventory["bytes"] = _require_int(
            record.payload.get("bytes"), f"{record.stage}/{record.name}.bytes"
        )
        return inventory

    if kind == "json":
        inventory["value_summary"] = _json_value_summary(record.payload.get("value"))
        return inventory

    raise TraceInventoryError(f"unsupported trace record kind: {kind}")


def _selected_qwen_layers(records: list[trace_reduce.TraceRecord]) -> list[int]:
    for record in records:
        if record.stage != "qwen.prompt" or record.name != "selected_out_layers":
            continue
        if record.kind != "json":
            raise TraceInventoryError("qwen selected_out_layers record must be JSON")
        value = record.payload.get("value")
        if not isinstance(value, dict):
            raise TraceInventoryError(
                "qwen selected_out_layers value must be an object"
            )
        layers = _require_list(value.get("layers"), "selected_out_layers.layers")
        return [
            _require_int(layer, f"selected_out_layers.layers[{index}]")
            for index, layer in enumerate(layers)
        ]
    return []


def _dit_recorded_layers(
    records: list[trace_reduce.TraceRecord],
) -> list[dict[str, Any]]:
    layer_map: dict[tuple[str, int], int] = {}
    for record in records:
        if record.stage != "graph.debug":
            continue
        match = LAYER_BOUNDARY_PATTERN.match(record.name)
        if not match:
            continue
        key = (match.group("branch"), int(match.group("layer")))
        layer_map[key] = layer_map.get(key, 0) + 1
    return [
        {"branch": branch, "layer": layer, "record_count": layer_map[(branch, layer)]}
        for branch, layer in sorted(layer_map)
    ]


def _sampler_steps(records: list[trace_reduce.TraceRecord]) -> list[int]:
    steps = set()
    for record in records:
        if not record.stage.startswith("sampler.step."):
            continue
        parts = record.stage.split(".")
        if len(parts) < 3:
            raise TraceInventoryError(f"invalid sampler step stage: {record.stage}")
        try:
            step = int(parts[2])
        except ValueError as exc:
            raise TraceInventoryError(
                f"invalid sampler step stage: {record.stage}"
            ) from exc
        steps.add(step)
    return sorted(steps)


def _coverage(records: list[trace_reduce.TraceRecord]) -> dict[str, Any]:
    qwen_record_names = {
        (record.stage, record.name)
        for record in records
        if record.stage.startswith("qwen.")
    }
    qwen_internal_record_count = sum(
        1
        for record in records
        if record.stage == "qwen.encoder"
        and record.name not in ("selected_hidden_states", "condition")
    )
    return {
        "qwen": {
            "has_prompt_tokens": ("qwen.prompt", "token_ids") in qwen_record_names,
            "has_prompt_attention_mask": ("qwen.prompt", "attention_mask")
            in qwen_record_names,
            "selected_out_layers": _selected_qwen_layers(records),
            "has_selected_hidden_states": (
                "qwen.encoder",
                "selected_hidden_states",
            )
            in qwen_record_names,
            "has_condition": ("qwen.encoder", "condition") in qwen_record_names,
            "internal_record_count": qwen_internal_record_count,
        },
        "ideogram4_dit": {
            "recorded_layer_boundaries": _dit_recorded_layers(records),
        },
        "sampler": {
            "steps": _sampler_steps(records),
        },
    }


def build_trace_inventory(trace_dir: Path, hash_payloads: bool) -> dict[str, Any]:
    records_by_key = trace_reduce.load_trace_manifest(trace_dir)
    records = sorted(records_by_key.values(), key=lambda record: record.ordinal)
    seen_ordinals: set[int] = set()
    for record in records:
        if record.ordinal in seen_ordinals:
            raise TraceInventoryError(f"duplicate trace ordinal: {record.ordinal}")
        seen_ordinals.add(record.ordinal)

    kind_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    stage_map: dict[str, dict[str, Any]] = {}
    total_payload_bytes = 0

    for record in records:
        record_inventory = _inventory_record(trace_dir, record, hash_payloads)
        kind = _require_string(record_inventory.get("kind"), "record kind")
        family = _require_string(record_inventory.get("family"), "record family")
        _increment_count(kind_counts, kind)
        _increment_count(family_counts, family)
        if "bytes" in record_inventory:
            total_payload_bytes += _require_int(record_inventory["bytes"], "bytes")

        stage_entry = stage_map.get(record.stage)
        if stage_entry is None:
            stage_entry = {
                "stage": record.stage,
                "record_count": 0,
                "ordinal_range": [record.ordinal, record.ordinal],
                "kind_counts": {},
                "family_counts": {},
                "records": [],
            }
            stage_map[record.stage] = stage_entry
        stage_entry["record_count"] += 1
        stage_entry["ordinal_range"][1] = record.ordinal
        _increment_count(stage_entry["kind_counts"], kind)
        _increment_count(stage_entry["family_counts"], family)
        stage_entry["records"].append(record_inventory)

    manifest_path = trace_dir / "manifest.jsonl"
    return {
        "schema_version": 1,
        "source_trace": {
            "trace_id": trace_dir.name,
            "manifest_sha256": _file_sha256(manifest_path),
        },
        "record_count": len(records),
        "total_payload_bytes": total_payload_bytes,
        "kind_counts": _sorted_counts(kind_counts),
        "family_counts": _sorted_counts(family_counts),
        "coverage": _coverage(records),
        "stages": [
            {
                "stage": stage_entry["stage"],
                "record_count": stage_entry["record_count"],
                "ordinal_range": stage_entry["ordinal_range"],
                "kind_counts": _sorted_counts(stage_entry["kind_counts"]),
                "family_counts": _sorted_counts(stage_entry["family_counts"]),
                "records": stage_entry["records"],
            }
            for stage_entry in sorted(
                stage_map.values(), key=lambda item: item["ordinal_range"][0]
            )
        ],
    }


def write_trace_inventory(
    trace_dir: Path, output: Path, hash_payloads: bool
) -> dict[str, Any]:
    inventory = build_trace_inventory(trace_dir, hash_payloads)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(inventory, file, indent=2, sort_keys=True)
        file.write("\n")
    return inventory


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact inventory from an ID4 raw trace."
    )
    parser.add_argument(
        "--trace-dir",
        required=True,
        type=Path,
        help="Raw trace directory containing manifest.jsonl.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Inventory JSON path to write.",
    )
    parser.add_argument(
        "--hash-payloads",
        action="store_true",
        help="Include SHA256 hashes for tensor and text payload files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        write_trace_inventory(args.trace_dir, args.output, args.hash_payloads)
    except (TraceInventoryError, trace_reduce.TraceReduceError) as exc:
        print(f"trace_inventory: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
