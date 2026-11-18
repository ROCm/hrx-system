# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Physical pipeline inventory, separate from per-function and loop reports."""

from __future__ import annotations

from loom.reporting.compile_report import CompileReportDocument, CompileReportError

_CODE_FIELDS = {
    "program_count": "images",
    "program_code_byte_count": "unique code bytes",
    "worker_code_byte_count": "loaded code bytes",
    "minimum_worker_code_headroom_byte_count": "minimum code headroom bytes",
}
_STORAGE_FIELDS = {
    "reserved_storage_byte_count": "reserved bytes",
    "worker_storage_byte_count": "program data bytes",
    "channel_storage_byte_count": "channel bytes",
}
_TRAFFIC_FIELDS = {
    "external_dma_byte_count": "external DMA",
    "routed_dma_byte_count": "routed on-array DMA",
}
_WORKER_FIELDS = (
    "code_byte_count",
    "code_capacity_byte_count",
    "worker_storage_byte_count",
    "local_memory_byte_count",
    "local_memory_capacity_byte_count",
)
_MEMORY_FIELDS = (
    "reserved_byte_count",
    "program_data_byte_count",
    "occupied_byte_count",
    "high_water_byte_count",
    "capacity_byte_count",
    "maximum_bank_storage_byte_count",
    "bank_storage_capacity_byte_count",
)
_CHANNEL_FIELDS = (
    "capacity",
    "record_count",
    "record_byte_count",
    "activation_byte_count",
    "local_storage_byte_count",
)


def build_pipeline_show(
    document: CompileReportDocument, *, summary_only: bool = False
) -> dict[str, object] | None:
    """Selects bounded physical facts, preserving absent logical provenance."""
    if "pipeline_plans" not in document.report:
        return None
    values = document.report["pipeline_plans"]
    if not isinstance(values, list):
        raise CompileReportError(f"{document.source}.pipeline_plans: expected array")
    plans = []
    for index, value in enumerate(values):
        path = f"{document.source}.pipeline_plans[{index}]"
        source = _object(value, path)
        plan: dict[str, object] = {
            "root": _string(source.get("root"), f"{path}.root"),
            "realization": _string(source.get("realization"), f"{path}.realization"),
        }
        for field in (*_CODE_FIELDS, *_STORAGE_FIELDS):
            if field in source:
                plan[field] = _count(source[field], f"{path}.{field}")
        for field in _TRAFFIC_FIELDS:
            plan[field] = (
                _count(source[field], f"{path}.{field}") if field in source else None
            )
        for name, index_field, fields in (
            ("workers", "worker_index", _WORKER_FIELDS),
            ("memories", "memory_index", _MEMORY_FIELDS),
            ("channels", "channel_index", _CHANNEL_FIELDS),
        ):
            if name in source:
                inventory = _inventory(
                    source[name], f"{path}.{name}", index_field, fields, summary_only
                )
                if name == "channels":
                    plan["channel_count"] = inventory["count"]
                    if "rows" in inventory:
                        plan["channel_rows"] = inventory["rows"]
                else:
                    plan[name] = inventory
        plans.append(plan)
    return {"count": len(plans), "rows": plans}


def _inventory(
    value: object,
    path: str,
    index_field: str,
    fields: tuple[str, ...],
    summary_only: bool,
) -> dict[str, object]:
    source = _object(value, path)
    count = _count(source.get("count"), f"{path}.count")
    result: dict[str, object] = {"count": count}
    if summary_only or "rows" not in source:
        return result
    values = source["rows"]
    if not isinstance(values, list) or len(values) != count:
        raise CompileReportError(f"{path}.rows: expected {count} rows")
    rows = []
    for index, value in enumerate(values):
        row_path = f"{path}.rows[{index}]"
        source_row = _object(value, row_path)
        if _count(source_row.get(index_field), f"{row_path}.{index_field}") != index:
            raise CompileReportError(f"{row_path}.{index_field}: expected {index}")
        row: dict[str, object] = {index_field: index}
        if index_field == "worker_index":
            row["entry"] = _string(source_row.get("entry"), f"{row_path}.entry")
        if index_field == "channel_index":
            row["transport"] = _string(
                source_row.get("transport"), f"{row_path}.transport"
            )
            for end in ("sender", "receiver"):
                endpoint_path = f"{row_path}.{end}"
                endpoint = _object(source_row.get(end), endpoint_path)
                row[end] = {
                    "owner": _string(endpoint.get("owner"), f"{endpoint_path}.owner"),
                    "owner_index": _count(
                        endpoint.get("owner_index"), f"{endpoint_path}.owner_index"
                    ),
                    "port": _count(endpoint.get("port"), f"{endpoint_path}.port"),
                }
            if "external_transfer" in source_row:
                transfer_path = f"{row_path}.external_transfer"
                transfer = _object(source_row["external_transfer"], transfer_path)
                row["external_transfer"] = {
                    key: _count(transfer.get(key), f"{transfer_path}.{key}")
                    for key in (
                        "binding_ordinal",
                        "task_byte_count",
                        "task_repeat_count",
                        "activation_byte_count",
                    )
                }
        if "placement" in source_row:
            placement = _object(source_row["placement"], f"{row_path}.placement")
            rank = _count(placement.get("rank"), f"{row_path}.placement.rank")
            if rank not in (1, 2, 3):
                raise CompileReportError(
                    f"{row_path}.placement.rank: expected 1, 2, or 3"
                )
            row["placement"] = [
                _count(placement.get(axis), f"{row_path}.placement.{axis}")
                for axis in ("x", "y", "z")[:rank]
            ]
        for field in fields:
            row[field] = _count(source_row.get(field), f"{row_path}.{field}")
        rows.append(row)
    result["rows"] = rows
    return result


def append_pipeline_show_text(lines: list[str], view: dict[str, object]) -> None:
    """Formats the selected inventory without inferring channels from addresses."""
    for plan in view["rows"]:
        lines.extend(("", f"Pipeline {plan['root']} ({plan['realization']})"))
        for field, label in _TRAFFIC_FIELDS.items():
            amount = f"{plan[field]} B" if plan[field] is not None else "unavailable"
            lines.append(f"  Planned {label} per activation: {amount}")
        if any(plan[field] is None for field in _TRAFFIC_FIELDS):
            lines.append(
                "  Incomplete transfer evidence. Count records crossing each missing "
                "boundary for the request shape, including repeats and checkpoints."
            )
        if "workers" in plan:
            lines.append(f"  workers: {plan['workers']['count']}")
        if "memories" in plan:
            lines.append(f"  physical memory owners: {plan['memories']['count']}")
        for title, fields in (("Code", _CODE_FIELDS), ("Storage", _STORAGE_FIELDS)):
            metrics = [
                f"{plan[field]} {label}"
                for field, label in fields.items()
                if field in plan
            ]
            if metrics:
                lines.append(f"  {title}: " + "; ".join(metrics))
        channels = plan.get("channel_count")
        lines.append(
            f"  logical channels: {channels if channels is not None else 'unavailable'}"
        )
        for channel in plan.get("channel_rows", []):
            sender, receiver = channel["sender"], channel["receiver"]
            lines.append(
                f"  channel[{channel['channel_index']}] {channel['transport']} "
                f"{sender['owner']}[{sender['owner_index']}]:{sender['port']} -> "
                f"{receiver['owner']}[{receiver['owner_index']}]:{receiver['port']}: "
                f"{channel['record_count']} records x {channel['record_byte_count']} B "
                f"= {channel['activation_byte_count']} B per activation, "
                f"{channel['capacity']} slots, "
                f"{channel['local_storage_byte_count']} B storage"
            )
            transfer = channel.get("external_transfer")
            if transfer is not None:
                lines.append(
                    f"    binding[{transfer['binding_ordinal']}] DMA: "
                    f"{transfer['task_repeat_count']} tasks x "
                    f"{transfer['task_byte_count']} B "
                    f"= {transfer['activation_byte_count']} B per activation"
                )
        for worker in plan.get("workers", {}).get("rows", []):
            placement = _placement_text(worker)
            lines.append(
                f"  worker[{worker['worker_index']}] {worker['entry']}{placement}: "
                f"{worker['code_byte_count']}/{worker['code_capacity_byte_count']} "
                "B code; "
                f"{worker['worker_storage_byte_count']} B program data"
            )
        for memory in plan.get("memories", {}).get("rows", []):
            placement = _placement_text(memory)
            lines.append(
                f"  memory[{memory['memory_index']}]{placement}: "
                f"{memory['occupied_byte_count']}/{memory['capacity_byte_count']} "
                "B occupied; "
                f"{memory['reserved_byte_count']} B reserved + "
                f"{memory['program_data_byte_count']} B program data; "
                f"extent {memory['high_water_byte_count']} B; "
                f"peak bank {memory['maximum_bank_storage_byte_count']}/"
                f"{memory['bank_storage_capacity_byte_count']} B"
            )


def _placement_text(row: dict[str, object]) -> str:
    placement = row.get("placement")
    return (
        " at [" + ",".join(str(value) for value in placement) + "]"
        if placement is not None
        else ""
    )


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{path}: expected object")
    return value


def _count(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompileReportError(f"{path}: expected nonnegative integer")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompileReportError(f"{path}: expected nonempty string")
    return value
