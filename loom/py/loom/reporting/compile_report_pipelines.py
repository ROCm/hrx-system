# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pipeline resource inventory and source-phase external copy demand."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fnmatch import fnmatchcase

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


@dataclass
class _PhaseCopyGroup:
    """Site multiplicities for one record width in one construction occurrence."""

    phase: str
    occurrence: int
    binding: int
    direction: str
    width: int | None
    visits: int | None = 0

    @property
    def byte_count(self) -> int | None:
        if self.visits == 0 or self.width == 0:
            return 0
        if self.width is None or self.visits is None:
            return None
        return self.width * self.visits

    def row(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "occurrence": self.occurrence,
            "binding": self.binding,
            "direction": self.direction,
            "bytes_per_copy": self.width,
            "copies_per_phase_call": self.visits,
            "bytes_per_phase_call": self.byte_count,
        }


def _copy_sites(document: CompileReportDocument) -> Iterator[tuple[str, dict]]:
    sites = document.report.get("pipeline_external_copies", [])
    if not isinstance(sites, list):
        raise CompileReportError("pipeline_external_copies: expected array")
    for i, site in enumerate(sites):
        path = f"pipeline_external_copies[{i}]"
        yield path, _object(site, path)


def single_slot_refills(
    document: CompileReportDocument,
) -> Iterator[tuple[str, str, int, int, int, int]]:
    """Yields reads into a fresh single-slot reservation on each worker visit."""
    for path, row in _copy_sites(document):
        if row.get("direction") != "read" or "reserved_fifo_capacity" not in row:
            continue
        capacity = _count(
            row["reserved_fifo_capacity"], f"{path}.reserved_fifo_capacity"
        )
        if capacity != 1 or "reservations_per_worker" not in row:
            continue
        visits = _count(
            row["reservations_per_worker"], f"{path}.reservations_per_worker"
        )
        width = _nullable_count(row, "bytes_per_copy", path)
        if visits > 1 and width:
            yield (
                path,
                _string(row.get("phase"), f"{path}.phase"),
                _count(row.get("binding"), f"{path}.binding"),
                _count(row.get("reserved_fifo"), f"{path}.reserved_fifo"),
                visits,
                width,
            )


def build_pipeline_phase_copies(
    document: CompileReportDocument, *, summary_only: bool
) -> list[dict[str, object]]:
    """Groups source payload at a single call site, independent of caller visits."""
    roots: dict[str, dict[tuple, _PhaseCopyGroup]] = {}
    for path, row in _copy_sites(document):
        root = _string(row.get("root"), f"{path}.root")
        phase = _string(row.get("phase"), f"{path}.phase")
        occurrence = _count(row.get("occurrence"), f"{path}.occurrence")
        binding = _count(row.get("binding"), f"{path}.binding")
        direction = row.get("direction")
        if direction not in ("read", "write"):
            raise CompileReportError(f"{path}.direction: expected read or write")
        width = _nullable_count(row, "bytes_per_copy", path)
        visits = _nullable_count(row, "copies_per_phase_call", path)
        groups = roots.setdefault(root, {})
        key = phase, occurrence, binding, direction, width
        if key not in groups:
            groups[key] = _PhaseCopyGroup(*key)
        group = groups[key]
        if visits is None:
            group.visits = None
        elif group.visits is not None:
            group.visits += visits

    results = []
    for root, groups in roots.items():
        largest: dict[str, _PhaseCopyGroup] = {}
        unknown = 0
        for group in groups.values():
            amount = group.byte_count
            if amount is None:
                unknown += 1
            elif amount and (
                group.direction not in largest
                or amount > largest[group.direction].byte_count
            ):
                largest[group.direction] = group
        entry = {
            "root": root,
            "group_count": len(groups),
            "unknown_group_count": unknown,
            "largest_counted": [
                largest[d].row() for d in ("read", "write") if d in largest
            ],
        }
        read, write = largest.get("read"), largest.get("write")
        if (
            read
            and write
            and read.occurrence != write.occurrence
            and read.binding == write.binding
            and read.width == write.width
        ):
            entry["range_check"] = (
                f"Both phases use binding[{read.binding}] for {read.width:,} B "
                "records. "
                "Check whether they save and restore the same byte ranges. If they do, "
                "compare keeping that state on chip through its last use."
            )
        if not summary_only:
            entry["groups"] = [group.row() for group in groups.values()]
        results.append(entry)
    return results


def _nullable_count(row: dict, field: str, path: str) -> int | None:
    if field not in row:
        raise CompileReportError(f"{path}.{field}: expected a count or null")
    return None if row[field] is None else _count(row[field], f"{path}.{field}")


def append_pipeline_phase_copies_text(lines: list[str], roots: list) -> None:
    for root in roots:
        lines.extend(("", f"Source-phase external copies in {root['root']}"))
        if root["largest_counted"]:
            lines.append("  Largest fully counted copies per phase call")
        lines.extend(
            f"    {group['direction']} binding[{group['binding']}] "
            f"{group['phase']} (call site {group['occurrence']}): "
            f"{group['copies_per_phase_call']:,} x {group['bytes_per_copy']:,} B "
            f"= {group['bytes_per_phase_call']:,} B"
            for group in root["largest_counted"]
        )
        lines.append(
            f"  {root['unknown_group_count']} of {root['group_count']} groups lack "
            "a proven byte total. Multiply each phase's bytes by its calls "
            "for the request shape."
        )
        if "range_check" in root:
            lines.append(f"  {root['range_check']}")


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


def _placement(value: object, path: str) -> list[int]:
    source = _object(value, path)
    rank = _count(source.get("rank"), f"{path}.rank")
    if rank not in (1, 2, 3):
        raise CompileReportError(f"{path}.rank: expected 1, 2, or 3")
    return [
        _count(source.get(axis), f"{path}.{axis}") for axis in ("x", "y", "z")[:rank]
    ]


def _placed_function(row: dict, path: str, root: str, name: str) -> dict:
    result = {
        "root": root,
        "function": name,
        "placement": _placement(row.get("placement"), f"{path}.placement"),
    }
    for field in (
        "worker_index",
        "group",
        "address",
        "code_byte_count",
        "stored_packet_count",
    ):
        result[field] = _count(row.get(field), f"{path}.{field}")
    for field in ("packets_per_call", "calls_per_entry"):
        if field not in row:
            raise CompileReportError(f"{path}.{field}: expected a count or null")
        result[field] = (
            None if row[field] is None else _count(row[field], f"{path}.{field}")
        )
    result["packet_loops"] = placed_packet_loops(row, path)
    return result


def placed_packet_loops(row: dict, path: str) -> list[dict[str, int]]:
    """Reads the hardware-loop evidence available in this report revision."""
    loops = row.get("packet_loops", [])
    if not isinstance(loops, list):
        raise CompileReportError(f"{path}.packet_loops: expected array")
    result = []
    for i, value in enumerate(loops):
        loop_path = f"{path}.packet_loops[{i}]"
        loop = _object(value, loop_path)
        fields = {
            field: _count(loop.get(field), f"{loop_path}.{field}")
            for field in ("byte_offset", "packet_count", "trip_count")
        }
        if "empty_packet_count" in loop:
            empty = _count(
                loop["empty_packet_count"], f"{loop_path}.empty_packet_count"
            )
            if empty > fields["packet_count"]:
                raise CompileReportError(
                    f"{loop_path}.empty_packet_count: exceeds packet count"
                )
            fields["empty_packet_count"] = empty
        result.append(fields)
    return result


def placed_function_view(document: CompileReportDocument, pattern: str) -> dict:
    """Selects final linked functions without expanding other report families."""
    functions = []
    found_inventory = False
    plans = document.report.get("pipeline_plans", [])
    if not isinstance(plans, list):
        raise CompileReportError(f"{document.source}.pipeline_plans: expected array")
    for i, value in enumerate(plans):
        path = f"{document.source}.pipeline_plans[{i}]"
        plan = _object(value, path)
        if "placed_functions" not in plan:
            continue
        found_inventory = True
        root = _string(plan.get("root"), f"{path}.root")
        rows = plan["placed_functions"]
        if not isinstance(rows, list):
            raise CompileReportError(f"{path}.placed_functions: expected array")
        for j, item in enumerate(rows):
            row_path = f"{path}.placed_functions[{j}]"
            row = _object(item, row_path)
            name = _string(row.get("function"), f"{row_path}.function")
            if fnmatchcase(name, pattern):
                functions.append(_placed_function(row, row_path, root, name))
    if not found_inventory:
        raise CompileReportError("report contains no placed native function inventory")
    if not functions:
        raise CompileReportError(f"no placed native function matches {pattern!r}")
    return {"pattern": pattern, "functions": functions}


def format_placed_function_text(view: dict) -> str:
    rows = view["functions"]
    label = "placement" if len(rows) == 1 else "placements"
    lines = [f"{len(rows)} callable {label} matching {view['pattern']}"]
    for row in rows:
        coordinates = ",".join(map(str, row["placement"]))
        packets = row["packets_per_call"]
        calls = row["calls_per_entry"]
        lines.append(
            f"{row['root']} / worker {row['worker_index']} ({coordinates}) "
            f"group {row['group']} @0x{row['address']:x}  {row['function']}  "
            f"{row['code_byte_count']} bytes, "
            f"{row['stored_packet_count']} stored packets, "
            f"{packets if packets is not None else 'unknown'} packets/call, "
            f"{calls if calls is not None else 'unknown'} calls/entry"
        )
        for loop in row["packet_loops"]:
            empty = loop.get("empty_packet_count")
            occupancy = f" ({empty} empty)" if empty is not None else ""
            lines.append(
                f"  loop +{loop['byte_offset']} bytes  "
                f"{loop['packet_count']} packets per trip{occupancy}, "
                f"{loop['trip_count']} trips"
            )
    return "\n".join(lines) + "\n"


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
            row["placement"] = _placement(
                source_row["placement"], f"{row_path}.placement"
            )
        for field in fields:
            row[field] = _count(source_row.get(field), f"{row_path}.{field}")
        rows.append(row)
    result["rows"] = rows
    return result


def append_pipeline_transfer_text(lines: list[str], plan: dict[str, object]) -> None:
    """Shows the root's transfer costs and the evidence needed to resolve a gap."""
    lines.extend(("", f"Pipeline {plan['root']} ({plan['realization']})"))
    for field, label in _TRAFFIC_FIELDS.items():
        amount = f"{plan[field]} B" if plan[field] is not None else "unavailable"
        lines.append(f"  Planned {label} per activation: {amount}")
    if any(plan[field] is None for field in _TRAFFIC_FIELDS):
        lines.append(
            "  Incomplete transfer evidence. Count records crossing each missing "
            "boundary for the request shape, including repeats and checkpoints."
        )


def append_pipeline_show_text(lines: list[str], view: dict[str, object]) -> None:
    """Formats the selected inventory without inferring channels from addresses."""
    for plan in view["rows"]:
        append_pipeline_transfer_text(lines, plan)
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
