# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native contraction layout views, diffs, and text formatting."""

from __future__ import annotations

from loom.reporting.compile_report import CompileReportDocument, CompileReportError

_ROLES = ("lhs", "rhs", "accumulator", "result")
_EVIDENCE_KINDS = frozenset(("exact", "metadata-dependent", "parametric", "opaque"))
_PHYSICAL_DIMENSIONS = frozenset(("participant", "position"))
_MISSING = object()


def build_native_layout_show(
    document: CompileReportDocument,
) -> dict[str, object] | None:
    """Builds native placement and transition views from detailed rows."""
    source_low_value = document.report.get("source_low")
    if source_low_value is None:
        return None
    source_low = _require_object(source_low_value, f"{document.source}.source_low")
    rows_value = source_low.get("rows")
    if rows_value is None:
        if document.mode == "details":
            raise CompileReportError(
                f"{document.source}.source_low.rows: detailed report is missing "
                "source rows"
            )
        return None
    if document.mode != "details":
        raise CompileReportError(
            f"{document.source}.source_low.rows: row-level native layout facts "
            "require mode='details'"
        )
    rows = _require_list(rows_value, f"{document.source}.source_low.rows")
    source_low_count = _require_count(
        source_low.get("count"), f"{document.source}.source_low.count"
    )
    if source_low_count != len(rows):
        raise CompileReportError(
            f"{document.source}.source_low.count: expected {len(rows)}, "
            f"got {source_low_count}"
        )

    native_rows = []
    for position, row_value in enumerate(rows):
        row_source = f"{document.source}.source_low.rows[{position}]"
        row = _require_object(row_value, row_source)
        report_index = _require_count(row.get("index"), f"{row_source}.index")
        if report_index != position:
            raise CompileReportError(
                f"{row_source}.index: expected {position}, got {report_index}"
            )
        if (
            row.get("native_contraction") is None
            and row.get("native_transition") is None
        ):
            continue
        native_rows.append(_build_native_row(row, row_source))

    if not native_rows:
        return None
    return {
        "row_count": len(native_rows),
        "rows": native_rows,
    }


def build_native_layout_diff(
    baseline: CompileReportDocument,
    candidate: CompileReportDocument,
    *,
    entry_function_pairs: tuple[tuple[str | None, str | None], ...] = (),
) -> dict[str, object] | None:
    """Builds a source-row-aware diff of retained native layout facts."""
    baseline_show = build_native_layout_show(baseline)
    candidate_show = build_native_layout_show(candidate)
    if baseline_show is None and candidate_show is None:
        return None
    if (baseline_show is None or candidate_show is None) and (
        baseline.mode != "details" or candidate.mode != "details"
    ):
        return {
            "availability": {
                "baseline": "available" if baseline_show is not None else "unavailable",
                "candidate": (
                    "available" if candidate_show is not None else "unavailable"
                ),
            },
            "changed_row_count": 0,
            "unchanged_row_count": 0,
            "rows": [],
        }

    function_aliases = _candidate_function_aliases(entry_function_pairs)
    baseline_rows = _rows_by_key(
        (
            _require_list(baseline_show["rows"], "baseline native layout rows")
            if baseline_show is not None
            else []
        ),
        baseline.source,
    )
    candidate_rows = _rows_by_key(
        (
            _require_list(candidate_show["rows"], "candidate native layout rows")
            if candidate_show is not None
            else []
        ),
        candidate.source,
        function_aliases=function_aliases,
    )

    changed_rows = []
    unchanged_row_count = 0
    for key in sorted(set(baseline_rows) | set(candidate_rows)):
        baseline_row = baseline_rows.get(key)
        candidate_row = candidate_rows.get(key)
        if baseline_row is None:
            changed_rows.append(
                {
                    "identity": candidate_row["identity"],
                    "status": "added",
                    "candidate": candidate_row,
                }
            )
            continue
        if candidate_row is None:
            changed_rows.append(
                {
                    "identity": baseline_row["identity"],
                    "status": "removed",
                    "baseline": baseline_row,
                }
            )
            continue

        baseline_comparable = _comparable_row(baseline_row)
        candidate_comparable = _comparable_row(
            candidate_row,
            function_aliases=function_aliases,
        )
        if baseline_comparable == candidate_comparable:
            unchanged_row_count += 1
            continue
        changes: list[dict[str, object]] = []
        _append_value_changes(changes, baseline_comparable, candidate_comparable)
        row_diff: dict[str, object] = {
            "identity": baseline_row["identity"],
            "status": "changed",
            "changes": changes,
            "baseline": baseline_row,
            "candidate": candidate_row,
        }
        if baseline_row["identity"] != candidate_row["identity"]:
            row_diff["identities"] = {
                "baseline": baseline_row["identity"],
                "candidate": candidate_row["identity"],
            }
        changed_rows.append(row_diff)

    return {
        "changed_row_count": len(changed_rows),
        "unchanged_row_count": unchanged_row_count,
        "rows": changed_rows,
    }


def append_native_layout_show_text(
    lines: list[str], native_layout: dict[str, object]
) -> None:
    """Appends the human-readable native layout view."""
    lines.extend(("", "Native layout economics (compiler analysis)"))
    lines.append(f"  source rows: {native_layout['row_count']}")
    for row_value in _require_list(native_layout["rows"], "native layout rows"):
        _append_native_row(lines, _require_object(row_value, "native layout row"))


def append_native_layout_diff_text(
    lines: list[str], native_layout: dict[str, object]
) -> None:
    """Appends the human-readable native layout diff."""
    lines.extend(("", "Native layout economics diff (compiler analysis)"))
    availability = native_layout.get("availability")
    if isinstance(availability, dict):
        lines.append(
            f"  evidence: {availability['baseline']} -> {availability['candidate']}"
        )
        return
    lines.append(
        f"  source rows: {native_layout['changed_row_count']} changed, "
        f"{native_layout['unchanged_row_count']} unchanged"
    )
    for row_value in _require_list(native_layout["rows"], "native layout diff rows"):
        row = _require_object(row_value, "native layout diff row")
        status = row["status"]
        identity = _require_object(row["identity"], "native layout row identity")
        lines.append(f"  {_format_identity(identity)}: {status}")
        if status == "added":
            _append_native_row(
                lines,
                _require_object(row["candidate"], "added native layout row"),
                indent="    ",
                include_identity=False,
            )
        elif status == "removed":
            _append_native_row(
                lines,
                _require_object(row["baseline"], "removed native layout row"),
                indent="    ",
                include_identity=False,
            )
        else:
            _append_changed_row(lines, row, indent="    ")


def _build_native_row(row: dict[str, object], source: str) -> dict[str, object]:
    identity = {
        "report_index": _require_count(row.get("index"), f"{source}.index"),
        "function": _optional_string(row.get("function"), f"{source}.function"),
        "source_op": _optional_string(row.get("source_op"), f"{source}.source_op"),
        "source_op_kind": _require_count(
            row.get("source_op_kind"), f"{source}.source_op_kind"
        ),
    }
    selection: dict[str, object] = {
        "kind": _require_string(row.get("selection"), f"{source}.selection"),
        "emitted_low_op_count": _require_count(
            row.get("emitted_low_op_count"), f"{source}.emitted_low_op_count"
        ),
    }
    for field in ("plan_key", "descriptor_key", "descriptor_semantic_tag"):
        value = _optional_string(row.get(field), f"{source}.{field}")
        if value is not None:
            selection[field] = value

    view: dict[str, object] = {
        "identity": identity,
        "selection": selection,
    }
    contraction_value = row.get("native_contraction")
    if contraction_value is not None:
        view["contraction"] = _build_contraction(
            contraction_value, f"{source}.native_contraction"
        )
    transition_value = row.get("native_transition")
    if transition_value is not None:
        contraction = view.get("contraction")
        if not isinstance(contraction, dict):
            raise CompileReportError(
                f"{source}.native_transition: transition requires retained "
                "native_contraction facts"
            )
        view["transition"] = _build_transition(
            transition_value,
            contraction,
            f"{source}.native_transition",
        )
    return view


def _build_contraction(value: object, source: str) -> dict[str, object]:
    contraction = _require_object(value, source)
    tile_value = _require_object(contraction.get("tile"), f"{source}.tile")
    tile = {
        dimension: _require_positive_count(
            tile_value.get(dimension), f"{source}.tile.{dimension}"
        )
        for dimension in ("blocks", "m", "n", "k")
    }
    participant_count = _require_positive_count(
        contraction.get("participants"), f"{source}.participants"
    )
    expected_logical_counts = {
        "lhs": tile["blocks"] * tile["m"] * tile["k"],
        "rhs": tile["blocks"] * tile["k"] * tile["n"],
        "accumulator": tile["blocks"] * tile["m"] * tile["n"],
        "result": tile["blocks"] * tile["m"] * tile["n"],
    }
    roles = {
        role: _build_role(
            contraction.get(role),
            role,
            participant_count,
            expected_logical_counts[role],
            f"{source}.{role}",
        )
        for role in _ROLES
    }
    return {
        "tile": tile,
        "participant_count": participant_count,
        "roles": roles,
    }


def _build_role(
    value: object,
    role: str,
    participant_count: int,
    expected_logical_count: int,
    source: str,
) -> dict[str, object]:
    role_value = _require_object(value, source)
    evidence = _require_string(role_value.get("evidence"), f"{source}.evidence")
    if evidence not in _EVIDENCE_KINDS:
        raise CompileReportError(
            f"{source}.evidence: unknown native layout evidence {evidence!r}"
        )
    result: dict[str, object] = {
        "evidence": evidence,
        "element_bits": _require_positive_count(
            role_value.get("element_bits"), f"{source}.element_bits"
        ),
        "registers_per_participant": _require_positive_count(
            role_value.get("registers_per_participant"),
            f"{source}.registers_per_participant",
        ),
        "payload_elements_per_participant": _require_positive_count(
            role_value.get("payload_elements_per_participant"),
            f"{source}.payload_elements_per_participant",
        ),
        "physical_positions": _require_positive_count(
            role_value.get("physical_positions"),
            f"{source}.physical_positions",
        ),
        "logical_coordinates": _require_positive_count(
            role_value.get("logical_coordinates"),
            f"{source}.logical_coordinates",
        ),
    }
    if result["physical_positions"] % participant_count != 0:
        raise CompileReportError(
            f"{source}.physical_positions: {result['physical_positions']} does "
            f"not divide across {participant_count} participants"
        )
    result["physical_positions_per_participant"] = (
        result["physical_positions"] // participant_count
    )
    if result["logical_coordinates"] != expected_logical_count:
        raise CompileReportError(
            f"{source}.logical_coordinates: expected {expected_logical_count} "
            f"from the native tile, got {result['logical_coordinates']}"
        )

    owners_value = role_value.get("owners_per_coordinate")
    if evidence == "exact":
        owners = _require_object(owners_value, f"{source}.owners_per_coordinate")
        minimum = _require_positive_count(
            owners.get("minimum"), f"{source}.owners_per_coordinate.minimum"
        )
        maximum = _require_positive_count(
            owners.get("maximum"), f"{source}.owners_per_coordinate.maximum"
        )
        if minimum > maximum:
            raise CompileReportError(
                f"{source}.owners_per_coordinate: minimum {minimum} exceeds "
                f"maximum {maximum}"
            )
        physical_positions = result["physical_positions"]
        logical_coordinates = result["logical_coordinates"]
        if not (
            logical_coordinates * minimum
            <= physical_positions
            <= logical_coordinates * maximum
        ):
            raise CompileReportError(
                f"{source}.owners_per_coordinate: multiplicity bounds do not "
                "cover the physical position count"
            )
        result["owners_per_coordinate"] = {
            "minimum": minimum,
            "maximum": maximum,
        }
    elif owners_value is not None:
        raise CompileReportError(
            f"{source}.owners_per_coordinate: non-exact evidence cannot claim "
            "exact ownership multiplicity"
        )
    return result


def _build_transition(
    value: object, contraction: dict[str, object], source: str
) -> dict[str, object]:
    transition = _require_object(value, source)
    source_role = _require_role(transition.get("source_role"), f"{source}.source_role")
    destination_role = _require_role(
        transition.get("destination_role"), f"{source}.destination_role"
    )
    roles = _require_object(contraction["roles"], "native contraction roles")
    source_role_facts = _require_object(roles[source_role], f"roles.{source_role}")
    destination_role_facts = _require_object(
        roles[destination_role], f"roles.{destination_role}"
    )
    for role_name, facts in (
        (source_role, source_role_facts),
        (destination_role, destination_role_facts),
    ):
        if facts["evidence"] != "exact":
            raise CompileReportError(
                f"{source}: transition role {role_name!r} requires exact "
                "placement evidence"
            )
    if (
        source_role_facts["logical_coordinates"]
        != destination_role_facts["logical_coordinates"]
    ):
        raise CompileReportError(
            f"{source}: source and destination roles have different logical domains"
        )

    destination_positions = _require_positive_count(
        transition.get("destination_positions"),
        f"{source}.destination_positions",
    )
    if destination_positions != destination_role_facts["physical_positions"]:
        raise CompileReportError(
            f"{source}.destination_positions: expected "
            f"{destination_role_facts['physical_positions']} from the "
            f"{destination_role} placement, got {destination_positions}"
        )
    participant_changes = _require_count(
        transition.get("participant_changes"),
        f"{source}.participant_changes",
    )
    local_position_changes = _require_count(
        transition.get("local_position_changes"),
        f"{source}.local_position_changes",
    )
    for name, count in (
        ("participant_changes", participant_changes),
        ("local_position_changes", local_position_changes),
    ):
        if count > destination_positions:
            raise CompileReportError(
                f"{source}.{name}: expected at most {destination_positions}, "
                f"got {count}"
            )

    replication_value = _require_object(
        transition.get("destination_positions_per_source"),
        f"{source}.destination_positions_per_source",
    )
    replication_minimum = _require_positive_count(
        replication_value.get("minimum"),
        f"{source}.destination_positions_per_source.minimum",
    )
    replication_maximum = _require_positive_count(
        replication_value.get("maximum"),
        f"{source}.destination_positions_per_source.maximum",
    )
    if replication_minimum > replication_maximum:
        raise CompileReportError(
            f"{source}.destination_positions_per_source: minimum "
            f"{replication_minimum} exceeds maximum {replication_maximum}"
        )
    source_coordinate_count = source_role_facts["logical_coordinates"]
    if not (
        source_coordinate_count * replication_minimum
        <= destination_positions
        <= source_coordinate_count * replication_maximum
    ):
        raise CompileReportError(
            f"{source}.destination_positions_per_source: replication bounds do "
            "not cover the destination position count"
        )

    factor_values = _require_list(
        transition.get("source_owner_factors"),
        f"{source}.source_owner_factors",
    )
    if not factor_values:
        raise CompileReportError(
            f"{source}.source_owner_factors: exact transition requires factors"
        )
    factors = []
    for index, factor_value in enumerate(factor_values):
        factor_source = f"{source}.source_owner_factors[{index}]"
        factor = _require_object(factor_value, factor_source)
        destination_dimension = _require_string(
            factor.get("destination_dimension"),
            f"{factor_source}.destination_dimension",
        )
        source_owner_dimension = _require_string(
            factor.get("source_owner_dimension"),
            f"{factor_source}.source_owner_dimension",
        )
        for field, dimension in (
            ("destination_dimension", destination_dimension),
            ("source_owner_dimension", source_owner_dimension),
        ):
            if dimension not in _PHYSICAL_DIMENSIONS:
                raise CompileReportError(
                    f"{factor_source}.{field}: unknown physical dimension {dimension!r}"
                )
        factors.append(
            {
                "destination_dimension": destination_dimension,
                "source_owner_dimension": source_owner_dimension,
                "divisor": _require_positive_count(
                    factor.get("divisor"), f"{factor_source}.divisor"
                ),
                "modulus": _require_count(
                    factor.get("modulus"), f"{factor_source}.modulus"
                ),
                "multiplier": _require_positive_count(
                    factor.get("multiplier"), f"{factor_source}.multiplier"
                ),
            }
        )

    return {
        "source": {
            "role": source_role,
            "type": _require_string(
                transition.get("source_type"), f"{source}.source_type"
            ),
        },
        "destination": {
            "role": destination_role,
            "type": _require_string(
                transition.get("destination_type"),
                f"{source}.destination_type",
            ),
        },
        "destination_position_count": destination_positions,
        "participant_change_count": participant_changes,
        "local_position_change_count": local_position_changes,
        "destination_positions_per_selected_source": {
            "minimum": replication_minimum,
            "maximum": replication_maximum,
        },
        "source_owner_factors": factors,
    }


def _candidate_function_aliases(
    entry_function_pairs: tuple[tuple[str | None, str | None], ...],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for baseline_function, candidate_function in entry_function_pairs:
        if (
            baseline_function is None
            or candidate_function is None
            or baseline_function == candidate_function
        ):
            continue
        previous = aliases.setdefault(candidate_function, baseline_function)
        if previous != baseline_function:
            raise CompileReportError(
                "explicit entry pairing maps candidate function "
                f"{candidate_function!r} to both {previous!r} and "
                f"{baseline_function!r}"
            )
    return aliases


def _rows_by_key(
    row_values: list[object],
    source: str,
    *,
    function_aliases: dict[str, str] | None = None,
) -> dict[tuple[str, int], dict[str, object]]:
    rows = {}
    for position, row_value in enumerate(row_values):
        row = _require_object(row_value, f"{source}.native_layout.rows[{position}]")
        identity = _require_object(row["identity"], "native layout identity")
        function = identity.get("function")
        if not isinstance(function, str):
            function = ""
        if function_aliases is not None:
            function = function_aliases.get(function, function)
        report_index = identity["report_index"]
        if not isinstance(report_index, int) or isinstance(report_index, bool):
            raise TypeError("invalid internal native layout report index")
        key = (function, report_index)
        if key in rows:
            raise CompileReportError(
                f"{source}: duplicate native layout source-row identity {key!r}"
            )
        rows[key] = row
    return rows


def _comparable_row(
    row: dict[str, object],
    *,
    function_aliases: dict[str, str] | None = None,
) -> dict[str, object]:
    identity = dict(_require_object(row["identity"], "native layout identity"))
    function = identity.get("function")
    if isinstance(function, str) and function_aliases is not None:
        identity["function"] = function_aliases.get(function, function)
    comparable = {"identity": identity, "selection": row["selection"]}
    for field in ("contraction", "transition"):
        if field in row:
            comparable[field] = row[field]
    return comparable


def _append_value_changes(
    changes: list[dict[str, object]],
    baseline: object,
    candidate: object,
    path: str = "",
) -> None:
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        for key in sorted(set(baseline) | set(candidate)):
            child_path = f"{path}.{key}" if path else key
            _append_value_changes(
                changes,
                baseline.get(key, _MISSING),
                candidate.get(key, _MISSING),
                child_path,
            )
        return
    if isinstance(baseline, list) and isinstance(candidate, list):
        for index in range(max(len(baseline), len(candidate))):
            _append_value_changes(
                changes,
                baseline[index] if index < len(baseline) else _MISSING,
                candidate[index] if index < len(candidate) else _MISSING,
                f"{path}[{index}]",
            )
        return
    if baseline == candidate:
        return
    change: dict[str, object] = {
        "path": path,
        "baseline": None if baseline is _MISSING else baseline,
        "candidate": None if candidate is _MISSING else candidate,
    }
    if _is_number(baseline) and _is_number(candidate):
        delta = candidate - baseline
        change["delta"] = delta
        if baseline != 0:
            change["change_percent"] = delta * 100.0 / baseline
    changes.append(change)


def _append_native_row(
    lines: list[str],
    row: dict[str, object],
    *,
    indent: str = "  ",
    include_identity: bool = True,
) -> None:
    identity = _require_object(row["identity"], "native layout identity")
    if include_identity:
        lines.append(f"{indent}{_format_identity(identity)}")
        indent += "  "
    selection = _require_object(row["selection"], "native layout selection")
    lines.append(f"{indent}selection: {selection['kind']}")
    plan_key = selection.get("plan_key")
    if plan_key is not None:
        lines.append(f"{indent}target recipe: {plan_key}")
    descriptor_key = selection.get("descriptor_key")
    descriptor_semantic_tag = selection.get("descriptor_semantic_tag")
    if descriptor_key is not None or descriptor_semantic_tag is not None:
        descriptor = descriptor_semantic_tag or "unavailable semantic"
        if descriptor_key is not None:
            descriptor += f" ({descriptor_key})"
        lines.append(f"{indent}selected descriptor: {descriptor}")
    lines.append(
        f"{indent}emitted Low operations (unscaled): "
        f"{selection['emitted_low_op_count']}"
    )
    contraction = row.get("contraction")
    if isinstance(contraction, dict):
        _append_contraction(lines, contraction, indent=indent)
    transition = row.get("transition")
    if isinstance(transition, dict):
        _append_transition(lines, transition, indent=indent)


def _append_contraction(
    lines: list[str], contraction: dict[str, object], *, indent: str
) -> None:
    tile = _require_object(contraction["tile"], "native contraction tile")
    lines.append(
        f"{indent}native contraction: {tile['blocks']} block(s) x "
        f"{tile['m']}m x {tile['n']}n x {tile['k']}k across "
        f"{contraction['participant_count']} participant(s)"
    )
    roles = _require_object(contraction["roles"], "native contraction roles")
    for role in _ROLES:
        facts = _require_object(roles[role], f"native contraction {role}")
        registers = facts["registers_per_participant"]
        payload_elements = facts["payload_elements_per_participant"]
        elements_per_register = ""
        if payload_elements % registers == 0:
            elements_per_register = (
                f", {payload_elements // registers} payload element(s)/register"
            )
        lines.append(
            f"{indent}  {role}: {facts['evidence']}; {facts['element_bits']}-bit; "
            f"{registers} register(s), {payload_elements} payload element(s) per "
            f"participant{elements_per_register}"
        )
        ownership = (
            _format_range(
                _require_object(
                    facts["owners_per_coordinate"],
                    f"native contraction {role} owners",
                )
            )
            if facts["evidence"] == "exact"
            else f"unavailable ({facts['evidence']} placement)"
        )
        lines.append(
            f"{indent}    placement: "
            f"{facts['physical_positions_per_participant']} coordinate-bearing "
            f"position(s)/participant, {facts['physical_positions']} total -> "
            f"{facts['logical_coordinates']} logical coordinate(s); "
            f"owners/coordinate: {ownership}"
        )


def _append_transition(
    lines: list[str], transition: dict[str, object], *, indent: str
) -> None:
    source = _require_object(transition["source"], "native transition source")
    destination = _require_object(
        transition["destination"], "native transition destination"
    )
    lines.append(
        f"{indent}native transition: {source['role']}:{source['type']} -> "
        f"{destination['role']}:{destination['type']}"
    )
    lines.append(
        f"{indent}  cross-participant movement: "
        f"{_format_movement(transition, 'participant_change_count')}"
    )
    lines.append(
        f"{indent}  participant-local position movement: "
        f"{_format_movement(transition, 'local_position_change_count')}"
    )
    replication = _require_object(
        transition["destination_positions_per_selected_source"],
        "native transition replication",
    )
    lines.append(
        f"{indent}  destination positions/selected source position: "
        f"{_format_range(replication)}"
    )
    for dimension, equation in _owner_equations(transition):
        lines.append(f"{indent}  source.{dimension} = {equation}")


def _append_changed_row(
    lines: list[str], row: dict[str, object], *, indent: str
) -> None:
    baseline = _require_object(row["baseline"], "baseline native layout row")
    candidate = _require_object(row["candidate"], "candidate native layout row")
    changes = [
        _require_object(value, "native layout value change")
        for value in _require_list(row["changes"], "native layout value changes")
    ]
    changed_paths = {str(change["path"]) for change in changes}
    handled_paths = set()

    baseline_transition = baseline.get("transition")
    candidate_transition = candidate.get("transition")
    if isinstance(baseline_transition, dict) != isinstance(candidate_transition, dict):
        baseline_availability = (
            "available" if isinstance(baseline_transition, dict) else "unavailable"
        )
        candidate_availability = (
            "available" if isinstance(candidate_transition, dict) else "unavailable"
        )
        lines.append(
            f"{indent}native transition: {baseline_availability} -> "
            f"{candidate_availability}"
        )
        handled_paths.update(
            path for path in changed_paths if path.startswith("transition")
        )
    if isinstance(baseline_transition, dict) and isinstance(candidate_transition, dict):
        for field, label in (
            ("participant_change_count", "cross-participant movement"),
            ("local_position_change_count", "participant-local position movement"),
        ):
            paths = {
                f"transition.{field}",
                "transition.destination_position_count",
            }
            if changed_paths & paths:
                lines.append(
                    f"{indent}{label}: {_format_movement(baseline_transition, field)}"
                    f" -> {_format_movement(candidate_transition, field)}"
                )
                handled_paths.update(paths)
        replication_prefix = "transition.destination_positions_per_selected_source."
        if any(path.startswith(replication_prefix) for path in changed_paths):
            baseline_replication = _require_object(
                baseline_transition["destination_positions_per_selected_source"],
                "baseline replication",
            )
            candidate_replication = _require_object(
                candidate_transition["destination_positions_per_selected_source"],
                "candidate replication",
            )
            lines.append(
                f"{indent}destination positions/selected source position: "
                f"{_format_range(baseline_replication)} -> "
                f"{_format_range(candidate_replication)}"
            )
            handled_paths.update(
                path for path in changed_paths if path.startswith(replication_prefix)
            )
        factor_prefix = "transition.source_owner_factors"
        if any(path.startswith(factor_prefix) for path in changed_paths):
            lines.append(
                f"{indent}source owner equations: "
                f"{_format_owner_equations(baseline_transition)} -> "
                f"{_format_owner_equations(candidate_transition)}"
            )
            handled_paths.update(
                path for path in changed_paths if path.startswith(factor_prefix)
            )

    for change in changes:
        path = str(change["path"])
        if path in handled_paths:
            continue
        suffix = ""
        if "delta" in change:
            suffix = f" (delta {change['delta']:+})"
        lines.append(
            f"{indent}{_format_path(path)}: {change['baseline']} -> "
            f"{change['candidate']}{suffix}"
        )


def _owner_equations(
    transition: dict[str, object],
) -> tuple[tuple[str, str], ...]:
    factors = [
        _require_object(value, "native transition owner factor")
        for value in _require_list(
            transition["source_owner_factors"], "native transition owner factors"
        )
    ]
    equations = []
    for owner_dimension in ("participant", "position"):
        terms = [
            _format_owner_factor(factor)
            for factor in factors
            if factor["source_owner_dimension"] == owner_dimension
        ]
        if terms:
            equations.append((owner_dimension, " + ".join(terms)))
    return tuple(equations)


def _format_owner_factor(factor: dict[str, object]) -> str:
    term = str(factor["destination_dimension"])
    divisor = factor["divisor"]
    modulus = factor["modulus"]
    multiplier = factor["multiplier"]
    if divisor != 1:
        term += f" / {divisor}"
    if modulus != 0:
        term += f" % {modulus}"
    if multiplier != 1:
        term = f"({term}) * {multiplier}"
    return term


def _format_owner_equations(transition: dict[str, object]) -> str:
    return "; ".join(
        f"source.{dimension} = {equation}"
        for dimension, equation in _owner_equations(transition)
    )


def _format_movement(transition: dict[str, object], field: str) -> str:
    count = transition[field]
    total = transition["destination_position_count"]
    percent = count * 100.0 / total
    return f"{count}/{total} destination position(s) ({percent:.2f}%)"


def _format_range(value: dict[str, object]) -> str:
    minimum = value["minimum"]
    maximum = value["maximum"]
    return str(minimum) if minimum == maximum else f"{minimum}..{maximum}"


def _format_identity(identity: dict[str, object]) -> str:
    function = identity.get("function") or "<unknown function>"
    source_op = identity.get("source_op") or "<unknown source op>"
    return f"source_low[{identity['report_index']}] {function}/{source_op}"


def _format_path(path: str) -> str:
    return path.replace("_", " ")


def _require_role(value: object, source: str) -> str:
    role = _require_string(value, source)
    if role not in _ROLES:
        raise CompileReportError(f"{source}: unknown contraction role {role!r}")
    return role


def _require_object(value: object, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompileReportError(f"{source}: expected object")
    return value


def _require_list(value: object, source: str) -> list[object]:
    if not isinstance(value, list):
        raise CompileReportError(f"{source}: expected array")
    return value


def _require_string(value: object, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompileReportError(f"{source}: expected non-empty string")
    return value


def _optional_string(value: object, source: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, source)


def _require_count(value: object, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CompileReportError(f"{source}: expected non-negative integer")
    return value


def _require_positive_count(value: object, source: str) -> int:
    count = _require_count(value, source)
    if count == 0:
        raise CompileReportError(f"{source}: expected positive integer")
    return count


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))
