# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.reporting.compile_report import (
    CompileReportDocument,
    CompileReportError,
    parse_compile_report,
)
from loom.reporting.compile_report_native_layout import (
    append_native_layout_diff_text,
    append_native_layout_show_text,
    build_native_layout_diff,
    build_native_layout_show,
)
from loom.reporting.compile_report_view import (
    build_compile_report_show,
    format_compile_report_show_text,
)


def _role(
    *,
    element_bits: int,
    registers: int,
    payload_elements: int,
    physical_positions: int,
    logical_coordinates: int,
    owners: int,
) -> dict[str, object]:
    return {
        "evidence": "exact",
        "element_bits": element_bits,
        "registers_per_participant": registers,
        "payload_elements_per_participant": payload_elements,
        "physical_positions": physical_positions,
        "logical_coordinates": logical_coordinates,
        "owners_per_coordinate": {"minimum": owners, "maximum": owners},
    }


def _f16_contraction() -> dict[str, object]:
    return {
        "tile": {"blocks": 1, "m": 16, "n": 16, "k": 16},
        "participants": 32,
        "lhs": _role(
            element_bits=16,
            registers=8,
            payload_elements=16,
            physical_positions=512,
            logical_coordinates=256,
            owners=2,
        ),
        "rhs": _role(
            element_bits=16,
            registers=8,
            payload_elements=16,
            physical_positions=512,
            logical_coordinates=256,
            owners=2,
        ),
        "accumulator": _role(
            element_bits=16,
            registers=8,
            payload_elements=16,
            physical_positions=256,
            logical_coordinates=256,
            owners=1,
        ),
        "result": _role(
            element_bits=16,
            registers=8,
            payload_elements=16,
            physical_positions=256,
            logical_coordinates=256,
            owners=1,
        ),
    }


def _f32_contraction() -> dict[str, object]:
    contraction = _f16_contraction()
    for role in ("accumulator", "result"):
        facts = contraction[role]
        facts["element_bits"] = 32
        facts["payload_elements_per_participant"] = 8
    return contraction


def _factor(
    destination_dimension: str,
    source_owner_dimension: str,
    divisor: int,
    modulus: int,
    multiplier: int,
) -> dict[str, object]:
    return {
        "destination_dimension": destination_dimension,
        "source_owner_dimension": source_owner_dimension,
        "divisor": divisor,
        "modulus": modulus,
        "multiplier": multiplier,
    }


def _result_to_rhs_transition() -> dict[str, object]:
    return {
        "source_role": "result",
        "destination_role": "rhs",
        "source_type": "f16",
        "destination_type": "f16",
        "destination_positions": 512,
        "participant_changes": 256,
        "local_position_changes": 480,
        "destination_positions_per_source": {"minimum": 2, "maximum": 2},
        "source_owner_factors": [
            _factor("participant", "participant", 1, 16, 1),
            _factor("position", "participant", 1, 2, 16),
            _factor("position", "position", 2, 0, 1),
        ],
    }


def _result_to_lhs_transition() -> dict[str, object]:
    return {
        "source_role": "result",
        "destination_role": "lhs",
        "source_type": "f32",
        "destination_type": "bf16",
        "destination_positions": 512,
        "participant_changes": 496,
        "local_position_changes": 480,
        "destination_positions_per_source": {"minimum": 2, "maximum": 2},
        "source_owner_factors": [
            _factor("participant", "participant", 1, 2, 16),
            _factor("position", "participant", 1, 0, 1),
            _factor("participant", "position", 2, 8, 1),
        ],
    }


def _native_row(
    *,
    transition: dict[str, object] | None = None,
    contraction: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "index": 1,
        "function": "kernel",
        "source_op": "vector.fragment.repack" if transition else "vector.mma",
        "source_op_kind": 3607 if transition else 3606,
        "selection": "plan",
        "plan_key": "target.result_to_rhs" if transition else None,
        "descriptor_key": "target.native.matrix",
        "descriptor_semantic_tag": "matrix.native",
        "emitted_low_op_count": 28 if transition else 2,
        "native_contraction": contraction or _f16_contraction(),
    }
    if transition is not None:
        row["native_transition"] = transition
    return row


def _document(
    native_row: dict[str, object], *, source: str = "report.json"
) -> CompileReportDocument:
    rows = [
        {
            "index": 0,
            "function": "kernel",
            "source_op": "index.constant",
            "source_op_kind": 1,
            "selection": "plan",
            "emitted_low_op_count": 0,
        },
        native_row,
    ]
    return parse_compile_report(
        {
            "kind": "loom.compile_report",
            "schema_version": 0,
            "mode": "details",
            "status": {"code": 0, "name": "OK"},
            "entries": {
                "count": 1,
                "rows": [{"index": 0, "function": "kernel"}],
            },
            "source_low": {"count": len(rows), "rows": rows},
        },
        source=source,
    )


def _document_without_native_rows(*, source: str) -> CompileReportDocument:
    return parse_compile_report(
        {
            "kind": "loom.compile_report",
            "schema_version": 0,
            "mode": "details",
            "status": {"code": 0, "name": "OK"},
            "entries": {
                "count": 1,
                "rows": [{"index": 0, "function": "kernel"}],
            },
            "source_low": {
                "count": 1,
                "rows": [
                    {
                        "index": 0,
                        "function": "kernel",
                        "source_op": "index.constant",
                        "source_op_kind": 1,
                        "selection": "plan",
                        "emitted_low_op_count": 0,
                    }
                ],
            },
        },
        source=source,
    )


def _make_result_opaque(row: dict[str, object]) -> None:
    result = row["native_contraction"]["result"]
    result["evidence"] = "opaque"
    del result["owners_per_coordinate"]


def test_show_explains_native_placement_and_transition_economics() -> None:
    view = build_native_layout_show(
        _document(_native_row(transition=_result_to_rhs_transition()))
    )

    assert view is not None
    assert view["row_count"] == 1
    row = view["rows"][0]
    assert row["identity"] == {
        "report_index": 1,
        "function": "kernel",
        "source_op": "vector.fragment.repack",
        "source_op_kind": 3607,
    }
    assert row["selection"]["plan_key"] == "target.result_to_rhs"
    assert row["contraction"]["roles"]["rhs"]["owners_per_coordinate"] == {
        "minimum": 2,
        "maximum": 2,
    }
    assert (
        row["contraction"]["roles"]["result"]["physical_positions_per_participant"] == 8
    )
    assert row["transition"]["participant_change_count"] == 256
    assert row["transition"]["destination_positions_per_selected_source"] == {
        "minimum": 2,
        "maximum": 2,
    }

    lines: list[str] = []
    append_native_layout_show_text(lines, view)
    text = "\n".join(lines)
    assert "Native layout economics (compiler analysis)" in text
    assert "target recipe: target.result_to_rhs" in text
    assert "emitted Low operations (unscaled): 28" in text
    assert "native contraction: 1 block(s) x 16m x 16n x 16k" in text
    assert "rhs: exact; 16-bit; 8 register(s), 16 payload element(s)" in text
    assert "owners/coordinate: 2" in text
    assert "cross-participant movement: 256/512" in text
    assert "participant-local position movement: 480/512" in text
    assert "8 coordinate-bearing position(s)/participant, 256 total" in text
    assert "destination positions/selected source position: 2" in text
    assert "source.participant = participant % 16 + (position % 2) * 16" in text
    assert "source.position = position / 2" in text


def test_compile_report_show_includes_native_layout_section() -> None:
    document = _document(_native_row(transition=_result_to_rhs_transition()))

    view = build_compile_report_show(document)

    assert view["native_layout"]["row_count"] == 1
    text = format_compile_report_show_text(view)
    assert "Native layout economics (compiler analysis)" in text
    assert "source_low[1] kernel/vector.fragment.repack" in text


def test_show_keeps_non_exact_placement_distinct_from_exact_ownership() -> None:
    row = _native_row()
    lhs = row["native_contraction"]["lhs"]
    lhs["evidence"] = "metadata-dependent"
    del lhs["owners_per_coordinate"]

    view = build_native_layout_show(_document(row))

    assert view is not None
    lhs_view = view["rows"][0]["contraction"]["roles"]["lhs"]
    assert lhs_view["evidence"] == "metadata-dependent"
    assert "owners_per_coordinate" not in lhs_view
    lines: list[str] = []
    append_native_layout_show_text(lines, view)
    assert "owners/coordinate: unavailable (metadata-dependent placement)" in "\n".join(
        lines
    )


def test_show_uses_same_contract_for_single_participant_x86_dot() -> None:
    row = _native_row()
    row["descriptor_key"] = "x86.avx_vnni_int8.vpdpbssd.ymm"
    row["descriptor_semantic_tag"] = "dot.s8s8.i32x8"
    row["native_contraction"] = {
        "tile": {"blocks": 8, "m": 1, "n": 1, "k": 4},
        "participants": 1,
        "lhs": _role(
            element_bits=8,
            registers=1,
            payload_elements=32,
            physical_positions=32,
            logical_coordinates=32,
            owners=1,
        ),
        "rhs": _role(
            element_bits=8,
            registers=1,
            payload_elements=32,
            physical_positions=32,
            logical_coordinates=32,
            owners=1,
        ),
        "accumulator": _role(
            element_bits=32,
            registers=1,
            payload_elements=8,
            physical_positions=8,
            logical_coordinates=8,
            owners=1,
        ),
        "result": _role(
            element_bits=32,
            registers=1,
            payload_elements=8,
            physical_positions=8,
            logical_coordinates=8,
            owners=1,
        ),
    }

    view = build_native_layout_show(_document(row))

    assert view is not None
    contraction = view["rows"][0]["contraction"]
    assert contraction["participant_count"] == 1
    assert contraction["roles"]["lhs"]["physical_positions"] == 32
    lines: list[str] = []
    append_native_layout_show_text(lines, view)
    text = "\n".join(lines)
    assert "8 block(s) x 1m x 1n x 4k across 1 participant(s)" in text
    assert (
        "32 payload element(s) per participant, 32 payload element(s)/register" in text
    )
    assert "x86.avx_vnni_int8.vpdpbssd.ymm" in text


def test_diff_compares_normalized_movement_facts_separately_from_recipe() -> None:
    baseline_row = _native_row(transition=_result_to_rhs_transition())
    candidate_row = _native_row(
        transition=_result_to_lhs_transition(),
        contraction=_f32_contraction(),
    )
    candidate_row["plan_key"] = "target.result_to_lhs"
    baseline = _document(baseline_row, source="baseline.json")
    candidate = _document(candidate_row, source="candidate.json")

    diff = build_native_layout_diff(baseline, candidate)

    assert diff is not None
    assert diff["changed_row_count"] == 1
    row = diff["rows"][0]
    changes = {change["path"]: change for change in row["changes"]}
    assert changes["selection.plan_key"] == {
        "path": "selection.plan_key",
        "baseline": "target.result_to_rhs",
        "candidate": "target.result_to_lhs",
    }
    assert changes["transition.participant_change_count"]["delta"] == 240
    assert changes["transition.destination.role"]["candidate"] == "lhs"
    assert changes["transition.source_owner_factors[0].modulus"] == {
        "path": "transition.source_owner_factors[0].modulus",
        "baseline": 16,
        "candidate": 2,
        "delta": -14,
        "change_percent": -87.5,
    }

    lines: list[str] = []
    append_native_layout_diff_text(lines, diff)
    text = "\n".join(lines)
    assert "selection.plan key: target.result_to_rhs -> target.result_to_lhs" in text
    assert "cross-participant movement: 256/512" in text
    assert "-> 496/512 destination position(s) (96.88%)" in text
    assert "source owner equations:" in text
    assert "source.participant = participant % 16 + (position % 2) * 16" in text
    assert "source.participant = (participant % 2) * 16 + position" in text


def test_diff_reports_new_detailed_native_row_as_added() -> None:
    baseline = _document_without_native_rows(source="baseline.json")
    candidate = _document(_native_row(), source="candidate.json")

    diff = build_native_layout_diff(baseline, candidate)

    assert diff is not None
    assert diff["changed_row_count"] == 1
    assert diff["unchanged_row_count"] == 0
    assert diff["rows"][0]["status"] == "added"
    assert diff["rows"][0]["identity"]["report_index"] == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda row: row["native_contraction"]["rhs"].__setitem__(
                "physical_positions", 511
            ),
            "physical_positions",
        ),
        (
            lambda row: row["native_transition"].__setitem__(
                "participant_changes", 513
            ),
            "participant_changes",
        ),
        (
            _make_result_opaque,
            "requires exact placement",
        ),
    ],
)
def test_rejects_inconsistent_native_layout_evidence(mutate, message: str) -> None:
    row = _native_row(transition=_result_to_rhs_transition())
    mutate(row)

    with pytest.raises(CompileReportError, match=message):
        build_native_layout_show(_document(row))
