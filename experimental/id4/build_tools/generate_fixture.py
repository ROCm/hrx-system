#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates ID4 reference fixtures from checked-in reduction plans."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import trace_reduce

TRACE_ROOT_ENV = "ID4_REFERENCE_TRACE_ROOT"
FIXTURE_ROOT_ENV = "ID4_REFERENCE_FIXTURE_ROOT"


class GenerateFixtureError(ValueError):
    """Raised when fixture generation inputs are missing or malformed."""


def default_plan_root() -> Path:
    return Path(__file__).resolve().parents[1] / "reference" / "plans"


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GenerateFixtureError(f"{field_name} must be a non-empty string")
    return value


def _required_path(
    explicit_value: Path | None, environment_name: str, description: str
) -> Path:
    if explicit_value is not None:
        return explicit_value
    environment_value = os.environ.get(environment_name)
    if environment_value:
        return Path(environment_value)
    raise GenerateFixtureError(
        f"{description} must be provided with --{description.replace('_', '-')} "
        f"or {environment_name}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise GenerateFixtureError(f"invalid fixture plan JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GenerateFixtureError(f"fixture plan must be a JSON object: {path}")
    return payload


def load_fixture_plan(plan_root: Path, fixture_id: str) -> dict[str, Any]:
    plan_path = plan_root / f"{fixture_id}.json"
    if not plan_path.is_file():
        raise GenerateFixtureError(f"fixture plan not found: {plan_path}")
    plan = _load_json(plan_path)
    plan_fixture_id = _require_string(plan.get("fixture_id"), "fixture_id")
    if plan_fixture_id != fixture_id:
        raise GenerateFixtureError(
            f"fixture plan {plan_path} declares fixture_id {plan_fixture_id}, "
            f"expected {fixture_id}"
        )
    _require_string(plan.get("source_trace_id"), "source_trace_id")
    trace_reduce.load_plan(plan_path)
    return plan


def generate_fixture(
    plan_root: Path,
    fixture_id: str,
    trace_root: Path,
    fixture_root: Path,
    trace_id: str | None = None,
    output_id: str | None = None,
) -> dict[str, Any]:
    plan = load_fixture_plan(plan_root, fixture_id)
    selected_trace_id = trace_id or _require_string(
        plan.get("source_trace_id"), "source_trace_id"
    )
    selected_output_id = output_id or fixture_id
    trace_dir = trace_root / selected_trace_id
    output_dir = fixture_root / selected_output_id
    if not trace_dir.is_dir():
        raise GenerateFixtureError(f"trace directory not found: {trace_dir}")
    return trace_reduce.reduce_trace(trace_dir, output_dir, plan)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an ID4 fixture from a checked-in reduction plan."
    )
    parser.add_argument(
        "--fixture-id",
        required=True,
        help="Checked-in fixture plan id to generate.",
    )
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=default_plan_root(),
        help="Directory containing checked-in fixture plan JSON files.",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        help=f"Root containing raw trace directories. Defaults to {TRACE_ROOT_ENV}.",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        help=(
            "Root that will receive generated fixture directories. Defaults to "
            f"{FIXTURE_ROOT_ENV}."
        ),
    )
    parser.add_argument(
        "--trace-id",
        help="Raw trace directory name under the trace root. Defaults to the plan.",
    )
    parser.add_argument(
        "--output-id",
        help="Output directory name under the fixture root. Defaults to fixture-id.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        trace_root = _required_path(args.trace_root, TRACE_ROOT_ENV, "trace_root")
        fixture_root = _required_path(
            args.fixture_root, FIXTURE_ROOT_ENV, "fixture_root"
        )
        generate_fixture(
            args.plan_root,
            args.fixture_id,
            trace_root,
            fixture_root,
            trace_id=args.trace_id,
            output_id=args.output_id,
        )
    except (GenerateFixtureError, trace_reduce.TraceReduceError) as exc:
        print(f"generate_fixture: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
