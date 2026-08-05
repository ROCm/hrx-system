#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import aggregate_recursion_summaries as aggregate


def summary(
    translation_unit: str,
    function_id: str,
    function_name: str,
    callee_id: str,
    callee_name: str,
) -> dict[str, object]:
    return {
        "version": 1,
        "translation_unit": translation_unit,
        "functions": [
            {
                "id": function_id,
                "name": function_name,
                "file": translation_unit,
                "line": 3,
                "column": 1,
            }
        ],
        "edges": [
            {
                "caller": function_id,
                "callee": callee_id,
                "caller_name": function_name,
                "callee_name": callee_name,
                "file": translation_unit,
                "line": 4,
                "column": 10,
                "indirect": False,
            }
        ],
    }


class AggregateRecursionSummariesTest(unittest.TestCase):
    def test_reports_one_cross_translation_unit_scc(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            a_path = root / "a.json"
            b_path = root / "b.json"
            a_path.write_text(
                json.dumps(summary("a.c", "external:a", "a", "external:b", "b")),
                encoding="utf-8",
            )
            b_path.write_text(
                json.dumps(summary("b.c", "external:b", "b", "external:a", "a")),
                encoding="utf-8",
            )

            functions_a, edges_a = aggregate.load_summary(a_path)
            functions_b, edges_b = aggregate.load_summary(b_path)
            output = aggregate.format_diagnostics(
                [*functions_a, *functions_b], [*edges_a, *edges_b]
            )

        self.assertEqual(output.count("[iree-unbounded-recursion]"), 1)
        self.assertIn(
            "cross-translation-unit call cycle containing 2 functions", output
        )
        self.assertIn("direct call from a to b", output)
        self.assertIn("direct call from b to a", output)

    def test_accepts_acyclic_summaries(self):
        functions = [
            aggregate.Function(
                id="external:a",
                name="a",
                point=aggregate.SourcePoint("a.c", 1, 1),
            )
        ]
        edges = [
            aggregate.Edge(
                caller="external:a",
                callee="external:b",
                caller_name="a",
                callee_name="b",
                point=aggregate.SourcePoint("a.c", 2, 3),
                indirect=False,
                dispatcher=None,
            )
        ]

        self.assertEqual(aggregate.format_diagnostics(functions, edges), "")

    def test_does_not_label_single_translation_unit_cycle_as_cross_tu(self):
        point = aggregate.SourcePoint("local.c", 1, 1)
        functions = [aggregate.Function(id="external:a", name="a", point=point)]
        edges = [
            aggregate.Edge(
                caller="external:a",
                callee="external:a",
                caller_name="a",
                callee_name="a",
                point=aggregate.SourcePoint("local.c", 2, 3),
                indirect=True,
                dispatcher="dispatch",
            )
        ]

        output = aggregate.format_diagnostics(functions, edges)

        self.assertIn("call cycle containing 1 function", output)
        self.assertIn("through callback dispatcher dispatch", output)
        self.assertNotIn("cross-translation-unit", output)


if __name__ == "__main__":
    unittest.main()
