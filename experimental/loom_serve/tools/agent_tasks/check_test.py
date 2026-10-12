# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experimental.loom_serve.tools.agent_tasks import check

SOLUTIONS = {
    "exclusive_scan": "return [sum(values[:i]) for i in range(len(values))]",
    "page_spans": """result = []
while length:
    count = min(length, block_size - start % block_size)
    result.append([start // block_size, start % block_size, count])
    start += count
    length -= count
return result""",
    "coalesce_ranges": """result = []
for start, end in sorted(ranges):
    if result and start <= result[-1][1]:
        result[-1][1] = max(result[-1][1], end)
    else:
        result.append([start, end])
return result""",
    "stable_softmax": """if not values:
    return []
weights = [math.exp(value - max(values)) for value in values]
return [value / sum(weights) for value in weights]""",
    "rms_norm": """if not values:
    return []
scale = math.sqrt(sum(value * value for value in values) / len(values) + epsilon)
return [value / scale for value in values]""",
    "masked_mean": """selected = [value for value, keep in zip(values, mask) if keep]
return sum(selected) / len(selected) if selected else 0.0""",
    "transpose": "return [list(column) for column in zip(*matrix)]",
    "segmented_sum": "return [sum(values[a:b]) for a, b in zip(offsets, offsets[1:])]",
}


class CheckTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.candidate = Path(directory.name) / "candidate.py"
        self.tasks = json.loads(
            Path(check.__file__).with_name("tasks.json").read_text()
        )

    def test_every_seed_fails_real_cases(self):
        self.assertEqual({task["id"] for task in self.tasks}, SOLUTIONS.keys())
        for task in self.tasks:
            with self.subTest(task=task["id"]):
                self.candidate.write_text(task["candidate"])
                result = check.evaluate(task["id"], self.candidate)
                self.assertFalse(result["passed"])
                self.assertNotIn("error", result)
                self.assertTrue(any(not case["passed"] for case in result["cases"]))

    def test_correct_functions_pass_each_corpus(self):
        for task in self.tasks:
            with self.subTest(task=task["id"]):
                signature = task["candidate"].splitlines()[0]
                body = "\n".join(
                    "    " + line for line in SOLUTIONS[task["id"]].splitlines()
                )
                self.candidate.write_text(signature + "\n" + body + "\n")
                result = check.evaluate(task["id"], self.candidate)
                self.assertTrue(result["passed"], result)
                self.assertEqual(len(result["cases"]), len(task["cases"]) + 1)

    def test_language_boundary_rejects_unsupported_code(self):
        sources = [
            "import os\ndef solve(values): return values\n",
            "def other(values): return values\n",
            "def solve(values): return values.__class__\n",
            "def solve(values): return __builtins__\n",
            "def solve(values):\n    def nested(): return 0\n    return []\n",
            "def solve(values): return []\n" + " " * 32768,
        ]
        for source in sources:
            with self.subTest(source=source[:80]):
                self.candidate.write_text(source)
                with self.assertRaises(ValueError):
                    check.evaluate("exclusive_scan", self.candidate)

    def test_cli_keeps_failed_outcomes_and_nonzero_status(self):
        self.candidate.write_text(self.tasks[0]["candidate"])
        process = subprocess.run(
            [sys.executable, check.__file__, "exclusive_scan", str(self.candidate)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 1, process.stderr)
        result = json.loads(process.stdout)
        self.assertFalse(result["passed"])
        self.assertEqual(result["cases"][1]["actual"], [2, 5, 9])

    def test_cli_bounds_nonterminating_candidate(self):
        self.candidate.write_text("def solve(values):\n    while True: pass\n")
        process = subprocess.run(
            [sys.executable, check.__file__, "exclusive_scan", str(self.candidate)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertIn(process.returncode, (-signal.SIGXCPU, -signal.SIGKILL))


if __name__ == "__main__":
    unittest.main()
