# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import unittest

from experimental.loom_serve.agent_trace import Request, Session, Trace
from experimental.loom_serve.simulate_packing import (
    ActiveRequest,
    admit,
    allocate,
    simulate,
)


def request(prefill, output, delay=0, retained=0):
    return Request(delay, 1000000, prefill, retained, output, "stop")


def trace(*histories):
    return Trace(
        "loom",
        "qwen3.8-27b",
        0,
        tuple(
            Session(f"{index:064x}", 0, tuple(history))
            for index, history in enumerate(histories)
        ),
    )


class PackingTest(unittest.TestCase):
    def test_short_tail_redistribution(self):
        self.assertEqual(allocate([0, 0, 1000, 2], 128, "single-pass"), [1, 1, 63, 2])
        self.assertEqual(allocate([0, 0, 1000, 2], 128, "fair-fill"), [1, 1, 124, 2])

    def test_small_frontiers_preserve_demand_progress_and_fair_fill(self):
        for size in range(1, 5):
            for demand in itertools.product((0, 1, 2, 5, 17), repeat=size):
                for capacity in (4, 8, 16, 32):
                    counts = allocate(demand, capacity, "fair-fill")
                    self.assertEqual(
                        sum(counts), min(capacity, sum(count or 1 for count in demand))
                    )
                    self.assertTrue(
                        all(
                            1 <= count <= (available or 1)
                            for count, available in zip(counts, demand)
                        )
                    )
                    unsatisfied = [
                        count
                        for count, available in zip(counts, demand)
                        if count < available
                    ]
                    if unsatisfied:
                        self.assertLessEqual(max(unsatisfied) - min(unsatisfied), 1)

    def replay(
        self,
        workload,
        *,
        policy="fair-fill",
        span_capacity=8,
        epoch_us=10,
        admission="round-robin",
        max_hold_us=0,
    ):
        epochs = []
        result = simulate(
            workload,
            policy=policy,
            capacities=(4, 8, 128),
            span_capacity=span_capacity,
            epoch_us=epoch_us,
            admission=admission,
            max_hold_us=max_hold_us,
            emit_epoch=epochs.append,
        )
        for epoch in epochs:
            useful = epoch["prefill_tokens"] + epoch["decode_tokens"]
            self.assertEqual(
                useful
                + sum(
                    epoch[key]
                    for key in (
                        "packing_gap_tokens",
                        "span_gap_tokens",
                        "causal_gap_tokens",
                    )
                ),
                128,
            )
            self.assertEqual(epoch["shape_padding_tokens"], epoch["capacity"] - useful)
            self.assertEqual(
                len({span["session"] for span in epoch["spans"]}), len(epoch["spans"])
            )
        return result, epochs

    def test_first_prediction_and_decode_are_causal(self):
        result, epochs = self.replay(trace([request(130, 3, retained=100)]))
        self.assertEqual([epoch["prefill_tokens"] for epoch in epochs], [128, 2, 0, 0])
        self.assertEqual([epoch["decode_tokens"] for epoch in epochs], [0, 0, 1, 1])
        self.assertEqual([epoch["selected_outputs"] for epoch in epochs], [0, 1, 1, 1])
        self.assertEqual(
            [epoch["spans"][0]["position"] for epoch in epochs], [100, 228, 230, 231]
        )
        self.assertEqual(result["requests"][0]["final_position"], 232)
        self.assertEqual(result["shape_epochs"], {4: 3, 128: 1})

    def test_followup_rebases_on_simulated_completion(self):
        workload = trace([request(1, 2), request(1, 1, delay=7)])
        fast, _ = self.replay(workload, epoch_us=10)
        slow, _ = self.replay(workload, epoch_us=100)
        self.assertEqual([row["arrival_us"] for row in fast["requests"]], [0, 27])
        self.assertEqual([row["arrival_us"] for row in slow["requests"]], [0, 207])
        self.assertEqual(
            (fast["simulated_finish_us"], slow["simulated_finish_us"]), (37, 307)
        )

    def test_span_limited_rows_rotate_without_starvation(self):
        result, epochs = self.replay(
            trace(*[[request(10, 2)] for _ in range(5)]), span_capacity=2
        )
        self.assertEqual(
            [[span["session"] for span in epoch["spans"]] for epoch in epochs],
            [[0, 1], [2, 3], [4, 0], [1, 2], [3, 4]],
        )
        self.assertGreater(result["span_gap_tokens"], 0)
        self.assertEqual(result["packing_gap_tokens"], 0)
        self.assertEqual(len(result["requests"]), 5)

    def test_epoch_underfill_and_equal_completed_work(self):
        workload = trace(
            [request(1, 2)], [request(1, 2)], [request(1063, 1)], [request(65, 1)]
        )
        baseline, baseline_epochs = self.replay(workload, policy="single-pass")
        candidate, candidate_epochs = self.replay(workload)
        self.assertEqual(baseline_epochs[1]["prefill_tokens"], 65)
        self.assertEqual(baseline_epochs[1]["decode_tokens"], 2)
        self.assertEqual(baseline_epochs[1]["packing_gap_tokens"], 61)
        self.assertEqual(candidate_epochs[1]["prefill_tokens"], 126)
        self.assertEqual(candidate_epochs[1]["packing_gap_tokens"], 0)
        for key in ("prefill_tokens", "decode_tokens", "selected_outputs"):
            self.assertEqual(baseline[key], candidate[key])

    def test_collection_coalesces_only_arrived_work_and_obeys_oldest_timer(self):
        workload = Trace(
            "loom",
            "qwen3.8-27b",
            0,
            (
                Session("0" * 64, 0, (request(60, 1),)),
                Session("1" * 64, 5, (request(68, 1),)),
            ),
        )
        immediate, _ = self.replay(workload)
        collected, epochs = self.replay(workload, max_hold_us=7)
        self.assertEqual((immediate["epochs"], collected["epochs"]), (2, 1))
        self.assertEqual(epochs[0]["start_us"], 5)
        self.assertEqual(collected["collection_us"], 5)
        # No second arrival: waiting must end at the timer, even with a gap.
        alone, epochs = self.replay(trace([request(60, 1)]), max_hold_us=7)
        self.assertEqual(epochs[0]["start_us"], 7)
        self.assertEqual(alone["simulated_finish_us"], 17)

    def test_size_order_yields_to_rows_due_before_another_epoch(self):
        rows = [
            ActiveRequest(index, 0, request(length, 1), 0, ready_since, 0, length, 1, 0)
            for index, (length, ready_since) in enumerate(((1, 0), (1000, 9), (10, 9)))
        ]
        self.assertEqual(
            [row.session for row in admit(rows, 3, "longest", 10, 20)], [1, 2, 0]
        )
        self.assertEqual(
            [row.session for row in admit(rows, 3, "shortest", 10, 20)], [0, 2, 1]
        )
        self.assertEqual(
            [row.session for row in admit(rows, 3, "longest", 20, 20)], [0, 1, 2]
        )

    def test_nonpreemptible_epoch_reports_hold_overrun(self):
        workload = Trace(
            "loom",
            "qwen3.8-27b",
            0,
            (
                Session("0" * 64, 0, (request(128, 1),)),
                Session("1" * 64, 1, (request(1, 1),)),
            ),
        )
        result, epochs = self.replay(workload, max_hold_us=2)
        self.assertEqual(result["collection_us"], 0)
        self.assertEqual(result["late_service_spans"], 1)
        self.assertEqual(result["max_hold_overrun_us"], 7)
        self.assertEqual(epochs[1]["spans"][0]["ready_wait_us"], 9)


if __name__ == "__main__":
    unittest.main()
