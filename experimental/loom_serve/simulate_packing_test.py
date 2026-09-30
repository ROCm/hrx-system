# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import unittest

from experimental.loom_serve.agent_trace import (
    COMPOSITION_FORMAT,
    ClientConfiguration,
    Request,
    Session,
    Trace,
    compose_trace,
)
from experimental.loom_serve.simulate_packing import (
    ActiveRequest,
    EpochCosts,
    accepted_drafts,
    admit,
    allocate,
    epoch_cost,
    expected_progress,
    plan_epoch,
    read_costs,
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

    def test_single_pass_selects_smallest_shape_without_changing_grants(self):
        rows = [
            ActiveRequest(index, 0, request(count, 1), 0, 0, 0, count, 1, 0)
            for index, count in enumerate((50, 1))
        ]
        plan = plan_epoch(
            rows,
            costs=EpochCosts("uniform", dict.fromkeys((32, 64, 128), 10)),
            capacities=(32, 64, 128),
            depths=(0,),
            acceptance=(),
            capture_bytes=0,
            policy="single-pass",
            cohort_policy="fill",
        )
        self.assertEqual((plan.grants, plan.capacity), ((50, 1), 64))

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
            costs=EpochCosts(
                "test uniform clock", dict.fromkeys((4, 8, 128), epoch_us)
            ),
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

    def test_overlay_pauses_and_stops_remain_causal_under_contention(self):
        original = trace(
            [
                request(130, 3),
                Request(7, 1000000, 20, 0, 2, "stop", "compaction"),
                request(3, 2, delay=11),
            ]
        )
        workload, _ = compose_trace(
            original,
            {
                "format": COMPOSITION_FORMAT,
                "instances": [
                    {
                        "source_session": 0,
                        "start_us": 0,
                        "request_count": 2,
                        "pauses": [{"before_request": 1, "duration_us": 100}],
                    },
                    {"source_session": 0, "start_us": 5},
                ],
            },
        )
        fast, _ = self.replay(workload, epoch_us=10, span_capacity=1)
        slow, _ = self.replay(workload, epoch_us=100, span_capacity=1)
        for result in (fast, slow):
            self.assertEqual(result["prefill_tokens"], 303)
            self.assertEqual(result["decode_tokens"], 7)
            self.assertEqual(result["selected_outputs"], 12)
            requests = result["requests"]
            self.assertEqual(len(requests), 5)
            self.assertEqual(
                [row["request"] for row in requests if row["session"] == 0], [0, 1]
            )
            for session_index, session in enumerate(workload.sessions):
                rows = [row for row in requests if row["session"] == session_index]
                self.assertEqual(rows[0]["arrival_us"], session.arrival_us)
                for previous, following in zip(rows, rows[1:]):
                    source_request = session.requests[following["request"]]
                    self.assertEqual(
                        following["arrival_us"],
                        previous["completion_us"] + source_request.delay_us,
                    )
                self.assertEqual(rows[1]["purpose"], "compaction")
                self.assertEqual(rows[1]["final_position"], 21)
        self.assertGreater(
            slow["requests"][1]["arrival_us"], fast["requests"][1]["arrival_us"]
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


class SpeculativePackingTest(unittest.TestCase):
    def costs(self, **coefficients):
        return EpochCosts(
            "test hypothesis",
            {4: 10, 8: 10, 32: 10},
            transition_bytes_per_token=2,
            **coefficients,
        )

    def replay(
        self,
        workload,
        *,
        acceptance=(1, 1, 1),
        capture_bytes=1024,
        cohort_policy="fill",
        costs=None,
    ):
        epochs = []
        result = simulate(
            workload,
            policy="fair-fill",
            capacities=(4, 8, 32),
            span_capacity=8,
            costs=costs or self.costs(),
            depths=(0, 3),
            acceptance=acceptance,
            capture_bytes=capture_bytes,
            cohort_policy=cohort_policy,
            emit_epoch=epochs.append,
        )
        return result, epochs

    def test_all_acceptance_prefixes_and_terminal_cuts_preserve_work(self):
        # Zero through three matching drafts, including EOS/budget cuts inside
        # an otherwise fully matching block. Serial decode is the count oracle.
        for matched in range(4):
            probabilities = (1,) * matched + (0,) * (3 - matched)
            for outputs in range(1, 12):
                workload = trace([request(3, outputs, retained=11)])
                result, epochs = self.replay(workload, acceptance=probabilities)
                control = simulate(
                    workload,
                    policy="fair-fill",
                    capacities=(4, 8, 32),
                    span_capacity=8,
                    costs=self.costs(),
                )
                self.assertEqual(result["prefill_tokens"], 3)
                self.assertEqual(result["decode_tokens"], outputs - 1)
                self.assertEqual(result["selected_outputs"], outputs)
                self.assertEqual(
                    result["requests"][0]["final_position"],
                    control["requests"][0]["final_position"],
                )
                self.assertEqual(
                    result["target_input_tokens"] - result["speculative_waste_tokens"],
                    outputs + 2,
                )
                for epoch in epochs:
                    self.assertEqual(
                        epoch["target_input_tokens"]
                        + epoch["packing_gap_tokens"]
                        + epoch["span_gap_tokens"]
                        + epoch["causal_gap_tokens"],
                        32,
                    )
                    for span in epoch["spans"]:
                        if span["kind"] == "decode":
                            self.assertEqual(span["input_tokens"], 4)
                            self.assertEqual(
                                span["committed_inputs"], span["selected_outputs"]
                            )

    def test_terminal_length_is_not_visible_to_planning(self):
        def choose(outputs):
            row = ActiveRequest(0, 0, request(1, outputs), 0, 0, 0, 0, outputs, 10)
            return plan_epoch(
                [row],
                costs=self.costs(),
                capacities=(4, 8, 32),
                depths=(0, 3),
                acceptance=(1, 1, 1),
                capture_bytes=1024,
                policy="fair-fill",
                cohort_policy="cost",
            )

        self.assertEqual(choose(1), choose(100))
        result, epochs = self.replay(trace([request(1, 2)]))
        self.assertEqual(epochs[-1]["target_input_tokens"], 4)
        self.assertEqual(epochs[-1]["decode_tokens"], 1)
        self.assertEqual(result["speculative_waste_tokens"], 3)

    def test_capture_pressure_selects_ordinary_progress(self):
        result, epochs = self.replay(trace([request(1, 7)]), capture_bytes=7)
        self.assertEqual(result["depth_epochs"], {0: 7})
        self.assertEqual(result["peak_capture_bytes"], 0)
        self.assertEqual(result["selected_outputs"], 7)
        self.assertTrue(all(epoch["capture_bytes"] <= 7 for epoch in epochs))

    def test_serial_costs_include_draft_catchup_and_commit(self):
        rows = [
            ActiveRequest(index, 0, request(1, 100), 0, 0, 0, 0, 100, position)
            for index, position in enumerate((10, 20))
        ]
        costs = self.costs(
            head_fixed_us=2,
            head_row_us=1,
            attention_pair_us=0.5,
            state_span_us=3,
            draft_round_us=5,
            draft_token_us=2,
            catchup_fixed_us=4,
            catchup_token_us=3,
            capture_token_us=7,
            commit_span_us=11,
            commit_token_us=13,
        )
        timing = epoch_cost(costs, rows, (4, 4), 8, (2, 4), True)
        self.assertEqual(
            timing,
            {
                "target_body_us": 10,
                "target_head_us": 10,
                "attention_us": 70,
                "state_us": 6,
                "draft_us": 27,
                "catchup_us": 22,
                "capture_us": 56,
                "commit_us": 100,
            },
        )
        result, _ = self.replay(trace([request(5, 6)]), costs=costs, capture_bytes=0)
        self.assertEqual(result["catchup_us"], 4 * result["epochs"] + 3 * 10)
        self.assertEqual(result["draft_us"], 0)

    def test_cost_selection_can_reject_deep_and_large_shapes(self):
        row = ActiveRequest(0, 0, request(1, 100), 0, 0, 0, 0, 100, 10)
        arguments = dict(
            rows=[row],
            capacities=(4, 8, 32),
            depths=(0, 3),
            acceptance=(1, 1, 1),
            capture_bytes=1024,
            policy="fair-fill",
            cohort_policy="cost",
        )
        self.assertEqual(plan_epoch(costs=self.costs(), **arguments).depth, 3)
        self.assertEqual(
            plan_epoch(costs=self.costs(draft_round_us=100), **arguments).depth, 0
        )
        row.prefill_remaining = 100
        costs = EpochCosts("shape hypothesis", {4: 10, 8: 100, 32: 1000})
        plan = plan_epoch(costs=costs, **arguments)
        self.assertEqual((plan.capacity, plan.grants), (4, (4,)))

    def test_followup_waits_for_actual_speculative_completion(self):
        workload = trace([request(1, 6), request(2, 2, delay=7, retained=6)])
        result, _ = self.replay(workload, costs=self.costs(draft_round_us=2))
        first, second = result["requests"]
        self.assertEqual(second["arrival_us"], first["completion_us"] + 7)
        self.assertEqual((first["final_position"], second["final_position"]), (6, 9))

    def test_conditional_acceptance_expectation(self):
        self.assertAlmostEqual(expected_progress(4, (0.5, 0.5, 0.5)), 1.875)
        self.assertEqual(expected_progress(1, ()), 1)

    def test_acceptance_draws_share_prefixes_and_ignore_cohort_order(self):
        rows = [
            ActiveRequest(index, 0, request(1, 100), 0, 0, 0, 0, 100, 10)
            for index in range(16)
        ]
        probabilities = (0.7,) * 7
        outcomes = {
            row.session: accepted_drafts(row, 8, probabilities, 123) for row in rows
        }
        self.assertEqual(
            outcomes,
            {
                row.session: accepted_drafts(row, 8, probabilities, 123)
                for row in reversed(rows)
            },
        )
        for row in rows:
            for width in range(1, 9):
                self.assertEqual(
                    accepted_drafts(row, width, probabilities, 123),
                    min(outcomes[row.session], width - 1),
                )

    def test_known_context_limits_clip_verification_without_peeking_at_eos(self):
        configuration = ClientConfiguration("test", 8, 8, "off", True, 1, 1)
        workload = Trace(
            "loom",
            "qwen3.8-27b",
            0,
            (Session("0" * 64, 0, (request(3, 6),), configuration),),
        )
        result, epochs = self.replay(workload)
        self.assertEqual(result["requests"][0]["final_position"], 8)
        self.assertEqual(
            [epoch["spans"][0]["input_tokens"] for epoch in epochs], [3, 4, 1]
        )
        invalid = Trace(
            "loom",
            "qwen3.8-27b",
            0,
            (Session("0" * 64, 0, (request(3, 7),), configuration),),
        )
        with self.assertRaisesRegex(ValueError, "context limit"):
            self.replay(invalid)

    def test_profile_validation_rejects_missing_shapes_and_invalid_costs(self):
        base = {
            "format": "loom-epoch-costs-v1",
            "provenance": "test",
            "target_us": {"4": 10},
        }
        self.assertEqual(read_costs(base, (4,)).target_us, {4: 10})
        for change in (
            {"target_us": {}},
            {"target_us": {"4": 0}},
            {"target_us": {"4": True}},
            {"provenance": ""},
            {"draft_round_us": -1},
            {"head_row_us": float("nan")},
            {"transition_bytes_per_token": 0.5},
            {"misspelled_cost": 1},
        ):
            with self.assertRaises(ValueError):
                read_costs({**base, **change}, (4,))


if __name__ == "__main__":
    unittest.main()
