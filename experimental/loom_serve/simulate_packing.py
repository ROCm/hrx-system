# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Closed-loop packing experiments over recorded agent work, without a device."""

import argparse
import hashlib
import heapq
import json
import sys
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from experimental.loom_serve.agent_trace import Request, compose_trace, load_trace

POLICIES = ("single-pass", "fair-fill")
ADMISSION_ORDERS = ("round-robin", "longest", "shortest")


def allocate(prefill_remaining, capacity, policy):
    """Partitions an admitted frontier; zero prefill means one ready decode.

    Admission has already bounded the span count by capacity. Every admitted
    row progresses; future decode tokens cannot enter this epoch's demand.
    """
    grants = [0 if count else 1 for count in prefill_remaining]
    remaining = capacity - sum(grants)
    pending = [index for index, count in enumerate(prefill_remaining) if count]
    if policy == "single-pass":
        # The count arithmetic used by qwen_workload_plan, in admission order.
        for offset, index in enumerate(pending):
            grants[index] = min(
                prefill_remaining[index], remaining // (len(pending) - offset)
            )
            remaining -= grants[index]
    else:
        while pending:
            share, extra = divmod(remaining, len(pending))
            short = [index for index in pending if prefill_remaining[index] <= share]
            if not short:
                for offset, index in enumerate(pending):
                    grants[index] = share + (offset < extra)
                break
            for index in short:
                grants[index] = prefill_remaining[index]
                remaining -= grants[index]
            pending = [index for index in pending if index not in short]
    return grants


@dataclass
class ActiveRequest:
    # Ordinal of the client in the imported trace.
    session: int
    # Ordinal of this request within its client history.
    index: int
    # Immutable observed request counts and client delay.
    request: Request
    # Simulated arrival after the preceding response and client delay.
    arrival_us: int
    # Time this row most recently became runnable, for fairness accounting.
    ready_since_us: int
    # Largest runnable-to-service delay for this request.
    max_ready_wait_us: int
    # Known input tokens not yet consumed.
    prefill_remaining: int
    # Predictions not yet selected, including the first prefill prediction.
    output_remaining: int
    # Retained prefix plus inputs consumed in completed epochs.
    position: int
    # Completion time of the first prediction, or None before final prefill.
    first_prediction_us: int | None = None


def admit(ready, count, order, urgent_by_us, max_hold_us):
    """Selects distinct rows without consuming input or changing queue order."""
    if order == "round-robin":
        return list(ready)[:count]

    def priority(row):
        if row.ready_since_us + max_hold_us <= urgent_by_us:
            return (0, row.ready_since_us, 0)
        length = row.prefill_remaining or 1
        return (1, -length if order == "longest" else length, row.ready_since_us)

    return sorted(ready, key=priority)[:count]


def simulate(
    trace,
    *,
    policy,
    capacities,
    span_capacity,
    epoch_us,
    admission="round-robin",
    max_hold_us=0,
    emit_epoch=None,
):
    """Replays trusted counts with resident state and an explicit uniform clock.

    Every client has at most one active request. Admission settings are shared
    by both allocators. Arrivals during an epoch wait for its completion; a
    finished response releases its successor only after the recorded client gap.
    Capacities are positive, sorted and unique; epoch_us and span_capacity are
    positive, and max_hold_us is nonnegative. The CLI validates these external
    configuration constraints. The hold is a ready-to-dispatch target, not a
    guarantee: overloaded queues and nonpreemptible work can overrun it.
    """
    budget = capacities[-1]
    arrivals = [
        (session.arrival_us, index, 0) for index, session in enumerate(trace.sessions)
    ]
    heapq.heapify(arrivals)
    ready = deque()
    counters = Counter()
    shapes = Counter()
    kinds = Counter()
    completed = []
    now_us = 0
    max_hold_overrun_us = 0

    def release_arrivals(time_us):
        while arrivals and arrivals[0][0] <= time_us:
            arrival_us, session, index = heapq.heappop(arrivals)
            request = trace.sessions[session].requests[index]
            ready.append(
                ActiveRequest(
                    session,
                    index,
                    request,
                    arrival_us,
                    arrival_us,
                    0,
                    request.prefill_tokens,
                    request.output_tokens,
                    request.retained_tokens,
                )
            )

    while ready or arrivals:
        if not ready:
            now_us = max(now_us, arrivals[0][0])
        release_arrivals(now_us)
        collection_start_us = now_us
        while True:
            admitted = admit(
                ready,
                min(len(ready), span_capacity, budget),
                admission,
                now_us + epoch_us,
                max_hold_us,
            )
            admitted_tokens = sum(row.prefill_remaining or 1 for row in admitted)
            hold_until_us = min(row.ready_since_us for row in ready) + max_hold_us
            if (
                admitted_tokens >= budget
                or len(admitted) == min(span_capacity, budget)
                or now_us >= hold_until_us
            ):
                break
            # The event engine supplies the next wakeup, not advance knowledge
            # of its demand. Without new work, the collection timer still fires.
            now_us = min(hold_until_us, arrivals[0][0]) if arrivals else hold_until_us
            release_arrivals(now_us)
        ready_tokens = sum(row.prefill_remaining or 1 for row in ready)
        ready_rows = len(ready)
        admitted_sessions = {row.session for row in admitted}
        ready = deque(row for row in ready if row.session not in admitted_sessions)
        grants = allocate([row.prefill_remaining for row in admitted], budget, policy)
        useful_tokens = sum(grants)
        capacity = next(
            capacity for capacity in capacities if capacity >= useful_tokens
        )
        end_us = now_us + epoch_us
        spans = []
        survivors = []
        late_service_spans = 0
        for row, count in zip(admitted, grants):
            is_prefill = row.prefill_remaining > 0
            selects_output = not is_prefill or count == row.prefill_remaining
            ready_wait_us = now_us - row.ready_since_us
            hold_overrun_us = max(0, ready_wait_us - max_hold_us)
            late_service_spans += hold_overrun_us > 0
            max_hold_overrun_us = max(max_hold_overrun_us, hold_overrun_us)
            spans.append(
                {
                    "session": row.session,
                    "request": row.index,
                    "purpose": row.request.purpose,
                    "kind": "prefill" if is_prefill else "decode",
                    "position": row.position,
                    "input_tokens": count,
                    "selects_output": selects_output,
                    "ready_wait_us": ready_wait_us,
                    "hold_overrun_us": hold_overrun_us,
                }
            )
            row.max_ready_wait_us = max(row.max_ready_wait_us, ready_wait_us)
            row.position += count
            if is_prefill:
                row.prefill_remaining -= count
            if selects_output:
                row.output_remaining -= 1
                if row.first_prediction_us is None:
                    row.first_prediction_us = end_us
            if row.output_remaining:
                row.ready_since_us = end_us
                survivors.append(row)
            else:
                completed.append(
                    {
                        "session": row.session,
                        "request": row.index,
                        "purpose": row.request.purpose,
                        "arrival_us": row.arrival_us,
                        "first_prediction_us": row.first_prediction_us,
                        "completion_us": end_us,
                        "max_ready_wait_us": row.max_ready_wait_us,
                        "final_position": row.position,
                    }
                )
                next_index = row.index + 1
                if next_index < len(trace.sessions[row.session].requests):
                    next_request = trace.sessions[row.session].requests[next_index]
                    heapq.heappush(
                        arrivals,
                        (end_us + next_request.delay_us, row.session, next_index),
                    )

        # Earlier queued rows retain their places. New arrivals precede rows
        # that just received service, including zero-delay tool continuations.
        release_arrivals(end_us)
        ready.extend(survivors)
        prefill_tokens = sum(
            span["input_tokens"] for span in spans if span["kind"] == "prefill"
        )
        decode_tokens = useful_tokens - prefill_tokens
        kind = (
            "mixed"
            if prefill_tokens and decode_tokens
            else "prefill"
            if prefill_tokens
            else "decode"
        )
        counts = {
            "collection_us": now_us - collection_start_us,
            "late_service_spans": late_service_spans,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "selected_outputs": sum(span["selects_output"] for span in spans),
            "token_slots": capacity,
            "shape_padding_tokens": capacity - useful_tokens,
            # These three gaps partition the unused maximum token budget,
            # not the padding of the smaller shape selected after packing.
            "packing_gap_tokens": min(budget, admitted_tokens) - useful_tokens,
            "span_gap_tokens": min(budget, ready_tokens) - min(budget, admitted_tokens),
            "causal_gap_tokens": budget - min(budget, ready_tokens),
        }
        counters.update(counts)
        shapes[capacity] += 1
        kinds[kind] += 1
        if emit_epoch:
            emit_epoch(
                {
                    "event": "epoch",
                    "policy": policy,
                    "epoch": sum(shapes.values()) - 1,
                    "start_us": now_us,
                    "end_us": end_us,
                    "kind": kind,
                    "ready_rows": ready_rows,
                    "ready_tokens": ready_tokens,
                    "admitted_ready_tokens": admitted_tokens,
                    "capacity": capacity,
                    **counts,
                    "spans": spans,
                }
            )
        now_us = end_us

    epochs = sum(shapes.values())
    useful_tokens = counters["prefill_tokens"] + counters["decode_tokens"]
    return {
        "policy": policy,
        "admission": admission,
        "max_hold_us": max_hold_us,
        "max_hold_overrun_us": max_hold_overrun_us,
        "epochs": epochs,
        **counters,
        "useful_input_tokens": useful_tokens,
        "shape_fill_fraction": useful_tokens / counters["token_slots"],
        "budget_fill_fraction": useful_tokens / (epochs * budget),
        "shape_epochs": dict(sorted(shapes.items())),
        "kind_epochs": dict(kinds),
        "simulated_finish_us": now_us,
        "simulated_busy_us": epochs * epoch_us,
        "requests": sorted(
            completed, key=lambda request: (request["session"], request["request"])
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="content-free agent_trace export")
    parser.add_argument(
        "--composition",
        type=Path,
        help="explicit synthetic instances, start times, pauses and request limits",
    )
    parser.add_argument(
        "--capacities", default="32,64,128", help="ascending token shapes"
    )
    parser.add_argument("--span-capacity", type=int, default=8)
    parser.add_argument("--admission", choices=ADMISSION_ORDERS, default="round-robin")
    parser.add_argument(
        "--max-hold-us",
        type=int,
        default=0,
        help="ready-to-dispatch target; zero dispatches immediately",
    )
    parser.add_argument(
        "--epoch-us",
        type=int,
        required=True,
        help="hypothetical duration of EVERY epoch; not measured throughput",
    )
    parser.add_argument("--policy", choices=(*POLICIES, "both"), default="both")
    parser.add_argument(
        "--output", type=Path, help="new summary file; defaults to stdout"
    )
    parser.add_argument("--epochs", type=Path, help="new per-epoch JSONL ledger")
    args = parser.parse_args()
    try:
        capacities = tuple(int(value) for value in args.capacities.split(","))
        if (
            not capacities
            or capacities[0] < 1
            or tuple(sorted(set(capacities))) != capacities
        ):
            raise ValueError("capacities must be positive, ascending, and unique")
        if args.span_capacity < 1 or args.epoch_us < 1:
            raise ValueError("span-capacity and epoch-us must be positive")
        if args.max_hold_us < 0:
            raise ValueError("max-hold-us must be nonnegative")
        trace = load_trace(args.trace)
        source_sessions = len(trace.sessions)
        composition = None
        if args.composition:
            with args.composition.open() as stream:
                trace, composition = compose_trace(trace, json.load(stream))
        policies = POLICIES if args.policy == "both" else (args.policy,)
        configuration = {
            "format": "loom-packing-replay-v1",
            "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
            "model": trace.model,
            "source_sessions": source_sessions,
            "composition": composition,
            "sessions": len(trace.sessions),
            "client_configurations": [
                asdict(session.configuration) if session.configuration else None
                for session in trace.sessions
            ],
            "capacities": capacities,
            "span_capacity": args.span_capacity,
            "admission": args.admission,
            "max_hold_us": args.max_hold_us,
            "uniform_epoch_us": args.epoch_us,
            "assumptions": [
                "Hypothetical uniform epoch time, not a throughput measurement.",
                "Recorded output lengths and retained-prefix outcomes stay fixed.",
                "All session state remains resident; no eviction or memory-capacity model.",
                "Ordinary causal decoding; no speculative tokens, backpressure, or cancellation.",
                "Smallest-fitting token shape; hold targets may be exceeded by queueing/nonpreemptible work.",
                "All recorded request input is ready at arrival; no incremental tokenizer publication timings.",
            ],
        }
        with ExitStack() as stack:
            output = (
                stack.enter_context(args.output.open("x"))
                if args.output
                else sys.stdout
            )
            ledger = stack.enter_context(args.epochs.open("x")) if args.epochs else None
            if ledger:
                ledger.write(
                    json.dumps({"event": "configuration", **configuration}) + "\n"
                )
            results = [
                simulate(
                    trace,
                    policy=policy,
                    capacities=capacities,
                    span_capacity=args.span_capacity,
                    epoch_us=args.epoch_us,
                    admission=args.admission,
                    max_hold_us=args.max_hold_us,
                    emit_epoch=(lambda epoch: ledger.write(json.dumps(epoch) + "\n"))
                    if ledger
                    else None,
                )
                for policy in policies
            ]
            output.write(
                json.dumps({**configuration, "results": results}, indent=2) + "\n"
            )
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
