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
import math
import sys
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from experimental.loom_serve.agent_trace import Request, compose_trace, load_trace

POLICIES = ("single-pass", "fair-fill")
ADMISSION_ORDERS = ("round-robin", "longest", "shortest")
COHORT_POLICIES = ("fill", "cost")


@dataclass(frozen=True)
class EpochCosts:
    # Description of the hypothesis or calibration evidence, retained in results.
    provenance: str
    # Target body cost by compiled token capacity, excluding components below.
    target_us: dict[int, float]
    # One target vocabulary weight traversal when any head positions are selected.
    head_fixed_us: float = 0
    # Incremental target head cost per selected position.
    head_row_us: float = 0
    # Context-dependent cost per causal query/key token pair, across all layers.
    attention_pair_us: float = 0
    # Target recurrent-state traffic/computation per active span.
    state_span_us: float = 0
    # Shared MTP block and head cost per sequential batched proposal round.
    draft_round_us: float = 0
    # Incremental proposal cost per row in a draft round.
    draft_token_us: float = 0
    # One batched MTP cache catch-up traversal when the drafter is kept warm.
    catchup_fixed_us: float = 0
    # MTP cache catch-up cost per committed target input, including prompt input.
    catchup_token_us: float = 0
    # Capture cost per provisional target input, including its pending anchor.
    capture_token_us: float = 0
    # Accepted GDN/history commit setup and state traffic per speculative span.
    commit_span_us: float = 0
    # Local recurrent/history replay cost per committed speculative input.
    commit_token_us: float = 0
    # Retained transition payload per provisional target input, in bytes.
    transition_bytes_per_token: int = 0


def read_costs(record, capacities):
    """Validates a cost hypothesis at the external JSON boundary."""
    if not isinstance(record, dict) or record.get("format") != "loom-epoch-costs-v1":
        raise ValueError("expected a loom-epoch-costs-v1 object")
    names = {field.name for field in fields(EpochCosts)}
    if set(record) - names - {"format"}:
        raise ValueError("unknown epoch cost field")
    provenance = record.get("provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("cost provenance must describe the hypothesis/evidence")
    target = record.get("target_us")
    if not isinstance(target, dict) or set(target) != {str(c) for c in capacities}:
        raise ValueError("target_us must specify every configured token shape exactly")
    for value in target.values():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("target costs must be finite positive microseconds")
    coefficients = {
        key: record[key] for key in names - {"provenance", "target_us"} if key in record
    }
    for key, value in coefficients.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if type(coefficients.get("transition_bytes_per_token", 0)) is not int:
        raise ValueError("transition_bytes_per_token must be an integer")
    return EpochCosts(
        provenance, {int(key): value for key, value in target.items()}, **coefficients
    )


def allocate(prefill_remaining, capacity, policy, decode_widths=None):
    """Partitions an admitted frontier; zero prefill means a ready decode span.

    Admission has already bounded the span count by capacity. Every admitted
    row progresses; future decode tokens cannot enter this epoch's demand.
    """
    widths = (
        decode_widths if decode_widths is not None else [1] * len(prefill_remaining)
    )
    grants = [0 if count else width for count, width in zip(prefill_remaining, widths)]
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
    # Recorded client context limit, or None when the trace did not specify one.
    context_capacity: int | None = None


def decode_width(row, depth):
    """Bounds speculative input by known context capacity, never by future EOS."""
    width = depth + 1
    if row.context_capacity is not None:
        width = min(width, row.context_capacity - row.position)
    return width


def expected_progress(width, acceptance):
    """One target correction/bonus plus the surviving conditional draft prefix."""
    progress = 1.0
    survival = 1.0
    for probability in acceptance[: width - 1]:
        survival *= probability
        progress += survival
    return progress


def accepted_drafts(row, width, acceptance, seed):
    """Samples a hypothesis, keyed independently of epoch and cohort order."""
    output_position = row.request.output_tokens - row.output_remaining
    for index in range(width - 1):
        key = f"{seed}:{row.session}:{row.index}:{output_position}:{index}".encode()
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") / 2**64
        if draw >= acceptance[index]:
            return index
    return width - 1


@dataclass(frozen=True)
class EpochPlan:
    # Actual target inputs for every admitted row, including rejected speculation.
    grants: tuple[int, ...]
    # Compiled target token shape.
    capacity: int
    # Requested common draft depth, before individual context-limit truncation.
    depth: int
    # Transition-payload credit reserved until accepted-state commit completes.
    capture_bytes: int
    # Estimated committed target inputs; planning cannot see sampled outcomes.
    expected_inputs: float
    # Estimated full serial draft/target/catch-up/commit duration.
    expected_us: int


def epoch_cost(costs, rows, grants, capacity, progress, mtp_enabled):
    """Additive single-GPU hypothesis with no assumed overlap between stages.

    Target shape cost excludes the separately modeled head, context, state,
    drafting and commit costs. A coefficient of zero explicitly omits that cost;
    it is not evidence that hardware execution is free.
    """
    head_rows = sum(
        count if not row.prefill_remaining else int(count == row.prefill_remaining)
        for row, count in zip(rows, grants)
    )
    drafts = [
        count - 1 if not row.prefill_remaining else 0
        for row, count in zip(rows, grants)
    ]
    speculative = [index for index, depth in enumerate(drafts) if depth]
    pairs = sum(
        count * row.position + count * (count + 1) // 2
        for row, count in zip(rows, grants)
    )
    return {
        "target_body_us": costs.target_us[capacity],
        "target_head_us": (costs.head_fixed_us if head_rows else 0)
        + costs.head_row_us * head_rows,
        "attention_us": costs.attention_pair_us * pairs,
        "state_us": costs.state_span_us * len(rows),
        "draft_us": costs.draft_round_us * max(drafts, default=0)
        + costs.draft_token_us * sum(drafts),
        "catchup_us": costs.catchup_fixed_us + costs.catchup_token_us * sum(progress)
        if mtp_enabled
        else 0,
        "capture_us": costs.capture_token_us
        * sum(grants[index] for index in speculative),
        "commit_us": costs.commit_span_us * len(speculative)
        + costs.commit_token_us * sum(progress[index] for index in speculative),
    }


def plan_epoch(
    rows, *, costs, capacities, depths, acceptance, capture_bytes, policy, cohort_policy
):
    """Searches the bounded shape/depth choices for an already admitted cohort.

    Cost mode maximizes expected committed inputs per predicted microsecond.
    This is a local objective, not a globally optimal completed-task schedule.
    Fill mode prefers deeper feasible proposals then more packed inputs.
    Neither policy reads future output lengths, arrivals or acceptance draws.
    """
    candidates = []
    has_decode = any(not row.prefill_remaining for row in rows)
    for depth in depths if has_decode else (0,):
        widths = [
            decode_width(row, depth) if not row.prefill_remaining else 1 for row in rows
        ]
        capture = costs.transition_bytes_per_token * sum(
            width for width in widths if width > 1
        )
        if capture > capture_bytes:
            continue
        for capacity in capacities:
            if sum(widths) > capacity:
                continue
            grants = tuple(
                allocate(
                    [row.prefill_remaining for row in rows], capacity, policy, widths
                )
            )
            # Single-pass can leave enough unfilled space for a smaller shape.
            # Preserve its grants when shrinking; reallocating at that shape
            # would change the deliberately wasteful comparison policy.
            compiled_capacity = (
                next(shape for shape in capacities if shape >= sum(grants))
                if cohort_policy == "fill"
                else capacity
            )
            progress = [
                count if row.prefill_remaining else expected_progress(count, acceptance)
                for row, count in zip(rows, grants)
            ]
            duration = math.ceil(
                sum(
                    epoch_cost(
                        costs,
                        rows,
                        grants,
                        compiled_capacity,
                        progress,
                        bool(depths[-1]),
                    ).values()
                )
            )
            candidates.append(
                EpochPlan(
                    grants, compiled_capacity, depth, capture, sum(progress), duration
                )
            )
    if cohort_policy == "fill":
        return max(
            candidates, key=lambda plan: (plan.depth, sum(plan.grants), -plan.capacity)
        )
    return max(
        candidates,
        key=lambda plan: (
            plan.expected_inputs / plan.expected_us,
            -plan.expected_us,
            -plan.depth,
            -plan.capacity,
        ),
    )


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
    costs,
    depths=(0,),
    acceptance=(),
    capture_bytes=0,
    cohort_policy="fill",
    seed=0,
    admission="round-robin",
    max_hold_us=0,
    emit_epoch=None,
):
    """Replays trusted counts with resident state and explicit stage cost hypotheses.

    Every client has at most one active request. Admission settings are shared
    by both allocators. Arrivals during an epoch wait for its completion; a
    finished response releases its successor only after the recorded client gap.
    Capacities and depths are sorted unique tuples; depth zero is always available.
    The CLI validates cost, capacity and probability inputs. The hold is a
    ready-to-dispatch target, not a guarantee: overloaded queues and
    nonpreemptible work can overrun it.
    """
    budget = capacities[-1]
    arrivals = [
        (session.arrival_us, index, 0) for index, session in enumerate(trace.sessions)
    ]
    heapq.heapify(arrivals)
    ready = deque()
    counters = Counter()
    shapes = Counter()
    depth_epochs = Counter()
    kinds = Counter()
    completed = []
    now_us = 0
    max_hold_overrun_us = 0
    busy_us = 0
    peak_capture_bytes = 0

    def release_arrivals(time_us):
        while arrivals and arrivals[0][0] <= time_us:
            arrival_us, session, index = heapq.heappop(arrivals)
            request = trace.sessions[session].requests[index]
            configuration = trace.sessions[session].configuration
            context_capacity = (
                configuration.context_window_tokens if configuration else None
            )
            if (
                context_capacity is not None
                and request.retained_tokens
                + request.prefill_tokens
                + request.output_tokens
                - 1
                > context_capacity
            ):
                raise ValueError("recorded request exceeds its client context limit")
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
                    context_capacity=context_capacity,
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
                now_us + max(costs.target_us.values()),
                max_hold_us,
            )
            admitted_tokens = sum(
                row.prefill_remaining or decode_width(row, depths[-1])
                for row in admitted
            )
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
        plan = plan_epoch(
            admitted,
            costs=costs,
            capacities=capacities,
            depths=depths,
            acceptance=acceptance,
            capture_bytes=capture_bytes,
            policy=policy,
            cohort_policy=cohort_policy,
        )
        ready_tokens = sum(
            row.prefill_remaining or decode_width(row, plan.depth) for row in ready
        )
        admitted_tokens = sum(
            row.prefill_remaining or decode_width(row, plan.depth) for row in admitted
        )
        ready_rows = len(ready)
        admitted_sessions = {row.session for row in admitted}
        ready = deque(row for row in ready if row.session not in admitted_sessions)
        grants = plan.grants
        processed_tokens = sum(grants)
        capacity = plan.capacity
        # Outcomes are revealed only after planning. A terminal response can
        # discard otherwise accepted suffix positions, but still pays to run them.
        matches = [
            accepted_drafts(row, count, acceptance, seed)
            if not row.prefill_remaining
            else 0
            for row, count in zip(admitted, grants)
        ]
        progress = [
            count if row.prefill_remaining else min(row.output_remaining, accepted + 1)
            for row, count, accepted in zip(admitted, grants, matches)
        ]
        timing = epoch_cost(
            costs, admitted, grants, capacity, progress, bool(depths[-1])
        )
        duration_us = math.ceil(sum(timing.values()))
        end_us = now_us + duration_us
        busy_us += duration_us
        peak_capture_bytes = max(peak_capture_bytes, plan.capture_bytes)
        spans = []
        survivors = []
        late_service_spans = 0
        for row, count, consumed, matched in zip(admitted, grants, progress, matches):
            is_prefill = row.prefill_remaining > 0
            selects_output = not is_prefill or count == row.prefill_remaining
            selected_outputs = int(selects_output) if is_prefill else consumed
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
                    "committed_inputs": consumed,
                    "draft_tokens": count - 1 if not is_prefill else 0,
                    "accepted_draft_tokens": min(matched, selected_outputs)
                    if not is_prefill
                    else 0,
                    "selects_output": selects_output,
                    "selected_outputs": selected_outputs,
                    "head_rows": int(selects_output) if is_prefill else count,
                    "ready_wait_us": ready_wait_us,
                    "hold_overrun_us": hold_overrun_us,
                }
            )
            row.max_ready_wait_us = max(row.max_ready_wait_us, ready_wait_us)
            row.position += consumed
            if is_prefill:
                row.prefill_remaining -= count
            if selects_output:
                row.output_remaining -= selected_outputs
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
        decode_tokens = sum(progress) - prefill_tokens
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
            "selected_outputs": sum(span["selected_outputs"] for span in spans),
            "target_input_tokens": processed_tokens,
            "target_head_rows": sum(span["head_rows"] for span in spans),
            "draft_tokens": sum(span["draft_tokens"] for span in spans),
            "accepted_draft_tokens": sum(
                span["accepted_draft_tokens"] for span in spans
            ),
            "speculative_waste_tokens": processed_tokens - sum(progress),
            "token_slots": capacity,
            "shape_padding_tokens": capacity - processed_tokens,
            # These three gaps partition the unused maximum token budget,
            # not the padding of the smaller shape selected after packing.
            "packing_gap_tokens": min(budget, admitted_tokens) - processed_tokens,
            "span_gap_tokens": min(budget, ready_tokens) - min(budget, admitted_tokens),
            "causal_gap_tokens": budget - min(budget, ready_tokens),
        }
        counters.update(counts)
        counters.update(timing)
        shapes[capacity] += 1
        depth_epochs[plan.depth] += 1
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
                    "depth": plan.depth,
                    "capture_bytes": plan.capture_bytes,
                    "expected_inputs": plan.expected_inputs,
                    "expected_us": plan.expected_us,
                    "timing": timing,
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
        "cohort_policy": cohort_policy,
        "epochs": epochs,
        **counters,
        "useful_input_tokens": useful_tokens,
        "shape_fill_fraction": counters["target_input_tokens"]
        / counters["token_slots"],
        "useful_shape_fraction": useful_tokens / counters["token_slots"],
        "budget_fill_fraction": counters["target_input_tokens"] / (epochs * budget),
        "shape_epochs": dict(sorted(shapes.items())),
        "depth_epochs": dict(sorted(depth_epochs.items())),
        "peak_capture_bytes": peak_capture_bytes,
        "kind_epochs": dict(kinds),
        "simulated_finish_us": now_us,
        "simulated_busy_us": busy_us,
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
    clock = parser.add_mutually_exclusive_group(required=True)
    clock.add_argument(
        "--epoch-us",
        type=int,
        help="hypothetical duration of EVERY epoch; not measured throughput",
    )
    clock.add_argument(
        "--costs", type=Path, help="explicit loom-epoch-costs-v1 hypothesis/calibration"
    )
    parser.add_argument(
        "--depths",
        default="0",
        help="ascending candidate MTP lookaheads, starting with 0",
    )
    parser.add_argument(
        "--acceptance",
        default="",
        help="conditional draft acceptance probabilities by position",
    )
    parser.add_argument(
        "--capture-mib",
        type=float,
        default=0,
        help="bounded transition-payload capacity, excluding scratch",
    )
    parser.add_argument("--cohort-policy", choices=COHORT_POLICIES, default="fill")
    parser.add_argument(
        "--seed", type=int, default=0, help="counter-keyed hypothetical acceptance seed"
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
        if args.span_capacity < 1 or (args.epoch_us is not None and args.epoch_us < 1):
            raise ValueError("span-capacity and epoch-us must be positive")
        if args.max_hold_us < 0:
            raise ValueError("max-hold-us must be nonnegative")
        depths = tuple(int(value) for value in args.depths.split(","))
        if not depths or depths[0] != 0 or tuple(sorted(set(depths))) != depths:
            raise ValueError("depths must be ascending, unique, and start with 0")
        acceptance = (
            tuple(float(value) for value in args.acceptance.split(","))
            if args.acceptance
            else ()
        )
        if len(acceptance) != depths[-1] or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in acceptance
        ):
            raise ValueError(
                "acceptance must contain one probability in [0,1] per maximum draft position"
            )
        if not math.isfinite(args.capture_mib) or args.capture_mib < 0:
            raise ValueError("capture-mib must be finite and nonnegative")
        capture_bytes = int(args.capture_mib * 1024 * 1024)
        if args.costs:
            with args.costs.open() as stream:
                costs = read_costs(json.load(stream), capacities)
        else:
            costs = EpochCosts(
                "Hypothetical uniform epoch clock, not measured throughput.",
                dict.fromkeys(capacities, args.epoch_us),
            )
        if depths[-1] and (not args.costs or not costs.transition_bytes_per_token):
            raise ValueError(
                "speculation requires explicit stage costs and transition_bytes_per_token"
            )
        trace = load_trace(args.trace)
        source_sessions = len(trace.sessions)
        composition = None
        if args.composition:
            with args.composition.open() as stream:
                trace, composition = compose_trace(trace, json.load(stream))
        policies = POLICIES if args.policy == "both" else (args.policy,)
        configuration = {
            "format": "loom-packing-replay-v2",
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
            "costs": asdict(costs),
            "depths": depths,
            "conditional_acceptance": acceptance,
            "capture_bytes": capture_bytes,
            "cohort_policy": args.cohort_policy,
            "seed": args.seed,
            "assumptions": [
                "Predicted serial stage times, not a throughput measurement or a qualified performance oracle.",
                "Recorded output lengths and retained-prefix outcomes stay fixed.",
                "Target/MTP retained state is assumed warm and resident; no prefix identity, pin, KV capacity or eviction model.",
                "Only transition-payload credit is bounded; output backpressure and cancellation are not modeled.",
                "Conditional acceptance is a seeded hypothesis, not observed MTP quality; planning cannot see its samples or future EOS.",
                "MTP catch-up is charged on every committed input when any nonzero depth is configured, even in depth-zero epochs.",
                "Shape/depth selection is local to the admitted cohort, not a global scheduling optimum.",
                "Hold targets may be exceeded by queueing/nonpreemptible work.",
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
                    costs=costs,
                    depths=depths,
                    acceptance=acceptance,
                    capture_bytes=capture_bytes,
                    cohort_policy=args.cohort_policy,
                    seed=args.seed,
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
