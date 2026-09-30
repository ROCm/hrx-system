# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exports content-free, causally linked workload counts from pi session files."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "loom-agent-trace-v2"
CALL_FORMAT = "loom-pi-calls-v1"


@dataclass(frozen=True)
class ClientConfiguration:
    # pi SDK version used by the call recorder.
    pi_version: str
    # This client's advertised window, not its physical KV allocation.
    context_window_tokens: int
    # Configured model output ceiling; individual calls may request less.
    max_output_tokens: int
    # Client reasoning setting, independent of the shared model identity.
    thinking_level: str
    # Whether pi may compact automatically between turns.
    compaction_enabled: bool
    # Headroom used by pi's automatic compaction threshold.
    reserve_tokens: int
    # Approximate recent history retained alongside the generated summary.
    keep_recent_tokens: int


@dataclass(frozen=True)
class Request:
    # Delay after the preceding client completion; zero for the first request.
    delay_us: int
    # Client invocation to completed-message persistence, not device time.
    observed_duration_us: int
    # New input to process, including any reported cache writes.
    prefill_tokens: int
    # Prefix already retained at recorded admission; not new work.
    retained_tokens: int
    # Selected predictions, including EOS when the provider counts it.
    output_tokens: int
    # Successful pi termination kind: stop, length, or toolUse.
    stop_reason: str
    # Ordinary agent generation or a separately recorded summarization call.
    purpose: str = "agent"


@dataclass(frozen=True)
class Session:
    # Hash of the raw recording, without its name, contents, or directory.
    source_sha256: str
    # First invocation offset from the trace origin.
    arrival_us: int
    # Sequential requests with completion-relative client delays.
    requests: tuple[Request, ...]
    # None when the native session file did not record the client's settings.
    configuration: ClientConfiguration | None = None


@dataclass(frozen=True)
class Trace:
    # Provider identity shared by every request in the trace.
    provider: str
    # Model identity shared by every request in the trace.
    model: str
    # Common client wall-clock origin in Unix microseconds.
    origin_unix_us: int
    # Independent client histories, each with at most one outstanding request.
    sessions: tuple[Session, ...]

    def to_dict(self):
        return {"format": FORMAT, **asdict(self)}


def _object(value):
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _integer(record, key, minimum=0):
    value = record.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _string(record, key):
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _timestamp_us(record):
    timestamp = datetime.fromisoformat(_string(record, "timestamp"))
    if timestamp.tzinfo is None:
        raise ValueError("entry timestamp must include its time zone")
    delta = timestamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def _stop_reason(value):
    if value not in ("stop", "length", "toolUse"):
        raise ValueError("only successful stop/length/toolUse requests are supported")
    return value


def _purpose(value):
    if value not in ("agent", "compaction"):
        raise ValueError("request purpose must be agent or compaction")
    return value


def _configuration(record):
    record = _object(record)
    enabled = record.get("compaction_enabled")
    if type(enabled) is not bool:
        raise ValueError("compaction_enabled must be a boolean")
    return ClientConfiguration(
        _string(record, "pi_version"),
        _integer(record, "context_window_tokens", 1),
        _integer(record, "max_output_tokens", 1),
        _string(record, "thinking_level"),
        enabled,
        _integer(record, "reserve_tokens", 1),
        _integer(record, "keep_recent_tokens", 1),
    )


def _request(usage, start_us, end_us, previous_end_us, stop_reason, purpose):
    usage = _object(usage)
    prefill = _integer(usage, "input") + _integer(usage, "cacheWrite")
    retained = _integer(usage, "cacheRead")
    output = _integer(usage, "output", 1)
    if prefill == 0:
        raise ValueError("each request needs at least one new input token")
    if _integer(usage, "totalTokens") != prefill + retained + output:
        raise ValueError("usage total disagrees with input/cache/output counts")
    delay_us = 0 if previous_end_us is None else start_us - previous_end_us
    if end_us < start_us or delay_us < 0:
        raise ValueError("client timestamps overlap or run backwards")
    return Request(
        delay_us,
        end_us - start_us,
        prefill,
        retained,
        output,
        _stop_reason(stop_reason),
        _purpose(purpose),
    )


def _read_pi_session(path):
    digest = hashlib.sha256()
    requests = []
    identities = set()
    entry_ids = set()
    parent_id = None
    first_start_us = None
    previous_end_us = None
    pending = False
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            try:
                entry = _object(json.loads(line))
                if line_number == 1:
                    if entry.get("type") != "session" or entry.get("version") != 3:
                        raise ValueError("expected a pi v3 session header")
                    if entry.get("parentSession"):
                        raise ValueError(
                            "forked sessions require an explicit branch replay"
                        )
                    continue
                entry_id = _string(entry, "id")
                if entry_id in entry_ids or entry.get("parentId") != parent_id:
                    raise ValueError("expected a linear history with unique entry IDs")
                entry_ids.add(entry_id)
                parent_id = entry_id
                kind = entry.get("type")
                if kind in ("model_change", "thinking_level_change", "session_info"):
                    continue
                if kind != "message":
                    raise ValueError(f"unsupported session entry type: {kind!r}")
                message = _object(entry.get("message"))
                role = message.get("role")
                if role in ("user", "toolResult"):
                    pending = True
                    continue
                if role != "assistant":
                    raise ValueError(f"unsupported message role: {role!r}")
                if message.get("api") != "openai-completions":
                    raise ValueError(
                        "timestamp/usage mapping requires openai-completions"
                    )
                identities.add(
                    (_string(message, "provider"), _string(message, "model"))
                )
                start_us = _integer(message, "timestamp") * 1000
                end_us = _timestamp_us(entry)
                requests.append(
                    _request(
                        message.get("usage"),
                        start_us,
                        end_us,
                        previous_end_us,
                        message.get("stopReason"),
                        "agent",
                    )
                )
                if first_start_us is None:
                    first_start_us = start_us
                previous_end_us = end_us
                pending = requests[-1].stop_reason == "toolUse"
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name}:{line_number}: {error}") from error
    if not requests or pending:
        raise ValueError(
            f"{path.name}: empty or incomplete session; finish recording first"
        )
    if len(identities) != 1:
        raise ValueError(
            f"{path.name}: model/provider changes cannot share one weight pass"
        )
    return (
        next(iter(identities)),
        first_start_us,
        digest.hexdigest(),
        tuple(requests),
        None,
    )


def _read_pi_calls(path):
    digest = hashlib.sha256()
    requests = []
    active = None
    finished = False
    first_start_us = None
    previous_end_us = None
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            try:
                entry = _object(json.loads(line))
                if line_number == 1:
                    if entry.get("format") != CALL_FORMAT:
                        raise ValueError(f"expected {CALL_FORMAT}")
                    identity = (_string(entry, "provider"), _string(entry, "model"))
                    configuration = _configuration(entry.get("configuration"))
                    continue
                if finished:
                    raise ValueError("data follows recording_end")
                kind = entry.get("type")
                if kind == "request_start":
                    if active is not None or _integer(entry, "index") != len(requests):
                        raise ValueError("expected sequential request starts")
                    if entry.get("api") != "openai-completions":
                        raise ValueError("usage mapping requires openai-completions")
                    active = (
                        _integer(entry, "start_us"),
                        _purpose(entry.get("purpose")),
                    )
                elif kind == "request_end":
                    if active is None or _integer(entry, "index") != len(requests):
                        raise ValueError("completion must match its request start")
                    start_us, purpose = active
                    end_us = _integer(entry, "end_us")
                    requests.append(
                        _request(
                            entry.get("usage"),
                            start_us,
                            end_us,
                            previous_end_us,
                            entry.get("stop_reason"),
                            purpose,
                        )
                    )
                    if first_start_us is None:
                        first_start_us = start_us
                    previous_end_us = end_us
                    active = None
                elif kind == "recording_end":
                    if active is not None or _integer(entry, "requests", 1) != len(
                        requests
                    ):
                        raise ValueError(
                            "recording_end must cover all completed requests"
                        )
                    if requests[-1].stop_reason == "toolUse":
                        raise ValueError("unfinished tool continuation")
                    finished = True
                else:
                    raise ValueError(f"failed or unsupported recording event: {kind!r}")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name}:{line_number}: {error}") from error
    if not finished:
        raise ValueError(f"{path.name}: empty or incomplete call recording")
    return identity, first_start_us, digest.hexdigest(), tuple(requests), configuration


def _read_recording(path):
    with path.open("rb") as stream:
        header = _object(json.loads(stream.readline()))
    if header.get("format") == CALL_FORMAT:
        return _read_pi_calls(path)
    return _read_pi_session(path)


def export_sessions(paths):
    """Reads completed linear recordings from one synchronized client clock."""
    if not paths:
        raise ValueError("at least one session recording is required")
    records = [_read_recording(Path(path)) for path in paths]
    if len({record[0] for record in records}) != 1:
        raise ValueError("all sessions must use the same provider and model")
    if len({record[2] for record in records}) != len(records):
        raise ValueError(
            "duplicate session recording; synthetic replication must be explicit"
        )
    origin = min(record[1] for record in records)
    provider, model = records[0][0]
    return Trace(
        provider,
        model,
        origin,
        tuple(
            Session(digest, start - origin, requests, configuration)
            for _, start, digest, requests, configuration in records
        ),
    )


def load_trace(path):
    """Validates the external count trace once, before simulation."""
    with Path(path).open() as stream:
        record = _object(json.load(stream))
    if record.get("format") != FORMAT:
        raise ValueError(f"expected {FORMAT}")
    sessions = []
    for session in record.get("sessions", []):
        session = _object(session)
        requests = []
        for request in session.get("requests", []):
            request = _object(request)
            requests.append(
                Request(
                    _integer(request, "delay_us"),
                    _integer(request, "observed_duration_us"),
                    _integer(request, "prefill_tokens", 1),
                    _integer(request, "retained_tokens"),
                    _integer(request, "output_tokens", 1),
                    _stop_reason(request.get("stop_reason")),
                    _purpose(request.get("purpose")),
                )
            )
        if not requests or requests[0].delay_us != 0:
            raise ValueError("session needs requests with zero initial delay")
        digest = _string(session, "source_sha256")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("source_sha256 must contain 64 lowercase hex digits")
        configuration = session.get("configuration")
        sessions.append(
            Session(
                digest,
                _integer(session, "arrival_us"),
                tuple(requests),
                _configuration(configuration) if configuration is not None else None,
            )
        )
    if not sessions or min(session.arrival_us for session in sessions) != 0:
        raise ValueError("trace needs sessions with a zero-origin first arrival")
    if len({session.source_sha256 for session in sessions}) != len(sessions):
        raise ValueError("duplicate session recording")
    return Trace(
        _string(record, "provider"),
        _string(record, "model"),
        _integer(record, "origin_unix_us"),
        tuple(sessions),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sessions",
        type=Path,
        nargs="+",
        help="completed pi v3 sessions or pi_record call logs",
    )
    parser.add_argument(
        "--output", type=Path, help="new output file; defaults to stdout"
    )
    args = parser.parse_args()
    try:
        trace = export_sessions(args.sessions)
        output = json.dumps(trace.to_dict(), indent=2) + "\n"
        if args.output:
            with args.output.open("x") as stream:
                stream.write(output)
        else:
            sys.stdout.write(output)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
