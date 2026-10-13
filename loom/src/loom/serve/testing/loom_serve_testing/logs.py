# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent, bounded /logz reader for tools and model harnesses.

Each client owns its cursors. The server never retains reader state. Responses
are copied and closed before the caller sees them; a slow consumer can lose
records, but cannot retain a server connection or block publication. A changed
stream incarnation is reported explicitly, never spliced into the old history.
"""

import dataclasses
import http.client
import json
import socket
from urllib.parse import urlsplit


@dataclasses.dataclass(frozen=True)
class LogBatch:
    # Routing and incarnation metadata shared by every record in this response.
    stream: dict
    # Copied wire records, including stream header, counters and cursor footer.
    records: list


class ModelNotFound(RuntimeError):
    """A selected model disappeared or has not been registered yet."""

    def __init__(self, model):
        super().__init__(f"GET /logz ({model!r}): HTTP 404: model is not registered")
        self.model = model


class LogClient:
    def __init__(self, url, timeout=10.0, *, unix_socket=None):
        """Read HTTP logs, optionally through a raw filesystem socket path.

        The origin still supplies the HTTP Host header. Unix connections use
        plain HTTP and never resolve the origin's host or fall back to TCP.
        """
        address = urlsplit(url)
        if (
            address.scheme not in ("http", "https")
            or not address.hostname
            or address.username is not None
            or address.password is not None
            or address.path not in ("", "/")
            or address.query
            or address.fragment
        ):
            raise ValueError("server URL must be an HTTP(S) origin")
        if unix_socket is not None:
            if address.scheme != "http":
                raise ValueError("Unix socket observation requires a plain HTTP origin")
            if (
                not isinstance(unix_socket, str)
                or not unix_socket
                or "\0" in unix_socket
            ):
                raise ValueError("Unix socket path must be a nonempty filesystem path")
            if not hasattr(socket, "AF_UNIX"):
                raise ValueError("Unix sockets are unavailable on this platform")
        self.address = address
        # Literal local filesystem path, or None for the origin's TCP endpoint.
        self.unix_socket = unix_socket
        # Network observation deadline, not a deadline on submitted model work.
        self.timeout = timeout
        self.cursors = {}

    def _get(self, path, headers=None):
        connection_type = (
            http.client.HTTPSConnection
            if self.address.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            self.address.hostname, self.address.port, timeout=self.timeout
        )
        try:
            if self.unix_socket is not None:
                connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                connection.sock.settimeout(self.timeout)
                connection.sock.connect(self.unix_socket)
            connection.request("GET", path, headers=headers or {})
            response = connection.getresponse()
            # /logz currently renders at most 64 fixed-size records per poll.
            # Bound memory even when pointed at a different or broken endpoint.
            payload = response.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError(f"{path}: observation response exceeds 1 MiB")
            return response.status, payload
        finally:
            connection.close()

    def models(self):
        """List registered models, including preparation and retirement.

        Observation follows the administrative inventory rather than the
        inference catalog so draining/removal cannot hide the final records.
        """
        status, payload = self._get("/v1/status")
        if status != 200:
            raise RuntimeError(f"GET /v1/status: HTTP {status}: {payload!r}")
        catalog = json.loads(payload)
        models = [item["id"] for item in catalog["data"]]
        if any(not isinstance(model, str) or not model for model in models):
            raise ValueError("/v1/status: expected nonempty model identifiers")
        if len(set(models)) != len(models):
            raise ValueError("/v1/status: duplicate model identifier")
        return models

    def forget(self, model):
        """Release a no-longer-observed cursor, returning its last position."""
        return self.cursors.pop(model, None)

    def poll(self, model=None):
        """Copy one batch, or raise ModelNotFound for an absent model selector.

        Absence preserves the last cursor until the caller records it and
        forgets the model, or attempts to resume a replacement incarnation.
        A missing server stream remains an endpoint error, not model churn.
        """
        headers = {} if model is None else {"X-Loom-Model": model}
        previous = self.cursors.get(model)
        if previous is not None:
            headers["X-Loom-Cursor"] = previous
        status, payload = self._get("/logz", headers)
        reset = status == 409 and previous is not None
        if reset:
            del headers["X-Loom-Cursor"]
            status, payload = self._get("/logz", headers)
        if status == 404 and model is not None:
            raise ModelNotFound(model)
        if status != 200:
            raise RuntimeError(f"GET /logz ({model!r}): HTTP {status}: {payload!r}")
        records = [json.loads(line) for line in payload.splitlines()]
        if (
            not payload.endswith(b"\n")
            or len(records) < 2
            or any(not isinstance(record, dict) for record in records)
            or records[0].get("event") != "stream"
            or records[-1].get("event") != "cursor"
            or any(not isinstance(record.get("event"), str) for record in records)
        ):
            raise ValueError("/logz: incomplete observation batch")
        header, footer = records[0], records[-1]
        identity = header.get("identity")
        if (
            header.get("scope") != ("server" if model is None else "model")
            or header.get("model") != model
            or not isinstance(identity, str)
            or len(identity.split(":")) != 2
            or not all(
                part.isascii() and part.isdecimal() for part in identity.split(":")
            )
        ):
            raise ValueError("/logz: invalid stream identity or selector")
        position = 0 if previous is None or reset else int(previous.rsplit(":", 1)[1])
        if (
            previous is not None
            and not reset
            and not previous.startswith(identity + ":")
        ):
            raise ValueError("/logz: incarnation changed without a cursor rejection")
        for record in records[1:-1]:
            if record["event"] == "gap":
                end = record.get("to")
                if (
                    record.get("from") != position
                    or type(end) is not int
                    or end <= position
                ):
                    raise ValueError("/logz: invalid gap extent")
                position = end
            elif record["event"] != "counters":
                if (
                    type(record.get("sequence")) is not int
                    or record["sequence"] != position
                ):
                    raise ValueError("/logz: discontinuity without a gap record")
                position += 1
        cursor = footer.get("value")
        if footer.get("next") != position or cursor != f"{identity}:{position}":
            raise ValueError("/logz: cursor does not cover the copied records")
        stream = {"scope": header["scope"], "identity": identity}
        if model is not None:
            stream["model"] = model
        if reset:
            records.insert(0, {"event": "stream_reset", "previous_cursor": previous})
        self.cursors[model] = cursor
        return LogBatch(stream, records)
