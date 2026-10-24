# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict byte-range reads for streaming parameter materialization."""

from __future__ import annotations

import base64
import http.client
import io
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .fetch_source import FetchError, Source, _HttpRedirects
from .manifest import _url

CHUNK_SIZE = 1024 * 1024


def _admit_range(status, fields, offset, length, size):
    if status != 206:
        raise FetchError(
            f"byte ranges require HTTP 206, got {status}; "
            "use a range-capable origin or a compact local --origin override"
        )
    ranges = fields("Content-Range")
    match = (
        re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", ranges[0])
        if len(ranges) == 1
        else None
    )
    if match is None:
        raise FetchError("range response requires one exact Content-Range")
    first, last, total = map(int, match.groups())
    if (first, last) != (offset, offset + length - 1) or not last < total < 2**64:
        raise FetchError("Content-Range does not match the requested interval")
    if size is not None and total != size:
        raise FetchError("range source changed its total byte count")
    if fields("Content-Encoding") not in ([], ["identity"]):
        raise FetchError("range source did not honor identity content encoding")
    lengths = fields("Content-Length")
    if lengths and (
        len(lengths) != 1
        or not re.fullmatch(r"[0-9]+", lengths[0])
        or int(lengths[0]) != length
    ):
        raise FetchError("range Content-Length does not match the requested interval")
    encoding = fields("Transfer-Encoding")
    if encoding and (encoding != ["chunked"] or lengths):
        raise FetchError("range source returned ambiguous HTTP message framing")
    return total


class RangeReader:
    """One source lifetime; a range failure never falls back to a whole transfer."""

    def __init__(self, source: Source, *, endpoint=None, token=False):
        # Effective transport after overrides and immutable repository resolution.
        self.source = source
        # Optional provider endpoint, used only by an admitted HF worker.
        self.endpoint = endpoint
        # Explicit credential passed by the parent's access coordinator.
        self.token = token
        # Parent size established by the first response or local file descriptor.
        self.size = None
        # Strong HTTP validator retained across a multi-range acquisition.
        self.etag = None
        # Local handle retained across reads, without copying the original file.
        self.stream = None

    def __enter__(self):
        if self.source.transport["kind"] == "local":
            self.stream = Path(self.source.transport["path"]).open("rb")
            self.size = os.fstat(self.stream.fileno()).st_size
        elif self.source.transport["kind"] == "inline":
            self.stream = io.BytesIO(
                base64.b64decode(self.source.transport["base64"], validate=True)
            )
            self.size = self.stream.getbuffer().nbytes
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.close()

    def _admit(self, status, fields, offset, length):
        self.size = _admit_range(status, fields, offset, length, self.size)
        validators = fields("ETag")
        if len(validators) > 1:
            raise FetchError("range source returned multiple ETags")
        if validators and not validators[0].startswith("W/"):
            if self.etag is not None and validators[0] != self.etag:
                raise FetchError("range source changed its ETag")
            self.etag = validators[0]

    def _http_chunks(self, offset, length, headers):
        request = urllib.request.Request(self.source.transport["url"], headers=headers)
        try:
            with urllib.request.build_opener(_HttpRedirects()).open(
                request, timeout=60
            ) as response:
                self._admit(
                    response.status,
                    lambda name: response.headers.get_all(name, []),
                    offset,
                    length,
                )
                remaining = length
                while True:
                    chunk = response.read(min(CHUNK_SIZE, remaining + 1))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise FetchError("range source exceeded the requested interval")
                    yield chunk
                if remaining:
                    raise FetchError("range transfer ended before its declared length")
        except urllib.error.HTTPError as error:
            raise FetchError(f"range source returned HTTP {error.code}") from None
        except (urllib.error.URLError, http.client.HTTPException):
            raise FetchError("range HTTP transfer failed") from None

    def _hf_chunks(self, offset, length, headers):
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import (
            build_hf_headers,
            get_session,
            hf_raise_for_status,
        )

        transport = self.source.transport
        location = hf_hub_url(
            transport["repository"],
            transport["path"],
            revision=transport["revision"],
            endpoint=self.endpoint,
        )
        headers = {
            **build_hf_headers(token=self.token, library_name="loom-fetch"),
            **headers,
        }
        client = get_session()
        for _ in range(20):
            with client.stream(
                "GET", location, headers=headers, follow_redirects=False
            ) as response:
                if response.has_redirect_location:
                    target = urljoin(location, response.headers["Location"])
                    _url(target, "Hugging Face range redirect")
                    previous, following = urlsplit(location), urlsplit(target)
                    if previous.scheme == "https" and following.scheme != "https":
                        raise FetchError(
                            "Hugging Face range redirected to insecure transport"
                        )
                    if (previous.scheme, previous.netloc) != (
                        following.scheme,
                        following.netloc,
                    ):
                        headers = {
                            name: value
                            for name, value in headers.items()
                            if name.lower() != "authorization"
                        }
                    location = target
                    continue
                hf_raise_for_status(response)
                self._admit(
                    response.status_code, response.headers.get_list, offset, length
                )
                remaining = length
                for chunk in response.iter_raw(chunk_size=CHUNK_SIZE):
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise FetchError(
                            "Hugging Face range exceeded the requested interval"
                        )
                    yield chunk
                if remaining:
                    raise FetchError(
                        "Hugging Face range ended before its declared length"
                    )
                return
        raise FetchError("too many Hugging Face range redirects")

    def chunks(self, offset, length):
        if not length:
            return
        if self.size is not None and offset + length > self.size:
            raise FetchError("requested interval exceeds its source")
        if self.stream is not None:
            self.stream.seek(offset)
            remaining = length
            while remaining:
                chunk = self.stream.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    raise FetchError("local range source was truncated")
                remaining -= len(chunk)
                yield chunk
            return
        headers = {
            "User-Agent": "loom-fetch",
            "Accept-Encoding": "identity",
            "Range": f"bytes={offset}-{offset + length - 1}",
        }
        if self.etag is not None:
            headers["If-Match"] = self.etag
        if self.source.transport["kind"] == "hf":
            yield from self._hf_chunks(offset, length, headers)
        else:
            yield from self._http_chunks(offset, length, headers)
