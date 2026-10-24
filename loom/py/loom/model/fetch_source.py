# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Logical source coordinates and bounded transport reads for model acquisition."""

from __future__ import annotations

import base64
import copy
import http.client
import io
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

from .lock import FileIdentity, ModelLock
from .manifest import ManifestError, _url


class FetchError(RuntimeError):
    """An acquisition, storage, or provider requirement could not be satisfied."""


@dataclass(frozen=True)
class Document:
    # Absolute authored URL or host-local file path, independent of redirects.
    location: str
    # UTF-8 document text admitted by its owning model/lock parser.
    text: str

    @property
    def remote(self) -> bool:
        return urlsplit(self.location).scheme in ("http", "https")


@dataclass(frozen=True)
class Source:
    # Authored logical locator used for lock coverage.
    origin: str | dict
    # Effective transport descriptor after explicit root overrides.
    transport: dict
    # Immutable revision for an explicit repository file, not an aliased root.
    resolved: dict | None = None


class _HttpRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        _url(new_url, "redirect")
        if (
            urlsplit(request.full_url).scheme == "https"
            and urlsplit(new_url).scheme != "https"
        ):
            raise FetchError("HTTPS source redirected to an insecure transport")
        return super().redirect_request(
            request, response, code, message, headers, new_url
        )


@contextmanager
def open_source(source: Source, *, offline: bool, size: int | None = None):
    """Yields a stream and transport length without contacting bypassed origins."""
    transport = source.transport
    kind = transport["kind"]
    if kind == "local":
        with Path(transport["path"]).open("rb") as stream:
            yield stream, os.fstat(stream.fileno()).st_size
    elif kind == "inline":
        with io.BytesIO(base64.b64decode(transport["base64"], validate=True)) as stream:
            yield stream, stream.getbuffer().nbytes
    elif kind == "url":
        if offline:
            raise FetchError(
                "--offline requires a cached file or local origin override"
            )
        request = urllib.request.Request(
            transport["url"],
            headers={"User-Agent": "loom-fetch", "Accept-Encoding": "identity"},
        )
        opener = urllib.request.build_opener(_HttpRedirects())
        try:
            with opener.open(request, timeout=60) as response:
                if response.status != 200:
                    raise FetchError(
                        f"whole-file request requires HTTP 200, got {response.status}"
                    )
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise FetchError("source did not honor identity content encoding")
                lengths = response.headers.get_all("Content-Length", [])
                if len(lengths) > 1:
                    raise FetchError("source returned multiple Content-Length headers")
                length = lengths[0] if lengths else None
                encoding = response.headers.get("Transfer-Encoding")
                if encoding is not None and (encoding.lower() != "chunked" or length):
                    raise FetchError("source returned ambiguous HTTP message framing")
                if length is not None:
                    if not re.fullmatch(r"[0-9]+", length):
                        raise FetchError("source returned an invalid Content-Length")
                    if size is not None and int(length) != size:
                        raise FetchError(
                            f"source length {length} does not match locked size {size}"
                        )
                yield response, int(length) if length is not None else None
        except urllib.error.HTTPError as error:
            # Signed redirect targets and credentials are not diagnostics or
            # persistent source identities.
            raise FetchError(
                f"source returned HTTP {error.code}: {error.reason}"
            ) from error
        except urllib.error.URLError as error:
            raise FetchError(f"source request failed: {error.reason}") from error
        except http.client.HTTPException as error:
            raise FetchError(f"source HTTP transfer failed: {error}") from error
    elif kind == "hf":
        raise FetchError(
            "Hugging Face transport requires the optional provider; "
            "use a local --origin override"
        )
    else:
        raise AssertionError("source resolver produced an unknown transport")


def document_location(value: str) -> str:
    if "://" in value:
        _url(value, "document")
        return value
    return str(Path(value).absolute())


def document_source(location: str) -> Source:
    if urlsplit(location).scheme in ("http", "https"):
        return Source(location, {"kind": "url", "url": location})
    return Source(location, {"kind": "local", "path": location})


def sibling_lock(location: str) -> str:
    parsed = urlsplit(location)
    if parsed.scheme in ("http", "https"):
        return parsed._replace(path=parsed.path + ".lock").geturl()
    return location + ".lock"


class SourceResolver:
    """Resolves declared origins, keeping lock and mirror coordinates separate."""

    def __init__(
        self,
        document: Document,
        roots: dict,
        overrides: dict[str, str],
        lock: ModelLock | None,
        *,
        pin_revisions: bool,
        revision_resolver: Callable[[dict], str] | None = None,
    ):
        self.document = document
        self.roots = roots
        unknown = overrides.keys() - self.roots.keys()
        if unknown:
            raise ManifestError(
                "unknown origin overrides: " + ", ".join(sorted(unknown))
            )
        self.overrides = {}
        for name, value in overrides.items():
            if "://" in value:
                _url(value, "override." + name)
                if urlsplit(value).query:
                    raise ManifestError(
                        "origin override directories cannot contain a query"
                    )
                self.overrides[name] = {"kind": "url", "url": value}
            else:
                self.overrides[name] = {
                    "kind": "local",
                    "path": str(Path(value).absolute()),
                }
        self.lock = lock
        self.pin_revisions = pin_revisions
        # Optional provider callback, invoked only for a non-overridden floating root.
        self.revision_resolver = revision_resolver
        # Only selected roots are resolved or retained in an update snapshot.
        self.origins: dict[str, dict] = {}

    def _revision(
        self, requested: dict, locked: dict | None, *, overridden: bool
    ) -> str:
        authored = requested.get("revision", "main")
        if re.fullmatch(r"[0-9a-f]{40}", authored):
            if locked is not None and locked["revision"] != authored:
                raise ManifestError(
                    "lock revision does not match the explicitly pinned origin"
                )
            return authored
        if locked is not None:
            return locked["revision"]
        if overridden:
            raise FetchError(
                "cannot resolve a floating Hugging Face revision through a plain "
                "mirror; pin the card revision before --update-lock or use a "
                "matching lock"
            )
        if self.revision_resolver is not None:
            return self.revision_resolver(requested)
        raise FetchError(
            "resolving a Hugging Face revision requires the optional provider"
        )

    def _file(self, descriptor: dict, *, caller_owned: bool) -> dict:
        result = copy.deepcopy(descriptor)
        if result["kind"] == "local":
            if self.document.remote and not caller_owned:
                raise FetchError(
                    "a remote model card cannot access host-local files; "
                    "use an explicit --origin override"
                )
            path = Path(result["path"])
            if not path.is_absolute():
                path = Path(self.document.location).parent / path
            result["path"] = str(path.absolute())
        return result

    def _root_file(self, root: dict, path: str, *, caller_owned: bool) -> dict:
        kind = root["kind"]
        if kind == "url":
            return {
                "kind": "url",
                "url": urljoin(root["url"].rstrip("/") + "/", quote(path, safe="/")),
            }
        if kind == "local":
            return self._file(
                {"kind": "local", "path": str(Path(root["path"]) / path)},
                caller_owned=caller_owned,
            )
        return {**root, "path": path}

    def resolve(self, origin: str | dict, identity: FileIdentity | None) -> Source:
        if isinstance(origin, str) and origin.startswith("@"):
            name, path = origin[1:].split("/", 1)
            requested = self.roots[name]
            overridden = name in self.overrides
            if name not in self.origins:
                record = {"requested": copy.deepcopy(requested)}
                if requested["kind"] == "hf" and (not overridden or self.pin_revisions):
                    locked = (
                        self.lock.origins[name].get("resolved") if self.lock else None
                    )
                    record["resolved"] = {
                        "revision": self._revision(
                            requested, locked, overridden=overridden
                        )
                    }
                self.origins[name] = record
            effective = copy.deepcopy(self.overrides.get(name, requested))
            if effective["kind"] == "hf":
                effective["revision"] = self.origins[name]["resolved"]["revision"]
            return Source(
                origin, self._root_file(effective, path, caller_owned=overridden)
            )
        if isinstance(origin, str):
            if "://" in origin:
                descriptor = {"kind": "url", "url": origin}
            elif self.document.remote:
                descriptor = {
                    "kind": "url",
                    "url": urljoin(self.document.location, quote(origin, safe="/")),
                }
            else:
                descriptor = {
                    "kind": "local",
                    "path": str(Path(self.document.location).parent / origin),
                }
        else:
            descriptor = origin
        resolved = None
        if descriptor["kind"] == "hf":
            resolved = {
                "revision": self._revision(
                    descriptor,
                    identity.resolved if identity else None,
                    overridden=False,
                )
            }
            descriptor = {**descriptor, **resolved}
        return Source(origin, self._file(descriptor, caller_owned=False), resolved)
