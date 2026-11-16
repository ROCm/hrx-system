# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small authenticated client for GitHub's JSON and download APIs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

API_VERSION = "2022-11-28"


class GitHubAPIError(RuntimeError):
    """Raised when scoped GitHub API access cannot proceed safely."""


def _url_origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.casefold()
    port = parsed.port
    if port is None:
        port = {"http": 80, "https": 443}.get(scheme)
    hostname = parsed.hostname.casefold() if parsed.hostname else None
    return scheme, hostname, port


class GitHubRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keeps GitHub API credentials within the request's origin."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        redirected_request = super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )
        if redirected_request is not None and _url_origin(
            request.full_url
        ) != _url_origin(new_url):
            redirected_request.remove_header("Authorization")
        return redirected_request


def _request(url: str, token: str, accept: str) -> urllib.request.Request:
    headers = {
        "Accept": accept,
        "X-GitHub-Api-Version": API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def get_bytes(
    url: str,
    token: str,
    *,
    accept: str = "application/vnd.github+json",
    timeout_seconds: int = 60,
) -> bytes:
    opener = urllib.request.build_opener(GitHubRedirectHandler())
    with opener.open(_request(url, token, accept), timeout=timeout_seconds) as response:
        return response.read()


def get_json(
    url: str,
    token: str,
    *,
    timeout_seconds: int = 60,
) -> Any:
    return json.loads(get_bytes(url, token, timeout_seconds=timeout_seconds))


class RepositoryClient:
    """Repository-scoped access to GitHub's JSON and download APIs."""

    def __init__(
        self,
        repository: str,
        token: str,
        *,
        get_json: Callable[[str, str], Any] = get_json,
        get_bytes: Callable[[str, str], bytes] = get_bytes,
    ):
        if not re.fullmatch(r"[^/]+/[^/]+", repository):
            raise GitHubAPIError(f"invalid GitHub repository {repository!r}")
        self.repository = repository
        self.token = token
        self._get_json = get_json
        self._get_bytes = get_bytes

    @property
    def api_root(self) -> str:
        return f"https://api.github.com/repos/{self.repository}"

    def _url(self, path_or_url: str) -> str:
        if path_or_url.startswith("/"):
            return self.api_root + path_or_url
        parsed = urllib.parse.urlsplit(path_or_url)
        if (
            parsed.scheme == "https"
            and parsed.hostname == "api.github.com"
            and parsed.path.startswith(f"/repos/{self.repository}/")
        ):
            return path_or_url
        raise GitHubAPIError(f"refusing unexpected GitHub API URL {path_or_url!r}")

    def json(self, path_or_url: str) -> Any:
        return self._get_json(self._url(path_or_url), self.token)

    def bytes(self, path_or_url: str) -> bytes:
        return self._get_bytes(self._url(path_or_url), self.token)

    def pages(self, path_or_url: str, field: str | None) -> list[Any]:
        url = self._url(path_or_url)
        results = []
        page = 1
        while True:
            parsed = urllib.parse.urlsplit(url)
            query = dict(urllib.parse.parse_qsl(parsed.query))
            query.update({"per_page": "100", "page": str(page)})
            page_url = urllib.parse.urlunsplit(
                parsed._replace(query=urllib.parse.urlencode(query))
            )
            payload = self._get_json(page_url, self.token)
            if field is None:
                page_results = payload
            elif isinstance(payload, dict):
                page_results = payload.get(field)
            else:
                page_results = None
            if not isinstance(page_results, list):
                raise GitHubAPIError(f"GitHub response has no {field or 'list'} page")
            results.extend(page_results)
            if len(page_results) < 100:
                return results
            page += 1


def resolve_token(environment: Mapping[str, str] = os.environ) -> str:
    """Resolves a GitHub token without placing it on a command line."""
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = environment.get(name, "").strip()
        if token:
            return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise GitHubAPIError("no GitHub token and gh is unavailable") from exc
    token = result.stdout.strip()
    if result.returncode or not token:
        raise GitHubAPIError("set GH_TOKEN/GITHUB_TOKEN or authenticate gh")
    return token
