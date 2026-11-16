# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Enables ROCm's shared Bazel cache for a GitHub Actions job."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

CACHE_STATUS_URL = "http://bazelremote-svc.bazelremote-ns.svc.cluster.local:8080/status"
CACHE_CONFIG_LINE = "build --config=rocm-ci-cache"
READ_ONLY_CONFIG_LINE = "build --remote_upload_local_results=false"


class ConfigurationError(RuntimeError):
    """Raised when the CI event or cache configuration is invalid."""


def load_event(environment: Mapping[str, str]) -> Mapping[str, Any]:
    """Loads the GitHub event used to establish the cache write boundary."""
    event_path = environment.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise ConfigurationError("GITHUB_EVENT_PATH is required")
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(
            f"cannot read GitHub event {event_path}: {exc}"
        ) from exc
    if not isinstance(event, dict):
        raise ConfigurationError("GitHub event must be a JSON object")
    return event


def uploads_are_authorized(
    environment: Mapping[str, str], event: Mapping[str, Any]
) -> bool:
    """Returns whether this workflow may publish results to the shared cache."""
    event_name = environment.get("GITHUB_EVENT_NAME")
    if event_name == "push":
        return environment.get("GITHUB_REF") == "refs/heads/main"
    if event_name != "pull_request":
        return False

    repository = environment.get("GITHUB_REPOSITORY")
    try:
        head_repository = event["pull_request"]["head"]["repo"]["full_name"]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(
            "pull_request event has no head repository identity"
        ) from exc
    if not isinstance(repository, str) or not repository:
        raise ConfigurationError("GITHUB_REPOSITORY is required")
    if not isinstance(head_repository, str) or not head_repository:
        raise ConfigurationError("pull_request head repository is invalid")
    return head_repository == repository


def fetch_cache_status(
    open_url: Callable[..., Any] = urllib.request.urlopen,
) -> Mapping[str, Any]:
    """Checks that the runner can reach the cache before starting Bazel."""
    try:
        with open_url(CACHE_STATUS_URL, timeout=10) as response:
            status = json.load(response)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"ROCm Bazel cache is unavailable: {exc}") from exc
    if not isinstance(status, dict):
        raise ConfigurationError("ROCm Bazel cache status must be a JSON object")
    return status


def configure_bazelrc(path: Path, *, allow_upload: bool) -> None:
    """Selects the shared cache in an existing checkout-local bazelrc."""
    try:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}") from exc

    lines = set(current.splitlines())
    additions = []
    if CACHE_CONFIG_LINE not in lines:
        additions.append(CACHE_CONFIG_LINE)
    if not allow_upload and READ_ONLY_CONFIG_LINE not in lines:
        additions.append(READ_ONLY_CONFIG_LINE)
    if allow_upload and READ_ONLY_CONFIG_LINE in lines:
        raise ConfigurationError(
            f"{path} already disables remote uploads for this trusted workflow"
        )
    if not additions:
        return

    separator = "" if not current or current.endswith("\n") else "\n"
    payload = separator + "\n".join(additions) + "\n"
    try:
        with path.open("a", encoding="utf-8") as file:
            file.write(payload)
    except OSError as exc:
        raise ConfigurationError(f"cannot update {path}: {exc}") from exc


def configure(
    bazelrc_path: Path,
    environment: Mapping[str, str] = os.environ,
    *,
    open_url: Callable[..., Any] = urllib.request.urlopen,
) -> Mapping[str, Any]:
    """Verifies cache reachability and configures the Bazel client."""
    event = load_event(environment)
    allow_upload = uploads_are_authorized(environment, event)
    status = fetch_cache_status(open_url)
    configure_bazelrc(bazelrc_path, allow_upload=allow_upload)
    print(f"ROCm CI Bazel cache: {json.dumps(status, separators=(',', ':'))}")
    print(f"ROCm CI Bazel cache uploads: {'enabled' if allow_upload else 'disabled'}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enable the ROCm shared Bazel cache for this GitHub Actions job."
    )
    parser.add_argument(
        "--bazelrc",
        type=Path,
        default=Path(".bazelrc.local"),
        help="checkout-local bazelrc to update (default: .bazelrc.local)",
    )
    args = parser.parse_args()
    try:
        configure(args.bazelrc)
    except ConfigurationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
