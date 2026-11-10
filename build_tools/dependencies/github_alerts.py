# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Normalizes Dependabot alert responses into stable action events."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

SCHEMA = "hrx.dependency.advisory.v0"
_SEVERITY_ORDER = {
    "unknown": 0,
    "low": 1,
    "moderate": 2,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def normalize_alert_pages(value: object) -> list[dict[str, Any]]:
    """Flattens one GitHub response or `gh api --paginate --slurp` output."""

    if not isinstance(value, list):
        raise ValueError("Dependabot alert input must be an array")
    if not value:
        return []
    pages = value if all(isinstance(item, list) for item in value) else [value]
    alerts: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, list):
            raise ValueError("Dependabot alert pages must contain arrays")
        for alert in page:
            if not isinstance(alert, dict):
                raise ValueError("Dependabot alerts must be objects")
            alerts.append(alert)
    return alerts


def build_advisory_events(
    alerts: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, object], ...]:
    """Collapses manifest instances into one package/advisory action event."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for alert in alerts:
        dependency = _mapping(alert, "dependency")
        package = _mapping(dependency, "package")
        advisory = _mapping(alert, "security_advisory")
        ecosystem = _string(package, "ecosystem")
        package_name = _string(package, "name")
        advisory_id = _string(advisory, "ghsa_id")
        grouped[(ecosystem, package_name, advisory_id)].append(alert)

    events = [
        _build_event(ecosystem, package_name, advisory_id, instances)
        for (ecosystem, package_name, advisory_id), instances in grouped.items()
    ]
    return tuple(sorted(events, key=lambda value: str(value["key"])))


def _build_event(
    ecosystem: str,
    package_name: str,
    advisory_id: str,
    alerts: list[Mapping[str, Any]],
) -> dict[str, object]:
    manifests = set()
    alert_numbers = set()
    alert_urls = set()
    version_requirements = set()
    vulnerable_ranges = set()
    fixed_versions = set()
    relationships = set()
    scopes = set()
    states = set()
    severities = set()
    summaries = set()
    updated_at = set()

    for alert in alerts:
        dependency = _mapping(alert, "dependency")
        vulnerability = _mapping(alert, "security_vulnerability")
        advisory = _mapping(alert, "security_advisory")
        manifests.add(_string(dependency, "manifest_path"))
        alert_numbers.add(_integer(alert, "number"))
        alert_urls.add(_string(alert, "html_url"))
        _add_optional_string(
            version_requirements, dependency.get("version_requirement")
        )
        _add_optional_string(
            vulnerable_ranges, vulnerability.get("vulnerable_version_range")
        )
        first_patched = vulnerability.get("first_patched_version")
        if first_patched is not None:
            if not isinstance(first_patched, dict):
                raise ValueError("first_patched_version must be an object or null")
            _add_optional_string(fixed_versions, first_patched.get("identifier"))
        _add_optional_string(relationships, dependency.get("relationship"))
        _add_optional_string(scopes, dependency.get("scope"))
        states.add(_string(alert, "state"))
        severities.add(_string(vulnerability, "severity").lower())
        _add_optional_string(summaries, advisory.get("summary"))
        _add_optional_string(updated_at, alert.get("updated_at"))

    key = f"dependabot:{ecosystem}:{package_name}:{advisory_id}"
    event: dict[str, object] = {
        "schema": SCHEMA,
        "key": key,
        "source": "github-dependabot",
        "state": "open" if "open" in states else sorted(states)[0],
        "severity": max(severities, key=_severity_rank),
        "package": {
            "ecosystem": ecosystem,
            "name": package_name,
        },
        "advisory": {
            "id": advisory_id,
            "url": f"https://github.com/advisories/{advisory_id}",
            "summaries": sorted(summaries),
        },
        "evidence": {
            "manifests": sorted(manifests),
            "alert_numbers": sorted(alert_numbers),
            "alert_urls": sorted(alert_urls),
            "version_requirements": sorted(version_requirements),
            "vulnerable_version_ranges": sorted(vulnerable_ranges),
            "relationships": sorted(relationships),
            "scopes": sorted(scopes),
            "latest_alert_update": max(updated_at) if updated_at else None,
        },
        "recommended_action": {
            "kind": (
                "upgrade_dependency"
                if fixed_versions
                else "investigate_without_fixed_version"
            ),
            "fixed_versions": sorted(fixed_versions),
        },
    }
    event["fingerprint"] = _fingerprint(event)
    return event


def _mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    member = value.get(name)
    if not isinstance(member, dict):
        raise ValueError(f"Dependabot alert {name!r} must be an object")
    return member


def _string(value: Mapping[str, Any], name: str) -> str:
    member = value.get(name)
    if not isinstance(member, str) or not member:
        raise ValueError(f"Dependabot alert {name!r} must be a nonempty string")
    return member


def _integer(value: Mapping[str, Any], name: str) -> int:
    member = value.get(name)
    if not isinstance(member, int):
        raise ValueError(f"Dependabot alert {name!r} must be an integer")
    return member


def _add_optional_string(values: set[str], value: object) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError("Dependabot optional string field has a non-string value")
    if value:
        values.add(value)


def _severity_rank(value: str) -> tuple[int, str]:
    return (_SEVERITY_ORDER.get(value, 0), value)


def _fingerprint(event: Mapping[str, object]) -> str:
    stable_event = dict(event)
    evidence = dict(stable_event["evidence"])
    # GitHub may refresh an alert timestamp without changing the action. Keep
    # the timestamp as useful evidence, but do not turn that refresh into a new
    # append-only event.
    evidence.pop("latest_alert_update", None)
    stable_event["evidence"] = evidence
    encoded = json.dumps(stable_event, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
