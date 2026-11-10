# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for Dependabot alert normalization."""

import unittest

from build_tools.dependencies.github_alerts import (
    build_advisory_events,
    normalize_alert_pages,
)


def _alert(
    number: int,
    manifest: str,
    *,
    fixed_version: str | None = "2.0",
) -> dict[str, object]:
    return {
        "number": number,
        "state": "open",
        "html_url": f"https://github.com/example/project/security/dependabot/{number}",
        "updated_at": f"2026-11-{number:02d}T00:00:00Z",
        "dependency": {
            "package": {"ecosystem": "pip", "name": "demo"},
            "manifest_path": manifest,
            "version_requirement": "= 1.0",
            "scope": "runtime",
            "relationship": "indirect",
        },
        "security_advisory": {
            "ghsa_id": "GHSA-demo-0000-0000",
            "severity": "high",
            "summary": "Demo advisory",
        },
        "security_vulnerability": {
            "package": {"ecosystem": "pip", "name": "demo"},
            "severity": "high",
            "vulnerable_version_range": ">= 1.0, < 2.0",
            "first_patched_version": (
                {"identifier": fixed_version} if fixed_version else None
            ),
        },
    }


class GithubAlertsTest(unittest.TestCase):
    def test_flattens_paginated_responses_and_collapses_manifest_instances(self):
        alerts = normalize_alert_pages(
            [
                [_alert(1, "requirements-analysis.lock.txt")],
                [_alert(2, "loom/docs/requirements.lock.txt")],
            ]
        )

        events = build_advisory_events(alerts)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["key"], "dependabot:pip:demo:GHSA-demo-0000-0000")
        self.assertEqual(event["severity"], "high")
        self.assertEqual(
            event["evidence"]["manifests"],
            [
                "loom/docs/requirements.lock.txt",
                "requirements-analysis.lock.txt",
            ],
        )
        self.assertEqual(event["evidence"]["alert_numbers"], [1, 2])
        self.assertEqual(
            event["evidence"]["latest_alert_update"],
            "2026-11-02T00:00:00Z",
        )
        self.assertEqual(
            event["recommended_action"],
            {"kind": "upgrade_dependency", "fixed_versions": ["2.0"]},
        )
        self.assertRegex(event["fingerprint"], r"^[0-9a-f]{64}$")

    def test_reports_advisories_without_a_fixed_version(self):
        event = build_advisory_events(
            [_alert(1, "requirements.lock.txt", fixed_version=None)]
        )[0]

        self.assertEqual(
            event["recommended_action"],
            {
                "kind": "investigate_without_fixed_version",
                "fixed_versions": [],
            },
        )

    def test_fingerprint_ignores_timestamp_only_refresh(self):
        first = _alert(1, "requirements.lock.txt")
        second = _alert(1, "requirements.lock.txt")
        second["updated_at"] = "2026-12-31T23:59:59Z"

        first_event = build_advisory_events([first])[0]
        second_event = build_advisory_events([second])[0]

        self.assertNotEqual(
            first_event["evidence"]["latest_alert_update"],
            second_event["evidence"]["latest_alert_update"],
        )
        self.assertEqual(first_event["fingerprint"], second_event["fingerprint"])

    def test_rejects_missing_advisory_identity(self):
        alert = _alert(1, "requirements.lock.txt")
        alert["security_advisory"]["ghsa_id"] = None

        with self.assertRaisesRegex(ValueError, "ghsa_id"):
            build_advisory_events([alert])


if __name__ == "__main__":
    unittest.main()
