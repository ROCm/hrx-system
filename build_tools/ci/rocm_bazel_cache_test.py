# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from build_tools.ci import rocm_bazel_cache


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class RocmBazelCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_event(self, payload):
        path = self.root / "event.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def environment(self, event_name, payload):
        return {
            "GITHUB_EVENT_NAME": event_name,
            "GITHUB_EVENT_PATH": str(self.write_event(payload)),
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_REPOSITORY": "ROCm/hrx-system",
        }

    @staticmethod
    def cache_status(url, *, timeout):
        if url != rocm_bazel_cache.CACHE_STATUS_URL or timeout != 10:
            raise AssertionError((url, timeout))
        return Response(b'{"CurrSize":123}')

    def test_main_push_can_publish(self):
        bazelrc = self.root / ".bazelrc.local"
        environment = self.environment("push", {})

        rocm_bazel_cache.configure(bazelrc, environment, open_url=self.cache_status)

        self.assertEqual(
            bazelrc.read_text(encoding="utf-8"),
            "build --config=rocm-ci-cache\n",
        )

    def test_same_repository_pull_request_can_publish(self):
        bazelrc = self.root / ".bazelrc.local"
        environment = self.environment(
            "pull_request",
            {"pull_request": {"head": {"repo": {"full_name": "ROCm/hrx-system"}}}},
        )

        rocm_bazel_cache.configure(bazelrc, environment, open_url=self.cache_status)

        self.assertNotIn(
            "remote_upload_local_results=false",
            bazelrc.read_text(encoding="utf-8"),
        )

    def test_fork_pull_request_is_read_only(self):
        bazelrc = self.root / ".bazelrc.local"
        environment = self.environment(
            "pull_request",
            {"pull_request": {"head": {"repo": {"full_name": "someone/fork"}}}},
        )

        rocm_bazel_cache.configure(bazelrc, environment, open_url=self.cache_status)

        self.assertEqual(
            bazelrc.read_text(encoding="utf-8"),
            "build --config=rocm-ci-cache\nbuild --remote_upload_local_results=false\n",
        )

    def test_non_main_push_is_read_only_and_preserves_existing_content(self):
        bazelrc = self.root / ".bazelrc.local"
        bazelrc.write_text("build --jobs=20\n", encoding="utf-8")
        environment = self.environment("push", {})
        environment["GITHUB_REF"] = "refs/heads/topic"

        rocm_bazel_cache.configure(bazelrc, environment, open_url=self.cache_status)

        self.assertEqual(
            bazelrc.read_text(encoding="utf-8"),
            "build --jobs=20\n"
            "build --config=rocm-ci-cache\n"
            "build --remote_upload_local_results=false\n",
        )

    def test_configuration_is_idempotent(self):
        bazelrc = self.root / ".bazelrc.local"
        environment = self.environment("workflow_dispatch", {})

        rocm_bazel_cache.configure(bazelrc, environment, open_url=self.cache_status)
        rocm_bazel_cache.configure(bazelrc, environment, open_url=self.cache_status)

        self.assertEqual(bazelrc.read_text(encoding="utf-8").count("rocm-ci-cache"), 1)

    def test_missing_pull_request_identity_fails_closed(self):
        environment = self.environment("pull_request", {})

        with self.assertRaisesRegex(
            rocm_bazel_cache.ConfigurationError,
            "no head repository identity",
        ):
            rocm_bazel_cache.configure(
                self.root / ".bazelrc.local",
                environment,
                open_url=self.cache_status,
            )

    def test_unavailable_cache_does_not_modify_bazelrc(self):
        bazelrc = self.root / ".bazelrc.local"
        environment = self.environment("push", {})

        def unavailable(url, *, timeout):
            raise OSError("name resolution failed")

        with self.assertRaisesRegex(
            rocm_bazel_cache.ConfigurationError,
            "cache is unavailable",
        ):
            rocm_bazel_cache.configure(bazelrc, environment, open_url=unavailable)

        self.assertFalse(bazelrc.exists())


if __name__ == "__main__":
    unittest.main()
