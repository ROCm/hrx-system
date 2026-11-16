# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest
import urllib.parse
import urllib.request

from build_tools.ci import github_api


class GitHubAPITest(unittest.TestCase):
    def test_authorization_does_not_cross_redirect_origins(self):
        handler = github_api.GitHubRedirectHandler()
        request = urllib.request.Request(
            "https://api.github.com/repos/ROCm/hrx-system/actions/jobs/1/logs",
            headers={"Authorization": "Bearer secret"},
        )

        cross_origin_request = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://results.example.com/job.log",
        )
        same_origin_request = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://API.GITHUB.COM:443/job.log",
        )

        self.assertIsNotNone(cross_origin_request)
        self.assertFalse(cross_origin_request.has_header("Authorization"))
        self.assertIsNotNone(same_origin_request)
        self.assertTrue(same_origin_request.has_header("Authorization"))

    def test_repository_client_pages_without_leaking_scope(self):
        requested_urls = []

        def get_json(url, _token):
            requested_urls.append(url)
            page = int(dict(urllib.parse.parse_qsl(url.split("?", 1)[1]))["page"])
            return {"jobs": list(range(100 if page == 1 else 1))}

        client = github_api.RepositoryClient(
            "ROCm/hrx-system",
            "secret",
            get_json=get_json,
        )

        results = client.pages("/actions/runs/1/jobs?filter=latest", "jobs")

        self.assertEqual(len(results), 101)
        self.assertIn("filter=latest", requested_urls[0])
        with self.assertRaises(github_api.GitHubAPIError):
            client.json("https://api.github.com/repos/other/project/actions/runs/1")

    def test_token_prefers_explicit_environment(self):
        self.assertEqual(
            github_api.resolve_token(
                {"GH_TOKEN": " primary ", "GITHUB_TOKEN": "other"}
            ),
            "primary",
        )


if __name__ == "__main__":
    unittest.main()
