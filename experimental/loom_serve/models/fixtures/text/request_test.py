# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CPU-only checks use the actual source policy and public request observer."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(sys.argv.pop(1)).resolve()
PACKAGE = Path(sys.argv.pop(1)).resolve().parent


class RequestTest(unittest.TestCase):
    def execute(self, request, response=None, capacity=96):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            request_path = directory / "request.json"
            request_path.write_text(request)
            command = [
                str(CHECKER),
                f"--control={PACKAGE / 'control.loom'}",
                f"--tokenizer={PACKAGE / 'tokenizer.json'}",
                f"--request={request_path}",
                f"--capacity={capacity}",
            ]
            if response is not None:
                response_path = directory / "response.txt"
                response_path.write_text(response)
                command.append(f"--response={response_path}")
            return subprocess.run(command, capture_output=True, text=True, check=False)

    @staticmethod
    def request(messages=None, **options):
        return json.dumps(
            {
                "model": "fixture-text",
                "stream": True,
                "messages": messages or [{"role": "user", "content": "abxa"}],
                **options,
            }
        )

    def test_prompt_completion_and_assistant_delimiter(self):
        result = self.execute(self.request(), "bx")
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["model"], "fixture-text")
        self.assertEqual(value["prompt"], "abxa")
        self.assertEqual(value["tokens"], [0, 1, 2, 0])
        self.assertEqual(value["checkpoint"], "abxabx")
        self.assertEqual(value["text_end"], 2)
        self.assertEqual(value["tool_calls"], [])
        result = self.execute(
            self.request(
                [
                    {"role": "user", "content": "abxa"},
                    {"role": "assistant", "content": "bx"},
                    {"role": "user", "content": "ba"},
                ],
                enable_thinking=False,
                model_options={},
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["prompt"], "abxabx#ba")
        self.assertEqual(value["tokens"], [0, 1, 2, 0, 1, 2, 3, 1, 0])

    def test_policy_errors_publish_no_partial_success(self):
        requests = [
            self.request(enable_thinking=True),
            self.request(model_options={"unknown": True}),
            self.request([{"role": "user", "content": "?"}]),
            self.request(
                [{"role": "assistant", "content": "a", "reasoning_content": "b"}]
            ),
            self.request(
                tools=[
                    {"type": "function", "function": {"name": "read", "parameters": {}}}
                ]
            ),
            self.request(enable_thinking=True).replace(
                "enable_thinking", "enable_think\\u0069ng"
            ),
            self.request(model_options={"unknown": True}).replace(
                "model_options", "model_opt\\u0069ons"
            ),
        ]
        for request in requests:
            with self.subTest(request=request):
                result = self.execute(request)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("INVALID_ARGUMENT", result.stderr)
                self.assertEqual(result.stdout, "")
        result = self.execute(self.request(), capacity=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input exceeds capacity", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
