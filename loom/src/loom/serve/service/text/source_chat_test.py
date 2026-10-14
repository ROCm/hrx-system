# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source chat-policy checks through the actual VM and tokenizer."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(sys.argv.pop(1)).resolve()
PACKAGE = Path(sys.argv.pop(1)).resolve().parent
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP4z8DA8J+"
    "BkYHh////DAAe9gT9Ce00PgAAAABJRU5ErkJggg=="
)


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
        self.assertEqual(value["output_cursor"], 2)
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

    def test_large_tool_catalog_renders(self):
        def tool(index):
            return {
                "type": "function",
                "function": {
                    "name": f"tool_{index}",
                    "description": f"Fixture tool {index}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }

        tools = [tool(index) for index in range(24)]
        result = self.execute(self.request(tools=tools))
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["declared_tool_count"], 24)
        self.assertEqual(value["prompt"], "#" * 24 + "abxa")
        self.assertEqual(value["tokens"], [3] * 24 + [0, 1, 2, 0])

        tools[-1]["function"]["name"] = tools[7]["function"]["name"]
        result = self.execute(self.request(tools=tools))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate tool name", result.stderr)
        self.assertEqual(result.stdout, "")

        result = self.execute(
            self.request(tools=[tool(index) for index in range(4097)])
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at most 4096 function objects", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_ordered_image_content_reaches_source_policy(self):
        image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{PNG}",
                "detail": "high",
            },
        }
        result = self.execute(
            self.request(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "a"},
                            image,
                            {"type": "text", "text": "b"},
                            image,
                        ],
                    }
                ]
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["prompt"], "axbx")
        self.assertEqual(value["tokens"], [0, 2, 1, 2])
        self.assertEqual(value["media_spans"], [[0, 1, 1], [1, 3, 1]])

        malformed = self.request(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        }
                    ],
                }
            ]
        )
        result = self.execute(malformed)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("base64 PNG, JPEG, GIF or WebP data URL", result.stderr)
        self.assertEqual(result.stdout, "")

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
            self.request(temperature=0.7),
            self.request(top_p=0.95),
            self.request(top_k=20),
            self.request(seed=1.5),
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

    def test_deterministic_sampling_options(self):
        result = self.execute(
            self.request(temperature=0, top_p=1, top_k=1, seed=9007199254740993)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["prompt"], "abxa")


if __name__ == "__main__":
    unittest.main()
