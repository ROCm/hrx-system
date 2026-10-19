# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete source-tool transactions through the production chat caller."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(sys.argv.pop(1)).resolve()
CONTROL = Path(sys.argv.pop(1)).resolve()


class ToolPolicyTest(unittest.TestCase):
    def execute(self, calls, stream=True, messages=None):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tokenizer = directory / "tokenizer.json"
            tokenizer.write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "BPE",
                            "vocab": {chr(index): index for index in range(128)},
                            "merges": [],
                        }
                    }
                ),
                encoding="utf-8",
            )
            request = directory / "request.json"
            request.write_text(
                json.dumps(
                    {
                        "model": "portable-chat",
                        "messages": messages or [{"role": "user", "content": "hello"}],
                        "stream": stream,
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                            for name in ("read", "list")
                        ],
                    }
                ),
                encoding="utf-8",
            )
            response = directory / "response.txt"
            response.write_text(calls, encoding="utf-8")
            return subprocess.run(
                [
                    str(CHECKER),
                    f"--control={CONTROL}",
                    f"--tokenizer={tokenizer}",
                    f"--request={request}",
                    f"--response={response}",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )

    @staticmethod
    def call(name="read", arguments='{"path":"a"}'):
        return {
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    def test_complete_records_have_stable_ids_and_decoded_values(self):
        arguments = json.dumps({"path": 'λ\n"a"'}, ensure_ascii=False)
        calls = [self.call(arguments=arguments), self.call("list", "{}")]
        source = json.dumps(calls).replace('"read"', '"r\\u0065ad"')
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = self.execute(source, stream)
                self.assertEqual(result.returncode, 0, result.stderr)
                value = json.loads(result.stdout)
                expected = [
                    {"id": f"call_1_{index}", **call}
                    for index, call in enumerate(calls)
                ]
                if stream:
                    for index, call in enumerate(expected):
                        call["index"] = index
                self.assertEqual(value["tool_calls"], expected)
                self.assertEqual(value["content"], "")
                self.assertEqual(
                    value["checkpoint"],
                    value["prompt"] + "read: " + arguments + "\nlist: {}\n\n",
                )
                self.assertFalse(value["reusable"])

    def test_later_invalid_record_publishes_no_partial_result(self):
        invalid = (
            {"type": "function", "function": {"name": "read"}},
            {"type": "function", "function": {"arguments": "{}"}},
            {"type": "function", "function": {"name": "read", "arguments": {}}},
            {"type": "unknown", "function": {"name": "read", "arguments": "{}"}},
            {
                "type": "function",
                "function": {"name": "read", "arguments": "{}", "extra": 1},
            },
            [],
        )
        for final in invalid:
            with self.subTest(final=final):
                result = self.execute(json.dumps([self.call(), final]))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("INVALID_ARGUMENT", result.stderr)

    def test_history_and_generated_calls_share_decoded_function_fields(self):
        call = {"id": "call_1_0", **self.call(arguments="{}")}
        call["function"]["name"] = "r\\u0065ad"
        # The request writer preserves this literal backslash. A decoded name
        # remains literal; only JSON syntax escapes are decoded by the parser.
        result = self.execute(
            "[]",
            messages=[
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("r\\u0065ad: {}\n", json.loads(result.stdout)["prompt"])
        del call["function"]["name"]
        result = self.execute(
            "[]",
            messages=[
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("missing chat field 'name'", result.stderr)


if __name__ == "__main__":
    unittest.main()
