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
    def execute(self, calls, stream=True, messages=None, tools=None, **options):
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
                        "tools": tools
                        if tools is not None
                        else [
                            {
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                            for name in ("read", "list")
                        ],
                        **options,
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
                    "--response_chunk_size=7",
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

    def test_argument_strings_are_opaque_to_the_protocol(self):
        calls = [
            self.call("unknown_tool", '{"q":"where?"}'),
            self.call(arguments='{"path":"one","path":"two"}'),
            self.call(arguments='{"path":'),
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = self.execute(json.dumps(calls), stream=stream)
                self.assertEqual(result.returncode, 0, result.stderr)
                value = json.loads(result.stdout)
                self.assertEqual(
                    [call["function"] for call in value["tool_calls"]],
                    [call["function"] for call in calls],
                )
                self.assertFalse(value["reusable"])

    def test_generated_calls_are_not_checked_against_request_policy(self):
        calls = [self.call("yield", '{"result":"done"}')]
        for stream in (False, True):
            for options in (
                {"tools": []},
                {"tool_choice": "none"},
                {"tool_choice": "required"},
                {"tool_choice": {"type": "function", "function": {"name": "read"}}},
            ):
                with self.subTest(stream=stream, options=options):
                    result = self.execute(json.dumps(calls), stream=stream, **options)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    value = json.loads(result.stdout)
                    retained_request = json.loads(bytes(value["request_options"]))
                    for name, expected in options.items():
                        self.assertEqual(retained_request[name], expected)
                    self.assertEqual(
                        value["tool_calls"][0]["function"], calls[0]["function"]
                    )
                    self.assertIn('yield: {"result":"done"}', value["checkpoint"])

    def test_native_admission_does_not_define_model_options(self):
        # This policy ignores model options. Native transport must not impose
        # another model's names, types, or template keyword vocabulary.
        options = {
            "temperature": {"schedule": [1, 0]},
            "top_k": None,
            "reasoning_effort": {"budget": 64},
            "model_options": ["custom"],
            "chat_template_kwargs": {"custom_option": True},
            "parallel_tool_calls": False,
            "response_format": {"type": "json_object"},
        }
        result = self.execute("[]", **options)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["prompt"], "user: hello\nassistant: ")
        retained_request = json.loads(bytes(value["request_options"]))
        for name, expected in options.items():
            self.assertEqual(retained_request[name], expected)

    def test_message_extensions_reach_source_unchanged(self):
        messages = [{"role": "user", "content": "hello", "model_metadata": {"mode": 7}}]
        result = self.execute("[]", messages=messages)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(
            json.loads(bytes(value["request_options"]))["messages"], messages
        )

    def test_harness_repairs_read_without_replaying_completed_image(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "$defs": {"path": {"type": "string"}},
                        "properties": {
                            "path": {"$ref": "#/$defs/path"},
                            "offset": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}]
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {"type": "function", "function": {"name": "generate_image"}},
        ]
        image = {
            "id": "call_previous_0",
            **self.call("generate_image", '{"prompt":"CI dashboard"}'),
        }
        for stream in (False, True):
            with self.subTest(stream=stream):
                messages = [
                    {
                        "role": "user",
                        "content": "Generate a dashboard, then read its CSS.",
                    },
                    {"role": "assistant", "content": None, "tool_calls": [image]},
                    {
                        "role": "tool",
                        "tool_call_id": image["id"],
                        "content": '{"image":"already-created.png"}',
                    },
                ]
                result = self.execute(
                    json.dumps([self.call(arguments='{"q":"styles.css"}')]),
                    stream=stream,
                    messages=messages,
                    tools=tools,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                failed_call = json.loads(result.stdout)["tool_calls"][0]
                failed_call.pop("index", None)
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [failed_call],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": failed_call["id"],
                            "content": "Invalid arguments for read: missing required property path",
                        },
                    ]
                )
                corrected = self.call(arguments='{"path":"styles.css"}')
                result = self.execute(
                    json.dumps([corrected]),
                    stream=stream,
                    messages=messages,
                    tools=tools,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                value = json.loads(result.stdout)
                self.assertEqual(len(value["tool_calls"]), 1)
                self.assertEqual(
                    value["tool_calls"][0]["function"], corrected["function"]
                )
                self.assertEqual(value["prompt"].count("generate_image:"), 1)
                self.assertIn("already-created.png", value["prompt"])
                self.assertIn('read: {"q":"styles.css"}', value["prompt"])
                self.assertIn("missing required property path", value["prompt"])

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
