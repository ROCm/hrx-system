# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The authoring CLI uses the production caller with two real source policies."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(sys.argv.pop(1)).resolve()
PACKAGE = Path(__file__).parent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


class RequestCheckTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        vocabulary = {chr(index): index for index in range(128)}
        vocabulary.update({"<bos>": 128, "<eos>": 129, "λ": 130})
        tokenizer = {
            "model": {"type": "BPE", "vocab": vocabulary, "merges": []},
            "added_tokens": [
                {
                    "id": index,
                    "content": spelling,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
                for spelling, index in (("<bos>", 128), ("<eos>", 129))
            ],
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<eos>", "type_id": 0}},
                ],
                "pair": [],
                "special_tokens": {
                    spelling: {"id": spelling, "ids": [index], "tokens": [spelling]}
                    for spelling, index in (("<bos>", 128), ("<eos>", 129))
                },
            },
        }
        self.tokenizer = self.directory / "tokenizer.json"
        self.tokenizer.write_text(json.dumps(tokenizer))
        self.model = self.directory / "model"
        self.model.mkdir()

    def execute(self, source, request, response=None, capacity=4096, chunk_size=0):
        (self.model / "control.loom").write_bytes(source.read_bytes())
        request_path = self.directory / "request.json"
        request_path.write_text(json.dumps(request))
        command = [
            str(CHECKER),
            f"--control={self.model / 'control.loom'}",
            f"--tokenizer={self.tokenizer}",
            f"--request={request_path}",
            f"--capacity={capacity}",
            f"--response_chunk_size={chunk_size}",
        ]
        if response is not None:
            response_path = self.directory / "response.txt"
            response_path.write_text(response)
            command.append(f"--response={response_path}")
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(
            value["tokens"],
            [
                130 if character == "λ" else ord(character)
                for character in value["prompt"]
            ],
        )
        return value

    def request(self, model="qwen3.8-27b", **options):
        return {
            "model": model,
            "stream": True,
            "enable_thinking": False,
            "messages": [{"role": "user", "content": "Hello λ."}],
            **options,
        }

    def test_production_prompt_and_plain_completion(self):
        request = self.request()
        value = self.success(self.execute(PACKAGE / "control.loom", request))
        prompt = "<|im_start|>user\nHello λ.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.assertEqual(value["model"], "qwen3.8-27b")
        self.assertEqual(value["prompt"], prompt)
        self.assertNotIn("checkpoint", value)
        value = self.success(self.execute(PACKAGE / "control.loom", request, "Hi."))
        self.assertEqual(value["checkpoint"], prompt + "Hi.<|im_end|>\n")
        self.assertEqual(value["output_cursor"], 3)
        self.assertEqual(value["tool_calls"], [])

    def test_production_xml_completion(self):
        call = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
        value = self.success(
            self.execute(
                PACKAGE / "control.loom", self.request(tools=TOOLS), "Ready.\n\n" + call
            )
        )
        self.assertEqual(value["output_cursor"], len("Ready.\n\n"))
        self.assertEqual(
            value["checkpoint"], value["prompt"] + "Ready.\n\n" + call + "<|im_end|>\n"
        )
        self.assertEqual(value["tool_calls"][0]["id"], "call_1_0")
        function = value["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "read")
        self.assertEqual(json.loads(function["arguments"]), {"path": "a.txt"})

    def test_independent_source_identity_roles_json_and_completion(self):
        response = json.dumps(
            [
                {
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"a"}'},
                }
            ]
        )
        value = self.success(
            self.execute(
                PACKAGE / "testdata/portable_chat.loom",
                self.request(
                    "portable-chat",
                    messages=[{"role": "developer", "content": "Hi."}],
                    tools=TOOLS,
                    enable_thinking=True,
                    model_options={"style": "plain"},
                ),
                response,
            )
        )
        self.assertEqual(value["model"], "portable-chat")
        self.assertEqual(value["prompt"], "developer: Hi.\nassistant: ")
        self.assertEqual(value["checkpoint"], value["prompt"] + '{"path":"a"}\n')
        self.assertEqual(value["output_cursor"], 0)
        self.assertEqual(
            value["tool_calls"][0]["function"]["arguments"], '{"path":"a"}'
        )

    def test_thinking_stream_fragments_and_retained_tool_history(self):
        call = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
        request = self.request(enable_thinking=True, tools=TOOLS)
        response = "Check λ first.\n</think>\n\nReady.\n\n" + call
        reference = self.success(
            self.execute(PACKAGE / "control.loom", request, response)
        )
        self.assertEqual(reference["reasoning_content"], "Check λ first.\n")
        self.assertEqual(reference["content"], "Ready.\n\n")
        self.assertTrue(reference["reusable"])
        self.assertIn("Reasoning effort is set to xhigh.", reference["prompt"])
        self.assertTrue(
            reference["prompt"].endswith("<|im_start|>assistant\n<think>\n")
        )
        for chunk_size in (1, 2, 3, 7, 8, 13, 64):
            with self.subTest(chunk_size=chunk_size):
                value = self.success(
                    self.execute(
                        PACKAGE / "control.loom",
                        request,
                        response,
                        chunk_size=chunk_size,
                    )
                )
                self.assertEqual(value, reference)
        following = self.request(
            enable_thinking=True,
            tools=TOOLS,
            messages=[
                *request["messages"],
                {
                    "role": "assistant",
                    "content": reference["content"],
                    "reasoning_content": reference["reasoning_content"],
                    "tool_calls": [
                        {key: value for key, value in call.items() if key != "index"}
                        for call in reference["tool_calls"]
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1_0", "content": "Found it."},
            ],
        )
        value = self.success(self.execute(PACKAGE / "control.loom", following))
        self.assertTrue(value["prompt"].startswith(reference["checkpoint"]))

    def test_truncated_thinking_cannot_reuse_closed_history(self):
        for response in (
            "still thinking",
            "unfinished </thin",
            "<tool_call> is an example",
        ):
            for chunk_size in (0, 1, 7):
                with self.subTest(response=response, chunk_size=chunk_size):
                    value = self.success(
                        self.execute(
                            PACKAGE / "control.loom",
                            self.request(enable_thinking=True),
                            response,
                            chunk_size=chunk_size,
                        )
                    )
                    self.assertEqual(value["reasoning_content"], response)
                    self.assertEqual(value["content"], "")
                    self.assertEqual(value["tool_calls"], [])
                    self.assertFalse(value["reusable"])

    def test_empty_reasoning_and_incomplete_answer_marker(self):
        for chunk_size in (0, 1, 7):
            value = self.success(
                self.execute(
                    PACKAGE / "control.loom",
                    self.request(enable_thinking=True),
                    "</think>\n\nUse <tool_ca",
                    chunk_size=chunk_size,
                )
            )
            self.assertEqual(value["reasoning_content"], "")
            self.assertEqual(value["content"], "Use <tool_ca")
            self.assertTrue(value["reusable"])

    def test_rejections_publish_no_partial_observation(self):
        source = PACKAGE / "testdata/portable_chat.loom"
        result = self.execute(source, self.request("portable-chat"), capacity=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input exceeds capacity", result.stderr)
        self.assertEqual(result.stdout, "")
        result = self.execute(source, self.request())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("select model portable-chat", result.stderr)
        self.assertEqual(result.stdout, "")
        result = self.execute(
            source, self.request("portable-chat", tools=TOOLS), "[truncated"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("INVALID_ARGUMENT", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_invalid_source_output_spans_fail_at_the_boundary(self):
        source = (PACKAGE / "testdata/portable_chat.loom").read_text()
        begin = source.index("func.def public @output_next(")
        end = source.index("func.def public @parse_tools(", begin)
        invalid = self.directory / "invalid.loom"
        for kind, start, stop, state in (
            (3, 0, 1, 1),
            (1, -1, 1, 1),
            (1, 2, 1, 1),
            (1, 0, 4, 1),
            (1, 0, 0, 2),
        ):
            with self.subTest(kind=kind, start=start, stop=stop, state=state):
                output = f"""func.def public @output_next(%response: buffer, %previous: i64, %state: i64, %phase: i32) -> (i64, i64, i64, i64) {{
  %kind = scalar.constant {kind} : i64
  %begin = scalar.constant {start} : i64
  %end = scalar.constant {stop} : i64
  %next = scalar.constant {state} : i64
  func.return %kind, %begin, %end, %next : i64, i64, i64, i64
}}

"""
                invalid.write_text(source[:begin] + output + source[end:])
                result = self.execute(invalid, self.request("portable-chat"), "abc")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid output span", result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
