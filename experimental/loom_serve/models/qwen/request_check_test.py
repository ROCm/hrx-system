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

    def execute(self, source, request, response=None, capacity=4096):
        (self.model / "control.loom").write_bytes(source.read_bytes())
        request_path = self.directory / "request.json"
        request_path.write_text(json.dumps(request))
        command = [
            str(CHECKER),
            f"--model={self.model}",
            f"--tokenizer={self.tokenizer}",
            f"--request={request_path}",
            f"--capacity={capacity}",
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
        self.assertEqual(value["text_end"], 3)
        self.assertEqual(value["tool_calls"], [])

    def test_production_xml_completion(self):
        call = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
        value = self.success(
            self.execute(
                PACKAGE / "control.loom", self.request(tools=TOOLS), "Ready.\n\n" + call
            )
        )
        self.assertEqual(value["text_end"], len("Ready.\n\n"))
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
        self.assertEqual(value["text_end"], 0)
        self.assertEqual(
            value["tool_calls"][0]["function"]["arguments"], '{"path":"a"}'
        )

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


if __name__ == "__main__":
    unittest.main()
