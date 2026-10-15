# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete counting/recall cohorts for Qwen's non-thinking chat template.

Servers are started, configured, isolated and retired by the caller. This client
preserves every outcome; only complete, verified cohorts have a throughput rate.
Tokenization and warmup are outside the measured windows. No EOS is suppressed.
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.chat import request
from loom_serve_testing.http_harness import control

from experimental.loom_serve.models.qwen.benchmark_service import WORDS

COUNTING = "\n".join(str(number) for number in range(1, 31))


def prompt(messages):
    """Render the text-only, thinking-disabled policy in control.loom."""
    result = ""
    for message in messages:
        result += "<|im_start|>" + message["role"] + "\n"
        if message["role"] == "assistant":
            result += "<think>\n\n</think>\n\n"
        result += message["content"].strip() + "<|im_end|>\n"
    return result + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def tokenize(address, engine, model, messages):
    def post(path, body):
        return control(
            address,
            "POST",
            path,
            200,
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )

    if engine == "llama":
        formatted = post("/apply-template", {"messages": messages})["prompt"]
        if formatted != prompt(messages):
            raise ValueError("llama chat template differs from the workload template")
        body = {"content": formatted, "add_special": False, "parse_special": True}
    elif engine == "loom":
        body = {"model": model, "content": prompt(messages)}
    else:
        body = {
            "model": model,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    return post("/tokenize", body)["tokens"]


def fixture(name, index, long_lines):
    word = WORDS[index % len(WORDS)]
    if index >= len(WORDS):
        word += f"-{index // len(WORDS)}"
    background = (
        "Background: the repository contains kernels, programs, tests, and documentation.\n"
        * (long_lines if index % 2 else 0)
    )
    messages = [
        {
            "role": "system",
            "content": "Follow the user's instructions exactly. Do not use tools.",
        },
        {
            "role": "user",
            "content": (
                f"Run identifier {name}, agent {index}.\n"
                f"Remember my secret codeword {word} for later.\n{background}\n"
                "Now count from 1 through 30, one integer per line, without commentary."
            ),
        },
    ]
    return {
        "session": f"{name}-{index}",
        "word": word,
        "messages": messages,
        "followup": [
            *messages,
            {"role": "assistant", "content": COUNTING},
            {
                "role": "user",
                "content": "What is my secret codeword? Reply with only that word.",
            },
        ],
    }


def compare(result, reference):
    """Compare actual work, not cache-policy counters or timing noise."""
    failures = []
    if (result["name"], result["clients"]) != (
        reference["name"],
        reference["clients"],
    ):
        return ["reference cohort identity differs"]
    if len(result["responses"]) != len(reference["responses"]):
        return ["reference response count differs"]
    for actual, expected in zip(
        result["responses"], reference["responses"], strict=True
    ):
        for field in ("session", "messages", "followup", "token_ids"):
            if actual[field] != expected[field]:
                failures.append(f"{actual['session']}: reference {field} differs")
        if len(actual["turns"]) != len(expected["turns"]):
            failures.append(
                f"{actual['session']}: reference completed turn count differs"
            )
        for index, (turn, baseline) in enumerate(
            zip(actual["turns"], expected["turns"])
        ):
            for field in ("text", "finish_reason"):
                if turn[field] != baseline[field]:
                    failures.append(
                        f"{actual['session']} turn {index}: {field} differs"
                    )
            for field in ("prompt_tokens", "completion_tokens"):
                if turn["usage"].get(field) != baseline["usage"].get(field):
                    failures.append(
                        f"{actual['session']} turn {index}: {field} differs"
                    )
    return failures


def cohort(address, engine, model, name, count, long_lines, *, reference=None):
    fixtures = [fixture(name, index, long_lines) for index in range(count)]
    result = {
        "protocol": "qwen-count-recall-v1",
        "engine": engine,
        "model": model,
        "name": name,
        "clients": count,
        "long_lines": long_lines,
        "planned_requests": count * 2,
        "attempted_requests": 0,
        "completed_requests": 0,
        "verified_requests": 0,
        "responses": fixtures,
        "failures": [],
        "output_tokens_per_second": None,
    }
    for item in fixtures:
        item["attempted_requests"] = 0
        item["turns"] = []
        item["failures"] = []
        item["token_ids"] = []
        try:
            item["token_ids"] = [
                tokenize(address, engine, model, item[field])
                for field in ("messages", "followup")
            ]
        except Exception as error:
            item["failures"].append(f"tokenization: {type(error).__name__}: {error}")
    if any(item["failures"] for item in fixtures):
        result["failures"].append("tokenization failed; cohort was not submitted")
        return result

    gate = threading.Barrier(count + 1)

    def client(item):
        gate.wait()
        for index, (field, expected, maximum) in enumerate(
            (("messages", COUNTING, 128), ("followup", item["word"], 16))
        ):
            item["attempted_requests"] += 1
            try:
                turn = request(
                    address, item["session"], item[field], maximum, model=model
                )
            except Exception as error:
                item["failures"].append(
                    f"turn {index}: {type(error).__name__}: {error}"
                )
                break
            item["turns"].append(turn)
            failures = []
            if turn["text"].strip() != expected or turn["finish_reason"] != "stop":
                failures.append(f"turn {index}: incorrect or incomplete response")
            if turn["usage"].get("prompt_tokens") != len(item["token_ids"][index]):
                failures.append(f"turn {index}: actual prompt token count differs")
            output_count = turn["usage"].get("completion_tokens")
            if type(output_count) is not int or output_count <= 0:
                failures.append(f"turn {index}: missing or invalid completion count")
            turn["verified"] = not failures
            item["failures"].extend(failures)
            if item["failures"]:
                # A failed first task has no valid continuation in this protocol.
                break

    with ThreadPoolExecutor(max_workers=count) as workers:
        futures = [workers.submit(client, item) for item in fixtures]
        started = time.perf_counter_ns()
        gate.wait()
        for future in futures:
            future.result()
        completed = time.perf_counter_ns()
    if any(item["failures"] for item in fixtures):
        result["failures"].append("one or more clients failed")
    if reference is not None:
        result["failures"].extend(compare(result, reference))
    turns = [turn for item in fixtures for turn in item["turns"]]
    output_counts = [turn["usage"].get("completion_tokens") for turn in turns]
    outputs = (
        sum(output_counts)
        if all(type(value) is int and value >= 0 for value in output_counts)
        else None
    )
    result.update(
        started_ns=started,
        completed_ns=completed,
        elapsed_ms=(completed - started) / 1e6,
        attempted_requests=sum(item["attempted_requests"] for item in fixtures),
        completed_requests=len(turns),
        verified_requests=sum(turn["verified"] for turn in turns),
        output_tokens=outputs,
    )
    if not result["failures"]:
        result["output_tokens_per_second"] = (
            outputs * 1e9 / (completed - started) if completed > started else None
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", choices=("loom", "llama", "vllm"), required=True)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--long-lines", type=int, default=32)
    parser.add_argument(
        "--run-id", default="", help="Identical across compared engines."
    )
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Prior complete JSON result.")
    options = parser.parse_args(argv)
    address = urlsplit(options.url)
    if (
        address.scheme != "http"
        or not address.hostname
        or address.path not in ("", "/")
        or address.query
        or address.fragment
        or address.username
    ):
        parser.error("url must be an HTTP origin")
    if not options.model or options.long_lines < 0:
        parser.error("model must be nonempty and long-lines nonnegative")
    if min(options.clients) < 1 or len(set(options.clients)) != len(options.clients):
        parser.error("clients must be distinct positive counts")
    phases = ("warm", "measure") if options.warmup else ("measure",)
    names = [
        f"{options.run_id}{phase}{count}"
        for phase in phases
        for count in options.clients
    ]
    references = {}
    if options.reference:
        reference_results = json.loads(options.reference.read_text())
        references = {item["name"]: item for item in reference_results}
        if (
            len(references) != len(reference_results)
            or set(references) != set(names)
            or any(item["failures"] for item in references.values())
        ):
            parser.error("reference must contain the same complete, successful cohorts")
    results = []
    with options.output.open("x") as sink:
        for name, count in zip(names, options.clients * len(phases)):
            print(
                json.dumps({"event": "cohort_start", "name": name, "clients": count}),
                file=sys.stderr,
                flush=True,
            )
            result = cohort(
                address,
                options.engine,
                options.model,
                name,
                count,
                options.long_lines,
                reference=references.get(name),
            )
            results.append(result)
            # Publish after each complete cohort so later failures retain earlier evidence.
            sink.seek(0)
            json.dump(results, sink, indent=2)
            sink.write("\n")
            sink.truncate()
            sink.flush()
            print(
                json.dumps(
                    {
                        "event": "cohort_complete",
                        **{
                            key: value
                            for key, value in result.items()
                            if key != "responses"
                        },
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
    return 1 if any(result["failures"] for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
