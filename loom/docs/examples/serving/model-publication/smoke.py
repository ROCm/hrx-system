#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Verify the published tiny text model, fetch it, and check a real CPU response.

Never updates a lock. Local checks also verify catalog metadata against the
authored cards. --catalog checks an uploaded copy instead. The private server
is stopped after the request; this checks inference, not graceful shutdown.
"""

import argparse
import http.client
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loom.model import ModelCard
from loom.model.catalog import ModelCatalog, generate


def check_catalog(root):
    cards = {
        path.relative_to(root).as_posix(): ModelCard.parse(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(root.glob("model/*/model.json"))
    }
    actual = ModelCatalog.parse((root / "catalog.json").read_text(encoding="utf-8"))
    if actual.document != generate(cards, previous=actual).document:
        raise ValueError(
            "catalog.json is stale; run python refresh.py and review the diff"
        )


def check_response(address, model):
    connection = http.client.HTTPConnection(address)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "abxa"}],
                    "max_tokens": 2,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            connection.request("GET", "/v1/status")
            status_response = connection.getresponse()
            status_payload = status_response.read()
            raise RuntimeError(
                f"chat returned HTTP {response.status}: {payload!r}\n"
                f"model status (HTTP {status_response.status}): {status_payload!r}"
            )
        result = json.loads(payload)
        choice = result["choices"][0]
        if (
            choice["message"]["content"] != "bx"
            or choice["finish_reason"] != "length"
            or result["usage"]["prompt_tokens"] != 4
            or result["usage"]["completion_tokens"] != 2
        ):
            raise ValueError(f"unexpected arithmetic text result: {result}")
        return result
    finally:
        connection.close()


def serve_and_check(server, card, model, directory):
    log_path = directory / "server.log"
    with (
        log_path.open("w", encoding="utf-8") as log,
        ThreadPoolExecutor(max_workers=1) as readers,
    ):
        process = subprocess.Popen(
            [
                server,
                "--device=task",
                "--workers=2",
                "--pool_backing=fixed",
                "--listen=127.0.0.1:0",
                "--model=" + card,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        drain = None
        failure = None

        def drain_log():
            for line in process.stderr:
                log.write(line)

        try:
            for line in process.stderr:
                log.write(line)
                if line.startswith('{"event":"ready"'):
                    address = json.loads(line)["address"]
                    drain = readers.submit(drain_log)
                    result = check_response(address, model)
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"server exited during smoke: {process.returncode}"
                        )
                    return result
            raise RuntimeError(
                f"server exited before listener readiness: {process.wait()}"
            )
        except BaseException as error:
            failure = error
            raise
        finally:
            # This disposable child owns no persistent sessions or other clients.
            # terminate is SIGTERM on Unix and process termination on Windows.
            try:
                if process.poll() is None:
                    process.terminate()
            finally:
                process.wait()
                if drain is not None:
                    drain.result()
                process.stderr.close()
                log.flush()
                if failure is not None:
                    print(log_path.read_text(encoding="utf-8"), file=sys.stderr)
            if os.name != "nt" and process.returncode != 0 and failure is None:
                raise RuntimeError(
                    f"server shutdown failed: {process.returncode}\n"
                    + log_path.read_text(encoding="utf-8")
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", help="published HF catalog URL instead of this folder"
    )
    parser.add_argument(
        "--model", default="example/tiny-text", help="published model ID"
    )
    parser.add_argument(
        "--serve", default="loom-serve", help="installed loom-serve executable"
    )
    parser.add_argument(
        "--cache", type=Path, help="reuse this cache instead of a temporary one"
    )
    args = parser.parse_args()
    root = Path(__file__).absolute().parent
    if args.catalog is None:
        check_catalog(root)
    with tempfile.TemporaryDirectory(prefix="loom-publication-") as temporary:
        directory = Path(temporary)
        fetched = subprocess.run(
            [
                sys.executable,
                "-m",
                "loom.model.fetch_main",
                "--catalog=" + (args.catalog or str(root / "catalog.json")),
                "--model=" + args.model,
                "--preset=default",
                "--output=" + str(directory / "installed"),
                "--cache="
                + str(args.cache.absolute() if args.cache else directory / "cache"),
                "--materialize=copy",
                "--json",
                "--no-input",
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        card = json.loads(fetched.stdout)["card"]
        result = serve_and_check(args.serve, card, args.model, directory)
        print(json.dumps({"status": "ok", "model": args.model, "response": result}))


if __name__ == "__main__":
    main()
