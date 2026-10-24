# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fetches only one partition, then serves its independent weighted response."""

import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from loom.model import ModelCard
from model_card_test import check_model


def main(fetch, serve, fixture, *built):
    fetch, serve, fixture = (
        Path(value).absolute() for value in (fetch, serve, fixture)
    )
    card = ModelCard.parse(fixture.read_text())
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        publication = root / "publication"
        publication.mkdir()
        for choice in card.choices():
            for file in card.select(**choice).files.values():
                shutil.copyfile(fixture.parent / file.origin, publication / file.origin)
        (publication / "model.json").write_text(json.dumps(card.document))
        requests = []

        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                super().do_GET()

            def log_message(self, *_):
                pass

        handler = functools.partial(Handler, directory=str(publication))
        outputs = []
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/model.json"

                def run(*arguments):
                    result = subprocess.run(
                        [str(fetch), url, "--json", *arguments],
                        capture_output=True,
                        text=True,
                    )
                    assert result.returncode == 0, result.stderr
                    return json.loads(result.stdout)

                snapshot = run(
                    "--update-lock",
                    "--all-variants",
                    "--lockfile=" + str(publication / "model.json.lock"),
                    "--cache=" + str(root / "publisher-cache"),
                )
                assert len(snapshot["selections"]) == 2
                for index, (partition, expected, excluded) in enumerate(
                    (
                        ("node-0", "abab", "/alternate.data"),
                        ("other side", "xbbx", "/bias.data"),
                    )
                ):
                    requests.clear()
                    output = root / str(index)
                    result = run(
                        "--preset=check",
                        "--partition=" + partition,
                        "--output=" + str(output),
                        "--cache=" + str(root / "client-cache"),
                    )
                    assert result["partition"] == partition
                    assert excluded not in requests
                    installed = ModelCard.parse(
                        (output / "model.json").read_text()
                    ).select()
                    assert installed.partition == partition
                    deployment = output / "serve.json"
                    deployment.write_text(
                        json.dumps(
                            {
                                "name": "fixture-text",
                                "model": {
                                    "path": "model.json",
                                    "preset": "check",
                                    "partition": partition,
                                },
                                "definition": {"kind": "text"},
                            }
                        )
                    )
                    outputs.append((output, deployment, expected))
            finally:
                server.shutdown()
                thread.join()
        assert os.path.samefile(
            outputs[0][0] / "control.loom", outputs[1][0] / "control.loom"
        )
        assert not os.path.samefile(
            outputs[0][0] / "bias.data", outputs[1][0] / "bias.data"
        )
        for _, deployment, expected in outputs:
            check_model(serve, deployment, expected)
        for deployment in built:
            check_model(serve, Path(deployment).absolute(), "xbbx")


if __name__ == "__main__":
    main(*sys.argv[1:])
