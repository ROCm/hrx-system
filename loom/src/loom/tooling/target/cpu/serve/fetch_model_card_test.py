# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fetches real source/bytecode packages, then checks native weighted execution."""

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
from loom.model.lock import FileIdentity, ModelLock, file_digest
from model_card_test import check_model


def main():
    fetch, serve, fixture_card = (Path(value).absolute() for value in sys.argv[1:])
    card = ModelCard.parse(fixture_card.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        publication = root / "publication"
        publication.mkdir()
        files = {}
        for programs in ("source", "archive", "mixed"):
            files.update(card.select(preset="check", programs=programs).files)
        identities = []
        for path in files:
            source = fixture_card.parent / path
            shutil.copyfile(source, publication / path)
            size, digest = file_digest(source)
            identities.append(FileIdentity(path, size, digest))
        document = card.document
        document.pop("$schema", None)
        (publication / "model.json").write_text(json.dumps(document), encoding="utf-8")
        (publication / "model.json.lock").write_text(
            json.dumps(ModelLock.snapshot({}, identities)), encoding="utf-8"
        )

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
                common = [
                    str(fetch),
                    url,
                    "--cache=" + str(root / "cache"),
                    "--preset=check",
                    "--json",
                ]
                listed = subprocess.run(
                    [*common, "--list"], check=True, capture_output=True, text=True
                )
                assert set(json.loads(listed.stdout)["programs"]) == {
                    "source",
                    "archive",
                    "mixed",
                }
                assert requests == ["/model.json"]
                dry_run = subprocess.run(
                    [
                        *common,
                        "--dry-run",
                        "--programs=source",
                        "--output=" + str(root / "preview"),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                assert len(json.loads(dry_run.stdout)["files"]) == len(
                    card.select(preset="check", programs="source").files
                )
                assert json.loads(dry_run.stdout)["memory"]["weights_bytes"] == 4
                assert not (root / "preview").exists()
                assert requests == ["/model.json", "/model.json", "/model.json.lock"]
                for programs in ("source", "archive", "mixed"):
                    output = root / programs
                    result = subprocess.run(
                        [*common, "--programs=" + programs, "--output=" + str(output)],
                        capture_output=True,
                        text=True,
                    )
                    assert result.returncode == 0, result.stderr
                    summary = json.loads(result.stdout)
                    assert summary["card"] == str(output / "model.json")
                    installed = ModelCard.parse(
                        (output / "model.json").read_text()
                    ).select("check")
                    if programs == "source":
                        assert summary["memory"] == installed.settings["memory"]
                        assert summary["memory"]["weights_bytes"] == 4
                    else:
                        assert "memory" not in summary
                        assert "memory" not in installed.settings
                    outputs.append(output)
                assert os.path.samefile(
                    outputs[0] / "bias.data", outputs[1] / "bias.data"
                )
                assert os.path.samefile(
                    outputs[1] / "bias.data", outputs[2] / "bias.data"
                )

                # Corrupt an executable after publishing its lock, and force a
                # fresh acquisition so a valid cached copy cannot satisfy it.
                program = publication / "prepare.loom"
                original = program.read_bytes()
                program.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                failed = subprocess.run(
                    [
                        str(fetch),
                        url,
                        "--preset=check",
                        "--programs=source",
                        "--cache=" + str(root / "tampered-cache"),
                        "--output=" + str(root / "tampered"),
                    ],
                    capture_output=True,
                    text=True,
                )
                assert failed.returncode == 1, failed.stdout
                assert "SHA-256" in failed.stderr, failed.stderr
                assert not (root / "tampered/model.json").exists()
                assert not (root / "tampered/prepare.loom").exists()
            finally:
                server.shutdown()
                thread.join()

        # The HTTP origin is gone. Fetch must use its verified local cache, and
        # native serving must use only the installation's ordinary local files.
        offline = root / "offline"
        result = subprocess.run(
            [*common, "--programs=source", "--offline", "--output=" + str(offline)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(offline)
        for output in outputs:
            check_model(serve, output / "model.json", "abab")


if __name__ == "__main__":
    main()
