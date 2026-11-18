# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Publishes compiled programs, acquires runtime assets, and checks native output."""

import functools
import http.server
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from loom.model import ModelCard
from loom.model.release import stage_release
from model_card_test import check_model, complete


def main():
    fetch, server_tool, fixture_card, *deployments = (
        Path(value).absolute() for value in sys.argv[1:]
    )
    builds = dict(zip(("source", "archive", "mixed"), deployments, strict=True))
    lock = fixture_card.parent / "native.card.json.lock"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        publication = root / "publication"
        publication.mkdir()
        for representation, deployment in builds.items():
            directory = publication / representation
            card = deployment.parent / "model.json"
            stage_release(card, directory, asset_lock=lock)
            # The same separately published runtime assets serve every encoding.
            for name in ("bias.data", "metadata.safetensors", "tokenizer.json"):
                assert not (card.parent / name).exists(), name
                shutil.copyfile(fixture_card.parent / name, directory / name)
            selected = ModelCard.parse((directory / "model.json").read_text()).select()
            compressed = [
                file
                for file in selected.files.values()
                if file.transform and file.transform["kind"] == "xz"
            ]
            if representation != "source":
                assert compressed, deployment
            for file in compressed:
                assert (directory / file.origin).is_file()
                assert not (directory / file.path).exists()

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(publication)
        )
        installations = {}
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                for representation in builds:
                    output = root / f"installed-{representation}"
                    subprocess.run(
                        [
                            str(fetch),
                            f"{base}/{representation}/model.json",
                            "--preset=check",
                            "--output=" + str(output),
                            "--cache=" + str(root / "cache"),
                        ],
                        check=True,
                    )
                    installations[representation] = output / "model.json"
            finally:
                server.shutdown()
                thread.join()
        offline = root / "offline"
        subprocess.run(
            [
                str(fetch),
                f"{base}/archive/model.json",
                "--offline",
                "--output=" + str(offline),
                "--cache=" + str(root / "cache"),
            ],
            check=True,
        )
        installations["archive"] = offline / "model.json"
        # Acquired files are sufficient after the publisher is gone, and all
        # representations reuse identical parameter bytes in the fetch cache.
        assert (installations["source"].parent / "bias.data").samefile(
            installations["archive"].parent / "bias.data"
        )
        for representation, deployment in installations.items():
            selected = ModelCard.parse(deployment.read_text()).select()
            for file in selected.files.values():
                if file.roles & {"program", "auxiliary"}:
                    assert (deployment.parent / file.path).read_bytes() == (
                        builds[representation].parent / file.path
                    ).read_bytes(), file.path
            check_model(
                server_tool,
                deployment,
                complete([0], bias=2 + selected.config["fixture.bias_delta"]),
                image="image" in selected.service.get("components", {}),
            )


if __name__ == "__main__":
    main()
