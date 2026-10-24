# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Publishes compiled programs, acquires runtime assets, and checks native output."""

import functools
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from loom.model import ModelCard
from loom.model.lock import FileIdentity, ModelLock, file_digest
from model_card_test import check_model, complete


def main():
    fetch, server_tool, fixture_card, *deployments = (
        Path(value).absolute() for value in sys.argv[1:]
    )
    runtime_files = {
        "bias.data": fixture_card.parent / "bias.data",
        "metadata.safetensors": fixture_card.parent / "metadata.safetensors",
        "tokenizer.json": fixture_card.parent / "tokenizer.json",
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        publication = root / "publication"
        publication.mkdir()
        for index, deployment in enumerate(deployments):
            directory = publication / str(index)
            directory.mkdir()
            card = deployment.parent / "model.json"
            selected = ModelCard.parse(card.read_text()).select("check")
            identities = []
            for path, file in selected.files.items():
                if path in runtime_files:
                    # Runtime assets are deliberately absent from the build package.
                    assert not (card.parent / path).exists(), path
                    source = runtime_files[path]
                else:
                    source = card.parent / path
                destination = directory / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                size, digest = file_digest(source)
                identities.append(FileIdentity(file.origin, size, digest))
            shutil.copyfile(card, directory / "model.json")
            (directory / "model.json.lock").write_text(
                json.dumps(ModelLock.snapshot({}, identities))
            )

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(publication)
        )
        installations = []
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                for index, deployment in enumerate(deployments):
                    output = root / f"installed-{index}"
                    subprocess.run(
                        [
                            str(fetch),
                            f"http://127.0.0.1:{server.server_port}/{index}/model.json",
                            "--preset=check",
                            "--output=" + str(output),
                            "--cache=" + str(root / "cache"),
                        ],
                        check=True,
                    )
                    installations.append(output / "model.json")
            finally:
                server.shutdown()
                thread.join()
        # Acquired files are sufficient after the publisher is gone, and all
        # representations reuse identical parameter bytes in the fetch cache.
        assert (installations[0].parent / "bias.data").samefile(
            installations[1].parent / "bias.data"
        )
        for deployment in installations:
            selected = ModelCard.parse(deployment.read_text()).select()
            check_model(
                server_tool,
                deployment,
                complete([0], bias=2 + selected.config["fixture.bias_delta"]),
                image="image" in selected.service.get("components", {}),
            )


if __name__ == "__main__":
    main()
