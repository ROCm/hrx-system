# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualifies gated SDK acquisition through native source/archive/mixed execution."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from loom.model import ModelCard
from loom.model.fetch_hf import HuggingFace
from loom.model.hf_test.hub import running_hub
from loom.model.lock import FileIdentity, ModelLock
from model_card_test import check_model


def main():
    serve, fixture = (Path(value).absolute() for value in sys.argv[1:])
    card = ModelCard.parse(fixture.read_text(encoding="utf-8"))
    files = {}
    for programs in ("source", "archive", "mixed"):
        for name in card.select(preset="check", programs=programs).files:
            files[name] = (fixture.parent / name).read_bytes()
    identities = [
        FileIdentity(name, len(content), hashlib.sha256(content).hexdigest())
        for name, content in files.items()
    ]
    publication = card.document
    publication.pop("$schema", None)
    files["model.json"] = json.dumps(publication).encode()
    files["model.json.lock"] = json.dumps(ModelLock.snapshot({}, identities)).encode()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        outputs = []
        with running_hub(root, files) as hub:
            hub.credentials.write_text("fixture-first")
            provider = HuggingFace(
                endpoint=hub.endpoint,
                cache_dir=hub.cache_dir,
            )
            url = hub.endpoint + "/fixture/model/blob/main/model.json"
            common = [
                sys.executable,
                "-m",
                "loom.model.fetch_main",
                url,
                "--cache=" + str(root / "cache"),
                "--preset=check",
                "--no-input",
                "--json",
            ]
            for programs in ("source", "archive", "mixed"):
                output = root / programs
                with subprocess.Popen(
                    [
                        *common,
                        "--programs=" + programs,
                        "--output=" + str(output),
                        "--wait-for-access",
                    ],
                    env=hub.environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ) as process:
                    try:
                        if not hub.approved:
                            for line in process.stderr:
                                if "Waiting for access" in line:
                                    hub.approved = True
                                    break
                            else:
                                raise AssertionError(
                                    "CLI exited before its access wait"
                                )
                        result, error = process.communicate()
                        assert process.returncode == 0, error
                        assert json.loads(result)["card"] == str(output / "model.json")
                    finally:
                        if process.poll() is None:
                            process.kill()
                outputs.append(output)
                assert os.path.samefile(
                    output / "bias.data", provider.cached_file(hub.source("bias.data"))
                )
            assert not (root / "cache/blob").exists()
            requests = list(hub.requests)
            hub.approved = False
            output = root / "offline"
            result = subprocess.run(
                [*common, "--offline", "--output=" + str(output)],
                env=hub.environment(),
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            outputs.append(output)
            assert hub.requests == requests

        # The provider is gone: serving owns only selected ordinary local files.
        for output in outputs:
            deployment = output / "serve.json"
            deployment.write_text(
                json.dumps(
                    {
                        "name": "fixture-text",
                        "model": {"path": "model.json", "preset": "check"},
                        "definition": {"kind": "text"},
                    }
                ),
                encoding="utf-8",
            )
            check_model(serve, deployment, "abab")


if __name__ == "__main__":
    main()
