# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs the copied author's scripts, including an HF publication-to-CPU check."""

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from loom.model.hf_test.hub import running_hub


def main():
    server, archive = (Path(value).absolute() for value in sys.argv[1:])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        # Exercise the downloadable website artifact, not a checkout with
        # private fixtures, read-only action inputs, or generated build files.
        with zipfile.ZipFile(archive) as bundle:
            assert not any("BUILD" in name for name in bundle.namelist())
            bundle.extractall(root)
        publication = root / "model-publication"
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(sys.path),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        def run(script, *arguments, succeeds=True, env=environment):
            result = subprocess.run(
                [sys.executable, str(publication / script), *arguments],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            assert (result.returncode == 0) == succeeds, result.stdout + result.stderr
            return result

        card = publication / "model/tiny-text/model.json"
        lock = card.with_suffix(".json.lock")
        catalog = publication / "catalog.json"
        original_lock = lock.read_bytes()
        original_catalog = catalog.read_bytes()
        run("refresh.py", "--cache=" + str(root / "author-cache"))
        assert lock.read_bytes() == original_lock
        assert catalog.read_bytes() == original_catalog
        catalog.write_text(
            json.dumps(json.loads(original_catalog) | {"x-owner": "publisher"}),
            encoding="utf-8",
        )
        original_catalog = catalog.read_bytes()

        def contents():
            return {
                path.relative_to(publication).as_posix(): path.read_bytes()
                for path in publication.rglob("*")
                if path.is_file() and path.name != ".loom-fetch.lock"
            }

        authored = contents()
        # Exercise the same standalone script used by authors, not a parallel
        # test implementation of its server launch or result assertions.
        result = run("smoke.py", "--serve=" + str(server))
        assert json.loads(result.stdout)["status"] == "ok"
        assert contents() == authored

        tokenizer = publication / "model/tiny-text/tokenizer.json"
        original_tokenizer = tokenizer.read_bytes()
        tokenizer.write_bytes(original_tokenizer + b"\n")
        result = run("smoke.py", "--serve=" + str(server), succeeds=False)
        assert "mismatch" in result.stderr, result.stderr
        assert lock.read_bytes() == original_lock
        tokenizer.write_bytes(original_tokenizer)

        changed = json.loads(original_catalog)
        changed["models"]["example/tiny-text"]["name"] = "stale name"
        catalog.write_text(json.dumps(changed), encoding="utf-8")
        result = run("smoke.py", "--serve=" + str(server), succeeds=False)
        assert "catalog.json is stale" in result.stderr
        assert (
            json.loads(catalog.read_text())["models"]["example/tiny-text"]["name"]
            == "stale name"
        )
        catalog.write_bytes(original_catalog)

        # The controlled HTTP dependency implements the Hub transport. Fetch,
        # its optional SDK, the copied scripts, JIT, and inference remain real.
        with running_hub(root, authored) as hub:
            hub.approved = True
            hub.credentials.write_text("fixture-first", encoding="utf-8")
            result = run(
                "smoke.py",
                "--serve=" + str(server),
                "--catalog="
                + hub.endpoint
                + "/fixture/model/resolve/main/catalog.json",
                env=hub.environment(),
            )
            assert json.loads(result.stdout)["status"] == "ok"
            requested = [path for method, path in hub.requests if method == "GET"]
            assert any(
                "/" + hub.revision + "/model/tiny-text/program/advance.loom" in path
                for path in requested
            ), requested
        assert contents() == authored


if __name__ == "__main__":
    main()
