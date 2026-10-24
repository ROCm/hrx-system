# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Acquires parameter ranges into normal archives and serves their locked result."""

import functools
import http.server
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from loom.model import ModelCard
from model_card_test import check_model


def archive(excluded):
    metadata = {
        "bias": {"dtype": "I32", "shape": [1], "data_offsets": [excluded, excluded + 4]}
    }
    if excluded:
        metadata["excluded"] = {
            "dtype": "U8",
            "shape": [excluded],
            "data_offsets": [0, excluded],
        }
    header = json.dumps(metadata, separators=(",", ":")).encode()
    return (
        struct.pack("<Q", len(header)) + header + b"x" * excluded + struct.pack("<i", 2)
    )


def main(fetch, serve, fixture):
    fetch, serve, fixture = (
        Path(value).absolute() for value in (fetch, serve, fixture)
    )
    card = ModelCard.parse(fixture.read_text())
    source = archive(4 * 1024 * 1024)
    header_length = struct.unpack("<Q", source[:8])[0]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        publication = root / "publication"
        publication.mkdir()
        for file in card.select().files.values():
            if file.transform is None:
                shutil.copyfile(fixture.parent / file.path, publication / file.path)
        (publication / "model.json").write_text(json.dumps(card.document))
        requests = []

        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/source.safetensors":
                    super().do_GET()
                    return
                requested = self.headers.get("Range")
                assert requested is not None, "whole parent download attempted"
                first, last = map(
                    int, re.fullmatch(r"bytes=(\d+)-(\d+)", requested).groups()
                )
                requests.append((first, last - first + 1))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(source)}")
                self.send_header("Content-Length", str(last - first + 1))
                self.end_headers()
                self.wfile.write(source[first : last + 1])

            def log_message(self, *_):
                pass

        handler = functools.partial(Handler, directory=str(publication))
        deployments = []
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            url = base + "/model.json"

            def run(*arguments):
                result = subprocess.run(
                    [str(fetch), url, "--json", *arguments],
                    capture_output=True,
                    text=True,
                )
                assert result.returncode == 0, result.stderr
                return json.loads(result.stdout)

            def install(name, variant="swatch", origin=base, *extra):
                output = root / name
                run(
                    "--preset=check",
                    "--variant=" + variant,
                    "--origin=checkpoint=" + origin,
                    "--cache=" + str(root / "client-cache"),
                    "--output=" + str(output),
                    *extra,
                )
                deployment = output / "serve.json"
                deployment.write_text(
                    json.dumps(
                        {
                            "name": "fixture-text",
                            "model": {"path": "model.json", "preset": "check"},
                            "definition": {"kind": "text"},
                        }
                    )
                )
                deployments.append((deployment, "abab"))
                return output

            try:
                run(
                    "--update-lock",
                    "--all-variants",
                    "--origin=checkpoint=" + base,
                    "--lockfile=" + str(publication / "model.json.lock"),
                    "--cache=" + str(root / "publisher-cache"),
                )
                requests.clear()
                subset = install("subset")
                assert requests == [(0, 8), (8, header_length), (len(source) - 4, 4)]
                assert (subset / "bias.data").stat().st_size == 4100
                assert not (root / "client-cache/preserved").exists()
                full = install("full", "full")
                assert os.path.samefile(subset / "control.loom", full / "control.loom")
                assert not os.path.samefile(subset / "bias.data", full / "bias.data")
                install("subset", "swatch", base, "--preserve-sources")
                receipt = json.loads(
                    next((root / "client-cache/preserved").iterdir()).read_text()
                )
                assert (
                    sum(item["length"] for item in receipt["ranges"])
                    == header_length + 12
                )
                mirror = root / "mirror"
                mirror.mkdir()
                (mirror / "source.safetensors").write_bytes(archive(0))
                # A separate cache makes this exercise the compact input, rather
                # than satisfying the request from the earlier materialized output.
                compact_cache = root / "compact-cache"
                run(
                    "--preset=check",
                    "--origin=checkpoint=" + str(mirror),
                    "--cache=" + str(compact_cache),
                    "--output=" + str(root / "compact"),
                )
                assert (root / "compact/bias.data").read_bytes() == (
                    subset / "bias.data"
                ).read_bytes()
                shutil.copyfile(subset / "serve.json", root / "compact/serve.json")
                deployments.append((root / "compact/serve.json", "abab"))
                # A developer mirror can change parameter values while code and
                # other inputs retain the publisher's expectations.
                changed = archive(0)[:-4] + struct.pack("<i", 3)
                (mirror / "source.safetensors").write_bytes(changed)
                developer = root / "developer"
                run(
                    "--preset=check",
                    "--origin=checkpoint=" + str(mirror),
                    "--cache=" + str(root / "developer-cache"),
                    "--output=" + str(developer),
                    "--no-verify-parameters",
                )
                receipt = json.loads((developer / ".loom-fetch.json").read_text())
                assert receipt["unverified_parameters"] == [
                    "bias.data",
                    "metadata.safetensors",
                ]
                shutil.copyfile(subset / "serve.json", developer / "serve.json")
                deployments.append((developer / "serve.json", "xbbx"))
            finally:
                server.shutdown()
                thread.join()
        install("offline", "swatch", base, "--offline", "--preserve-sources")
        assert os.path.samefile(subset / "bias.data", root / "offline/bias.data")
        for deployment, expected in deployments:
            check_model(serve, deployment, expected)


if __name__ == "__main__":
    main(*sys.argv[1:])
