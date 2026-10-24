# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""One publication lock serves distinct weighted variants at the same local path."""

import copy
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

from loom.model import ModelCard
from loom.model.fetch import install, plan_fetch, plan_lock, update_lock
from loom.model.fetch_storage import ContentCache
from model_card_test import check_model


def completion(bias):
    # Dense arithmetic for the prompt "a", independent of device packet layout.
    hidden, history, total = bias, bias, bias
    result = ""
    for position in range(1, 5):
        token = (hidden + total) % 3
        result += "abx"[token]
        hidden = (5 * hidden + token + bias) % 97
        history = (7 * history + hidden + position) % 101
        total += history
    return result


def main(server, fixture):
    server = Path(server).absolute()
    # Preserve the declared source/generated runfiles namespace.
    fixture = Path(fixture).absolute()
    authored = json.loads(fixture.read_text())
    authored["origins"] = {"fixture": {"kind": "local", "path": str(fixture.parent)}}
    for program in authored["programs"].values():
        program["origin"] = "@fixture/"
    for service in authored["variants"]["fixture"]["services"].values():
        service["parameters"]["model"]["origin"] = "@fixture/"
        for name, tokenizer in service.get("tokenizers", {}).items():
            service["tokenizers"][name] = {
                "path": tokenizer,
                "origin": "@fixture/" + tokenizer,
            }
    authored["variants"]["alternate"] = copy.deepcopy(authored["variants"]["fixture"])
    for service in authored["variants"]["alternate"]["services"].values():
        service["parameters"]["model"]["files"][1]["origin"] = "alternate.data"
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        card_path = directory / "model.json"
        card_path.write_text(json.dumps(authored))
        checkpoint = (fixture.parent / "bias.data").read_bytes()
        (directory / "alternate.data").write_bytes(
            checkpoint[:-4] + struct.pack("<i", 3)
        )
        cache = ContentCache(directory / "cache")
        document = cache.read_document(str(card_path), offline=True)
        card = ModelCard.parse(document.text)
        publication = plan_lock(document, card, cache, offline=True)
        assert len(publication.selections) == 18
        update_lock(publication, cache)
        installed = []
        for variant, bias in (("fixture", 3), ("alternate", 4)):
            plan = plan_fetch(
                document, card.select("check", variant=variant), cache, offline=True
            )
            output = directory / variant
            install(plan, cache, output)
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
            check_model(server, deployment, completion(bias))
            installed.append(output)
        assert os.path.samefile(
            installed[0] / "control.loom", installed[1] / "control.loom"
        )
        assert not os.path.samefile(
            installed[0] / "bias.data", installed[1] / "bias.data"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
