# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stages built programs with externally provisioned model-test assets."""

import json
from pathlib import Path

from loom.model import ModelCard
from loom.model.fetch_storage import share_file


def stage_model(card: Path, assets: Path, output: Path) -> Path:
    """Shares selected inputs into a fresh, caller-owned test directory.

    Providers and importer data always come from the built card's directory.
    Parameters and tokenizers come from the supplied installation using their
    card-relative destination names. The caller declares the card, acquisition
    lock, and built programs as test inputs; the lock is a trusted identity for
    the externally provisioned assets. This function neither reads asset bytes
    nor acquires content. The caller owns cleanup, including partial failure.
    """
    selection = ModelCard.parse(card.read_text(encoding="utf-8")).select()
    output.mkdir(parents=True, exist_ok=False)
    for path, file in selection.files.items():
        root = card.parent if file.roles & {"program", "auxiliary"} else assets
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        share_file(root / path, destination)
    staged = output / "model.json"
    staged.write_text(json.dumps(selection.document, indent=2) + "\n", encoding="utf-8")
    return staged
