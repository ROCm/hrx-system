# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks current program outputs with separately provisioned model assets."""

import argparse
import shutil
import tempfile
from pathlib import Path

from loom.model import ModelCard
from loom.model.lock import ModelLock
from loom_serve_testing.model_staging import stage_model
from model_card_test import check_model, complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    parser.add_argument("fixture_card", type=Path)
    parser.add_argument("deployments", type=Path, nargs="+")
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    # The model target propagates the authored lock independently of assets.
    lockfile = args.fixture_card.with_name(args.fixture_card.name + ".lock")
    ModelLock.parse(lockfile.read_text(encoding="utf-8")).cover(
        ModelCard.parse(args.fixture_card.read_text(encoding="utf-8")).select("check")
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        if args.assets is not None:
            assets = args.assets.absolute()
        else:
            assets = root / "assets"
            assets.mkdir()
            for name in ("bias.data", "metadata.safetensors", "tokenizer.json"):
                shutil.copyfile(args.fixture_card.parent / name, assets / name)
            # Installed programs may be stale. Only assets belong to this root.
            (assets / "prepare.loom").write_text("invalid stale program\n")
            (assets / "weights.loom").write_text("invalid stale importer input\n")

        for index, deployment in enumerate(args.deployments):
            card = deployment.absolute().parent / "model.json"
            selection = ModelCard.parse(card.read_text()).select()
            staged = stage_model(card, assets, root / str(index))
            for path, file in selection.files.items():
                source = (
                    card.parent if file.roles & {"program", "auxiliary"} else assets
                )
                assert (staged.parent / path).samefile(source / path), path
            check_model(
                args.server.absolute(),
                staged,
                complete([0], bias=2 + selection.config["fixture.bias_delta"]),
            )

        missing = root / "missing"
        missing.mkdir()
        try:
            stage_model(card, missing, root / "incomplete")
        except FileNotFoundError:
            assert not (root / "incomplete/model.json").exists()
        else:
            raise AssertionError("missing external assets must fail staging")


if __name__ == "__main__":
    main()
