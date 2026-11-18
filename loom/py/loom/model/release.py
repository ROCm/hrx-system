# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stage one built model selection and its lock for publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from loom.model import ManifestError, ModelCard, ModelFile
from loom.model.lock import FileIdentity, ModelLock, lock_files


def _open_program_output(path, bytecode):
    if not bytecode:
        return path.open("xb")
    try:
        import lzma
    except ImportError:
        raise ManifestError(
            "publishing XZ bytecode requires Python's lzma module"
        ) from None
    # Independent, small dictionaries bound the working set of parallel readers.
    return lzma.open(
        path,
        "xb",
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": 1 << 20}],
    )


def _stage_program(root: Path, stage: Path, file: ModelFile) -> FileIdentity:
    bytecode = "program" in file.roles and (
        file.format == "loombc"
        or (file.format is None and file.path.endswith(".loombc"))
    )
    origin = file.path + ".xz" if bytecode else file.path
    output = stage / origin
    output.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    digest = hashlib.sha256()
    with (
        (root / file.path).open("rb") as source,
        _open_program_output(output, bytecode) as destination,
    ):
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
            destination.write(chunk)
    return FileIdentity(
        origin,
        size,
        digest.hexdigest(),
        transform={"kind": "xz", "version": 1} if bytecode else None,
    )


def stage_release(
    card: Path,
    output: Path,
    *,
    asset_lock: Path | None = None,
    preset: str | None = None,
) -> None:
    """Publishes local providers and retains separately provisioned asset origins."""
    selected = ModelCard.parse(card.read_text(encoding="utf-8")).select(preset)
    assets = lock_files(selected, "assets")
    identities = {}
    roots = {}
    if assets:
        lock_path = asset_lock or card.with_name(card.name + ".lock")
        lock = ModelLock.parse(lock_path.read_text(encoding="utf-8"))
        identities = lock.cover(selected, assets_only=True)
        roots = lock.origins
    document = selected.document
    program = document["programs"][selected.programs]
    program.pop("origin", None)

    def published(value):
        record = {"path": value} if isinstance(value, str) else value.copy()
        name = record["path"]
        identity = identities[name]
        if identity.origin == name:
            record.pop("origin", None)
        else:
            record["origin"] = identity.origin
        if identity.transform:
            record["transform"] = identity.transform
        else:
            record.pop("transform", None)
        if "data" in record:
            record["data"] = [published(item) for item in record["data"]]
        return record if len(record) != 1 else name

    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ManifestError(
            f"release output must be a new or empty directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "release"
        stage.mkdir()
        for name, file in selected.files.items():
            if name not in assets:
                identities[name] = _stage_program(card.parent, stage, file)
        for role in ("inputs", "libraries"):
            program[role] = [published(value) for value in program[role]]
        snapshot = ModelLock.snapshot(roots, list(identities.values()))
        for name, contents in (("model.json", document), ("model.json.lock", snapshot)):
            with (stage / name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(contents, indent=2) + "\n")
        # A relative runtime asset must still have a place in the publication.
        for file in assets.values():
            origin = file.origin
            if (
                isinstance(origin, str)
                and not origin.startswith(("@", "../"))
                and "://" not in origin
            ):
                path = Path(origin)
                if (stage / path).exists() or any(
                    (stage / part).is_file() for part in path.parents
                ):
                    raise ManifestError(
                        f"release file conflicts with asset origin {origin!r}"
                    )
        # Bazel creates the empty tree artifact before running the action.
        # Removing only an empty destination keeps publication portable to Windows.
        if output.is_dir():
            output.rmdir()
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", type=Path, help="built model card beside its providers")
    parser.add_argument(
        "--output", required=True, type=Path, help="new or empty directory"
    )
    parser.add_argument(
        "--preset", help="choose a preset when the card declares several"
    )
    parser.add_argument(
        "--asset-lock", type=Path, help="authored lock (default: CARD.lock)"
    )
    args = parser.parse_args()
    try:
        stage_release(
            args.card, args.output, asset_lock=args.asset_lock, preset=args.preset
        )
    except (ManifestError, OSError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
