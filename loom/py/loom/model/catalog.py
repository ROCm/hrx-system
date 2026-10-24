# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small discovery indexes generated from portable model cards."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from loom.model.manifest import (
    ManifestError,
    ModelCard,
    _json_value,
    _mapping,
    _object,
    _path,
    _string,
    _strings,
    parse_json,
)


class ModelCatalog:
    """An owned metadata index; selected cards own executable model semantics."""

    def __init__(self, document: dict):
        document = copy.deepcopy(document)
        _json_value(document, "catalog")
        _object(document, "catalog", ("schema_version", "models"), ("$schema",))
        if (
            type(document["schema_version"]) is not int
            or document["schema_version"] != 1
        ):
            raise ManifestError("unsupported model-catalog version; supported: 1")
        if "$schema" in document:
            _string(document["$schema"], "catalog.$schema")
        for identity, entry in _mapping(
            document["models"], "catalog.models", empty=True
        ).items():
            path = "catalog.models." + identity
            _object(
                entry,
                path,
                ("name", "card", "services", "variants", "presets"),
                ("description",),
            )
            _string(entry["name"], path + ".name")
            _path(entry["card"], path + ".card")
            if "description" in entry and not isinstance(entry["description"], str):
                raise ManifestError(path + ".description: expected a string")
            _strings(entry["variants"], path + ".variants", empty=False)
            _strings(entry["presets"], path + ".presets")
            for name, service in _mapping(
                entry["services"], path + ".services"
            ).items():
                current = path + ".services." + name
                _object(service, current, ("kind",), ("version",))
                kind = _string(service["kind"], current + ".kind")
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", kind):
                    raise ManifestError(
                        current + ".kind: expected a service symbol identifier"
                    )
                if "version" in service and (
                    type(service["version"]) is not int
                    or not 1 <= service["version"] <= 0xFFFFFFFF
                ):
                    raise ManifestError(
                        current + ".version: expected an ABI version in [1, 2^32-1]"
                    )
        self._document = document

    @classmethod
    def parse(cls, text: str) -> ModelCatalog:
        return cls(parse_json(text))

    @property
    def document(self) -> dict:
        return copy.deepcopy(self._document)

    def card_path(self, identity: str) -> str:
        if identity not in self._document["models"]:
            raise ManifestError(f"model {identity!r} is not in the selected catalog")
        return self._document["models"][identity]["card"]


def generate(cards: dict[str, ModelCard]) -> ModelCatalog:
    """Summarizes validated cards keyed by their publication-relative paths."""
    models = {}
    for path, card in cards.items():
        _path(path, "catalog card path")
        document = card.document
        identity = document["model"]["id"]
        if identity in models:
            raise ManifestError(f"duplicate catalog model {identity!r}")
        entry = {
            "name": document["model"]["name"],
            "card": path,
            "services": {
                name: {
                    key: value
                    for key, value in service.items()
                    if key in ("kind", "version")
                }
                for name, service in document["services"].items()
            },
            "variants": list(document["variants"]),
            "presets": list(document.get("presets", {})),
        }
        if "description" in document["model"]:
            entry["description"] = document["model"]["description"]
        models[identity] = entry
    return ModelCatalog({"schema_version": 1, "models": dict(sorted(models.items()))})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate model discovery metadata without reading model payloads."
    )
    parser.add_argument(
        "--card", action="append", default=[], metavar="PUBLISHED_PATH=LOCAL_JSON"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cards = {}
    for value in args.card:
        path, separator, local = value.partition("=")
        if not separator or not local or path in cards:
            parser.error("--card requires a unique PUBLISHED_PATH=LOCAL_JSON")
        cards[path] = ModelCard.parse(Path(local).read_text(encoding="utf-8"))
    args.output.write_text(
        json.dumps(generate(cards).document, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
