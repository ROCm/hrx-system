# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Catalog discovery with one publication snapshot per acquisition operation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urljoin

from .catalog import ModelCatalog
from .fetch_source import Document, FetchError, document_location
from .fetch_storage import ContentCache, atomic_write
from .manifest import ManifestError, ModelCard, parse_json


@dataclass(frozen=True)
class CatalogSnapshot:
    # Authored local catalog or GitHub repository/tag association.
    source: str
    # Catalog document at its immutable publication coordinate, when remote.
    document: Document
    # Validated discovery facts, independent of selected model semantics.
    catalog: ModelCatalog
    # Selected Git commit, absent for a caller-owned local catalog.
    revision: str | None

    def summary(self) -> dict:
        return {
            "source": self.source,
            "location": self.document.location,
            "revision": self.revision,
        }

    def read_model(
        self, identity: str, cache: ContentCache, *, offline: bool
    ) -> tuple[Document, ModelCard]:
        path = self.catalog.card_path(identity)
        location = (
            urljoin(self.document.location, quote(path, safe="/"))
            if self.document.remote
            else str(Path(self.document.location).parent / path)
        )
        document = cache.read_document(location, offline=offline)
        card = ModelCard.parse(document.text)
        if card.document["model"]["id"] != identity:
            raise ManifestError(
                "selected model card identity differs from its catalog entry"
            )
        return document, card


def installed_catalog() -> str:
    """Reads only the active installed SDK's explicitly authored association."""
    try:
        distribution = importlib.metadata.distribution("hrx-loom")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    # A source checkout on PYTHONPATH must not inherit another installed SDK's
    # release association merely because its distribution metadata is visible.
    if distribution is not None and Path(__file__).resolve().is_relative_to(
        Path(distribution.locate_file("loom/model")).resolve()
    ):
        text = distribution.read_text("loom.json")
        if text is not None:
            source = json.loads(text)["catalog"]
            if source:
                return source
    raise FetchError(
        "this SDK has no associated model catalog; use --catalog=LOCAL_JSON "
        "or --catalog=github:OWNER/REPOSITORY@TAG"
    )


def _commit(cache: ContentCache, repository: str, tag: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", tag):
        return tag
    root = "https://api.github.com/repos/" + repository
    location = root + "/git/ref/tags/" + quote(tag, safe="")
    visited = set()
    requested_revision = None
    while True:
        document = parse_json(cache.read_document(location, offline=False).text)
        if not isinstance(document, dict) or (
            document.get("ref") != "refs/tags/" + tag
            if requested_revision is None
            else document.get("sha") != requested_revision
        ):
            raise FetchError("GitHub returned a different catalog reference")
        target = document.get("object")
        if (
            not isinstance(target, dict)
            or not isinstance(target.get("sha"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", target["sha"])
        ):
            raise FetchError("GitHub returned an invalid catalog reference")
        revision = target["sha"]
        if target.get("type") == "commit":
            return revision
        if target.get("type") != "tag" or revision in visited or len(visited) >= 32:
            raise FetchError("catalog reference does not resolve to a commit")
        visited.add(revision)
        requested_revision = revision
        location = root + "/git/tags/" + revision


def load_catalog(source: str, cache: ContentCache, *, offline: bool) -> CatalogSnapshot:
    if not source.startswith("github:"):
        if "://" in source:
            raise ManifestError(
                "catalog discovery requires a local JSON or "
                "github:OWNER/REPOSITORY@TAG; direct model URLs do not "
                "require a catalog"
            )
        location = document_location(source)
        document = cache.read_document(location, offline=offline)
        return CatalogSnapshot(
            location, document, ModelCatalog.parse(document.text), None
        )
    match = re.fullmatch(r"github:([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@(.+)", source)
    if match is None or any(part in (".", "..") for part in match[1].split("/")):
        raise ManifestError("expected github:OWNER/REPOSITORY@TAG_OR_COMMIT")
    repository, tag = match.groups()
    pointer = cache.root / "catalog" / hashlib.sha256(source.encode()).hexdigest()
    if offline:
        if re.fullmatch(r"[0-9a-f]{40}", tag):
            revision = tag
        else:
            if not pointer.is_file():
                raise FetchError("--offline has no cached snapshot for this catalog")
            with pointer.open("rb") as stream:
                encoded = stream.read(41)
            if not re.fullmatch(rb"[0-9a-f]{40}", encoded):
                raise FetchError("cached catalog snapshot is invalid")
            revision = encoded.decode("ascii")
    else:
        revision = _commit(cache, repository, tag)
    location = f"https://raw.githubusercontent.com/{repository}/{revision}/catalog.json"
    document = cache.read_document(location, offline=offline)
    catalog = ModelCatalog.parse(document.text)
    if not offline:
        atomic_write(pointer, revision.encode("ascii"))
    return CatalogSnapshot(source, document, catalog, revision)
