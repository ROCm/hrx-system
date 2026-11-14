# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Parameter payloads the native runtime loader derives at load time.

Acquisition installs the declared source and leaves the derived payload absent.
The loader computes a content key from the source bytes and recipe, derives into
the card's cache directory on first load, and reuses that entry afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from .manifest import ManifestError, Selection


@dataclass(frozen=True)
class Recipe:
    """One loader implementation, keyed by the card's transform declaration."""

    # Payload layout identifier included in the loader's content key.
    layout_version: str
    # Numerical recipe identifier included in the loader's content key.
    recipe_version: str
    # Declared output -> payload file name under the content-keyed directory.
    outputs: dict[str, str]


# (kind, version, layout, kernel_version) -> recipe implemented by
# loom/src/loom/serve/runtime/parameter/i8_panel.c. Keep both in step.
_RECIPES = {
    ("i8-panel", 1, "v2", "resident-full-k-i8-i32-bf16-column-scale-v1"): Recipe(
        layout_version="full-k-colscale-n128-k128-nfrag-kfrag-k8-n8-v2",
        recipe_version="loom-f16-dequant-bf16-absmax-rne-i8-v1",
        outputs={"resident-halves": "panels_gu.i8", "colscale": "colscale_gu.f32"},
    ),
}

# Transform kinds whose outputs only the loader produces.
LOADER_DERIVED_KINDS = frozenset(key[0] for key in _RECIPES)


def is_loader_derived(transform: dict | None) -> bool:
    return transform is not None and transform.get("kind") in LOADER_DERIVED_KINDS


def recipe(transform: dict) -> Recipe | None:
    return _RECIPES.get(
        (
            transform.get("kind"),
            transform.get("version"),
            transform.get("layout"),
            transform.get("kernel_version"),
        )
    )


@dataclass(frozen=True)
class DerivedFile:
    """A selected parameter path that acquisition defers to the loader."""

    # Installation path the loader publishes; never created by acquisition.
    path: str
    # Lock coordinate: the destination path, since nothing is transferred.
    origin: str
    # Card transform, retained verbatim in the installed card and lock.
    transform: dict
    # Installation path of the source the loader reads.
    source: str
    # Logical origin of that source, which the lock commits separately.
    source_origin: str | dict
    # Loader content key named by the path's parent directory.
    cache_key: str
    # Loader implementation of this declaration.
    recipe: Recipe

    def facts(self, source_origin: str | dict | None = None) -> dict:
        """Lock record of the derivation, independent of payload bytes."""
        return {
            "source": self.source_origin if source_origin is None else source_origin,
            "layout_version": self.recipe.layout_version,
            "recipe_version": self.recipe.recipe_version,
        }

    def summary(self) -> dict:
        return {
            "by": "native runtime loader",
            "source": self.source,
            "source_origin": self.source_origin,
            "cache_key": self.cache_key,
            "layout_version": self.recipe.layout_version,
            "recipe_version": self.recipe.recipe_version,
        }


def parameter_label(selection: Selection, path: str) -> str:
    """Names a file by its declaring parameter scopes and installation path."""
    scopes = sorted(
        scope
        for scope, parameter in selection.parameters.items()
        for value in parameter["files"]
        if (value if isinstance(value, str) else value["path"]) == path
    )
    return (f"parameter {', '.join(scopes)} " if scopes else "") + repr(path)


def derived_files(selection: Selection) -> dict[str, DerivedFile]:
    """Selects loader-derived outputs and checks the loader can produce them."""
    result = {}
    for path, file in selection.files.items():
        transform = file.transform
        if transform is None or transform.get("kind") == "safetensors":
            continue
        label = parameter_label(selection, path)
        found = recipe(transform)
        if found is None or not is_loader_derived(transform):
            raise ManifestError(
                f"{label}: transform {transform.get('kind')!r} version "
                f"{transform.get('version')!r} layout {transform.get('layout')!r} "
                f"kernel_version {transform.get('kernel_version')!r} has no native "
                "loader implementation; supported: "
                + ", ".join(
                    f"{kind} v{version} layout {layout} {kernel}"
                    for kind, version, layout, kernel in _RECIPES
                )
            )
        if file.format != "raw":
            raise ManifestError(
                f"{label}: the native loader publishes {transform['kind']} "
                f"payloads as format 'raw', not {file.format!r}"
            )
        if file.roles != {"parameter"}:
            raise ManifestError(f"{label}: loader-derived files are parameters only")
        if file.origin != path:
            raise ManifestError(
                f"{label}: loader-derived files have no acquisition origin; "
                "remove the file or group origin"
            )
        destination = PurePosixPath(path)
        key = destination.parent.name
        if (
            str(destination.parent.parent) != transform["cache"]
            or not re.fullmatch(r"[0-9a-f]{64}", key)
            or destination.name != found.outputs[transform["output"]]
        ):
            raise ManifestError(
                f"{label}: the loader publishes {transform['output']!r} at "
                f"{transform['cache']}/<64-hex content key>/"
                f"{found.outputs[transform['output']]}"
            )
        source = selection.files.get(transform["source"])
        if source is None:
            raise ManifestError(
                f"{label}: source {transform['source']!r} is not a file of this "
                "selection, so it would not be installed for the native loader"
            )
        source_label = parameter_label(selection, source.path)
        if source.transform is not None or "parameter" not in source.roles:
            raise ManifestError(
                f"{label}: source {source_label} must be an untransformed "
                "parameter file"
            )
        if source.format not in (None, "gguf"):
            raise ManifestError(
                f"{label}: {transform['kind']} derives from a GGUF source, "
                f"not {source_label} with format {source.format!r}"
            )
        result[path] = DerivedFile(
            path, file.origin, transform, source.path, source.origin, key, found
        )
    return result
