# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan, verify, and publish one portable model selection."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin

from .fetch_hf import AccessRequired
from .fetch_source import (
    Document,
    FetchError,
    Source,
    SourceResolver,
    document_location,
    sibling_lock,
)
from .fetch_storage import (
    AcquiredFile,
    ContentCache,
    atomic_write,
    installation_lock,
    share_file,
)
from .lock import FileIdentity, ModelLock, file_digest, file_key
from .manifest import ManifestError, ModelCard, Selection, parse_json

_RESERVED_PATHS = frozenset(
    {"model.json", "model.json.lock", ".loom-fetch.lock", ".loom-fetch.json"}
)


@dataclass(frozen=True)
class FetchPlan:
    # Authored document coordinates used for relative acquisition sources.
    document: Document
    # Owned, validated exact program/service/variant selection.
    selection: Selection
    # Effective sources, with transport overrides already applied.
    sources: dict[str, Source]
    # Required byte identities after any explicit parameter verification exception.
    identities: dict[str, FileIdentity]
    # Selected logical roots and their immutable resolutions, never mirrors.
    origins: dict
    # Network prohibition, including provider metadata and authentication.
    offline: bool
    # Whether installation/source lock documents are emitted.
    locked: bool
    # Explicit author's lock update destination; absent for ordinary acquisition.
    update_path: Path | None
    # Acquisition provenance for catalog or developer verification selections.
    provenance: dict | None = None
    # Parameter-only paths whose publisher byte expectations are explicitly skipped.
    unverified_parameters: frozenset[str] = frozenset()

    def summary(self) -> dict:
        """Machine-readable acquisition facts; unknown sizes remain null."""
        return {
            "model": self.selection.model_id,
            "preset": self.selection.preset,
            "service": self.selection.service_name,
            "variant": self.selection.variant,
            "partition": self.selection.partition,
            "programs": self.selection.programs,
            "offline": self.offline,
            "lock": "verify"
            if self.identities
            else "update"
            if self.locked
            else "none",
            "unverified_parameters": sorted(self.unverified_parameters),
            **(
                {"memory": copy.deepcopy(self.selection.settings["memory"])}
                if "memory" in self.selection.settings
                else {}
            ),
            "files": [
                {
                    "path": path,
                    "origin": source.origin,
                    "transport": source.transport,
                    "roles": sorted(self.selection.files[path].roles),
                    "transform": self.selection.files[path].transform,
                    "size": self.identities[path].size
                    if path in self.identities
                    else None,
                    "sha256": self.identities[path].sha256
                    if path in self.identities
                    else None,
                }
                for path, source in self.sources.items()
            ],
        }


def plan_fetch(
    document: Document,
    selection: Selection,
    cache: ContentCache,
    *,
    overrides: dict[str, str] | None = None,
    lock_location: str | None = None,
    offline: bool = False,
    no_lock: bool = False,
    update_lock: bool = False,
    no_verify_parameters: bool = False,
    catalog: dict | None = None,
) -> FetchPlan:
    """Checks selected lock coverage and resolves sources before payload transfer."""
    if no_lock and (update_lock or lock_location is not None):
        raise ManifestError(
            "--no-lock cannot be combined with --update-lock or --lockfile"
        )
    if no_verify_parameters and (no_lock or update_lock):
        raise ManifestError(
            "--no-verify-parameters requires ordinary locked acquisition; "
            "it cannot be combined with --no-lock or --update-lock"
        )
    for path in selection.files:
        if PurePosixPath(path).parts[0] in _RESERVED_PATHS:
            raise ManifestError(f"reserved installation path {path!r}")
    location = (
        document_location(lock_location)
        if lock_location
        else sibling_lock(document.location)
    )
    lock = None
    if not no_lock and not update_lock:
        try:
            lock = ModelLock.parse(cache.read_document(location, offline=offline).text)
        except AccessRequired:
            raise
        except (OSError, FetchError) as error:
            raise FetchError(
                f"cannot read model lock: {error}; provide --lockfile, "
                "explicitly --update-lock, or use --no-lock"
            ) from error
    identities = lock.cover(selection) if lock else {}
    resolver = SourceResolver(
        document,
        selection.document.get("origins", {}),
        overrides or {},
        lock,
        pin_revisions=not no_lock,
        revision_resolver=lambda root: cache.huggingface.resolve_revision(
            root, offline=offline
        ),
    )
    sources = {
        path: resolver.resolve(file.origin, identities.get(path))
        for path, file in selection.files.items()
    }
    protected = {
        file_key(file.origin, file.transform)
        for file in selection.files.values()
        if file.roles - {"parameter"}
    }
    unverified_parameters = frozenset(
        path
        for path, file in selection.files.items()
        if no_verify_parameters
        and file_key(file.origin, file.transform) not in protected
    )
    identities = {
        path: identity
        for path, identity in identities.items()
        if path not in unverified_parameters
    }
    sources = _prepare_sources(
        sources,
        identities,
        cache,
        offline=offline,
        transforms={
            path: file.transform
            for path, file in selection.files.items()
            if file.transform is not None
        },
    )
    update_path = None
    if update_lock:
        if "://" in location:
            if lock_location:
                raise ManifestError(
                    "--update-lock requires a local --lockfile destination"
                )
        else:
            update_path = Path(location)
            _check_lock_sources(update_path, document, sources)
    return FetchPlan(
        document,
        selection,
        sources,
        identities,
        resolver.origins,
        offline,
        not no_lock,
        update_path,
        {
            "schema_version": 1,
            "card": {
                "location": document.location,
                "sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
            },
            "catalog": copy.deepcopy(catalog),
            **(
                {"unverified_parameters": sorted(unverified_parameters)}
                if unverified_parameters
                else {}
            ),
        }
        if catalog is not None or unverified_parameters
        else None,
        unverified_parameters,
    )


@dataclass(frozen=True)
class LockPlan:
    # Training-source model identity shared by all included choices.
    model_id: str
    # Explicit compatible variant/service/program selections, without execution.
    selections: list[dict[str, str]]
    # One source per origin/transform pair, independent of installation paths.
    sources: dict[str, Source]
    # Fixed operations included in each unique output's identity.
    transforms: dict[str, dict]
    # Selected authored roots and their immutable repository resolutions.
    origins: dict
    # Caller-owned local publication lock destination.
    path: Path
    # Network prohibition retained across planning and acquisition.
    offline: bool

    def summary(self) -> dict:
        return {
            "model": self.model_id,
            "lock": "update-all-variants",
            "lockfile": str(self.path),
            "offline": self.offline,
            "selections": copy.deepcopy(self.selections),
            "files": [
                {
                    "origin": copy.deepcopy(source.origin),
                    "transport": source.transport,
                    "transform": self.transforms.get(key),
                }
                for key, source in self.sources.items()
            ],
        }


def plan_lock(
    document: Document,
    card: ModelCard,
    cache: ContentCache,
    *,
    programs: str | None = None,
    overrides: dict[str, str] | None = None,
    lock_location: str | None = None,
    offline: bool = False,
) -> LockPlan:
    """Resolves a publication union, optionally restricted to one program set."""
    selections = card.choices()
    if programs is not None:
        selections = [choice for choice in selections if choice["programs"] == programs]
        if not selections:
            raise ManifestError(f"no compatible choices for program set {programs!r}")
    location = document_location(lock_location or sibling_lock(document.location))
    if "://" in location:
        raise ManifestError("publication lock updates require a local --lockfile")
    path = Path(location)
    if path.name == ".loom-fetch.lock" or (path.exists() and not path.is_file()):
        raise ManifestError("publication lock destination must be an ordinary file")
    authored = card.document
    resolver = SourceResolver(
        document,
        authored.get("origins", {}),
        overrides or {},
        None,
        pin_revisions=True,
        revision_resolver=lambda root: cache.huggingface.resolve_revision(
            root, offline=offline
        ),
    )
    sources = {}
    transforms = {}
    for choice in selections:
        selected = card.select(**choice)
        for name, file in selected.files.items():
            if PurePosixPath(name).parts[0] in _RESERVED_PATHS:
                raise ManifestError(f"reserved installation path {name!r}")
            key = file_key(file.origin, file.transform)
            if key not in sources:
                sources[key] = resolver.resolve(file.origin, None)
                if file.transform is not None:
                    transforms[key] = file.transform
    sources = _prepare_sources(
        sources, {}, cache, offline=offline, transforms=transforms
    )
    _check_lock_sources(path, document, sources)
    return LockPlan(
        authored["model"]["id"],
        selections,
        sources,
        transforms,
        resolver.origins,
        path,
        offline,
    )


def _prepare_sources(
    sources: dict[str, Source],
    identities: dict[str, FileIdentity],
    cache: ContentCache,
    *,
    offline: bool,
    transforms: dict[str, dict],
) -> dict[str, Source]:
    sources = {
        name: source
        if name in transforms
        else cache.prepare_source(source, identities.get(name), offline=offline)
        for name, source in sources.items()
    }
    for name, source in sources.items():
        if source.transport["kind"] == "local" and name not in transforms:
            path = Path(source.transport["path"])
            if not path.is_file():
                raise FetchError(f"local source is missing or not a file: {path}")
    return sources


def _check_lock_sources(path: Path, document: Document, sources: dict[str, Source]):
    protected = [Path(document.location)] if not document.remote else []
    protected += [
        Path(source.transport["path"])
        for source in sources.values()
        if source.transport["kind"] == "local"
    ]
    if any(
        path == source
        or (path.exists() and source.exists() and os.path.samefile(path, source))
        for source in protected
    ):
        raise ManifestError("lock update destination overlaps a source file")


def _acquire_sources(
    sources: dict[str, Source],
    identities: dict[str, FileIdentity],
    cache: ContentCache,
    *,
    offline: bool,
    transforms: dict[str, dict],
    preserve_sources: bool = False,
    present: dict[str, AcquiredFile] | None = None,
) -> dict[str, AcquiredFile]:
    """Joins a bounded transfer batch, deduplicated by logical source identity."""
    acquired = dict(present or {})
    unique = {}
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="loom-fetch") as executor:
        try:
            for name, source in sources.items():
                transform = transforms.get(name)
                key = file_key(source.origin, transform)
                if key not in unique and key not in acquired:
                    unique[key] = (
                        executor.submit(
                            cache.acquire_transformed,
                            source,
                            transform,
                            identities.get(name),
                            offline=offline,
                            preserve_sources=preserve_sources,
                        )
                        if transform is not None
                        else executor.submit(
                            cache.acquire, source, identities.get(name), offline=offline
                        )
                    )
            names = {future: key for key, future in unique.items()}
            pending = set(names)
            while pending:
                # Windows Python before 3.14 cannot interrupt an indefinite lock
                # wait. Return to Python to deliver Ctrl-C and join the writers.
                completed, pending = wait(
                    pending, timeout=0.1, return_when=FIRST_COMPLETED
                )
                for future in completed:
                    acquired[names[future]] = future.result()
        except BaseException:
            cache.cancel()
            for future in unique.values():
                future.cancel()
            raise
    return acquired


def update_lock(
    plan: LockPlan, cache: ContentCache, *, preserve_sources: bool = False
) -> Path:
    """Atomically replaces one publication lock after all source reads succeed."""
    with installation_lock(plan.path.parent):
        acquired = _acquire_sources(
            plan.sources,
            {},
            cache,
            offline=plan.offline,
            transforms=plan.transforms,
            preserve_sources=preserve_sources,
        )
        document = ModelLock.snapshot(
            plan.origins, [item.identity for item in acquired.values()]
        )
        if cache.cancelled.is_set():
            raise FetchError("model acquisition cancelled")
        atomic_write(plan.path, (json.dumps(document, indent=2) + "\n").encode("utf-8"))
    return plan.path


def _installation_card(plan: FetchPlan) -> tuple[dict, dict[str, str | dict], dict]:
    """Relocates acquisition coordinates while keeping runtime paths relative."""
    document = copy.deepcopy(plan.selection.document)
    roots = {}
    for name, record in plan.origins.items():
        root = copy.deepcopy(record)
        if root["requested"]["kind"] == "local":
            path = Path(root["requested"]["path"])
            if not path.is_absolute():
                path = Path(plan.document.location).parent / path
            root["requested"]["path"] = str(path.absolute())
        roots[name] = root
    document_alias = "document"
    while document_alias in plan.selection.document.get("origins", {}):
        document_alias += "_"
    origins = {}
    for path, source in plan.sources.items():
        origin = copy.deepcopy(source.origin)
        if (
            isinstance(origin, str)
            and not origin.startswith("@")
            and "://" not in origin
        ):
            if document_alias not in roots:
                requested = (
                    {"kind": "url", "url": urljoin(plan.document.location, ".")}
                    if plan.document.remote
                    else {
                        "kind": "local",
                        "path": str(Path(plan.document.location).parent),
                    }
                )
                roots[document_alias] = {"requested": requested}
            origin = f"@{document_alias}/{origin}"
        elif isinstance(origin, dict) and origin["kind"] == "local":
            original = Path(origin["path"])
            if not original.is_absolute():
                origin["path"] = str(Path(plan.document.location).parent / original)
        origins[path] = origin

    def file(value):
        record = {"path": value} if isinstance(value, str) else value
        record["origin"] = copy.deepcopy(origins[record["path"]])
        if "data" in record:
            record["data"] = [file(auxiliary) for auxiliary in record["data"]]
        return record

    for program in document["programs"].values():
        program.pop("origin", None)
        for role in ("inputs", "libraries"):
            program[role] = [file(value) for value in program[role]]
    for variant in document["variants"].values():
        scope = (
            variant["partitions"][plan.selection.partition]
            if plan.selection.partition is not None
            else variant
        )
        for service in scope["services"].values():
            for parameter in service.get("parameters", {}).values():
                parameter.pop("origin", None)
                parameter["files"] = [file(value) for value in parameter["files"]]
            service["tokenizers"] = {
                name: file(value)
                for name, value in service.get("tokenizers", {}).items()
            }
    if roots:
        document["origins"] = {
            name: record["requested"] for name, record in roots.items()
        }
    else:
        document.pop("origins", None)
    schema = document.get("$schema")
    if schema and "://" not in schema:
        document["$schema"] = (
            urljoin(plan.document.location, schema)
            if plan.document.remote
            else (Path(plan.document.location).parent / schema).resolve().as_uri()
        )
    return document, origins, roots


def _destination(directory: Path, relative: str) -> Path:
    path = directory / relative
    for parent in path.parents:
        if parent == directory:
            break
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise FetchError(
                f"destination parent is not an ordinary directory: {parent}"
            )
    if path.is_symlink() and not path.exists():
        raise FetchError(f"destination is a dangling symlink: {path}")
    if path.exists() and not path.is_file():
        raise FetchError(f"destination is not a file: {path}")
    return path


def _check_document(path: Path, expected: dict) -> None:
    if path.exists() and parse_json(path.read_text(encoding="utf-8")) != expected:
        raise FetchError(
            f"conflicting installation document: {path}; choose another output"
        )


def _check_outputs(
    plan: FetchPlan, directory: Path, card: dict
) -> dict[str, AcquiredFile]:
    _check_document(_destination(directory, "model.json"), card)
    _destination(directory, "model.json.lock")
    if plan.provenance is not None:
        _check_document(_destination(directory, ".loom-fetch.json"), plan.provenance)
    present = {}
    for name, source in plan.sources.items():
        destination = _destination(directory, name)
        if not destination.exists():
            continue
        identity = plan.identities.get(name)
        if identity is None:
            if (
                plan.selection.files[name].transform is not None
                or source.transport["kind"] != "local"
                or not os.path.samefile(source.transport["path"], destination)
            ):
                raise FetchError(
                    f"cannot verify existing destination {destination} "
                    "before transfer; "
                    "use a matching lock or choose another output"
                )
            size, digest = file_digest(destination)
            identity = FileIdentity(source.origin, size, digest, source.resolved)
        else:
            identity.verify(destination)
        present[name] = AcquiredFile(destination, identity)
    return present


def check_output(plan: FetchPlan, output: Path) -> None:
    """Checks an explicit output without creating it or acquiring payloads."""
    card, relocated, roots = _installation_card(plan)
    _check_outputs(plan, output.absolute(), card)
    _check_lock_output(plan, output.absolute(), relocated, roots)


def _check_lock_output(plan: FetchPlan, directory: Path, relocated: dict, roots: dict):
    path = directory / "model.json.lock"
    if not plan.locked or not path.exists():
        return
    if plan.unverified_parameters:
        raise FetchError(
            "choose another output for --no-verify-parameters; "
            "this installation already records parameter identities"
        )
    if not plan.identities:
        raise FetchError(
            "an installation lock already exists; choose another output "
            "for --update-lock"
        )
    expected = ModelLock.snapshot(
        roots,
        [
            replace(identity, origin=relocated[name])
            for name, identity in plan.identities.items()
        ],
    )
    _check_document(path, expected)


def _probe_sharing(plan: FetchPlan, cache: ContentCache, directory: Path) -> None:
    roots = {cache.root}
    if any(
        source.transport["kind"] == "hf"
        and plan.selection.files[name].transform is None
        for name, source in plan.sources.items()
    ):
        roots.add(cache.huggingface.cache_directory)
    with ExitStack() as stack:
        sources = set()
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
            temporary = stack.enter_context(
                tempfile.TemporaryDirectory(prefix=".loom-fetch-", dir=root)
            )
            path = Path(temporary) / "probe"
            path.write_bytes(b"")
            sources.add(path)
        sources.update(
            Path(source.transport["path"])
            for name, source in plan.sources.items()
            if source.transport["kind"] == "local"
            and plan.selection.files[name].transform is None
        )
        with tempfile.TemporaryDirectory(prefix=".loom-fetch-", dir=directory) as probe:
            destination = Path(probe) / "shared"
            devices = set()
            for source in sources:
                device = source.stat().st_dev
                if device in devices:
                    continue
                share_file(source, destination)
                destination.unlink()
                devices.add(device)


def install(
    plan: FetchPlan,
    cache: ContentCache,
    output: Path,
    *,
    materialize: str = "share",
    preserve_sources: bool = False,
) -> Path:
    """Publishes a card only after its complete selected acquisition succeeds."""
    if cache.cancelled.is_set():
        raise FetchError("model acquisition cancelled")
    if materialize not in ("share", "copy"):
        raise ManifestError("materialize must be 'share' or 'copy'")
    directory = output.absolute()
    card, relocated, roots = _installation_card(plan)
    if plan.update_path is not None:
        for name in (*plan.sources, *_RESERVED_PATHS):
            if plan.update_path == directory / name:
                raise ManifestError(
                    "lock update destination overlaps installation content"
                )
    with installation_lock(directory):
        present = _check_outputs(plan, directory, card)
        _check_lock_output(plan, directory, relocated, roots)
        if materialize == "share":
            _probe_sharing(plan, cache, directory)
        acquired = _acquire_sources(
            plan.sources,
            plan.identities,
            cache,
            offline=plan.offline,
            transforms={
                name: file.transform
                for name, file in plan.selection.files.items()
                if file.transform is not None
            },
            preserve_sources=preserve_sources,
            present={
                file_key(
                    plan.sources[name].origin, plan.selection.files[name].transform
                ): item
                for name, item in present.items()
                if not preserve_sources or plan.selection.files[name].transform is None
            },
        )
        identities = [
            replace(
                acquired[
                    file_key(source.origin, plan.selection.files[name].transform)
                ].identity,
                origin=relocated[name],
            )
            for name, source in plan.sources.items()
        ]
        installation_lock_document = (
            ModelLock.snapshot(roots, identities) if plan.locked else None
        )
        if installation_lock_document is not None:
            _check_document(directory / "model.json.lock", installation_lock_document)
        published = []
        try:
            with tempfile.TemporaryDirectory(
                prefix=".loom-fetch-", dir=directory
            ) as temporary:
                staging = Path(temporary)
                for name, source in plan.sources.items():
                    if cache.cancelled.is_set():
                        raise FetchError("model acquisition cancelled")
                    if name in present:
                        continue
                    destination = directory / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    path = acquired[
                        file_key(source.origin, plan.selection.files[name].transform)
                    ].path
                    if materialize == "copy":
                        copied = staging / "payload"
                        shutil.copyfile(path, copied)
                        os.link(copied, destination)
                        copied.unlink()
                    else:
                        share_file(path, destination)
                    published.append(destination)
                # The complete card is the publication boundary. No runtime ever
                # sees a card advertising a partially acquired closure.
                metadata = {
                    "model.json.lock": installation_lock_document,
                    ".loom-fetch.json": plan.provenance,
                    "model.json": card,
                }
                for name, document in metadata.items():
                    if cache.cancelled.is_set():
                        raise FetchError("model acquisition cancelled")
                    destination = directory / name
                    if document is None or destination.exists():
                        continue
                    staged = staging / name
                    atomic_write(
                        staged, (json.dumps(document, indent=2) + "\n").encode("utf-8")
                    )
                    os.link(staged, destination)
                    published.append(destination)
            if plan.update_path is not None:
                document = ModelLock.snapshot(
                    plan.origins, [item.identity for item in acquired.values()]
                )
                atomic_write(
                    plan.update_path,
                    (json.dumps(document, indent=2) + "\n").encode("utf-8"),
                )
        except BaseException:
            for path in reversed(published):
                path.unlink()
            raise
    return directory / "model.json"
