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
import tempfile
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin

from .derived import DerivedFile, derived_files
from .fetch_filesystem import SharingUnavailable, copy_file, publish_file, share_file
from .fetch_hf import AccessRequired
from .fetch_progress import ByteProgress
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
)
from .lock import (
    DerivedIdentity,
    FileIdentity,
    ModelLock,
    file_digest,
    file_key,
    lock_files,
)
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
    # Acquisition policy: verify an existing lock, update it, or omit locks.
    lock_mode: str
    # Required commitment coverage, independent of transport or materialization.
    lock_scope: str
    # Explicit author's lock update destination; absent for ordinary acquisition.
    update_path: Path | None
    # Acquisition provenance for catalog or developer verification selections.
    provenance: dict | None = None
    # Parameter-only paths whose publisher byte expectations are explicitly skipped.
    unverified_parameters: frozenset[str] = frozenset()
    # Outputs the native loader derives at load; never acquired or created here.
    derived: dict[str, DerivedFile] = field(default_factory=dict)
    # Their locked records, including any publisher-observed payload bytes.
    derived_identities: dict[str, DerivedIdentity] = field(default_factory=dict)

    def _file_summary(self, path: str) -> dict:
        file = self.selection.files[path]
        if path in self.derived:
            identity = self.derived_identities.get(path)
            return {
                "path": path,
                "origin": file.origin,
                "transport": None,
                "roles": sorted(file.roles),
                "transform": file.transform,
                "derived": self.derived[path].summary(),
                "size": identity.size if identity else None,
                "sha256": identity.sha256 if identity else None,
            }
        identity = self.identities.get(path)
        return {
            "path": path,
            "origin": self.sources[path].origin,
            "transport": self.sources[path].transport,
            "roles": sorted(file.roles),
            "transform": file.transform,
            "size": identity.size if identity else None,
            "sha256": identity.sha256 if identity else None,
        }

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
            "lock": self.lock_mode,
            "lock_scope": self.lock_scope if self.lock_mode != "none" else None,
            "unverified_parameters": sorted(self.unverified_parameters),
            **(
                {"memory": copy.deepcopy(self.selection.settings["memory"])}
                if "memory" in self.selection.settings
                else {}
            ),
            "files": [self._file_summary(path) for path in self.selection.files],
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
    lock_scope: str | None = None,
    no_verify_parameters: bool = False,
    catalog: dict | None = None,
) -> FetchPlan:
    """Checks selected lock coverage and resolves sources before payload transfer."""
    if no_lock and (update_lock or lock_location is not None):
        raise ManifestError(
            "--no-lock cannot be combined with --update-lock or --lockfile"
        )
    if lock_scope is not None and not update_lock:
        raise ManifestError("--lock-scope requires --update-lock")
    if no_verify_parameters and (no_lock or update_lock):
        raise ManifestError(
            "--no-verify-parameters requires ordinary locked acquisition; "
            "it cannot be combined with --no-lock or --update-lock"
        )
    for path in selection.files:
        if PurePosixPath(path).parts[0] in _RESERVED_PATHS:
            raise ManifestError(f"reserved installation path {path!r}")
    derived = derived_files(selection)
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
    scope = lock.scope if lock else lock_scope or "assets"
    if scope not in ("assets", "all"):
        raise ManifestError("lock scope must be 'assets' or 'all'")
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
        if path not in derived
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
    derived_identities = {
        path: identities.pop(path) for path in derived if path in identities
    }
    sources = _prepare_sources(
        sources,
        identities,
        cache,
        offline=offline,
        transforms={
            path: selection.files[path].transform
            for path in sources
            if selection.files[path].transform is not None
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
        "none" if no_lock else "update" if update_lock else "verify",
        scope,
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
        derived,
        derived_identities,
    )


@dataclass(frozen=True)
class LockPlan:
    # Training-source model identity shared by all included choices.
    model_id: str
    # Explicit compatible variant/service/program selections, without execution.
    selections: list[dict[str, str]]
    # One source per origin/transform pair, independent of installation paths.
    sources: dict[str, Source]
    # First authored destination for each unique source, retained for observation.
    paths: dict[str, str]
    # Fixed operations included in each unique output's identity.
    transforms: dict[str, dict]
    # Selected authored roots and their immutable repository resolutions.
    origins: dict
    # Explicit asset-only or complete commitment coverage for this update.
    scope: str
    # Caller-owned local lock destination.
    path: Path
    # Network prohibition retained across planning and acquisition.
    offline: bool
    # Loader-derived outputs by logical identity; recorded, never acquired.
    derived: dict[str, DerivedFile] = field(default_factory=dict)
    # Local card directory where a loader may already have derived payloads.
    directory: Path | None = None

    def summary(self) -> dict:
        return {
            "model": self.model_id,
            "lock": "update-all-variants",
            "lock_scope": self.scope,
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
            ]
            + [
                {
                    "origin": output.origin,
                    "transport": None,
                    "transform": copy.deepcopy(output.transform),
                    "derived": output.summary(),
                }
                for output in self.derived.values()
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
    lock_scope: str = "assets",
) -> LockPlan:
    """Resolves the selected lock scope across all compatible card choices."""
    selections = card.choices()
    if programs is not None:
        selections = [choice for choice in selections if choice["programs"] == programs]
        if not selections:
            raise ManifestError(f"no compatible choices for program set {programs!r}")
    location = document_location(lock_location or sibling_lock(document.location))
    if "://" in location:
        raise ManifestError("lock updates require a local --lockfile")
    path = Path(location)
    if path.name == ".loom-fetch.lock" or (path.exists() and not path.is_file()):
        raise ManifestError("lock destination must be an ordinary file")
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
    paths = {}
    transforms = {}
    derived = {}
    local_inputs = set()
    for choice in selections:
        selected = card.select(**choice)
        outputs = derived_files(selected)
        for name, file in selected.files.items():
            if PurePosixPath(name).parts[0] in _RESERVED_PATHS:
                raise ManifestError(f"reserved installation path {name!r}")
            local = resolver.local_file(file.origin) if name not in outputs else None
            if local is not None:
                local_inputs.add(local)
        for name, file in lock_files(selected, lock_scope).items():
            key = file_key(file.origin, file.transform)
            if name in outputs:
                if derived.setdefault(key, outputs[name]) != outputs[name]:
                    raise ManifestError(
                        f"choices derive {name!r} from different sources; "
                        "give each derived output its own cache path"
                    )
                continue
            if key not in sources:
                sources[key] = resolver.resolve(file.origin, None)
                paths[key] = name
                if file.transform is not None:
                    transforms[key] = file.transform
    sources = _prepare_sources(
        sources, {}, cache, offline=offline, transforms=transforms
    )
    _check_lock_sources(path, document, sources, local_inputs=local_inputs)
    return LockPlan(
        authored["model"]["id"],
        selections,
        sources,
        paths,
        transforms,
        resolver.origins,
        lock_scope,
        path,
        offline,
        derived,
        None if document.remote else Path(document.location).parent,
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


def _check_lock_sources(
    path: Path,
    document: Document,
    sources: dict[str, Source],
    *,
    local_inputs: Iterable[Path] = (),
):
    protected = [Path(document.location)] if not document.remote else []
    protected.extend(local_inputs)
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
    paths: dict[str, str],
    preserve_sources: bool = False,
    present: dict[str, AcquiredFile] | None = None,
) -> dict[str, AcquiredFile]:
    """Joins a bounded transfer batch, deduplicated by logical source identity."""
    acquired = dict(present or {})
    unique = {}
    total = len(
        set(acquired)
        | {
            file_key(source.origin, transforms.get(name))
            for name, source in sources.items()
        }
    )
    cache.progress.emit("acquire", completed_files=len(acquired), total_files=total)
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
                            progress=cache.progress.file(paths[name]),
                        )
                        if transform is not None
                        else executor.submit(
                            cache.acquire,
                            source,
                            identities.get(name),
                            offline=offline,
                            progress=cache.progress.file(paths[name]),
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
                    cache.progress.emit(
                        "acquire", completed_files=len(acquired), total_files=total
                    )
        except BaseException:
            cache.cancel()
            for future in unique.values():
                future.cancel()
            raise
    return acquired


def _previous_lock(path: Path) -> ModelLock | None:
    """Reads a lock about to be replaced; an unreadable one carries nothing."""
    try:
        return ModelLock.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ManifestError):
        return None


def _observe_derivation(
    output: DerivedFile,
    source: FileIdentity,
    directory: Path | None,
    previous: ModelLock | None,
    cache: ContentCache,
) -> DerivedIdentity:
    """Records a derived output's recipe and, when known, its payload bytes.

    Acquisition never derives. Payload bytes come from a loader-published file
    beside a local card, or from the replaced lock when it recorded them for the
    same recipe and source bytes. Otherwise the record has no size or SHA-256.
    """
    size = digest = None
    payload = directory / output.path if directory is not None else None
    if payload is not None and payload.is_file():
        size, digest = file_digest(
            payload,
            cancelled=cache.cancelled,
            progress=ByteProgress(cache.progress.file(output.path), "verify"),
        )
    elif previous is not None:
        prior = previous.files.get(file_key(output.origin, output.transform))
        prior_source = previous.files.get(file_key(output.source_origin))
        if (
            isinstance(prior, DerivedIdentity)
            and prior.size is not None
            and prior.derived in (None, output.facts())
            and isinstance(prior_source, FileIdentity)
            and (prior_source.size, prior_source.sha256) == (source.size, source.sha256)
        ):
            size, digest = prior.size, prior.sha256
    return DerivedIdentity(
        output.origin, output.transform, output.facts(), size, digest
    )


def _installed_derivations(
    plan: FetchPlan, relocated: dict, identities: dict[str, DerivedIdentity]
) -> list[DerivedIdentity]:
    """Derived records for an installed card, whose sources may be relocated."""
    return [
        replace(
            identities.get(name) or DerivedIdentity(output.origin, output.transform),
            derived=output.facts(relocated[output.source]),
        )
        for name, output in plan.derived.items()
    ]


def update_lock(
    plan: LockPlan, cache: ContentCache, *, preserve_sources: bool = False
) -> Path:
    """Atomically replaces one scoped lock after all required source reads succeed."""
    with installation_lock(plan.path.parent):
        acquired = _acquire_sources(
            plan.sources,
            {},
            cache,
            offline=plan.offline,
            transforms=plan.transforms,
            paths=plan.paths,
            preserve_sources=preserve_sources,
        )
        previous = _previous_lock(plan.path)
        derived = [
            _observe_derivation(
                output,
                acquired[file_key(output.source_origin)].identity,
                plan.directory,
                previous,
                cache,
            )
            for output in plan.derived.values()
        ]
        document = ModelLock.snapshot(
            plan.origins,
            [item.identity for item in acquired.values()] + derived,
            scope=plan.scope,
        )
        if cache.cancelled.is_set():
            raise FetchError("model acquisition cancelled")
        cache.progress.emit("install", completed_files=0, total_files=1)
        atomic_write(plan.path, (json.dumps(document, indent=2) + "\n").encode("utf-8"))
        cache.progress.emit("complete", completed_files=1, total_files=1)
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
        if record["path"] in plan.derived:
            # Runtime coordinates only: the loader derives this from its source.
            return record
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
            for voice in service.get("voices", {}).values():
                voice["file"] = file(voice["file"])
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


def _check_provenance(path: Path, expected: dict) -> None:
    if not path.exists():
        return
    actual = parse_json(path.read_text(encoding="utf-8"))
    # An authored URL and its shorthand can name the same pinned publication.
    # Keep the original receipt, but compare its immutable location and revision
    # instead of the user's spelling. Card identity and all other facts still match.
    catalog = expected.get("catalog")
    if isinstance(catalog, dict) and "location" in catalog:
        expected = {
            **expected,
            "catalog": {
                key: value for key, value in catalog.items() if key != "source"
            },
        }
        if isinstance(actual, dict) and isinstance(actual.get("catalog"), dict):
            actual["catalog"].pop("source", None)
    if actual != expected:
        raise FetchError(
            f"conflicting installation document: {path}; choose another output"
        )


def _check_installation_lock(path: Path, document: dict) -> None:
    if not path.exists():
        return
    actual = ModelLock.parse(path.read_text(encoding="utf-8"))
    canonical = ModelLock.snapshot(
        actual.origins, list(actual.files.values()), scope=actual.scope
    )
    if canonical != document:
        raise FetchError(
            f"conflicting installation document: {path}; choose another output"
        )


def _check_outputs(
    plan: FetchPlan, directory: Path, card: dict, *, cancelled=None, progress=None
) -> dict[str, AcquiredFile]:
    _check_document(_destination(directory, "model.json"), card)
    _destination(directory, "model.json.lock")
    if plan.provenance is not None:
        _check_provenance(_destination(directory, ".loom-fetch.json"), plan.provenance)
    present = {}
    for name, source in plan.sources.items():
        destination = _destination(directory, name)
        if not destination.exists():
            continue
        counter = ByteProgress(progress.file(name) if progress else None, "verify")
        identity = plan.identities.get(name)
        if identity is None:
            editable = plan.selection.files[name].roles <= {"program", "auxiliary"}
            if not editable and (
                plan.selection.files[name].transform is not None
                or source.transport["kind"] != "local"
            ):
                raise FetchError(
                    f"cannot verify existing destination {destination} "
                    "before transfer; "
                    "use a matching lock or choose another output"
                )
            if not editable and not os.path.samefile(
                source.transport["path"], destination
            ):
                # An independent local copy can be verified before transfer too.
                # Without a lock, its expected identity must come from the source.
                size, digest = file_digest(
                    Path(source.transport["path"]),
                    cancelled=cancelled,
                    progress=counter,
                )
                identity = FileIdentity(source.origin, size, digest, source.resolved)
        if identity is None:
            size, digest = file_digest(
                destination, cancelled=cancelled, progress=counter
            )
            identity = FileIdentity(source.origin, size, digest, source.resolved)
        else:
            identity.verify(destination, cancelled=cancelled, progress=counter)
        present[name] = AcquiredFile(destination, identity)
    return present


def check_output(plan: FetchPlan, output: Path) -> None:
    """Checks an explicit output without creating it or acquiring payloads."""
    card, relocated, roots = _installation_card(plan)
    _check_outputs(plan, output.absolute(), card)
    _check_lock_output(plan, output.absolute(), relocated, roots)


def _check_lock_output(plan: FetchPlan, directory: Path, relocated: dict, roots: dict):
    path = directory / "model.json.lock"
    if plan.lock_mode == "none" or not path.exists():
        return
    if plan.unverified_parameters:
        raise FetchError(
            "choose another output for --no-verify-parameters; "
            "this installation already records parameter identities"
        )
    if plan.lock_mode == "update":
        raise FetchError(
            "an installation lock already exists; choose another output "
            "for --update-lock"
        )
    expected = ModelLock.snapshot(
        roots,
        [
            replace(identity, origin=relocated[name])
            for name, identity in plan.identities.items()
        ]
        + _installed_derivations(plan, relocated, plan.derived_identities),
        scope=plan.lock_scope,
    )
    _check_installation_lock(path, expected)


def _try_share(source: Path, destination: Path, *, automatic: bool, progress) -> bool:
    try:
        share_file(source, destination)
        return True
    except SharingUnavailable as error:
        if not automatic:
            raise
        progress.emit(
            "warning",
            message=f"{error}. Automatic materialization will use independent "
            "copies for this source/output path, requiring additional disk space.",
        )
        return False


def _probe_sharing(
    plan: FetchPlan,
    cache: ContentCache,
    directory: Path,
    sources: dict[str, Source],
    *,
    automatic: bool,
) -> set[str]:
    shared = set()
    with ExitStack() as stack:
        probes = {}
        destinations = {}
        results = {}
        for name, source in sources.items():
            transform = plan.selection.files[name].transform
            kind = source.transport["kind"]
            if transform is None and kind == "local":
                path = Path(source.transport["path"])
            else:
                identity = plan.identities.get(name)
                path = cache.blob_path(identity.sha256) if identity else None
                if path is None or not path.is_file():
                    root = (
                        cache.huggingface.cache_directory
                        if kind == "hf" and transform is None
                        else cache.root / "blob" / "sha256"
                    )
                    if root not in probes:
                        root.mkdir(parents=True, exist_ok=True)
                        temporary = stack.enter_context(
                            tempfile.TemporaryDirectory(prefix=".loom-fetch-", dir=root)
                        )
                        probes[root] = Path(temporary) / "probe"
                        probes[root].write_bytes(b"probe")
                    path = probes[root]
            # A subdirectory may be a mount point with different capabilities.
            # st_dev also cannot distinguish local and UNC spellings on Windows.
            parent = (directory / name).parent
            if parent not in destinations:
                parent.mkdir(parents=True, exist_ok=True)
                temporary = stack.enter_context(
                    tempfile.TemporaryDirectory(prefix=".loom-fetch-", dir=parent)
                )
                destinations[parent] = Path(temporary) / "shared"
            key = (path, parent)
            if key not in results:
                destination = destinations[parent]
                results[key] = _try_share(
                    path,
                    destination,
                    automatic=automatic,
                    progress=cache.progress.file(name),
                )
                if results[key]:
                    destination.unlink()
            if results[key]:
                shared.add(name)
    return shared


def install(
    plan: FetchPlan,
    cache: ContentCache,
    output: Path,
    *,
    materialize: str = "auto",
    preserve_sources: bool = False,
) -> Path:
    """Publishes a card only after its complete selected acquisition succeeds."""
    if cache.cancelled.is_set():
        raise FetchError("model acquisition cancelled")
    if materialize not in ("auto", "share", "copy"):
        raise ManifestError("materialize must be 'auto', 'share', or 'copy'")
    shared = {
        name: source
        for name, source in plan.sources.items()
        if materialize == "share"
        or (
            materialize == "auto"
            and not plan.selection.files[name].roles & {"program", "auxiliary"}
        )
    }
    directory = output.absolute()
    card, relocated, roots = _installation_card(plan)
    if plan.update_path is not None:
        for name in (*plan.sources, *_RESERVED_PATHS):
            if plan.update_path == directory / name:
                raise ManifestError(
                    "lock update destination overlaps installation content"
                )
    with installation_lock(directory):
        present = _check_outputs(
            plan, directory, card, cancelled=cache.cancelled, progress=cache.progress
        )
        _check_lock_output(plan, directory, relocated, roots)
        if shared:
            shared = _probe_sharing(
                plan,
                cache,
                directory,
                {
                    name: source
                    for name, source in shared.items()
                    if name not in present
                },
                automatic=materialize == "auto",
            )
        required = lock_files(plan.selection, plan.lock_scope)
        acquired = _acquire_sources(
            {
                name: source
                for name, source in plan.sources.items()
                if name not in present
                or (
                    preserve_sources
                    and plan.selection.files[name].transform is not None
                )
            },
            plan.identities,
            cache,
            offline=plan.offline,
            transforms={
                name: file.transform
                for name, file in plan.selection.files.items()
                if file.transform is not None
            },
            paths={name: name for name in plan.sources},
            preserve_sources=preserve_sources,
            present={
                file_key(
                    plan.sources[name].origin, plan.selection.files[name].transform
                ): item
                for name, item in present.items()
                if name in required
                and (
                    not preserve_sources or plan.selection.files[name].transform is None
                )
            },
        )
        derived = plan.derived_identities
        if plan.lock_mode == "update":
            previous = (
                _previous_lock(plan.update_path)
                if plan.update_path is not None
                else None
            )
            derived = {
                name: _observe_derivation(
                    output,
                    acquired[file_key(output.source_origin)].identity,
                    None
                    if plan.document.remote
                    else Path(plan.document.location).parent,
                    previous,
                    cache,
                )
                for name, output in plan.derived.items()
            }
        identities = [
            replace(
                acquired[
                    file_key(source.origin, plan.selection.files[name].transform)
                ].identity,
                origin=relocated[name],
            )
            for name, source in plan.sources.items()
            if name in required
        ] + _installed_derivations(plan, relocated, derived)
        installation_lock_document = (
            ModelLock.snapshot(roots, identities, scope=plan.lock_scope)
            if plan.lock_mode != "none"
            else None
        )
        if installation_lock_document is not None:
            _check_installation_lock(
                directory / "model.json.lock", installation_lock_document
            )
        published = []
        try:
            cache.progress.emit(
                "install", completed_files=len(present), total_files=len(plan.sources)
            )
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
                    cache.progress.file(name).emit("install")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    item = acquired[
                        file_key(source.origin, plan.selection.files[name].transform)
                    ]
                    if name not in shared or not _try_share(
                        item.path,
                        destination,
                        automatic=materialize == "auto",
                        progress=cache.progress.file(name),
                    ):
                        try:
                            copy_file(
                                item.path,
                                destination,
                                item.identity.size,
                                cancelled=cache.cancelled,
                                progress=cache.progress.file(name),
                            )
                        except OSError as error:
                            raise FetchError(
                                f"cannot copy {name} to {destination}: {error}"
                            ) from error
                    published.append(destination)
                    cache.progress.emit(
                        "install",
                        completed_files=len(present) + len(published),
                        total_files=len(plan.sources),
                    )
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
                    publish_file(staged, destination)
                    published.append(destination)
            if plan.update_path is not None:
                document = ModelLock.snapshot(
                    plan.origins,
                    [
                        acquired[file_key(file.origin, file.transform)].identity
                        for name, file in required.items()
                        if name not in plan.derived
                    ]
                    + list(derived.values()),
                    scope=plan.lock_scope,
                )
                atomic_write(
                    plan.update_path,
                    (json.dumps(document, indent=2) + "\n").encode("utf-8"),
                )
        except BaseException:
            for path in reversed(published):
                path.unlink()
            raise
    cache.progress.emit(
        "complete", completed_files=len(plan.sources), total_files=len(plan.sources)
    )
    return directory / "model.json"
