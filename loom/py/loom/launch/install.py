# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Acquisition of published profiles through the SDK's own fetch command.

Acquisition stays with `loom-fetch`: it owns locks, caches, materialization and
gated-repository handling, and it is what the publisher tested. The launcher
adds the parts a launcher needs on top: a default destination, a captured log,
streamed progress, and the registry that later launches resolve names against.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from loom.launch.errors import InstallError, LauncherError
from loom.launch.hub import Hub, HubError, HubNotFound
from loom.launch.scenario import (
    DEPLOYMENT_DOCUMENT,
    DEVICES_DOCUMENT,
    Scenario,
    is_scenario,
)
from loom.launch.store import (
    InstalledModel,
    InstallRegistry,
    read_json,
    state_directory,
)
from loom.launch.values import dict_value, list_value, str_value

#: What a fetch result reports, in the order the launcher explains it.
INSTALL_DOCUMENT = "model.json"


def fetch_command(explicit: str | None = None) -> list[str]:
    """Finds the acquisition command, preferring an explicit installation."""
    if explicit:
        return [explicit]
    from_environment = os.environ.get("LOOM_FETCH")
    if from_environment:
        return [from_environment]
    found = shutil.which("loom-fetch")
    if found:
        return [found]
    # A development checkout can run the module without installing scripts.
    sibling = Path(sys.executable).parent / "loom-fetch"
    if sibling.is_file():
        return [str(sibling)]
    try:
        import loom.model.fetch_main  # noqa: F401
    except ImportError as exc:
        raise InstallError(
            "loom-fetch was not found: install the Loom SDK "
            '(`pip install "hrx-loom[hf]"`), or pass --loom-fetch=PATH'
        ) from exc
    return [sys.executable, "-m", "loom.model.fetch_main"]


#: How an install came to carry (or not) the profile's own deployment and
#: device documents, which `loom-fetch` does not install because the card does
#: not declare them.
DOCUMENTS_PRESENT = "present"  # already in the installed directory
DOCUMENTS_CACHE = "cache"  # copied from the local hub cache's snapshot
DOCUMENTS_HUB = "hub"  # read from the hub at the pinned revision
DOCUMENTS_ABSENT = "absent"  # the profile publishes neither (the common case)
DOCUMENTS_OFFLINE = "offline"  # --offline, and nothing local to use
DOCUMENTS_NOT_HUB = "not-hub"  # a URL or another provider: nothing to look up
DOCUMENTS_INCOMPLETE = "incomplete"  # only one of the two is published
DOCUMENTS_INVALID = "invalid"  # published, but not a servable scenario
DOCUMENTS_FAILED = "failed"  # the hub could not be read; the install stands

_INSTALLED_STATUSES = (DOCUMENTS_PRESENT, DOCUMENTS_CACHE, DOCUMENTS_HUB)


@dataclass(frozen=True)
class ScenarioDocuments:
    """What became of `deployment.json` and `devices.json` for one install."""

    status: str = DOCUMENTS_ABSENT
    #: The revision the documents were read at (or looked up at).
    revision: str | None = None
    #: Where they came from: a cache directory or the hub URL prefix.
    source: str | None = None
    #: The scenario they describe, once installed.
    scenario: Scenario | None = None
    #: What the user is told when the documents were expected but not installed.
    warning: str | None = None

    @property
    def installed(self) -> bool:
        return self.status in _INSTALLED_STATUSES

    def summary(self) -> str | None:
        """The `install` summary line for an installed scenario."""
        if self.scenario is None or not self.installed:
            return None
        revision = self.revision or "main"
        origin = {
            DOCUMENTS_PRESENT: "shipped with the card",
            DOCUMENTS_CACHE: f"from the local hub cache at {revision}",
            DOCUMENTS_HUB: f"from the hub at {revision}",
        }[self.status]
        return (
            f"deployment {self.scenario.name} on {self.scenario.device_text} "
            f"({DEPLOYMENT_DOCUMENT} + {DEVICES_DOCUMENT} {origin})"
        )

    def describe(self) -> dict:
        return {
            "status": self.status,
            "scenario": self.installed,
            "revision": self.revision,
            "source": self.source,
            "devices": self.scenario.device_text if self.scenario else None,
            "deployment": self.scenario.name if self.scenario else None,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class InstallResult:
    """One completed acquisition."""

    repo: str
    path: Path
    preset: str | None = None
    revision: str | None = None
    seconds: float = 0.0
    command: tuple[str, ...] = ()
    log_path: Path | None = None
    documents: ScenarioDocuments = field(default_factory=ScenarioDocuments)

    @property
    def card_path(self) -> Path:
        return self.path / INSTALL_DOCUMENT

    def describe(self) -> dict:
        return {
            "repo": self.repo,
            "path": str(self.path),
            "card": str(self.card_path),
            "preset": self.preset,
            "revision": self.revision,
            "seconds": round(self.seconds, 3),
            "command": list(self.command),
            "log": str(self.log_path) if self.log_path else None,
            "scenario_documents": self.documents.describe(),
        }


def default_models_directory(state_root: Path | None = None) -> Path:
    """Where the launcher installs packages unless told otherwise."""
    return state_directory(state_root) / "models"


def install(
    repo: str,
    output: str | Path,
    *,
    preset: str | None = None,
    revision: str | None = None,
    materialize: str = "copy",
    offline: bool = False,
    explicit_fetch: str | None = None,
    log_path: Path | None = None,
    on_line: Callable[[str], None] | None = None,
    timeout: float | None = None,
) -> InstallResult:
    """Installs one published profile into `output`."""
    command = [
        *fetch_command(explicit_fetch),
        selector(repo, preset, revision),
        f"--output={Path(output).expanduser()}",
        f"--materialize={materialize}",
        "--no-input",
        "--json",
    ]
    if offline:
        command.append("--offline")
    started = time.monotonic()
    log_file = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")
    stdout_lines: list[str] = []
    try:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise InstallError(f"cannot run {command[0]}: {exc}") from exc
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\n")
            stdout_lines.append(line)
            if log_file is not None:
                log_file.write(line + "\n")
                log_file.flush()
            if on_line is not None:
                on_line(line)
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise InstallError(f"acquisition of {repo} timed out") from exc
    finally:
        if log_file is not None:
            log_file.close()
    elapsed = time.monotonic() - started
    if return_code != 0:
        tail = "\n".join(stdout_lines[-8:])
        raise InstallError(
            f"loom-fetch failed for {repo} with exit code {return_code}:\n{tail}"
        )
    path = Path(output).expanduser()
    if not (path / INSTALL_DOCUMENT).is_file():
        raise InstallError(f"{repo} did not install a {INSTALL_DOCUMENT} under {path}")
    result = _parse_result(stdout_lines)
    preset = preset or str_value(result.get("preset"))
    if preset is None:
        # The installed card retains the selection fetch made.
        card = read_json(path / INSTALL_DOCUMENT)
        preset = str_value(card.get("preset"))
    return InstallResult(
        repo=repo,
        path=path,
        preset=preset,
        revision=result_revision(result, repo) or revision,
        seconds=elapsed,
        command=tuple(command),
        log_path=log_path,
    )


def selector(repo: str, preset: str | None, revision: str | None) -> str:
    """The fetch selector: `huggingface:<repo>[:<preset>][@<revision>]`."""
    if repo.startswith(("huggingface:", "github:", "http://", "https://")):
        return repo
    source = f"huggingface:{repo}"
    if preset:
        source += f":{preset}"
    if revision:
        source += f"@{revision}"
    return source


def result_revision(document: dict, repo: str) -> str | None:
    """The revision a fetch result pinned: a top-level `revision`, else the
    first hub transport revision of the installed repository."""
    revision = str_value(document.get("revision"))
    if revision:
        return revision
    fallback = None
    for entry in list_value(document.get("files")):
        transport = dict_value(dict_value(entry).get("transport"))
        found = str_value(transport.get("revision"))
        if found is None:
            continue
        if str_value(transport.get("repository")) == repo:
            return found
        fallback = fallback or found
    return fallback


def install_and_record(
    repo: str,
    *,
    output: str | Path | None = None,
    preset: str | None = None,
    revision: str | None = None,
    materialize: str = "copy",
    offline: bool = False,
    explicit_fetch: str | None = None,
    state_root: Path | None = None,
    on_line: Callable[[str], None] | None = None,
    hub: Hub | None = None,
) -> tuple[InstallResult, InstalledModel]:
    """Installs a profile and records it for later launches."""
    root = state_directory(state_root)
    destination = (
        Path(output).expanduser()
        if output is not None
        else default_models_directory(state_root) / repo.replace("/", "-")
    )
    result = install(
        repo,
        destination,
        preset=preset,
        revision=revision,
        materialize=materialize,
        offline=offline,
        explicit_fetch=explicit_fetch,
        log_path=root / "logs" / "fetch.log",
        on_line=on_line,
    )
    documents = acquire_scenario_documents(
        result.path,
        repo,
        result.revision,
        offline=offline,
        hub=hub if hub is not None else Hub(),
    )
    result = replace(result, documents=documents)
    model = InstalledModel(
        repo=repo,
        path=str(result.path),
        preset=result.preset,
        revision=result.revision,
        installed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        size_bytes=directory_size(result.path),
        scenario=True if documents.installed else None,
        scenario_revision=documents.revision if documents.installed else None,
    )
    InstallRegistry(root).record(model)
    return result, model


def _hub_repository(repo: str) -> str | None:
    """The hub repository an install selector names, when it names one."""
    if repo.startswith("huggingface:"):
        repo = repo.removeprefix("huggingface:")
    elif repo.startswith("github:") or "://" in repo:
        return None
    bare = repo.split("@", 1)[0].replace(":", "/")
    parts = bare.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return f"{parts[0]}/{parts[1]}"


def hub_cache_root() -> Path | None:
    """The local Hugging Face hub cache (`huggingface_hub`'s own resolution)."""
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache) if cache else Path.home() / ".cache"
    return base / "huggingface" / "hub"


def _cached_snapshot(cache: Path, repo: str, revision: str | None) -> Path | None:
    """The cached snapshot of `repo` at `revision` (a commit, or a ref the
    cache resolved), when one exists."""
    root = cache / f"models--{repo.replace('/', '--')}"
    revision = revision or "main"
    commit = revision
    if not (
        len(revision) == 40 and all(c in "0123456789abcdefABCDEF" for c in revision)
    ):
        try:
            commit = (root / "refs" / revision).read_text(encoding="utf-8").strip()
        except OSError:
            return None
    snapshot = root / "snapshots" / commit
    return snapshot if snapshot.is_dir() else None


def _place_documents(directory: Path, deployment: bytes, devices: bytes) -> Scenario:
    """Writes both documents into `directory`, then checks they describe a
    servable scenario; nothing is left behind when they do not."""
    bodies = ((DEPLOYMENT_DOCUMENT, deployment), (DEVICES_DOCUMENT, devices))
    for name, body in bodies:
        try:
            document = json.loads(body)
        except json.JSONDecodeError:
            document = None
        if not isinstance(document, dict):
            raise InstallError(f"{name} is not a JSON object")

    def remove() -> None:
        for name, _body in bodies:
            (directory / name).unlink(missing_ok=True)

    for name, body in bodies:
        staged = directory / f".{name}.partial"
        try:
            staged.write_bytes(body)
            os.replace(staged, directory / name)
        except OSError as exc:
            staged.unlink(missing_ok=True)
            remove()
            raise InstallError(f"cannot write {directory / name}: {exc}") from exc
    try:
        scenario = Scenario.detect(directory)
    except LauncherError:
        remove()
        raise
    if scenario is None:
        remove()
        raise InstallError("the documents did not land")
    return scenario


def acquire_scenario_documents(
    directory: str | Path,
    repo: str,
    revision: str | None,
    *,
    offline: bool,
    hub: Hub,
    cache: Path | None = None,
) -> ScenarioDocuments:
    """Gives an installed profile its own `deployment.json` and `devices.json`
    when the profile publishes them.

    They come from the installed directory itself, the local hub cache's
    snapshot of the pinned revision, or the hub at that revision (never when
    `offline`). Nothing here fails the install: a profile without the documents
    is the common case, and a hub failure is reported in `warning` while the
    installed package stays as `loom-fetch` left it.
    """
    directory = Path(directory)
    if is_scenario(directory):
        try:
            scenario = Scenario.detect(directory)
        except LauncherError as exc:
            return ScenarioDocuments(
                status=DOCUMENTS_INVALID,
                revision=revision,
                warning=f"{directory} carries {DEPLOYMENT_DOCUMENT} and "
                f"{DEVICES_DOCUMENT}, but they do not describe a servable scenario: "
                f"{exc}",
            )
        return ScenarioDocuments(
            status=DOCUMENTS_PRESENT,
            revision=revision,
            source=str(directory),
            scenario=scenario,
        )
    repository = _hub_repository(repo)
    if repository is None:
        return ScenarioDocuments(status=DOCUMENTS_NOT_HUB, revision=revision)
    label = f"{repository}@{revision or 'main'}"
    cache = cache if cache is not None else hub_cache_root()
    snapshot = _cached_snapshot(cache, repository, revision) if cache else None
    if snapshot is not None:
        try:
            scenario = _place_documents(
                directory,
                (snapshot / DEPLOYMENT_DOCUMENT).read_bytes(),
                (snapshot / DEVICES_DOCUMENT).read_bytes(),
            )
        except (OSError, LauncherError):
            pass
        else:
            return ScenarioDocuments(
                status=DOCUMENTS_CACHE,
                revision=revision,
                source=str(snapshot),
                scenario=scenario,
            )
    if offline:
        return ScenarioDocuments(status=DOCUMENTS_OFFLINE, revision=revision)

    def read(name: str) -> bytes | None:
        try:
            return hub.raw(repository, name, revision)
        except HubNotFound:
            return None

    try:
        deployment = read(DEPLOYMENT_DOCUMENT)
        devices = read(DEVICES_DOCUMENT)
    except (HubError, OSError) as exc:
        return ScenarioDocuments(
            status=DOCUMENTS_FAILED,
            revision=revision,
            warning=f"could not check {label} for {DEPLOYMENT_DOCUMENT} and "
            f"{DEVICES_DOCUMENT}: {exc}. The install itself succeeded, but if the "
            "profile is a GPU+NPU (or other multi-device) scenario, `serve` will run "
            f"a plain single-device deployment until both files are in {directory}; "
            "rerun the install when the hub is reachable, or copy them from the "
            "repository at that revision",
        )
    if deployment is None and devices is None:
        return ScenarioDocuments(status=DOCUMENTS_ABSENT, revision=revision)
    if deployment is None or devices is None:
        has, lacks = (
            (DEPLOYMENT_DOCUMENT, DEVICES_DOCUMENT)
            if deployment is not None
            else (DEVICES_DOCUMENT, DEPLOYMENT_DOCUMENT)
        )
        return ScenarioDocuments(
            status=DOCUMENTS_INCOMPLETE,
            revision=revision,
            warning=f"{label} publishes {has} but not {lacks}; installed as an "
            "ordinary profile (a scenario needs both)",
        )
    try:
        scenario = _place_documents(directory, deployment, devices)
    except LauncherError as exc:
        return ScenarioDocuments(
            status=DOCUMENTS_INVALID,
            revision=revision,
            warning=f"{label} publishes {DEPLOYMENT_DOCUMENT} and {DEVICES_DOCUMENT}, "
            f"but they do not describe a servable scenario ({exc}); installed as an "
            "ordinary profile",
        )
    return ScenarioDocuments(
        status=DOCUMENTS_HUB,
        revision=revision,
        source=f"{hub.endpoint}/{repository}",
        scenario=scenario,
    )


def directory_size(path: Path) -> int:
    """Sums regular file sizes, ignoring what cannot be read."""
    total = 0
    for root, _directories, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def _parse_result(lines: list[str]) -> dict:
    """Reads the JSON result fetch reports on stdout, when it reports one.

    `--json` prints one document, but progress and file lists can precede it, so
    every line that opens an object starts a candidate document.
    """
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("{"):
            continue
        candidate = "\n".join(lines[index:]).strip()
        try:
            document = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(document, dict):
            return document
    return {}
