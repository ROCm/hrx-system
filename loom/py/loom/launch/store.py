# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Where the launcher keeps its runs, installs, cached documents and results.

Everything persistent lives under one state directory (`LOOM_LAUNCH_STATE`, by
default `~/.local/state/loom-launch`), so a launcher installation owns no data
in the model directories it serves and no evidence in a scratch filesystem.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from loom.launch.values import dict_value, human_bytes, str_value

DEFAULT_STATE_DIRECTORY = Path.home() / ".local" / "state" / "loom-launch"


def state_directory(explicit: str | Path | None = None) -> Path:
    """Resolves the launcher's state directory, creating nothing.

    The result is absolute. Run records, deployment documents and the install
    registry all persist paths derived from it, and the native server resolves a
    relative model path against the deployment document's own directory: a
    relative state directory would silently point a later command elsewhere.
    """
    if explicit:
        return Path(os.path.abspath(Path(explicit).expanduser()))
    from_environment = os.environ.get("LOOM_LAUNCH_STATE")
    if from_environment:
        return Path(os.path.abspath(Path(from_environment).expanduser()))
    return DEFAULT_STATE_DIRECTORY


def write_json(path: Path, document: object) -> Path:
    """Publishes a JSON document whole, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def read_json(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


class JsonlWriter:
    """An append-only event log; one record per line, flushed as it is written."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"-{os.getpid()}"


@dataclass
class RunDirectory:
    """One `loom-launch serve` run: its command, logs, events and requests."""

    root: Path
    run_id: str
    metadata: dict = field(default_factory=dict)

    @property
    def serve_log(self) -> Path:
        return self.root / "serve.log"

    @property
    def event_log(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def request_log(self) -> Path:
        return self.root / "requests.jsonl"

    @property
    def deployment_path(self) -> Path:
        return self.root / "serve.json"

    @property
    def metadata_path(self) -> Path:
        return self.root / "meta.json"

    def events(self) -> JsonlWriter:
        return JsonlWriter(self.event_log)

    def requests(self) -> JsonlWriter:
        return JsonlWriter(self.request_log)

    def save(self) -> None:
        write_json(self.metadata_path, self.metadata)

    def read_metadata(self) -> dict:
        return read_json(self.metadata_path)

    def describe(self) -> dict:
        return {
            "id": self.run_id,
            "directory": str(self.root),
            "serve_log": str(self.serve_log),
            "event_log": str(self.event_log),
            "request_log": str(self.request_log),
            "metadata": dict(self.metadata),
        }


class RunStore:
    """Creates and enumerates run directories under one state directory."""

    def __init__(self, root: Path):
        self.root = root

    def create(
        self, run_id: str | None = None, metadata: dict | None = None
    ) -> RunDirectory:
        run_id = run_id or new_run_id()
        directory = RunDirectory(
            root=self.root / "runs" / run_id,
            run_id=run_id,
            metadata=dict(metadata or {}),
        )
        directory.root.mkdir(parents=True, exist_ok=True)
        directory.save()
        return directory

    def open(self, run_id: str) -> RunDirectory:
        return RunDirectory(root=self.root / "runs" / run_id, run_id=run_id)

    def latest(self) -> RunDirectory | None:
        directory = self.root / "runs"
        if not directory.is_dir():
            return None
        candidates = sorted(
            (path for path in directory.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        if not candidates:
            return None
        return RunDirectory(root=candidates[-1], run_id=candidates[-1].name)

    def list(self) -> list[RunDirectory]:
        directory = self.root / "runs"
        if not directory.is_dir():
            return []
        return [
            RunDirectory(root=path, run_id=path.name)
            for path in sorted(
                (path for path in directory.iterdir() if path.is_dir()),
                key=lambda path: path.name,
                reverse=True,
            )
        ]


@dataclass(frozen=True)
class InstalledModel:
    """One installed package the launcher can serve without the hub."""

    repo: str
    path: str
    preset: str | None = None
    revision: str | None = None
    installed_at: str | None = None
    size_bytes: int | None = None
    #: Set when the installed directory carries its own `deployment.json` and
    #: `devices.json`; absent for an ordinary profile.
    scenario: bool | None = None
    #: The hub revision those two documents were read at.
    scenario_revision: str | None = None

    @property
    def card_path(self) -> Path:
        return Path(self.path) / "model.json"

    def describe(self) -> dict:
        return {
            "repo": self.repo,
            "path": self.path,
            "card": str(self.card_path),
            "preset": self.preset,
            "revision": self.revision,
            "installed_at": self.installed_at,
            "size_bytes": self.size_bytes,
            "size": human_bytes(self.size_bytes),
            "scenario": bool(self.scenario),
            "scenario_revision": self.scenario_revision,
        }


class InstallRegistry:
    """The launcher's index of installed packages."""

    def __init__(self, root: Path):
        self.root = root
        self.path = root / "installs.json"

    def _document(self) -> dict:
        document = read_json(self.path)
        document.setdefault("installs", {})
        return document

    def list(self) -> list[InstalledModel]:
        installs = dict_value(self._document().get("installs"))
        models = []
        for repo, entry in sorted(installs.items()):
            entry = dict_value(entry)
            path = self._recorded_path(str_value(entry.get("path")))
            if not path:
                continue
            models.append(
                InstalledModel(
                    repo=str_value(entry.get("repo")) or repo,
                    path=path,
                    preset=str_value(entry.get("preset")),
                    revision=str_value(entry.get("revision")),
                    installed_at=str_value(entry.get("installed_at")),
                    size_bytes=entry.get("size_bytes")
                    if isinstance(entry.get("size_bytes"), int)
                    else None,
                    scenario=entry.get("scenario")
                    if isinstance(entry.get("scenario"), bool)
                    else None,
                    scenario_revision=str_value(entry.get("scenario_revision")),
                )
            )
        return models

    def _recorded_path(self, path: str) -> str:
        """Resolves a recorded install path, tolerating a relative one.

        Records written before the state directory was made absolute stored the
        path as it was typed, so a relative one is resolved against the directory
        holding the state root rather than against this process's own directory.
        """
        if not path:
            return path
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        bases = [Path.cwd(), *Path(self.root).absolute().parents]
        for base in bases:
            resolved = base / candidate
            if resolved.is_dir():
                return str(resolved)
        return str(Path.cwd() / candidate)

    def get(self, repo: str) -> InstalledModel | None:
        for model in self.list():
            if model.repo == repo:
                return model
        return None

    def find_by_path(self, path: str | Path) -> InstalledModel | None:
        resolved = Path(path).expanduser().resolve()
        for model in self.list():
            if Path(model.path).expanduser().resolve() == resolved:
                return model
        return None

    def record(self, model: InstalledModel) -> None:
        document = self._document()
        document["installs"][model.repo] = {
            "repo": model.repo,
            "path": model.path,
            "preset": model.preset,
            "revision": model.revision,
            "installed_at": model.installed_at,
            "size_bytes": model.size_bytes,
        }
        entry = document["installs"][model.repo]
        if model.scenario is not None:
            entry["scenario"] = model.scenario
        if model.scenario_revision is not None:
            entry["scenario_revision"] = model.scenario_revision
        write_json(self.path, document)

    def forget(self, repo: str) -> bool:
        document = self._document()
        if repo not in document["installs"]:
            return False
        del document["installs"][repo]
        write_json(self.path, document)
        return True


class ProfileCache:
    """Cached profile documents, so listing and showing work offline."""

    def __init__(self, root: Path, *, max_age_seconds: float = 6 * 3600):
        self.root = root / "cache" / "profiles"
        self.max_age_seconds = max_age_seconds

    def _path(self, repo: str) -> Path:
        return self.root / (repo.replace("/", "-") + ".json")

    def get(self, repo: str) -> dict | None:
        path = self._path(repo)
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return None
        if age > self.max_age_seconds:
            return None
        document = read_json(path)
        return document or None

    def put(self, repo: str, documents: dict) -> None:
        write_json(self._path(repo), documents)

    def clear(self) -> int:
        if not self.root.is_dir():
            return 0
        removed = 0
        for path in self.root.glob("*.json"):
            path.unlink()
            removed += 1
        return removed


class BenchmarkStore:
    """Stored local benchmark runs, one JSON record each, plus an index."""

    def __init__(self, root: Path):
        self.root = root / "benchmarks"
        self.index_path = self.root / "index.jsonl"

    def save(self, record: dict) -> Path:
        slug = str_value(record.get("profile")) or "unknown"
        slug = slug.replace("/", "-")
        run_id = str_value(record.get("id")) or new_run_id()
        record["id"] = run_id
        path = self.root / slug / (run_id + ".json")
        write_json(path, record)
        line = {
            "id": run_id,
            "profile": record.get("profile"),
            "suite": record.get("suite"),
            "started_at": record.get("started_at"),
            "path": str(path),
            "ok": record.get("ok"),
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "a", encoding="utf-8") as index:
            index.write(json.dumps(line, separators=(",", ":")) + "\n")
        return path

    def index(self) -> list[dict]:
        try:
            text = self.index_path.read_text(encoding="utf-8")
        except OSError:
            return []
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def list(self, profile: str | None = None) -> list[dict]:
        records = [record for record in self.index() if record.get("path")]
        if profile:
            records = [record for record in records if record.get("profile") == profile]
        return records

    def latest(self, profile: str | None = None) -> dict | None:
        records = self.list(profile)
        if not records:
            return None
        path = Path(str(records[-1]["path"]))
        document = read_json(path)
        return document or None

    def load(self, run_id: str) -> dict | None:
        for record in self.index():
            if record.get("id") == run_id and record.get("path"):
                document = read_json(Path(str(record["path"])))
                return document or None
        return None
