# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Supervision of one native server, its logs and its instrumentation proxy.

The launcher owns process lifetime, readiness and the run directory; the native
executable owns devices, compilation and inference. Every line the server writes
is kept, its JSONL events are indexed separately, and the client-facing endpoint
is the instrumentation proxy unless the user asks for the server's own address.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from loom.launch.client import Client
from loom.launch.deployment import (
    ContextSettings,
    deployment_document,
    write_deployment,
)
from loom.launch.errors import LaunchError
from loom.launch.hardware import Hardware, detect
from loom.launch.metrics import Metrics
from loom.launch.profile import Profile
from loom.launch.proxy import Proxy, Upstream
from loom.launch.scenario import Scenario
from loom.launch.store import JsonlWriter, RunDirectory, state_directory, write_json
from loom.launch.values import dict_value, list_value, str_value

SERVE_BINARY = "loom-serve"
CURRENT_RECORD = "current.json"


def serve_command(explicit: str | None = None) -> str:
    """Finds the native server, preferring an explicit installation."""
    if explicit:
        return explicit
    from_environment = os.environ.get("LOOM_SERVE")
    if from_environment:
        return from_environment
    found = shutil.which(SERVE_BINARY)
    if found:
        return found
    sibling = Path(sys.executable).parent / SERVE_BINARY
    if sibling.is_file():
        return str(sibling)
    raise LaunchError(
        f"{SERVE_BINARY} was not found: install the Loom SDK "
        '(`pip install "hrx-loom[hf]"`), or pass --loom-serve=PATH'
    )


@dataclass(frozen=True)
class LaunchOptions:
    """Everything a launch needs beyond the profile itself."""

    name: str | None = None
    device: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    workers: int | None = None
    pool_backing: str | None = None
    memory_bytes: int | None = None
    proxy: bool = True
    proxy_port: int = 0
    auth_token: str | None = None
    startup_timeout: float = 1800.0
    extra_arguments: tuple[str, ...] = ()
    loom_serve: str | None = None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "device": self.device,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "pool_backing": self.pool_backing,
            "memory_bytes": self.memory_bytes,
            "proxy": self.proxy,
            "proxy_port": self.proxy_port,
            "auth": bool(self.auth_token),
            "startup_timeout": self.startup_timeout,
        }


@dataclass(frozen=True)
class LaunchPlan:
    """The resolved command, document and addresses of one launch."""

    profile: Profile
    run: RunDirectory
    command: tuple[str, ...]
    deployment: dict
    device_uri: str
    server_host: str
    server_port: int
    settings: ContextSettings = field(default_factory=ContextSettings)
    #: The prepared scenario this launch serves as its package declares it.
    scenario: Scenario | None = None
    #: What a run-local devices.json changed (`--memory-bytes`/`--pool-backing`).
    overrides: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "scenario" if self.scenario is not None else "composed"


def plan_launch(
    profile: Profile,
    card_path: str | Path,
    settings: ContextSettings,
    run: RunDirectory,
    options: LaunchOptions,
    *,
    hardware: Hardware | None = None,
) -> LaunchPlan:
    """Composes the deployment document and the server command line."""
    hardware = hardware or detect()
    device_uri = options.device or hardware.default_device_uri()
    name = options.name or profile.slug
    document = deployment_document(
        profile, card_path, settings, name=name, preset=_preset(profile)
    )
    deployment_path = write_deployment(run.deployment_path, document)
    listen = f"{options.host}:{options.port}"
    command = [
        serve_command(options.loom_serve),
        f"--model={deployment_path}",
        f"--listen={listen}",
        f"--device={device_uri}",
    ]
    if options.workers is not None:
        command.append(f"--workers={options.workers}")
    if options.pool_backing:
        command.append(f"--pool_backing={options.pool_backing}")
    # A card records the memory budget it was validated with; use it unless the
    # caller chose one, since the package's own preparation expects it.
    memory_bytes = (
        options.memory_bytes
        if options.memory_bytes is not None
        else profile.memory_bytes
    )
    if memory_bytes:
        command.append(f"--memory_bytes={memory_bytes}")
    command.extend(options.extra_arguments)
    return LaunchPlan(
        profile=profile,
        run=run,
        command=tuple(command),
        deployment=document,
        device_uri=device_uri,
        server_host=options.host,
        server_port=options.port,
        settings=settings,
    )


def plan_scenario(
    profile: Profile,
    scenario: Scenario,
    run: RunDirectory,
    options: LaunchOptions,
) -> LaunchPlan:
    """The server command line of a prepared scenario: its own deployment and
    devices, no composed deployment and no `--device`."""
    devices_path, overrides = scenario.devices_for(
        options.memory_bytes, options.pool_backing, run.root
    )
    command = [
        serve_command(options.loom_serve),
        f"--devices={devices_path}",
        f"--model={scenario.deployment_path}",
        f"--listen={options.host}:{options.port}",
    ]
    if options.workers is not None:
        command.append(f"--workers={options.workers}")
    command.extend(options.extra_arguments)
    return LaunchPlan(
        profile=profile,
        run=run,
        command=tuple(command),
        deployment=scenario.deployment,
        device_uri=scenario.device_text,
        server_host=options.host,
        server_port=options.port,
        settings=scenario.settings,
        scenario=scenario,
        overrides=overrides,
    )


class Supervisor:
    """Runs one launch: child process, readiness, proxy and shutdown."""

    def __init__(
        self,
        plan: LaunchPlan,
        options: LaunchOptions,
        *,
        on_event: Callable[[dict], None] | None = None,
        on_output: Callable[[str], None] | None = None,
        state_root: Path | None = None,
    ):
        self.plan = plan
        self.options = options
        self._on_event = on_event
        self._on_output = on_output
        self._state_root = state_directory(state_root)
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._events: JsonlWriter | None = None
        self._serve_log = None
        self._proxy: Proxy | None = None
        self._metrics = Metrics()
        self._request_log: JsonlWriter | None = None
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._failure: str | None = None
        self._server_address: str | None = None
        self._started_at = time.time()
        self._exit_code: int | None = None
        self._tail: list[str] = []
        self._lock = threading.Lock()

    # -- properties ---------------------------------------------------------
    @property
    def endpoint(self) -> str:
        if self._proxy is not None:
            return self._proxy.address
        return f"http://{self.plan.server_host}:{self.plan.server_port}"

    @property
    def server_endpoint(self) -> str:
        return f"http://{self.plan.server_host}:{self.plan.server_port}"

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    def client(self, timeout: float = 900.0) -> Client:
        return Client(self.endpoint, token=self.options.auth_token, timeout=timeout)

    # -- lifetime -----------------------------------------------------------
    def start(self) -> None:
        self._events = self.plan.run.events()
        self._serve_log = open(self.plan.run.serve_log, "a", encoding="utf-8")
        self.plan.run.metadata.update(
            {
                "run_id": self.plan.run.run_id,
                "repo": self.plan.profile.repo,
                "revision": self.plan.profile.revision,
                "model_id": self.plan.profile.model_id,
                "deployment_name": self.plan.deployment["name"],
                "command": list(self.plan.command),
                "device": self.plan.device_uri,
                "settings": self.plan.settings.describe(),
                "deployment": self.plan.deployment,
                "kind": self.plan.kind,
                "scenario": self.plan.scenario.describe()
                if self.plan.scenario
                else None,
                "options": self.options.describe(),
                "started_at": _isoformat(self._started_at),
                "state": "starting",
            }
        )
        self.plan.run.save()
        self._record_event(
            {
                "event": "launch",
                "repo": self.plan.profile.repo,
                "command": list(self.plan.command),
                "deployment": self.plan.deployment["name"],
                "device": self.plan.device_uri,
            }
        )
        try:
            self._process = subprocess.Popen(
                list(self.plan.command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._fail(f"cannot run {self.plan.command[0]}: {exc}")
            raise LaunchError(self._failure or str(exc)) from exc
        self._reader = threading.Thread(
            target=self._read_output, name="loom-launch-server", daemon=True
        )
        self._reader.start()
        try:
            self._wait_until_prepared()
        except LaunchError:
            # A launch that never prepared must not leave its server behind: the
            # process owns a device and a port, and a caller's next launch would
            # find both taken.
            self._abandon()
            raise
        if self.options.proxy:
            self._start_proxy()
        self.plan.run.metadata.update(
            {
                "state": "serving",
                "endpoint": self.endpoint,
                "server_endpoint": self.server_endpoint,
                "pid": self._process.pid,
                "ready_at": _isoformat(time.time()),
                "prepare_seconds": round(time.time() - self._started_at, 3),
            }
        )
        self.plan.run.save()
        write_current(
            self._state_root,
            {
                "run_id": self.plan.run.run_id,
                "endpoint": self.endpoint,
                "server_endpoint": self.server_endpoint,
                "pid": self._process.pid,
                "repo": self.plan.profile.repo,
                "deployment": self.plan.deployment["name"],
                "kind": self.plan.kind,
                "device": self.plan.device_uri,
                "command": list(self.plan.command),
                "started_at": self.plan.run.metadata["started_at"],
                "context_length": self.plan.settings.context_length,
                "max_tokens": self.plan.settings.max_tokens,
                "auth": bool(self.options.auth_token),
            },
        )
        self._record_event(
            {
                "event": "serving",
                "endpoint": self.endpoint,
                "server_endpoint": self.server_endpoint,
            }
        )

    def wait(self) -> int:
        """Blocks until the server exits, stopping it on interruption."""
        if self._process is None:
            return 0
        try:
            self._exit_code = self._process.wait()
        except KeyboardInterrupt:
            self.stop()
            self._exit_code = self._process.returncode if self._process else 0
        finally:
            self._shutdown()
        return self._exit_code or 0

    def stop(self, timeout: float = 30.0) -> None:
        self._terminate(timeout)

    def _terminate(self, timeout: float = 30.0) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        self._record_event({"event": "stopping"})
        try:
            process.send_signal(signal.SIGINT)
        except OSError:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.05)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    def _abandon(self) -> None:
        """Stops a launch that never prepared, keeping its failure on record."""
        self._terminate(timeout=10.0)
        self._close_streams()

    def _close_streams(self) -> None:
        if self._proxy is not None:
            self._proxy.stop()
            self._proxy = None
        if self._reader is not None:
            self._reader.join(timeout=5)
        for stream in (self._serve_log, self._events, self._request_log):
            if stream is not None:
                stream.close()
        self._serve_log = None
        self._events = None
        self._request_log = None

    def _shutdown(self) -> None:
        self._close_streams()
        self.plan.run.metadata.update(
            {
                "state": "stopped",
                "stopped_at": _isoformat(time.time()),
                "exit_code": self._process.returncode if self._process else None,
            }
        )
        self.plan.run.save()
        clear_current(self._state_root)
        self._record_event(
            {
                "event": "stopped",
                "exit_code": self._process.returncode if self._process else None,
            }
        )

    # -- readiness ----------------------------------------------------------
    def _wait_until_prepared(self) -> None:
        client = Client(
            self.server_endpoint, timeout=max(5.0, self.options.startup_timeout)
        )
        names = {
            str_value(self.plan.deployment.get("name")),
            self.plan.profile.model_id,
        }
        names = {name for name in names if name}
        deadline = time.monotonic() + self.options.startup_timeout
        prepared = False
        while time.monotonic() < deadline:
            if self._process is None or self._process.poll() is not None:
                self._fail(
                    "the server exited before the model became ready "
                    f"(exit code {self._process.returncode if self._process else -1})"
                )
                raise LaunchError(self._failure or "server exited")
            if not client.healthy(timeout=1.0):
                time.sleep(0.1)
                continue
            available = set(client.model_names(timeout=5.0))
            if names & available:
                prepared = True
                break
            if self._registration_failed(client, names):
                raise LaunchError(self._failure or "model preparation failed")
            time.sleep(0.25)
        if not prepared:
            self._fail(
                "the model did not become ready within "
                f"{self.options.startup_timeout:.0f}s"
            )
            raise LaunchError(self._failure or "model preparation timed out")

    def _registration_failed(self, client: Client, names: set[str]) -> bool:
        status = client.server_status(timeout=5.0)
        for entry in _status_entries(status):
            reported = str_value(entry.get("name")) or str_value(entry.get("id"))
            if reported and names and reported not in names:
                continue
            state = _entry_state(entry)
            if state in ("failed", "error"):
                self._fail("model preparation failed: " + (_error_text(entry) or state))
                return True
        return False

    # -- proxy --------------------------------------------------------------
    def _start_proxy(self) -> None:
        self._request_log = self.plan.run.requests()
        self._proxy = Proxy(
            Upstream(self.plan.server_host, self.plan.server_port),
            metrics=self._metrics,
            request_log=self._request_log,
            bind_host=self.plan.server_host,
            bind_port=self.options.proxy_port,
            status_provider=self._launch_status,
            auth_token=self.options.auth_token,
        )
        self._proxy.start()

    def _launch_status(self) -> dict:
        return {
            "launch": {
                "repo": self.plan.profile.repo,
                "revision": self.plan.profile.revision,
                "deployment": self.plan.deployment.get("name"),
                "device": self.plan.device_uri,
                "launch_kind": self.plan.kind,
                "command": list(self.plan.command),
                "run_directory": str(self.plan.run.root),
                "started_at": self.plan.run.metadata.get("started_at"),
                "pid": self._process.pid if self._process else None,
                "server_endpoint": self.server_endpoint,
            },
            "settings": self.plan.settings.describe(),
        }

    # -- output -------------------------------------------------------------
    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            line = line.rstrip("\n")
            if self._serve_log is not None:
                self._serve_log.write(line + "\n")
                self._serve_log.flush()
            with self._lock:
                self._tail.append(line)
                if len(self._tail) > 200:
                    del self._tail[: len(self._tail) - 200]
            event = _parse_event(line)
            if event is not None:
                self._record_event(event)
                if event.get("event") == "ready":
                    address = str_value(event.get("address"))
                    if address:
                        self._server_address = address
                    self._ready.set()
            if self._on_output is not None:
                self._on_output(line)

    def _record_event(self, event: dict) -> None:
        if self._events is not None:
            record = {"time": _isoformat(time.time())}
            record.update(event)
            self._events.write(record)
        if self._on_event is not None:
            self._on_event(event)

    def _fail(self, message: str) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = message
        self._failed.set()
        self.plan.run.metadata.update({"state": "failed", "error": message})
        self.plan.run.save()
        self._record_event({"event": "failed", "error": message})

    # -- observation --------------------------------------------------------
    def tail(self, lines: int = 40) -> list[str]:
        with self._lock:
            return list(self._tail[-lines:])

    def status(self) -> dict:
        alive = self._process is not None and self._process.poll() is None
        status = {
            "run": self.plan.run.describe(),
            "repo": self.plan.profile.repo,
            "deployment": self.plan.deployment.get("name"),
            "device": self.plan.device_uri,
            "endpoint": self.endpoint,
            "server_endpoint": self.server_endpoint,
            "alive": alive,
            "pid": self._process.pid if self._process else None,
            "settings": self.plan.settings.describe(),
            "metrics": self._metrics.snapshot(),
        }
        client = Client(self.server_endpoint, timeout=5.0)
        try:
            status["models"] = client.models(timeout=5.0)
        except LaunchError:
            status["models"] = []
        return status


def _preset(profile: Profile) -> str | None:
    return profile.execution.preset if profile.execution else None


def _status_entries(status: dict) -> list[dict]:
    """Every per-model record a status document carries, however it nests it.

    The server reports a model as `data[]` entries whose registration detail sits
    under a nested `loom` object, and its shapes drift, so a record and each of
    its nested views is visited and named after the model it belongs to.
    """
    entries: list[dict] = []

    def visit(document: dict, name: str | None) -> None:
        entry = dict(document)
        resolved = (
            str_value(entry.get("name"))
            or str_value(entry.get("id"))
            or str_value(entry.get("model"))
            or name
        )
        if resolved and not str_value(entry.get("name")):
            entry["name"] = resolved
        entries.append(entry)
        nested = dict_value(entry.get("loom"))
        if nested:
            visit(nested, resolved)

    models = status.get("models")
    if isinstance(models, dict):
        for name, value in models.items():
            candidates = value if isinstance(value, list) else [value]
            for entry in candidates:
                if isinstance(entry, dict):
                    visit(entry, str_value(entry.get("name")) or name)
    for key in ("data", "registrations", "deployments"):
        for entry in list_value(status.get(key)):
            if isinstance(entry, dict):
                visit(entry, None)
    loom = status.get("loom")
    if isinstance(loom, dict):
        visit(loom, None)
    else:
        for entry in list_value(loom):
            if isinstance(entry, dict):
                visit(entry, None)
    return entries


def _entry_state(entry: dict) -> str | None:
    for key in ("state", "status", "registration"):
        value = entry.get(key)
        if isinstance(value, dict):
            value = value.get("phase") or value.get("state")
        if isinstance(value, str):
            return value.lower()
    return None


def _error_text(entry: dict) -> str | None:
    """The message of an error a status entry reports, however it is spelled."""
    error = entry.get("error")
    message = str_value(error)
    if message:
        return message
    detail = dict_value(error)
    return str_value(detail.get("message")) or str_value(detail.get("details"))


def _parse_event(line: str) -> dict | None:
    line = line.strip()
    if not line.startswith("{") or not line.endswith("}"):
        return None
    try:
        document = json.loads(line)
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None


def _isoformat(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


# -- the current-launch record ---------------------------------------------
def current_path(root: Path | None = None) -> Path:
    return state_directory(root) / CURRENT_RECORD


def write_current(root: Path, record: dict) -> Path:
    path = write_json(current_path(root), record)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def read_current(root: Path | None = None) -> dict | None:
    from loom.launch.store import read_json

    record = read_json(current_path(root))
    return record or None


def clear_current(root: Path | None = None) -> None:
    try:
        current_path(root).unlink()
    except OSError:
        pass


def running_endpoint(root: Path | None = None) -> tuple[str, dict] | None:
    """The endpoint of a live launch, when one is running and answering."""
    record = read_current(root)
    if not record:
        return None
    pid = record.get("pid")
    if isinstance(pid, int) and not _pid_alive(pid):
        return None
    endpoint = str_value(record.get("endpoint"))
    if not endpoint:
        return None
    client = Client(endpoint, timeout=2.0)
    if not client.healthy(timeout=2.0):
        return None
    return endpoint, record


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def describe_state(root: Path | None = None) -> dict:
    """Summarizes the launcher's persistent state for `loom-launch status`."""
    root = state_directory(root)
    return {
        "state_directory": str(root),
        "current": read_current(root),
    }
