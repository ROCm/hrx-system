# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Prepared scenarios: packages that carry their own deployment and machine.

A package directory holding both `deployment.json` and `devices.json` was
prepared for one machine shape (for example a GPU plus an XDNA NPU sharing an
exchange pool). The launcher serves it exactly as prepared:
`loom-serve --devices=<devices.json> --model=<deployment.json>`, without a
composed deployment or a `--device`. The scenario's documents own its
capacities and devices, so launcher options that would change them are refused
rather than silently dropped.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from loom.launch.deployment import ContextSettings
from loom.launch.errors import LaunchError
from loom.launch.profile import Execution
from loom.launch.store import write_json
from loom.launch.values import dict_value, int_value, str_value

DEPLOYMENT_DOCUMENT = "deployment.json"
DEVICES_DOCUMENT = "devices.json"
CARD_DOCUMENT = "model.json"


def is_scenario(directory: str | Path) -> bool:
    """True when `directory` holds both scenario documents."""
    directory = Path(directory)
    return (directory / DEPLOYMENT_DOCUMENT).is_file() and (
        directory / DEVICES_DOCUMENT
    ).is_file()


def _read_document(path: Path) -> dict:
    try:
        document = json.loads(path.read_bytes())
    except OSError as exc:
        raise LaunchError(f"{path} is unreadable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LaunchError(f"{path} is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise LaunchError(f"{path} is not a JSON object")
    return document


@dataclass(frozen=True)
class Scenario:
    """One package prepared with its own deployment and device documents."""

    directory: Path
    deployment_path: Path
    devices_path: Path
    deployment: dict = field(default_factory=dict)
    devices: dict = field(default_factory=dict)
    #: The capacities the deployment declares over its card's preset.
    settings: ContextSettings = field(default_factory=ContextSettings)

    @classmethod
    def detect(cls, directory: str | Path) -> Scenario | None:
        """Reads the scenario in `directory` (a package or its card), `None`
        when the directory is an ordinary profile."""
        directory = Path(directory).expanduser()
        if directory.is_file():
            directory = directory.parent
        if not is_scenario(directory):
            return None
        directory = directory.absolute()
        deployment_path = directory / DEPLOYMENT_DOCUMENT
        devices_path = directory / DEVICES_DOCUMENT
        deployment = _read_document(deployment_path)
        devices = _read_document(devices_path)
        if not dict_value(devices.get("devices")):
            raise LaunchError(f"{devices_path} names no devices")
        return cls(
            directory=directory,
            deployment_path=deployment_path,
            devices_path=devices_path,
            deployment=deployment,
            devices=devices,
            settings=_declared_settings(directory, deployment),
        )

    @property
    def name(self) -> str:
        return str_value(self.deployment.get("name")) or self.directory.name

    @property
    def device_names(self) -> list[str]:
        return list(dict_value(self.devices.get("devices")))

    @property
    def device_text(self) -> str:
        """`amdgpu + xdna`."""
        return " + ".join(self.device_names)

    @property
    def pool(self) -> str | None:
        return str_value(self.deployment.get("pool"))

    @property
    def pool_names(self) -> list[str]:
        return list(dict_value(self.devices.get("pools")))

    def describe(self) -> dict:
        devices = dict_value(self.devices.get("devices"))
        return {
            "kind": "scenario",
            "directory": str(self.directory),
            "deployment": self.name,
            "deployment_path": str(self.deployment_path),
            "devices_path": str(self.devices_path),
            "devices": self.device_text,
            "device_uris": {
                name: dict_value(entry).get("uri") for name, entry in devices.items()
            },
            "pool": self.pool,
            "pools": self.pool_names,
            "preset": dict_value(self.deployment.get("model")).get("preset"),
            "context_length": self.settings.context_length,
            "rows": self.settings.rows,
            "max_tokens": self.settings.max_tokens,
            "prefill_capacity": self.settings.prefill_capacity,
            "draft_depth": self.settings.draft_depth,
        }

    def refuse(self, options: list[str]) -> None:
        """Refuses launcher options that would change what the scenario declares."""
        if not options:
            return

        def dash(value: int | None) -> str:
            return "-" if value is None else str(value)

        settings = self.settings
        raise LaunchError(
            f"{self.directory} is a prepared scenario: it owns its capacities and "
            f"devices, so {', '.join(options)} cannot change them. Its capacities "
            f"come from {self.deployment_path} (deployment {self.name}: context "
            f"{dash(settings.context_length)}, max tokens {dash(settings.max_tokens)}, "
            f"rows {dash(settings.rows)}, draft depth {dash(settings.draft_depth)}) "
            f"and its devices from {self.devices_path} ({self.device_text}); drop "
            f"{'it' if len(options) == 1 else 'them'} or edit those files"
        )

    def devices_for(
        self,
        memory_bytes: int | None,
        pool_backing: str | None,
        run_directory: Path,
    ) -> tuple[Path, dict]:
        """The devices document the server reads.

        `loom-serve` refuses `--memory_bytes` and `--pool_backing` next to
        `--devices`, so a launch that sets them gets a copy of `devices.json` in
        its run directory with the deployment's pool changed; otherwise the
        package's own file is used.
        """
        overrides: dict = {}
        if memory_bytes is not None:
            if memory_bytes < 0:
                raise LaunchError("memory bytes must not be negative")
            overrides["memory_bytes"] = memory_bytes
        if pool_backing:
            if pool_backing not in ("elastic", "fixed"):
                raise LaunchError("pool backing must be elastic or fixed")
            overrides["pool_backing"] = pool_backing
        if not overrides:
            return self.devices_path, overrides
        pools = self.pool_names
        pool = self.pool or (pools[0] if len(pools) == 1 else None)
        if pool is None:
            raise LaunchError(
                f"{self.deployment_path} names no pool and {self.devices_path} has "
                f"{len(pools)} pools, so --memory-bytes and --pool-backing cannot "
                f"tell which to change; edit the pool in {self.devices_path}"
            )
        devices = copy.deepcopy(self.devices)
        entry = dict_value(devices.get("pools")).get(pool)
        if not isinstance(entry, dict):
            raise LaunchError(
                f"{self.deployment_path} places the model in pool {pool}, which "
                f"{self.devices_path} does not declare"
            )
        entry.update(overrides)
        if entry.get("pool_backing") == "fixed" and int_value(
            entry.get("memory_bytes")
        ):
            raise LaunchError(
                f"pool {pool} would be fixed with a memory budget; loom-serve requires "
                "elastic backing for a physical budget (drop --memory-bytes or "
                "--pool-backing fixed)"
            )
        overrides["pool"] = pool
        path = write_json(Path(run_directory) / DEVICES_DOCUMENT, devices)
        return path, overrides


def _declared_settings(directory: Path, deployment: dict) -> ContextSettings:
    """The capacities a scenario serves: its card's preset execution, overlaid
    with whatever the deployment's `definition` sets, as `loom-serve` resolves
    them."""
    model = dict_value(deployment.get("model"))
    card_path = directory / (str_value(model.get("path")) or CARD_DOCUMENT)
    if not card_path.is_file():
        raise LaunchError(
            f"{directory / DEPLOYMENT_DOCUMENT} loads {card_path}, which does not exist"
        )
    card = _read_document(card_path)
    execution = Execution.from_card(card, str_value(model.get("preset"))) or Execution()
    definition = dict_value(deployment.get("definition"))

    def declared(key: str, fallback: int | None = None) -> int | None:
        value = int_value(definition.get(key))
        return value if value is not None else fallback

    return ContextSettings(
        context_length=declared("context_capacity", execution.context_capacity),
        max_tokens=declared("max_tokens", execution.max_tokens),
        rows=declared("rows", execution.rows),
        max_sessions=declared("max_sessions"),
        prefill_capacity=declared("prefill_capacity", execution.prefill_capacity),
        checkpoint_capacity=declared("checkpoint_capacity"),
        session_cache_capacity=declared("session_cache_capacity"),
        prefix_cache_capacity=declared("prefix_cache_capacity"),
        cache_target_bytes=declared("cache_target_bytes"),
        draft_depth=declared("draft_depth", execution.draft_depth),
        continuation_epochs=declared("continuation_epochs"),
        pending_requests=declared("pending_requests"),
    )
