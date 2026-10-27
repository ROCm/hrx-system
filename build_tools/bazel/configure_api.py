# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generic contracts for repository Bazel configuration contributors."""

from __future__ import annotations

import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import cast

TRUE_VALUES = frozenset(("1", "ON", "TRUE", "YES"))
FALSE_VALUES = frozenset(("0", "OFF", "FALSE", "NO"))


@dataclass
class ConfigRequest:
    """One parsed configuration request with contributor-owned state."""

    output: Path  # Generated Bazelrc destination.
    # State keyed by its single owning contributor.
    _project_states: dict[str, object] = field(default_factory=dict, init=False)

    def add_project_state(self, name: str, state: object) -> None:
        if name in self._project_states:
            raise RuntimeError(f"duplicate project configuration state: {name}")
        self._project_states[name] = state

    def project_state(self, name: str) -> object:
        return self._project_states[name]


OptionHandler = Callable[[ConfigRequest, str, str], None]
ConfigHandler = Callable[[ConfigRequest, str], None]
RemovedOptionHandler = Callable[[ConfigRequest, str], None]
RequestPhase = Callable[[ConfigRequest], None]


def _no_request_phase(request: ConfigRequest) -> None:
    del request


def _no_sections(request: ConfigRequest) -> tuple[BazelrcSection, ...]:
    del request
    return ()


def _no_summary(request: ConfigRequest) -> tuple[str, ...]:
    del request
    return ()


@dataclass(frozen=True)
class BazelrcSection:
    """One contributor-owned section of the generated Bazel configuration."""

    heading: str  # Human-readable comment heading.
    lines: tuple[str, ...]  # Complete Bazelrc lines below the heading.


@dataclass(frozen=True)
class Contributor:
    """Declared option ownership and lifecycle hooks for one project owner."""

    name: str  # Unique state and diagnostic owner name.
    initialize: RequestPhase = _no_request_phase  # Installs owner state.
    defines: Mapping[str, OptionHandler] = field(default_factory=dict)  # -D names.
    # Bazel build-setting flags accepted by the public configurator.
    native_options: Mapping[str, OptionHandler] = field(default_factory=dict)
    # Named --config values accepted by the public configurator.
    bazel_configs: Mapping[str, ConfigHandler] = field(default_factory=dict)
    # Named --repo_env values accepted by the public configurator.
    repo_env: Mapping[str, OptionHandler] = field(default_factory=dict)
    # Removed command-line flags with owner-specific replacement diagnostics.
    removed_options: Mapping[str, RemovedOptionHandler] = field(default_factory=dict)
    # Normalizes owner state and publishes cross-owner requirements.
    resolve: RequestPhase = _no_request_phase
    finalize: RequestPhase = _no_request_phase  # Applies all published requirements.
    validate: RequestPhase = _no_request_phase  # Rejects contradictory resolved state.
    # Ordered Bazelrc sections emitted for this owner.
    sections: Callable[[ConfigRequest], tuple[BazelrcSection, ...]] = _no_sections
    # Concise terminal summary lines emitted after configuration.
    summary: Callable[[ConfigRequest], tuple[str, ...]] = _no_summary


@dataclass
class BooleanOptionState:
    enabled: bool  # Resolved build-setting value.
    source: str | None = None  # Portable or native spelling, when explicit.


def boolean_configuration(request: ConfigRequest, name: str) -> BooleanOptionState:
    return cast(BooleanOptionState, request.project_state(name))


def boolean_option_contributor(
    *,
    name: str,
    default: bool,
    portable_option: str,
    native_option: str,
    section_heading: str,
) -> Contributor:
    """Builds a contributor for one equivalent portable/native boolean option."""

    def initialize(request: ConfigRequest) -> None:
        request.add_project_state(name, BooleanOptionState(default))

    def set_value(request: ConfigRequest, option: str, value: str) -> None:
        state = boolean_configuration(request, name)
        source = "portable" if option == portable_option else "native"
        if state.source not in (None, source):
            raise SystemExit(
                f"Do not mix portable -D{portable_option} with the native "
                f"{native_option}=... Bazel option."
            )
        state.enabled = parse_bool(option, value)
        state.source = source

    def sections(request: ConfigRequest) -> tuple[BazelrcSection, ...]:
        state = boolean_configuration(request, name)
        return (
            BazelrcSection(
                section_heading,
                (
                    bazelrc_line(
                        "build", native_option + "=" + str(state.enabled).lower()
                    ),
                ),
            ),
        )

    return Contributor(
        name=name,
        initialize=initialize,
        defines={portable_option: set_value},
        native_options={native_option: set_value},
        sections=sections,
    )


def parse_bool(name: str, value: str) -> bool:
    normalized_value = value.upper()
    if normalized_value in TRUE_VALUES:
        return True
    if normalized_value in FALSE_VALUES:
        return False
    raise SystemExit(f"{name} expects ON or OFF, got {value!r}.")


def parse_string_list(value: str, separator: str = ",") -> set[str]:
    return set(filter(None, value.split(separator)))


def as_bazel_path(path: PurePath) -> str:
    """Returns a filesystem path in Bazel's platform-neutral spelling."""
    return path.as_posix()


def bazelrc_line(command: str, option: str) -> str:
    return f"{command} {shlex.quote(option)}"
