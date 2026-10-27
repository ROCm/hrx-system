# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""libamdf-owned Bazel configuration policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from build_tools.bazel.configure_api import (
    BazelrcSection,
    ConfigRequest,
    Contributor,
    bazelrc_line,
    parse_bool,
    parse_string_list,
)

FAMILY_DEFINES = {
    "AMDF_FAMILY_RDNA": "rdna",
    "AMDF_FAMILY_CDNA": "cdna",
    "AMDF_FAMILY_XDNA": "xdna",
}
FAMILIES = tuple(FAMILY_DEFINES.values())
NATIVE_ENABLED_FLAG = "--//libamdf/config:enabled"
NATIVE_FAMILIES_FLAG = "--//libamdf/config:families"

_NAME = "libamdf"


@dataclass
class State:
    enabled: bool = False  # Whether the library is built.
    enabled_source: str | None = None  # Portable or native explicit selection.
    # Selected implementation families.
    families: set[str] = field(default_factory=lambda: set(FAMILIES))
    family_source: str | None = None  # Portable or native family spelling.
    # Other owners that require the library in this request.
    requirements: set[str] = field(default_factory=set)


def configuration(request: ConfigRequest) -> State:
    return cast(State, request.project_state(_NAME))


def require(request: ConfigRequest, owner: str) -> None:
    configuration(request).requirements.add(owner)


def _initialize(request: ConfigRequest) -> None:
    request.add_project_state(_NAME, State())


def _set_enabled(state: State, enabled: bool, source: str) -> None:
    if state.enabled_source not in (None, source):
        raise SystemExit(
            "Do not mix portable -DAMDF_BUILD with the native "
            f"{NATIVE_ENABLED_FLAG}=... Bazel option."
        )
    state.enabled = enabled
    state.enabled_source = source


def _apply_define(request: ConfigRequest, name: str, value: str) -> None:
    state = configuration(request)
    if name == "AMDF_BUILD":
        _set_enabled(state, parse_bool(name, value), "portable")
        return
    family = FAMILY_DEFINES[name]
    if state.family_source == "native":
        raise SystemExit(
            "Do not mix portable -DAMDF_FAMILY_* options with the native "
            f"{NATIVE_FAMILIES_FLAG}=... Bazel option."
        )
    state.family_source = "portable"
    if parse_bool(name, value):
        state.families.add(family)
    else:
        state.families.discard(family)


def _apply_native_option(request: ConfigRequest, name: str, value: str) -> None:
    state = configuration(request)
    if name == NATIVE_ENABLED_FLAG:
        _set_enabled(state, parse_bool(name, value), "native")
        return
    if state.family_source == "portable":
        raise SystemExit(
            "Do not mix portable -DAMDF_FAMILY_* options with the native "
            f"{NATIVE_FAMILIES_FLAG}=... Bazel option."
        )
    families = parse_string_list(value)
    unknown = families.difference(FAMILIES)
    if unknown:
        raise SystemExit(
            "Unknown libamdf family or families: " + ", ".join(sorted(unknown))
        )
    state.family_source = "native"
    state.families = families


def _finalize(request: ConfigRequest) -> None:
    state = configuration(request)
    if state.requirements and state.enabled_source is None:
        state.enabled = True


def _validate(request: ConfigRequest) -> None:
    state = configuration(request)
    if state.requirements and not state.enabled:
        raise SystemExit(
            "AMDF_BUILD=OFF conflicts with " + ", ".join(sorted(state.requirements))
        )


def _sections(request: ConfigRequest) -> tuple[BazelrcSection, ...]:
    state = configuration(request)
    families = ",".join(family for family in FAMILIES if family in state.families)
    return (
        BazelrcSection(
            "libamdf library and implementation-family scope.",
            (
                bazelrc_line(
                    "build", NATIVE_ENABLED_FLAG + "=" + str(state.enabled).lower()
                ),
                bazelrc_line("build", NATIVE_FAMILIES_FLAG + "=" + families),
            ),
        ),
    )


CONTRIBUTOR = Contributor(
    name=_NAME,
    initialize=_initialize,
    defines={
        "AMDF_BUILD": _apply_define,
        **{name: _apply_define for name in FAMILY_DEFINES},
    },
    native_options={
        NATIVE_ENABLED_FLAG: _apply_native_option,
        NATIVE_FAMILIES_FLAG: _apply_native_option,
    },
    finalize=_finalize,
    validate=_validate,
    sections=_sections,
)
