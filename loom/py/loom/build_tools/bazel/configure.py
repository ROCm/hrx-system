# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom-owned Bazel configuration policy."""

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

from loom.build_tools import target_presets

IMPORT_DEFINES = {
    "LOOM_IMPORT_CXX": "cxx",
    "LOOM_IMPORT_MLIR": "mlir",
    "LOOM_IMPORT_TILELANG": "tilelang",
}
ARCHITECTURES = target_presets.ARCHITECTURES
ARTIFACT_FORMATS = target_presets.ARTIFACT_FORMATS
EXECUTION_SUBSTRATES = target_presets.EXECUTION_SUBSTRATES
IMPORTERS = tuple(IMPORT_DEFINES.values())
IMAGE_CODECS = ("png", "jpeg", "gif", "webp")
AUDIO_CODECS = ("wav", "mp3", "flac")

NATIVE_ARCH_FLAG = "--//loom/config/arch:enable"
NATIVE_AMDGPU_TARGETS_FLAG = "--//loom/config/arch/amdgpu:targets"
NATIVE_EMIT_FLAG = "--//loom/config/emit:enable"
NATIVE_EXECUTE_FLAG = "--//loom/config/execute:enable"
NATIVE_IMPORT_FLAG = "--//loom/config/import:enable"
NATIVE_IMAGE_INPUT_FLAG = "--//loom/config/serve:image_input"
NATIVE_IMAGE_CODECS_FLAG = "--//loom/config/serve:image_codecs"
NATIVE_AUDIO_CODECS_FLAG = "--//loom/config/serve:audio_codecs"

_NAME = "loom"


@dataclass
class SetSelection:
    allowed: tuple[str, ...]  # Canonical value order and membership.
    portable_family: str  # Portable -D family used in diagnostics.
    native_flag: str  # Equivalent exact-list Bazel flag.
    description: str  # Value kind used in diagnostics.
    # Explicit values before preset capabilities are added.
    values: set[str] = field(default_factory=set)
    source: str | None = None  # Portable or native explicit selection.

    def set_member(self, value: str, enabled: bool) -> None:
        if self.source == "native":
            self._raise_mixed()
        self.source = "portable"
        if enabled:
            self.values.add(value)
        else:
            self.values.discard(value)

    def set_list(self, value: str, source: str, separator: str = ",") -> None:
        if self.source not in (None, source):
            self._raise_mixed()
        values = parse_string_list(value, separator)
        unknown = values.difference(self.allowed)
        if unknown:
            raise SystemExit(
                f"Unknown Loom {self.description}(s): " + ", ".join(sorted(unknown))
            )
        self.values = values
        self.source = source

    def _raise_mixed(self) -> None:
        raise SystemExit(
            f"Do not mix portable -D{self.portable_family} options with the "
            f"native {self.native_flag}=... Bazel option."
        )


@dataclass
class State:
    # Compiler architecture provider selection.
    architectures: SetSelection = field(
        default_factory=lambda: SetSelection(
            ARCHITECTURES, "LOOM_ARCH_*", NATIVE_ARCH_FLAG, "architecture"
        )
    )
    # Emitted artifact-format selection.
    artifact_formats: SetSelection = field(
        default_factory=lambda: SetSelection(
            ARTIFACT_FORMATS,
            "LOOM_EMIT_*",
            NATIVE_EMIT_FLAG,
            "artifact format",
        )
    )
    # In-process execution-substrate selection.
    execution_substrates: SetSelection = field(
        default_factory=lambda: SetSelection(
            EXECUTION_SUBSTRATES,
            "LOOM_EXECUTE_*",
            NATIVE_EXECUTE_FLAG,
            "execute substrate",
        )
    )
    # Optional source importer selection.
    importers: SetSelection = field(
        default_factory=lambda: SetSelection(
            IMPORTERS, "LOOM_IMPORT_*", NATIVE_IMPORT_FLAG, "importer"
        )
    )
    amdgpu_targets: set[str] | None = None  # AMDGPU processor selectors.
    image_input: bool = True  # Whether image inputs are accepted.
    image_input_source: str | None = None  # Portable or native spelling.
    # Encoded image provider selection.
    image_codecs: SetSelection = field(
        default_factory=lambda: SetSelection(
            IMAGE_CODECS,
            "LOOM_SERVE_IMAGE_CODECS",
            NATIVE_IMAGE_CODECS_FLAG,
            "image codec",
            set(IMAGE_CODECS),
        )
    )
    # Encoded audio provider selection; raw PCM is always present.
    audio_codecs: SetSelection = field(
        default_factory=lambda: SetSelection(
            AUDIO_CODECS,
            "LOOM_SERVE_AUDIO_CODECS",
            NATIVE_AUDIO_CODECS_FLAG,
            "audio codec",
            set(AUDIO_CODECS),
        )
    )
    # Presets selected before fallback replacement and transitive expansion.
    enabled_presets: set[str] = field(default_factory=lambda: {"default"})
    explicit_presets: set[str] = field(default_factory=set)  # User-named presets.
    preset_provenance: tuple[str, ...] = ()  # Effective top-level presets.
    exact_capabilities: bool = False  # Whether native lists replace presets.


def configuration(request: ConfigRequest) -> State:
    return cast(State, request.project_state(_NAME))


def _initialize(request: ConfigRequest) -> None:
    request.add_project_state(_NAME, State())


_MEMBER_DEFINES = {
    **{
        name: ("architectures", value)
        for name, value in target_presets.ARCH_DEFINES.items()
    },
    **{
        name: ("artifact_formats", value)
        for name, value in target_presets.EMIT_DEFINES.items()
    },
    **{
        name: ("execution_substrates", value)
        for name, value in target_presets.EXECUTE_DEFINES.items()
    },
    **{name: ("importers", value) for name, value in IMPORT_DEFINES.items()},
}

_LIST_DEFINES = {
    "LOOM_SERVE_IMAGE_CODECS": "image_codecs",
    "LOOM_SERVE_AUDIO_CODECS": "audio_codecs",
}

_NATIVE_SELECTIONS = {
    NATIVE_ARCH_FLAG: "architectures",
    NATIVE_EMIT_FLAG: "artifact_formats",
    NATIVE_EXECUTE_FLAG: "execution_substrates",
    NATIVE_IMPORT_FLAG: "importers",
    NATIVE_IMAGE_CODECS_FLAG: "image_codecs",
    NATIVE_AUDIO_CODECS_FLAG: "audio_codecs",
}


def _apply_define(request: ConfigRequest, name: str, value: str) -> None:
    state = configuration(request)
    preset = target_presets.PRESETS_BY_OPTION.get(name)
    if preset is not None:
        state.explicit_presets.add(preset.key)
        if parse_bool(name, value):
            state.enabled_presets.add(preset.key)
        else:
            state.enabled_presets.discard(preset.key)
        return
    member = _MEMBER_DEFINES.get(name)
    if member is not None:
        selection = cast(SetSelection, getattr(state, member[0]))
        selection.set_member(member[1], parse_bool(name, value))
        return
    list_attribute = _LIST_DEFINES.get(name)
    if list_attribute is not None:
        selection = cast(SetSelection, getattr(state, list_attribute))
        selection.set_list(value, "portable", ";")
        return
    if name == "LOOM_SERVE_IMAGE_INPUT":
        _set_image_input(state, parse_bool(name, value), "portable")
        return
    raise RuntimeError(f"unhandled declared Loom configure option {name}")


def _set_image_input(state: State, enabled: bool, source: str) -> None:
    if state.image_input_source not in (None, source):
        raise SystemExit("Do not mix portable and native Loom image-input options.")
    state.image_input_source = source
    state.image_input = enabled


def _apply_native_option(request: ConfigRequest, name: str, value: str) -> None:
    state = configuration(request)
    if name == NATIVE_AMDGPU_TARGETS_FLAG:
        state.amdgpu_targets = parse_string_list(value)
        return
    if name == NATIVE_IMAGE_INPUT_FLAG:
        _set_image_input(state, parse_bool(name, value), "native")
        return
    selection = cast(SetSelection, getattr(state, _NATIVE_SELECTIONS[name]))
    selection.set_list(value, "native")


def _apply_bazel_config(request: ConfigRequest, name: str) -> None:
    state = configuration(request)
    preset = target_presets.PRESETS_BY_BAZEL_CONFIG[name]
    state.explicit_presets.add(preset.key)
    state.enabled_presets.add(preset.key)


def _resolve(request: ConfigRequest) -> None:
    state = configuration(request)
    sources = (
        state.architectures.source,
        state.artifact_formats.source,
        state.execution_substrates.source,
    )
    native_count = sum(source == "native" for source in sources)
    if native_count not in (0, len(sources)):
        raise SystemExit(
            "Native Loom capability selection must name complete architecture, "
            "artifact, and execution lists together."
        )
    state.exact_capabilities = native_count == len(sources)
    if state.exact_capabilities and state.explicit_presets:
        raise SystemExit(
            "Do not mix Loom target presets with native complete capability lists."
        )
    enabled_presets = set() if state.exact_capabilities else set(state.enabled_presets)
    family_presets = enabled_presets.difference(("default",))
    if family_presets:
        enabled_presets = family_presets
    resolved = target_presets.resolve_presets(enabled_presets)
    state.architectures.values.update(resolved.architectures)
    state.artifact_formats.values.update(resolved.artifacts)
    state.execution_substrates.values.update(resolved.execution)
    state.preset_provenance = tuple(
        preset.key
        for preset in target_presets.TARGET_PRESETS
        if preset.key in enabled_presets
    )


def _ordered(values: set[str], order: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value for value in order if value in values)


def _list_line(flag: str, values: tuple[str, ...] | list[str]) -> str:
    return bazelrc_line("build", flag + "=" + ",".join(values))


def _summary(request: ConfigRequest) -> tuple[str, ...]:
    state = configuration(request)
    if state.preset_provenance:
        selection = ",".join(state.preset_provenance)
    elif state.exact_capabilities:
        selection = "exact native lists"
    else:
        selection = "explicit leaves"
    return (
        "Loom target selection: " + selection,
        "Loom architectures: "
        + ",".join(_ordered(state.architectures.values, ARCHITECTURES)),
        "Loom artifact formats: "
        + ",".join(_ordered(state.artifact_formats.values, ARTIFACT_FORMATS)),
        "Loom execution substrates: "
        + ",".join(_ordered(state.execution_substrates.values, EXECUTION_SUBSTRATES)),
    )


def _sections(request: ConfigRequest) -> tuple[BazelrcSection, ...]:
    state = configuration(request)
    return (
        BazelrcSection(
            "Optional encoded audio providers; raw PCM is always available.",
            (
                _list_line(
                    NATIVE_AUDIO_CODECS_FLAG,
                    _ordered(state.audio_codecs.values, AUDIO_CODECS),
                ),
            ),
        ),
        BazelrcSection(
            "Loom architecture, artifact, execution, and importer scope.",
            (
                *("# " + line for line in _summary(request)),
                bazelrc_line(
                    "build",
                    NATIVE_IMAGE_INPUT_FLAG + "=" + str(state.image_input).lower(),
                ),
                _list_line(
                    NATIVE_IMAGE_CODECS_FLAG,
                    _ordered(state.image_codecs.values, IMAGE_CODECS),
                ),
                _list_line(
                    NATIVE_ARCH_FLAG,
                    _ordered(state.architectures.values, ARCHITECTURES),
                ),
                *(
                    (
                        _list_line(
                            NATIVE_AMDGPU_TARGETS_FLAG,
                            sorted(state.amdgpu_targets),
                        ),
                    )
                    if state.amdgpu_targets is not None
                    else ()
                ),
                _list_line(
                    NATIVE_EXECUTE_FLAG,
                    _ordered(state.execution_substrates.values, EXECUTION_SUBSTRATES),
                ),
                _list_line(
                    NATIVE_IMPORT_FLAG, _ordered(state.importers.values, IMPORTERS)
                ),
                _list_line(
                    NATIVE_EMIT_FLAG,
                    _ordered(state.artifact_formats.values, ARTIFACT_FORMATS),
                ),
            ),
        ),
    )


_DEFINE_NAMES = (
    *target_presets.PRESETS_BY_OPTION,
    *target_presets.ARCH_DEFINES,
    *target_presets.EMIT_DEFINES,
    *target_presets.EXECUTE_DEFINES,
    *IMPORT_DEFINES,
    "LOOM_SERVE_IMAGE_INPUT",
    "LOOM_SERVE_IMAGE_CODECS",
    "LOOM_SERVE_AUDIO_CODECS",
)
_NATIVE_FLAGS = (
    *_NATIVE_SELECTIONS,
    NATIVE_AMDGPU_TARGETS_FLAG,
    NATIVE_IMAGE_INPUT_FLAG,
)

CONTRIBUTOR = Contributor(
    name=_NAME,
    initialize=_initialize,
    defines={name: _apply_define for name in _DEFINE_NAMES},
    native_options={name: _apply_native_option for name in _NATIVE_FLAGS},
    bazel_configs={
        preset.bazel_config: _apply_bazel_config
        for preset in target_presets.TARGET_PRESETS
    },
    resolve=_resolve,
    sections=_sections,
    summary=_summary,
)
