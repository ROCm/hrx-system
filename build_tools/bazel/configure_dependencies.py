# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository source-dependency configuration policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from build_tools.bazel.configure_api import (
    BazelrcSection,
    ConfigRequest,
    Contributor,
    as_bazel_path,
    bazelrc_line,
)
from runtime.build_tools.bazel import configure as runtime_configure

DEPENDENCY_MODES = frozenset(("pinned", "package", "auto"))

_NAME = "dependencies"


@dataclass
class State:
    dependency_mode: str = "pinned"  # Repository-wide source mode.
    rocm_dependency_mode: str | None = None  # Optional ROCm-specific override.
    rocm_path: str | None = None  # Validated package root in Bazel path syntax.

    @property
    def effective_rocm_dependency_mode(self) -> str:
        if self.rocm_dependency_mode is not None:
            return self.rocm_dependency_mode
        if self.rocm_path is not None:
            return "package"
        return self.dependency_mode


def configuration(request: ConfigRequest) -> State:
    return cast(State, request.project_state(_NAME))


def _initialize(request: ConfigRequest) -> None:
    request.add_project_state(_NAME, State())


def _resolve_rocm_path(path: str) -> str:
    if not path:
        raise SystemExit("IREE_ROCM_PATH must not be empty.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"IREE_ROCM_PATH does not exist: {resolved}")
    if not (resolved / "include").is_dir():
        raise SystemExit(f"IREE_ROCM_PATH has no include directory: {resolved}")
    return as_bazel_path(resolved)


def _set_rocm_path(state: State, path: str) -> None:
    rocm_path = _resolve_rocm_path(path)
    if state.rocm_path is not None and state.rocm_path != rocm_path:
        raise SystemExit(f"Conflicting ROCm roots: {state.rocm_path} and {rocm_path}")
    state.rocm_path = rocm_path


def _set_dependency_mode(state: State, name: str, mode: str) -> None:
    dependency_mode = mode.lower()
    if dependency_mode not in DEPENDENCY_MODES:
        raise SystemExit(
            f"{name} must be one of pinned, package, or auto; got {mode!r}."
        )
    if name == "IREE_DEPENDENCY_MODE":
        state.dependency_mode = dependency_mode
    else:
        state.rocm_dependency_mode = dependency_mode


def _apply_option(request: ConfigRequest, name: str, value: str) -> None:
    state = configuration(request)
    if name == "IREE_ROCM_PATH":
        _set_rocm_path(state, value)
    else:
        _set_dependency_mode(state, name, value)


def _finalize(request: ConfigRequest) -> None:
    state = configuration(request)
    runtime = runtime_configure.configuration(request)
    if (
        runtime.drivers.intersection(runtime_configure.ROCM_DRIVERS)
        and not state.rocm_path
    ):
        environment_path = os.environ.get("IREE_ROCM_PATH")
        if environment_path:
            _set_rocm_path(state, environment_path)


def _validate(request: ConfigRequest) -> None:
    state = configuration(request)
    rocm_drivers = runtime_configure.configuration(request).drivers.intersection(
        runtime_configure.ROCM_DRIVERS
    )
    if (
        rocm_drivers
        and state.effective_rocm_dependency_mode == "package"
        and not state.rocm_path
    ):
        raise SystemExit(
            "ROCm package dependency mode requires -DIREE_ROCM_PATH=/path/to/rocm, "
            "--repo_env=IREE_ROCM_PATH=/path/to/rocm, or IREE_ROCM_PATH in the "
            "environment when enabling {}. Set IREE_ROCM_DEPENDENCY_MODE=pinned "
            "or auto to use pinned ROCm headers without a ROCm root.".format(
                ", ".join(sorted(rocm_drivers))
            )
        )


def _sections(request: ConfigRequest) -> tuple[BazelrcSection, ...]:
    state = configuration(request)
    runtime = runtime_configure.configuration(request)
    dependency_lines = [
        bazelrc_line(
            "common", "--repo_env=IREE_DEPENDENCY_MODE=" + state.dependency_mode
        )
    ]
    if state.rocm_dependency_mode or state.rocm_path:
        dependency_lines.append(
            bazelrc_line(
                "common",
                "--repo_env=IREE_ROCM_DEPENDENCY_MODE="
                + state.effective_rocm_dependency_mode,
            )
        )
    device_toolchain = (
        "rocm"
        if "amdgpu" in runtime.drivers and state.rocm_path is not None
        else "none"
    )
    dependency_lines.append(
        bazelrc_line(
            "common",
            "--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=" + device_toolchain,
        )
    )
    if state.rocm_path is not None:
        dependency_lines.append(
            bazelrc_line("common", "--repo_env=IREE_ROCM_PATH=" + state.rocm_path)
        )
    return (
        BazelrcSection(
            "Source dependency and optional AMDGPU toolchain configuration.",
            tuple(dependency_lines),
        ),
    )


_OPTIONS = ("IREE_ROCM_PATH", "IREE_DEPENDENCY_MODE", "IREE_ROCM_DEPENDENCY_MODE")

CONTRIBUTOR = Contributor(
    name=_NAME,
    initialize=_initialize,
    defines={name: _apply_option for name in _OPTIONS},
    repo_env={name: _apply_option for name in _OPTIONS},
    finalize=_finalize,
    validate=_validate,
    sections=_sections,
)
