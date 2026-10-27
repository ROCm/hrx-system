# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime-owned Bazel configuration policy."""

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
from libamdf.build_tools.bazel import configure as libamdf_configure

HOST_DRIVERS = ("task",)
AMD_ENGINES = ("amd-gpu", "amd-xdna")
DRIVER_DEFINES = {
    "IREE_HAL_DRIVER_TASK": "task",
    "IREE_HAL_DRIVER_AMDGPU": "amdgpu",
    "IREE_HAL_DRIVER_VULKAN": "vulkan",
    "IREE_HAL_DRIVER_WEBGPU": "webgpu",
    "IREE_HAL_DRIVER_AMD": "amd",
    "IREE_HAL_DRIVER_AMD_GPU": "amd-gpu",
    "IREE_HAL_DRIVER_AMD_XDNA": "amd-xdna",
}
ALL_DRIVERS = tuple(DRIVER_DEFINES.values())
DEFAULT_DRIVERS = (*HOST_DRIVERS, "amd")
SUPPORTED_ENABLED_DRIVERS = frozenset(ALL_DRIVERS).difference(("amd",))
ROCM_DRIVERS = frozenset(("amdgpu",))

NATIVE_DRIVER_FLAG = "--//runtime/config/hal:drivers"
NATIVE_AMDGPU_TARGETS_FLAG = "--//runtime/src/iree/hal/drivers/amdgpu:targets"
NATIVE_RDMA_FLAG = "--//runtime/config/net:rdma"

_REMOVED_OPTIONS = (
    "--enable-driver",
    "--include-driver",
    "--exclude-driver",
    "--rocm-path",
)
_NAME = "runtime"


@dataclass
class State:
    # Selected registry drivers, including any unresolved AMD umbrella.
    drivers: set[str] = field(default_factory=lambda: set(DEFAULT_DRIVERS))
    driver_source: str | None = None  # Portable or native selection spelling.
    # Explicit per-engine choices that override the AMD umbrella in either order.
    amd_engine_overrides: dict[str, bool] = field(default_factory=dict)
    amdgpu_targets: set[str] | None = None  # Legacy AMDGPU device binaries.
    rdma_enabled: bool = False  # Whether native RDMA resources are built.
    rdma_source: str | None = None  # Portable or native RDMA spelling.


def configuration(request: ConfigRequest) -> State:
    return cast(State, request.project_state(_NAME))


def _initialize(request: ConfigRequest) -> None:
    request.add_project_state(_NAME, State())


def _set_driver(state: State, driver: str, enabled: bool) -> None:
    if state.driver_source == "native":
        raise SystemExit(
            "Do not mix portable -DIREE_HAL_DRIVER_* options with the "
            f"native {NATIVE_DRIVER_FLAG}=... Bazel option."
        )
    state.driver_source = "portable"
    if driver in AMD_ENGINES:
        state.amd_engine_overrides[driver] = enabled
    if enabled:
        state.drivers.add(driver)
    else:
        state.drivers.discard(driver)


def _set_driver_list(state: State, value: str) -> None:
    if state.driver_source == "portable":
        raise SystemExit(
            "Do not mix portable -DIREE_HAL_DRIVER_* options with the "
            f"native {NATIVE_DRIVER_FLAG}=... Bazel option."
        )
    drivers = parse_string_list(value)
    unknown = drivers.difference(ALL_DRIVERS)
    if unknown:
        raise SystemExit("Unknown runtime HAL driver(s): " + ", ".join(sorted(unknown)))
    state.driver_source = "native"
    state.drivers = drivers


def _set_rdma(state: State, enabled: bool, source: str) -> None:
    if state.rdma_source not in (None, source):
        raise SystemExit(
            "Do not mix portable -DIREE_NET_RDMA with the native "
            f"{NATIVE_RDMA_FLAG}=... Bazel option."
        )
    state.rdma_source = source
    state.rdma_enabled = enabled


def _apply_define(request: ConfigRequest, name: str, value: str) -> None:
    state = configuration(request)
    if name == "IREE_NET_RDMA":
        _set_rdma(state, parse_bool(name, value), "portable")
    else:
        _set_driver(state, DRIVER_DEFINES[name], parse_bool(name, value))


def _apply_native_option(request: ConfigRequest, name: str, value: str) -> None:
    state = configuration(request)
    if name == NATIVE_DRIVER_FLAG:
        _set_driver_list(state, value)
    elif name == NATIVE_AMDGPU_TARGETS_FLAG:
        state.amdgpu_targets = parse_string_list(value)
    elif name == NATIVE_RDMA_FLAG:
        _set_rdma(state, parse_bool(name, value), "native")
    else:
        raise RuntimeError(f"unhandled declared runtime native option {name}")


def _removed_option(request: ConfigRequest, arg: str) -> None:
    del request
    name = arg.split("=", 1)[0]
    raise SystemExit(
        f"{name} was removed. Use portable -DIREE_HAL_DRIVER_* options and "
        "-DIREE_ROCM_PATH=/path/to/rocm when package-mode ROCm headers are "
        "required. The native Bazel form is "
        f"{NATIVE_DRIVER_FLAG}=amdgpu,task "
        "--repo_env=IREE_ROCM_PATH=/path/to/rocm."
    )


def _resolve(request: ConfigRequest) -> None:
    state = configuration(request)
    if "amd" in state.drivers:
        state.drivers.remove("amd")
        state.drivers.update(AMD_ENGINES)
    for engine, enabled in state.amd_engine_overrides.items():
        if enabled:
            state.drivers.add(engine)
        else:
            state.drivers.discard(engine)
    if state.drivers.intersection(AMD_ENGINES):
        libamdf_configure.require(request, "an enabled native AMD HAL engine")


def _validate(request: ConfigRequest) -> None:
    unsupported = configuration(request).drivers.difference(SUPPORTED_ENABLED_DRIVERS)
    if unsupported:
        raise SystemExit(
            "Driver enablement is not yet configured for: {}. Leave these drivers "
            "disabled until their repositories have been ported.".format(
                ", ".join(sorted(unsupported))
            )
        )


def _sections(request: ConfigRequest) -> tuple[BazelrcSection, ...]:
    state = configuration(request)
    driver_list = ",".join(driver for driver in ALL_DRIVERS if driver in state.drivers)
    driver_lines = [bazelrc_line("build", NATIVE_DRIVER_FLAG + "=" + driver_list)]
    if state.amdgpu_targets is not None:
        driver_lines.append(
            bazelrc_line(
                "build",
                NATIVE_AMDGPU_TARGETS_FLAG
                + "="
                + ",".join(sorted(state.amdgpu_targets)),
            )
        )
    return (
        BazelrcSection("Runtime driver registry scope.", tuple(driver_lines)),
        BazelrcSection(
            "Native RDMA resources; no host carrier is implicitly registered.",
            (
                bazelrc_line(
                    "build", NATIVE_RDMA_FLAG + "=" + str(state.rdma_enabled).lower()
                ),
            ),
        ),
    )


CONTRIBUTOR = Contributor(
    name=_NAME,
    initialize=_initialize,
    defines={
        **{name: _apply_define for name in DRIVER_DEFINES},
        "IREE_NET_RDMA": _apply_define,
    },
    native_options={
        NATIVE_DRIVER_FLAG: _apply_native_option,
        NATIVE_AMDGPU_TARGETS_FLAG: _apply_native_option,
        NATIVE_RDMA_FLAG: _apply_native_option,
    },
    removed_options={name: _removed_option for name in _REMOVED_OPTIONS},
    resolve=_resolve,
    validate=_validate,
    sections=_sections,
)
