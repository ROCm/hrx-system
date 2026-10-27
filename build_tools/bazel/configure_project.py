# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository composition for project-owned Bazel configuration policy."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOOM_PY_ROOT = str(_REPO_ROOT / "loom" / "py")
if _LOOM_PY_ROOT not in sys.path:
    sys.path.insert(0, _LOOM_PY_ROOT)

from build_tools.bazel import configure_dependencies
from build_tools.bazel.configure_api import BazelrcSection, ConfigRequest
from build_tools.d3d12.config import configure as d3d12_configure
from build_tools.vulkan.config import configure as vulkan_configure
from libamdf.build_tools.bazel import configure as libamdf_configure
from loom.build_tools.bazel import configure as loom_configure
from runtime.build_tools.bazel import configure as runtime_configure

_CONTRIBUTORS = (
    runtime_configure.CONTRIBUTOR,
    libamdf_configure.CONTRIBUTOR,
    vulkan_configure.CONTRIBUTOR,
    d3d12_configure.CONTRIBUTOR,
    loom_configure.CONTRIBUTOR,
    configure_dependencies.CONTRIBUTOR,
)

_Handler = Callable[..., None]


def _index_handlers(attribute: str) -> dict[str, _Handler]:
    handlers: dict[str, _Handler] = {}
    owners: dict[str, str] = {}
    for contributor in _CONTRIBUTORS:
        contributor_handlers = cast(
            Mapping[str, _Handler], getattr(contributor, attribute)
        )
        for option, handler in contributor_handlers.items():
            if option in handlers:
                raise RuntimeError(
                    f"project configure option {option} has multiple owners: "
                    f"{owners[option]}, {contributor.name}"
                )
            handlers[option] = handler
            owners[option] = contributor.name
    return handlers


_DEFINE_HANDLERS = _index_handlers("defines")
_NATIVE_OPTION_HANDLERS = _index_handlers("native_options")
_BAZEL_CONFIG_HANDLERS = _index_handlers("bazel_configs")
_REPO_ENV_HANDLERS = _index_handlers("repo_env")
_REMOVED_OPTION_HANDLERS = _index_handlers("removed_options")


def _dispatch(handlers: Mapping[str, _Handler], key: str, *args: object) -> bool:
    handler = handlers.get(key)
    if handler is None:
        return False
    handler(*args)
    return True


def initialize_request(request: ConfigRequest) -> None:
    for contributor in _CONTRIBUTORS:
        contributor.initialize(request)


def apply_define(request: ConfigRequest, name: str, value: str) -> bool:
    return _dispatch(_DEFINE_HANDLERS, name, request, name, value)


def apply_native_bazel_arg(request: ConfigRequest, arg: str) -> bool:
    option_name = arg.split("=", 1)[0]
    if _dispatch(_REMOVED_OPTION_HANDLERS, option_name, request, arg):
        return True

    if arg.startswith("--config="):
        config_name = arg.removeprefix("--config=")
        return _dispatch(_BAZEL_CONFIG_HANDLERS, config_name, request, config_name)

    if arg.startswith("--repo_env="):
        repo_env = arg.removeprefix("--repo_env=")
        if "=" not in repo_env:
            raise SystemExit("--repo_env must use --repo_env=NAME=VALUE syntax.")
        name, value = repo_env.split("=", 1)
        return _dispatch(_REPO_ENV_HANDLERS, name, request, name, value)

    if option_name not in _NATIVE_OPTION_HANDLERS:
        return False
    if "=" not in arg:
        raise SystemExit(f"{option_name} must use --flag=value syntax.")
    return _dispatch(
        _NATIVE_OPTION_HANDLERS,
        option_name,
        request,
        option_name,
        arg.split("=", 1)[1],
    )


def resolve_request(request: ConfigRequest) -> None:
    for contributor in _CONTRIBUTORS:
        contributor.resolve(request)
    for contributor in _CONTRIBUTORS:
        contributor.finalize(request)


def validate_request(request: ConfigRequest) -> None:
    for contributor in _CONTRIBUTORS:
        contributor.validate(request)


def config_sections(request: ConfigRequest) -> tuple[BazelrcSection, ...]:
    return tuple(
        section
        for contributor in _CONTRIBUTORS
        for section in contributor.sections(request)
    )


def request_summary(request: ConfigRequest) -> tuple[str, ...]:
    return tuple(
        line for contributor in _CONTRIBUTORS for line in contributor.summary(request)
    )
