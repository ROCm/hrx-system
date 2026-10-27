# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates build-system views of Loom target capability presets."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from loom.build_tools import target_presets
from loom.gen import bootstrap
from loom.gen.support.generated_file import (
    GeneratedFileSet,
    line_comment_header,
    maintain_generated_file_set,
)

DESCRIPTION = "Loom target capability presets"
REGENERATE_COMMAND = "python3 loom/py/loom/gen/run.py capability_presets --in-place"
_GENERATOR = "loom.gen.target.capability_presets"


def _starlark_list(name: str, values: Sequence[str]) -> list[str]:
    return [f"{name} = [", *(f'    "{value}",' for value in values), "]"]


def _cmake_list(name: str, values: Sequence[str]) -> list[str]:
    return [f"set({name}", *(f'  "{value}"' for value in values), ")"]


def render_capabilities_bzl() -> str:
    default = target_presets.resolve_presets(("default",))
    lines = [
        "# Copyright 2026 The IREE Authors",
        "#",
        "# Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "# See https://llvm.org/LICENSE.txt for license information.",
        "# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("#", generator=_GENERATOR, regenerate=REGENERATE_COMMAND),
        "",
        '"""Canonical Loom compiler capability values and ordinary defaults."""',
        "",
        *_starlark_list("LOOM_ARCHITECTURES", target_presets.ARCHITECTURES),
        "",
        *_starlark_list("LOOM_ARTIFACT_FORMATS", target_presets.ARTIFACT_FORMATS),
        "",
        *_starlark_list("LOOM_EXECUTION_SUBSTRATES", target_presets.EXECUTION_SUBSTRATES),
        "",
        "# The ordinary build contains every production target except WebAssembly.",
        *_starlark_list("LOOM_DEFAULT_ARCHITECTURES", default.architectures),
        "",
        *_starlark_list("LOOM_DEFAULT_ARTIFACT_FORMATS", default.artifacts),
        "",
        *_starlark_list("LOOM_DEFAULT_EXECUTION_SUBSTRATES", default.execution),
        "",
    ]
    return "\n".join(lines)


def render_bazelrc() -> str:
    lines = [
        *line_comment_header("#", generator=_GENERATOR, regenerate=REGENERATE_COMMAND),
        "# Each named config writes complete canonical capability lists.",
    ]
    for preset in target_presets.TARGET_PRESETS:
        capabilities = target_presets.resolve_presets((preset.key,))
        lines.extend(
            [
                "",
                f"# {preset.description}",
                f"build:{preset.bazel_config} --//loom/config/arch:enable=" + ",".join(capabilities.architectures),
                f"build:{preset.bazel_config} --//loom/config/emit:enable=" + ",".join(capabilities.artifacts),
                f"build:{preset.bazel_config} --//loom/config/execute:enable=" + ",".join(capabilities.execution),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_cmake() -> str:
    preset_keys = tuple(preset.key.upper() for preset in target_presets.TARGET_PRESETS)
    family_keys = tuple(key for key in preset_keys if key != "DEFAULT")
    lines = [
        *line_comment_header("#", generator=_GENERATOR, regenerate=REGENERATE_COMMAND),
        "# Data only: loom/project.cmake owns preset resolution.",
        "",
        *_cmake_list("_LOOM_CAPABILITY_ARCHITECTURES", target_presets.ARCHITECTURES),
        *_cmake_list(
            "_LOOM_CAPABILITY_ARTIFACT_FORMATS",
            target_presets.ARTIFACT_FORMATS,
        ),
        *_cmake_list(
            "_LOOM_CAPABILITY_EXECUTION_SUBSTRATES",
            target_presets.EXECUTION_SUBSTRATES,
        ),
        *_cmake_list("_LOOM_TARGET_PRESET_KEYS", preset_keys),
        *_cmake_list("_LOOM_TARGET_FAMILY_PRESET_KEYS", family_keys),
    ]
    for preset in target_presets.TARGET_PRESETS:
        key = preset.key.upper()
        resolved = target_presets.resolve_presets((preset.key,))
        prefix = f"_LOOM_TARGET_PRESET_{key}"
        lines.extend(
            [
                "",
                f'set({prefix}_OPTION "{preset.option}")',
                f'set({prefix}_DESCRIPTION "{preset.description}")',
                *_cmake_list(prefix + "_ARCHITECTURES", resolved.architectures),
                *_cmake_list(prefix + "_ARTIFACTS", resolved.artifacts),
                *_cmake_list(prefix + "_EXECUTION", resolved.execution),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def checked_in_file_set() -> GeneratedFileSet:
    return GeneratedFileSet.from_mapping(
        {
            "loom/config/capabilities.bzl": render_capabilities_bzl(),
            "loom/config/target_presets.bazelrc": render_bazelrc(),
            "loom/config/target_presets.cmake": render_cmake(),
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    args = parser.parse_args(argv)
    result = maintain_generated_file_set(
        bootstrap.find_repo_root(),
        checked_in_file_set(),
        mode="update" if args.in_place else "check",
        description=DESCRIPTION,
        regenerate_command=REGENERATE_COMMAND,
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
