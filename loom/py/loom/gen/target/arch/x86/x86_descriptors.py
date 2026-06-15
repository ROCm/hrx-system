# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: x86 target-low descriptor views -> C descriptor tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TypeVar


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[5]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.target.low.low_descriptors import (  # noqa: E402
    generate_descriptor_set,
    generate_descriptor_set_shared_header,
    generate_descriptor_set_shared_source,
    write_descriptor_set_to_paths,
)
from loom.target.arch.x86.target_info import (  # noqa: E402
    X86DescriptorSetInfo,
    x86_descriptor_set_info_by_generator_target,
    x86_descriptor_set_storage_info_by_generator_target,
    x86_descriptor_set_view_infos_by_storage_generator_target,
)
from loom.target.descriptor_sets import resolve_descriptor_set  # noqa: E402
from loom.target.low_descriptors import (  # noqa: E402
    DescriptorSet,
    EnumDomain,
    RegClass,
    Resource,
    ScheduleClass,
)

_T = TypeVar("_T", RegClass, Resource, ScheduleClass, EnumDomain)


def _parse_view_headers(values: Sequence[str]) -> dict[str, Path]:
    view_headers: dict[str, Path] = {}
    for value in values:
        target, separator, path = value.partition("=")
        if not separator or not target or not path:
            raise ValueError("x86 descriptor --view-header must have form <target>=<path>")
        if target in view_headers:
            raise ValueError(f"duplicate x86 descriptor view header for {target}")
        view_headers[target] = Path(path)
    return view_headers


def _descriptor_set_for_info(info: X86DescriptorSetInfo) -> DescriptorSet:
    return resolve_descriptor_set(info.key)


def _view_infos_for_storage_target(
    storage_info: X86DescriptorSetInfo,
    view_headers: dict[str, Path],
) -> tuple[X86DescriptorSetInfo, ...]:
    view_infos = x86_descriptor_set_view_infos_by_storage_generator_target(storage_info.generator_target)
    expected_view_targets = {info.generator_target for info in view_infos}
    unknown_view_headers = set(view_headers) - expected_view_targets
    if unknown_view_headers:
        unknown_targets = ", ".join(sorted(unknown_view_headers))
        raise ValueError(f"x86 descriptor target {storage_info.generator_target} cannot emit view headers for: {unknown_targets}")
    return tuple(info for info in view_infos if info.generator_target in view_headers)


def _shared_storage_descriptor_set(
    storage_descriptor_set: DescriptorSet,
    view_descriptor_sets: Sequence[DescriptorSet],
) -> DescriptorSet:
    descriptor_sets = (storage_descriptor_set, *view_descriptor_sets)
    descriptors = []
    seen_descriptor_keys: set[str] = set()
    for descriptor_set in descriptor_sets:
        for descriptor in descriptor_set.descriptors:
            if descriptor.key in seen_descriptor_keys:
                continue
            descriptors.append(descriptor)
            seen_descriptor_keys.add(descriptor.key)

    def merge_by_name(item_groups: Sequence[Sequence[_T]]) -> tuple[_T, ...]:
        merged: list[_T] = []
        seen_names: set[str] = set()
        for item_group in item_groups:
            for item in item_group:
                item_name = item.name
                if item_name in seen_names:
                    continue
                merged.append(item)
                seen_names.add(item_name)
        return tuple(merged)

    return replace(
        storage_descriptor_set,
        c_table_prefix=f"{storage_descriptor_set.c_table_prefix}Storage",
        c_enum_prefix=f"{storage_descriptor_set.c_enum_prefix}_STORAGE",
        descriptors=tuple(descriptors),
        reg_classes=merge_by_name([item.reg_classes for item in descriptor_sets]),
        resources=merge_by_name([item.resources for item in descriptor_sets]),
        schedule_classes=merge_by_name([item.schedule_classes for item in descriptor_sets]),
        enum_domains=merge_by_name([item.enum_domains for item in descriptor_sets]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate x86 target-low descriptor C tables.")
    parser.add_argument(
        "--target",
        required=True,
        help="x86 descriptor generator target shard to generate.",
    )
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated descriptor header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated descriptor source path.",
    )
    parser.add_argument(
        "--view-header",
        action="append",
        default=[],
        help="Generated descriptor view header as <target>=<path>.",
    )
    args = parser.parse_args(argv)

    view_headers = _parse_view_headers(args.view_header)
    descriptor_set_info = x86_descriptor_set_info_by_generator_target(args.target)
    storage_info = x86_descriptor_set_storage_info_by_generator_target(args.target)
    if storage_info != descriptor_set_info:
        raise ValueError(f"x86 descriptor target {args.target} is a view of storage target {storage_info.generator_target}; generate the storage target with --view-header instead")
    view_infos = _view_infos_for_storage_target(descriptor_set_info, view_headers)
    descriptor_set = _descriptor_set_for_info(descriptor_set_info)
    if view_infos:
        view_descriptor_sets = tuple(_descriptor_set_for_info(info) for info in view_infos)
        storage_descriptor_set = _shared_storage_descriptor_set(
            descriptor_set,
            view_descriptor_sets,
        )
        generated = generate_descriptor_set(descriptor_set)
        args.header.parent.mkdir(parents=True, exist_ok=True)
        args.source.parent.mkdir(parents=True, exist_ok=True)
        args.header.write_text(generated.header, encoding="utf-8")
        args.source.write_text(
            generate_descriptor_set_shared_source(
                storage_descriptor_set,
                (*view_descriptor_sets, descriptor_set),
            ),
            encoding="utf-8",
        )
        for view_info, view_descriptor_set in zip(
            view_infos,
            view_descriptor_sets,
            strict=True,
        ):
            view_header_path = view_headers[view_info.generator_target]
            view_header = generate_descriptor_set_shared_header(
                storage_descriptor_set,
                view_descriptor_set,
            )
            view_header_path.parent.mkdir(parents=True, exist_ok=True)
            view_header_path.write_text(
                view_header,
                encoding="utf-8",
            )
        return 0

    write_descriptor_set_to_paths(
        descriptor_set,
        header_path=args.header,
        source_path=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
