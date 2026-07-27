# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU ISA XML -> target-low descriptor C tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.target.low.low_descriptors import (  # noqa: E402
    generate_descriptor_set,
    generate_descriptor_set_shared_source,
    write_descriptor_set_to_paths,
)
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS,
    build_amdgpu_core_descriptor_set_from_spec,
)
from loom.target.arch.amdgpu.isa_xml import parse_amdgpu_isa_xml_path  # noqa: E402
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AmdgpuDescriptorSetInfo,
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_storage_info_by_generator_target,
    amdgpu_descriptor_set_view_infos_by_storage_generator_target,
)
from loom.target.low_descriptors import DescriptorSet  # noqa: E402


def _parse_view_headers(values: Sequence[str]) -> dict[str, Path]:
    view_headers: dict[str, Path] = {}
    for value in values:
        target, separator, path = value.partition("=")
        if not separator or not target or not path:
            raise ValueError("AMDGPU descriptor --view-header must have form <target>=<path>")
        if target in view_headers:
            raise ValueError(f"duplicate AMDGPU descriptor view header for {target}")
        view_headers[target] = Path(path)
    return view_headers


def _view_infos_for_storage_target(
    storage_info: AmdgpuDescriptorSetInfo,
    view_headers: dict[str, Path],
) -> tuple[AmdgpuDescriptorSetInfo, ...]:
    view_infos = amdgpu_descriptor_set_view_infos_by_storage_generator_target(storage_info.generator_target)
    expected_view_targets = {info.generator_target for info in view_infos}
    unknown_view_headers = set(view_headers) - expected_view_targets
    if unknown_view_headers:
        unknown_targets = ", ".join(sorted(unknown_view_headers))
        raise ValueError(f"AMDGPU descriptor target {storage_info.generator_target} cannot emit view headers for: {unknown_targets}")
    return view_infos


def _shared_storage_descriptor_set(
    storage_descriptor_set: DescriptorSet,
    view_descriptor_sets: Sequence[DescriptorSet],
) -> DescriptorSet:
    backing_descriptor_set = max(
        (storage_descriptor_set, *view_descriptor_sets),
        key=lambda descriptor_set: (
            len(descriptor_set.descriptors),
            len(descriptor_set.resources),
            len(descriptor_set.schedule_classes),
        ),
    )
    return replace(
        storage_descriptor_set,
        descriptors=backing_descriptor_set.descriptors,
        resources=backing_descriptor_set.resources,
        schedule_classes=backing_descriptor_set.schedule_classes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU target-low descriptor C tables from vendor ISA XML.")
    parser.add_argument(
        "--target",
        required=True,
        choices=AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS,
        help="AMDGPU descriptor target shard to generate.",
    )
    parser.add_argument(
        "--xml",
        required=True,
        type=Path,
        help="Path to the AMD machine-readable ISA XML file for the target family.",
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
    descriptor_set_info = amdgpu_descriptor_set_info_by_generator_target(args.target)
    storage_info = amdgpu_descriptor_set_storage_info_by_generator_target(args.target)
    if storage_info != descriptor_set_info:
        raise ValueError(f"AMDGPU descriptor target {args.target} is a view of storage target {storage_info.generator_target}; generate the storage target with --view-header instead")
    view_infos = _view_infos_for_storage_target(descriptor_set_info, view_headers)
    spec = parse_amdgpu_isa_xml_path(args.xml)
    descriptor_set = build_amdgpu_core_descriptor_set_from_spec(args.target, spec)
    if view_infos:
        view_descriptor_sets = tuple(build_amdgpu_core_descriptor_set_from_spec(info.generator_target, spec) for info in view_infos)
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
                (descriptor_set, *view_descriptor_sets),
            ),
            encoding="utf-8",
        )
        for view_info, view_descriptor_set in zip(view_infos, view_descriptor_sets, strict=True):
            view_header_path = view_headers.get(view_info.generator_target)
            if view_header_path is None:
                continue
            view_generated = generate_descriptor_set(view_descriptor_set)
            view_header_path.parent.mkdir(parents=True, exist_ok=True)
            view_header_path.write_text(
                view_generated.header,
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
