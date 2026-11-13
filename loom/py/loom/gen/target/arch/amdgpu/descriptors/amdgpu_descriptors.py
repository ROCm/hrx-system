# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU ISA XML -> target-low descriptor C tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.support.string_pool import CStringBank  # noqa: E402
from loom.gen.target.arch.amdgpu.amdgpu_target_table_family import (  # noqa: E402
    AmdgpuTargetTableFamily,
    amdgpu_target_table_family,
)
from loom.gen.target.low.compiled import (  # noqa: E402
    CompiledNativeAsmValue,
    DescriptorIdentityDomain,
    GeneratedDescriptorSetFamily,
)
from loom.gen.target.low.low_descriptors import (  # noqa: E402
    compile_descriptor_set_family,
    descriptor_identity_domain,
    descriptor_set_family_string_values,
    generate_descriptor_set_family,
    write_descriptor_set_to_paths,
)
from loom.gen.target.low.native_assembly import NativeAsmValuePool  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS,
    amdgpu_core_descriptor_set_instruction_names_by_isa_key,
    amdgpu_descriptor_ref_keys,
    build_amdgpu_core_descriptor_sets_from_specs,
)
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    parse_amdgpu_isa_xml_paths_for_instructions,
)
from loom.target.arch.amdgpu.names import (  # noqa: E402
    amdgpu_low_descriptor_storage_view_provider_symbol,
)
from loom.target.low_descriptors import (  # noqa: E402
    Descriptor,
    DescriptorSet,
)

AMDGPU_DESCRIPTOR_IDENTITY_PUBLIC_HEADER = "loom/target/arch/amdgpu/descriptors/descriptor_identities.h"
AMDGPU_DESCRIPTOR_IDENTITY_TABLE_SYMBOL = "loom_amdgpu_descriptor_identities"


def amdgpu_descriptor_identity_domain(
    descriptors: Sequence[Descriptor] = (),
) -> DescriptorIdentityDomain:
    """Returns the complete AMDGPU identity domain in target-ref order."""

    keys = amdgpu_descriptor_ref_keys()
    return descriptor_identity_domain(
        descriptors,
        keys=keys,
        table_symbol=AMDGPU_DESCRIPTOR_IDENTITY_TABLE_SYMBOL,
        public_header=AMDGPU_DESCRIPTOR_IDENTITY_PUBLIC_HEADER,
    )


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


def _parse_isa_xml_paths(
    values: Sequence[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition(":")
        if not separator or not key or not path:
            raise ValueError("AMDGPU descriptor --isa-xml entries must be key:path pairs")
        if key in paths:
            raise ValueError(f"AMDGPU descriptor ISA XML key '{key}' is duplicate")
        paths[key] = Path(path)
    return paths


def _validate_view_headers(
    family: AmdgpuTargetTableFamily,
    view_headers: Mapping[str, Path],
) -> None:
    expected_view_targets = {info.generator_target for info in family.view_infos}
    unknown_view_headers = set(view_headers) - expected_view_targets
    if unknown_view_headers:
        unknown_targets = ", ".join(sorted(unknown_view_headers))
        raise ValueError(f"AMDGPU descriptor target {family.storage_info.generator_target} cannot emit view headers for: {unknown_targets}")


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


def _amdgpu_descriptor_table_family_specs(
    family: AmdgpuTargetTableFamily,
    descriptor_sets: Mapping[str, DescriptorSet],
    *,
    source_public_header: str,
) -> tuple[DescriptorSet, tuple[DescriptorSet, ...]]:
    """Materializes the shared storage and view specs for one table family."""

    storage_descriptor_set = descriptor_sets[family.storage_info.generator_target]
    representation_targets = {info.generator_target for info in family.representation_infos}
    view_descriptor_sets = tuple(
        replace(
            descriptor_sets[info.generator_target],
            function_name=amdgpu_low_descriptor_storage_view_provider_symbol(info.key, family.storage_info.generator_target),
            descriptor_set_ordinal=storage_descriptor_set.descriptor_set_ordinal,
        )
        if info.generator_target in representation_targets
        else descriptor_sets[info.generator_target]
        for info in family.descriptor_view_infos
    )
    shared_storage_descriptor_set = replace(
        _shared_storage_descriptor_set(
            storage_descriptor_set,
            view_descriptor_sets,
        ),
        public_header=source_public_header,
    )
    return shared_storage_descriptor_set, (
        storage_descriptor_set,
        *view_descriptor_sets,
    )


def amdgpu_descriptor_table_family_string_values(
    family: AmdgpuTargetTableFamily,
    descriptor_sets: Mapping[str, DescriptorSet],
) -> frozenset[str]:
    """Returns a vocabulary covering every string retained by one family."""

    storage_spec, view_specs = _amdgpu_descriptor_table_family_specs(
        family,
        descriptor_sets,
        source_public_header=descriptor_sets[family.storage_info.generator_target].public_header,
    )
    return descriptor_set_family_string_values(storage_spec, view_specs)


def amdgpu_descriptor_table_family_native_spans(
    family: AmdgpuTargetTableFamily,
    descriptor_sets: Mapping[str, DescriptorSet],
) -> tuple[tuple[CompiledNativeAsmValue, ...], ...]:
    """Returns the native recipes used by storage and every representation view."""
    storage_spec, view_specs = _amdgpu_descriptor_table_family_specs(family, descriptor_sets, source_public_header=descriptor_sets[family.storage_info.generator_target].public_header)
    compiled, _ = compile_descriptor_set_family(storage_spec, view_specs)
    storage = compiled.asm_table_storage
    return tuple(tuple(storage.native_values[layout.native_assembly_value_start : layout.native_assembly_value_start + layout.native_assembly_value_count]) for layout in storage.layouts)


def generate_amdgpu_descriptor_table_family(
    family: AmdgpuTargetTableFamily,
    descriptor_sets: Mapping[str, DescriptorSet],
    *,
    source_public_header: str,
    primary_strings: CStringBank | None = None,
    shared_strings: CStringBank | None = None,
    native_asm_values: NativeAsmValuePool | None = None,
) -> GeneratedDescriptorSetFamily:
    """Generates descriptor storage and views from a materialized corpus."""

    storage_spec, view_specs = _amdgpu_descriptor_table_family_specs(
        family,
        descriptor_sets,
        source_public_header=source_public_header,
    )
    return generate_descriptor_set_family(
        storage_spec,
        view_specs,
        primary_strings=primary_strings,
        shared_strings=shared_strings,
        native_asm_values=native_asm_values,
        identity_domain=amdgpu_descriptor_identity_domain(tuple(storage_spec.descriptors) + tuple(descriptor for view_spec in view_specs for descriptor in view_spec.descriptors)),
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
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as <key>:<path>.",
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
    family = amdgpu_target_table_family(args.target)
    _validate_view_headers(family, view_headers)
    isa_specs = parse_amdgpu_isa_xml_paths_for_instructions(
        _parse_isa_xml_paths(args.isa_xml),
        amdgpu_core_descriptor_set_instruction_names_by_isa_key(family.descriptor_set_infos),
    )
    descriptor_sets = build_amdgpu_core_descriptor_sets_from_specs(
        family.generator_targets,
        isa_specs,
    )
    descriptor_set = descriptor_sets[family.storage_info.generator_target]
    if family.descriptor_view_infos:
        generated = generate_amdgpu_descriptor_table_family(
            family,
            descriptor_sets,
            source_public_header=descriptor_set.public_header,
        )
        write_text_file(args.header, generated.view_headers[0])
        write_text_file(args.source, generated.source)
        for view_info, view_header in zip(
            family.view_infos,
            generated.view_headers[1 : len(family.view_infos) + 1],
            strict=True,
        ):
            view_header_path = view_headers.get(view_info.generator_target)
            if view_header_path is None:
                continue
            write_text_file(view_header_path, view_header)
        return 0

    write_descriptor_set_to_paths(
        descriptor_set,
        header_path=args.header,
        source_path=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
