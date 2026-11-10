# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU ISA XML -> global descriptor table family."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.support.string_pool import (  # noqa: E402
    CStringBank,
    CStringPool,
    emit_c_string_pool,
    layout_c_string_pool,
)
from loom.gen.target.arch.amdgpu.amdgpu_target_table_family import (  # noqa: E402
    amdgpu_target_table_family,
)
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_descriptors import (  # noqa: E402
    AMDGPU_DESCRIPTOR_IDENTITY_PUBLIC_HEADER,
    AMDGPU_DESCRIPTOR_IDENTITY_TABLE_SYMBOL,
    amdgpu_descriptor_identity_domain,
    amdgpu_descriptor_table_family_native_spans,
    amdgpu_descriptor_table_family_string_values,
)
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_lower_capabilities import (  # noqa: E402
    generate_lower_capability_header,
)
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_planning_table_inputs import (  # noqa: E402
    AmdgpuPlanningTableInputs,
    load_amdgpu_planning_table_inputs,
)
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_vopd_component_tables import (  # noqa: E402
    amdgpu_vopd_instruction_encoding_names_by_isa_key,
    generate_vopd_component_table_outputs,
)
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_wait_packet_tables import (  # noqa: E402
    generate_wait_packet_table_outputs,
)
from loom.gen.target.arch.amdgpu.refs.amdgpu_target_refs import (  # noqa: E402
    generate_target_ref_outputs,
    select_target_ref_descriptor_set_infos,
)
from loom.gen.target.low import c_spelling  # noqa: E402
from loom.gen.target.low.native_assembly import NativeAsmValuePool  # noqa: E402
from loom.target.arch.amdgpu.descriptors.matrix import MATRIX_OPERAND_PLACEMENT_QUALIFICATIONS  # noqa: E402
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    amdgpu_descriptor_set_ordinal,
    amdgpu_descriptor_set_storage_info_by_generator_target,
    amdgpu_target_descriptor_set_key,
    sorted_processor_infos,
    sorted_target_infos,
)
from loom.target.low_descriptors import (  # noqa: E402
    Descriptor,
    DescriptorSet,
    OperandRole,
)


def _descriptor_identity_header(identity_count: int) -> str:
    guard = "LOOM_TARGET_ARCH_AMDGPU_DESCRIPTORS_DESCRIPTOR_IDENTITIES_H_"
    return "\n".join(
        [
            *line_comment_header(
                "//",
                generator=("loom.gen.target.arch.amdgpu.descriptors.amdgpu_descriptor_tables"),
            ),
            "",
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            '#include "loom/codegen/low/descriptors.h"',
            "",
            f"#define LOOM_AMDGPU_DESCRIPTOR_IDENTITY_COUNT UINT16_C({identity_count})",
            "",
            (f"extern const loom_low_descriptor_identity_t {AMDGPU_DESCRIPTOR_IDENTITY_TABLE_SYMBOL}[LOOM_AMDGPU_DESCRIPTOR_IDENTITY_COUNT];"),
            "",
            f"#endif  // {guard}",
            "",
        ]
    )


def _descriptor_identity_source(
    descriptor_sets: Sequence[DescriptorSet],
    descriptor_strings: CStringBank,
    shared_strings: CStringBank,
) -> str:
    descriptors_by_key: dict[str, Descriptor] = {}
    descriptors: list[Descriptor] = []
    for descriptor_set in descriptor_sets:
        for descriptor in descriptor_set.descriptors:
            previous = descriptors_by_key.get(descriptor.key)
            if previous is not None and previous.semantic_tag != descriptor.semantic_tag:
                raise ValueError(f"descriptor identity '{descriptor.key}' has inconsistent semantic tags")
            descriptors_by_key[descriptor.key] = descriptor
            descriptors.append(descriptor)
    identity_domain = amdgpu_descriptor_identity_domain(descriptors)
    selected_identities = tuple((ordinal, key, descriptors_by_key[key]) for ordinal, key in enumerate(identity_domain.keys) if key in descriptors_by_key)

    pool = CStringPool("AMDGPU_DESCRIPTOR_IDENTITY")
    for _, key, descriptor in selected_identities:
        pool.intern(f"descriptor_{key}", key)
        if descriptor.semantic_tag is not None:
            pool.intern(f"semantic_{key}", descriptor.semantic_tag)
    layout = layout_c_string_pool(
        pool,
        shared_strings,
        primary=descriptor_strings,
    )
    lines = [
        *line_comment_header(
            "//",
            generator=("loom.gen.target.arch.amdgpu.descriptors.amdgpu_descriptor_tables"),
        ),
        "",
        f'#include "{AMDGPU_DESCRIPTOR_IDENTITY_PUBLIC_HEADER}"',
        "",
        f'#include "{descriptor_strings.header}"',
        f'#include "{shared_strings.header}"',
        "",
        *emit_c_string_pool(
            pool,
            "kAmdgpuDescriptorIdentityStringData",
            layout=layout,
        ),
        (f"const loom_low_descriptor_identity_t {AMDGPU_DESCRIPTOR_IDENTITY_TABLE_SYMBOL}[LOOM_AMDGPU_DESCRIPTOR_IDENTITY_COUNT] = {{"),
    ]
    for ordinal, key, descriptor in selected_identities:
        lines.extend(
            [
                f"    [{ordinal}] = {{",
                f"        .key_string_ref = {pool.ref(f'descriptor_{key}')},",
                (f"        .semantic_tag_string_ref = {c_spelling.optional_string_expr(pool, f'semantic_{key}' if descriptor.semantic_tag is not None else None)},"),
                "    },",
            ]
        )
    lines.extend(
        [
            "};",
            f'static_assert(IREE_ARRAYSIZE({AMDGPU_DESCRIPTOR_IDENTITY_TABLE_SYMBOL}) == LOOM_AMDGPU_DESCRIPTOR_IDENTITY_COUNT, "descriptor identity count mismatch");',
            "",
        ]
    )
    return "\n".join(lines)


def _instruction_placement_bindings(inputs: AmdgpuPlanningTableInputs) -> str:
    """Joins qualifications to the actual generated descriptor/processor rows."""
    lookups: dict[tuple[str, str, int], list[int]] = {}
    bindings: list[tuple[int, int, tuple[str, str, int]]] = []
    ranges: list[tuple[int, int]] = []
    qualifications = set(MATRIX_OPERAND_PLACEMENT_QUALIFICATIONS)
    processors = sorted_processor_infos()
    unknown = {row[0] for row in qualifications} - {info.processor for info in processors}
    if unknown:
        raise ValueError(f"unknown placement qualification processors: {sorted(unknown)}")
    for processor in processors:
        first_binding = len(bindings)
        descriptor_set_keys = sorted({amdgpu_target_descriptor_set_key(target, processor) for target in sorted_target_infos() if target.processor == processor.processor})
        for name, subgroup_size, descriptor_key in sorted(qualifications):
            if name != processor.processor:
                continue
            for key in descriptor_set_keys:
                descriptor_set = inputs.descriptor_sets_by_key[key]
                ordinal = next((i for i, descriptor in enumerate(descriptor_set.descriptors) if descriptor.key == descriptor_key), None)
                if ordinal is None:
                    raise ValueError(f"placement qualification '{descriptor_key}' is absent from '{key}'")
                descriptor = descriptor_set.descriptors[ordinal]
                operands = tuple(operand for operand in descriptor.operands if operand.role is OperandRole.OPERAND)
                if tuple(operand.field_name for operand in operands) != ("lhs", "rhs", "acc") or any(
                    operand.unit_count < minimum or not any(alt.reg_class is not None for alt in operand.reg_alts) for operand, minimum in zip(operands, (1, 1, 2), strict=True)
                ):
                    raise ValueError(f"placement qualification '{descriptor_key}' requires register lhs/rhs/acc coordinates and two accumulator units")
                # Descriptor vocabulary may be shared without sharing measured
                # processor/subgroup qualifications.
                lookup_key = (key, name, subgroup_size)
                lookup = lookups.setdefault(lookup_key, [0] * len(descriptor_set.descriptors))
                lookup[ordinal] = 1
                binding = (amdgpu_descriptor_set_ordinal(key), subgroup_size, lookup_key)
                if binding not in bindings[first_binding:]:
                    bindings.append(binding)
        ranges.append((first_binding, len(bindings) - first_binding))
    lines = [*line_comment_header("//", generator="loom.gen.target.arch.amdgpu.descriptors.amdgpu_descriptor_tables"), ""]
    lookup_names = {key: f"kInstructionPreferenceIndices{i}" for i, key in enumerate(lookups)}
    for key, lookup in lookups.items():
        lines.append(f"static const uint16_t {lookup_names[key]}[{len(lookup)}] = {{")
        lines.extend(f"    [{i}] = {value}," for i, value in enumerate(lookup) if value)
        lines.append("};")
    lines.append("static const loom_amdgpu_placement_binding_t kInstructionPreferenceBindings[] = {")
    lines.extend(f"    {{{ordinal}, {subgroup}, {lookup_names[key]}}}," for ordinal, subgroup, key in bindings)
    lines.append("};")
    lines.append(f"static const loom_amdgpu_placement_binding_range_t kInstructionPreferenceRanges[{len(processors)}] = {{")
    lines.extend(f"    [{i}] = {{{first}, {count}}}," for i, (first, count) in enumerate(ranges) if count)
    lines.append("};")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=("Generate global AMDGPU descriptor-derived tables from one ISA corpus."))
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as <key>:<path>.",
    )
    parser.add_argument(
        "--descriptor-set",
        action="append",
        default=[],
        help="Descriptor-set key to include in target-reference tables.",
    )
    parser.add_argument("--target-ref-header", type=Path, required=True)
    parser.add_argument("--target-ref-source", type=Path, required=True)
    parser.add_argument(
        "--target-ref-public-header",
        default="loom/target/arch/amdgpu/refs/target_refs.h",
    )
    parser.add_argument("--wait-packet-source", type=Path, required=True)
    parser.add_argument("--vopd-source", type=Path, required=True)
    parser.add_argument("--placement-bindings", type=Path, required=True)
    parser.add_argument("--lower-capabilities-header", type=Path, required=True)
    parser.add_argument("--descriptor-strings-header", type=Path, required=True)
    parser.add_argument("--descriptor-strings-source", type=Path, required=True)
    parser.add_argument("--descriptor-strings-layout", type=Path, required=True)
    parser.add_argument("--descriptor-identities-header", type=Path, required=True)
    parser.add_argument("--descriptor-identities-source", type=Path, required=True)
    parser.add_argument("--descriptor-strings-public-header", required=True)
    parser.add_argument("--descriptor-strings-symbol", required=True)
    parser.add_argument(
        "--descriptor-shared-strings",
        type=Path,
        required=True,
        help="Secondary string bank shared with descriptor tables.",
    )
    parser.add_argument("--native-asm-values-header", type=Path, required=True)
    parser.add_argument("--native-asm-values-source", type=Path, required=True)
    parser.add_argument("--native-asm-values-layout", type=Path, required=True)
    args = parser.parse_args(argv)

    inputs = load_amdgpu_planning_table_inputs(
        args.isa_xml,
        amdgpu_vopd_instruction_encoding_names_by_isa_key(),
    )
    selected_descriptor_set_infos = select_target_ref_descriptor_set_infos(args.descriptor_set)
    selected_descriptor_sets = tuple(inputs.descriptor_sets_by_key[info.key] for info in selected_descriptor_set_infos)
    storage_generator_targets = tuple(dict.fromkeys(amdgpu_descriptor_set_storage_info_by_generator_target(info.generator_target).generator_target for info in selected_descriptor_set_infos))
    descriptor_string_values = frozenset().union(
        *(
            amdgpu_descriptor_table_family_string_values(
                amdgpu_target_table_family(storage_generator_target),
                inputs.descriptor_sets_by_generator_target,
            )
            for storage_generator_target in storage_generator_targets
        )
    )
    descriptor_shared_strings = CStringBank.read(args.descriptor_shared_strings)
    descriptor_strings = CStringBank.build(
        descriptor_string_values,
        symbol=args.descriptor_strings_symbol,
        header=args.descriptor_strings_public_header,
        shared=descriptor_shared_strings,
    )
    write_text_file(args.descriptor_strings_header, descriptor_strings.emit_header())
    write_text_file(args.descriptor_strings_source, descriptor_strings.emit_source())
    write_text_file(args.descriptor_strings_layout, descriptor_strings.manifest())
    native_asm_values = NativeAsmValuePool.build(
        (
            span
            for storage_target in storage_generator_targets
            for span in amdgpu_descriptor_table_family_native_spans(amdgpu_target_table_family(storage_target), inputs.descriptor_sets_by_generator_target)
        ),
        symbol="loom_amdgpu_native_asm_values",
        header="loom/target/arch/amdgpu/descriptors/native_asm_values.h",
    )
    write_text_file(args.native_asm_values_header, native_asm_values.emit_header())
    write_text_file(args.native_asm_values_source, native_asm_values.emit_source(descriptor_strings, descriptor_shared_strings))
    write_text_file(args.native_asm_values_layout, native_asm_values.manifest())
    write_text_file(
        args.descriptor_identities_header,
        _descriptor_identity_header(len(amdgpu_descriptor_identity_domain().keys)),
    )
    write_text_file(
        args.descriptor_identities_source,
        _descriptor_identity_source(
            selected_descriptor_sets,
            descriptor_strings,
            descriptor_shared_strings,
        ),
    )
    generate_target_ref_outputs(
        public_header=args.target_ref_public_header,
        descriptor_set_infos=selected_descriptor_set_infos,
        descriptor_sets_by_key=inputs.descriptor_sets_by_key,
        header_path=args.target_ref_header,
        source_path=args.target_ref_source,
    )
    generate_lower_capability_header(selected_descriptor_sets, args.lower_capabilities_header)
    generate_wait_packet_table_outputs(
        inputs,
        source_path=args.wait_packet_source,
    )
    generate_vopd_component_table_outputs(
        inputs,
        source_path=args.vopd_source,
    )
    write_text_file(args.placement_bindings, _instruction_placement_bindings(inputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
