#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates shared AMDGPU target and code-object map fragments.

Canonical processor, code-object, target-ID qualification, and artifact facts
live in target_map_data.py so Python consumers can share them without importing
this generator. This module owns build-family selectors, the legacy ELF machine
catalog, and projections for Bazel, CMake, Loom, and the runtime. Runtime target
rows also carry processor facts imported from
loom.target.arch.amdgpu.target_info so those facts stay tied to Loom's target
table. Keep build logic in Starlark/CMake; keep target facts in Python tables.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path

if __package__:
    from .target_map_data import (
        AMDGPU_EXACT_TARGET_INFOS,
        AMDGPU_GENERIC_CODE_OBJECT_INFOS,
        AMDGPU_PHYSICAL_TARGET_INFOS,
        AMDGPU_TARGET_OVERLAY_INFOS,
        TARGET_ID_FEATURE_SRAMECC,
        TARGET_ID_FEATURE_XNACK,
        AmdgpuDeviceBinaryTarget,
        target_id_features_for_processor,
        validate_target_map_data,
    )
    from .target_map_data import (
        target_processor as _target_processor,
    )
else:
    from target_map_data import (
        AMDGPU_EXACT_TARGET_INFOS,
        AMDGPU_GENERIC_CODE_OBJECT_INFOS,
        AMDGPU_PHYSICAL_TARGET_INFOS,
        AMDGPU_TARGET_OVERLAY_INFOS,
        TARGET_ID_FEATURE_SRAMECC,
        TARGET_ID_FEATURE_XNACK,
        AmdgpuDeviceBinaryTarget,
        target_id_features_for_processor,
        validate_target_map_data,
    )
    from target_map_data import (
        target_processor as _target_processor,
    )

DEFAULT_TARGET_SELECTIONS = (
    "gfx9-generic",
    "gfx90a",
    "gfx9-4-generic",
    "gfx10-1-generic",
    "gfx10-3-generic",
    "gfx11-generic",
    "gfx12-generic",
)


ELF_MACHINE_PROCESSORS = (
    (0x020, "gfx600"),
    (0x021, "gfx601"),
    (0x022, "gfx700"),
    (0x023, "gfx701"),
    (0x024, "gfx702"),
    (0x025, "gfx703"),
    (0x026, "gfx704"),
    (0x028, "gfx801"),
    (0x029, "gfx802"),
    (0x02A, "gfx803"),
    (0x02B, "gfx810"),
    (0x02C, "gfx900"),
    (0x02D, "gfx902"),
    (0x02E, "gfx904"),
    (0x02F, "gfx906"),
    (0x030, "gfx908"),
    (0x031, "gfx909"),
    (0x032, "gfx90c"),
    (0x033, "gfx1010"),
    (0x034, "gfx1011"),
    (0x035, "gfx1012"),
    (0x036, "gfx1030"),
    (0x037, "gfx1031"),
    (0x038, "gfx1032"),
    (0x039, "gfx1033"),
    (0x03A, "gfx602"),
    (0x03B, "gfx705"),
    (0x03C, "gfx805"),
    (0x03D, "gfx1035"),
    (0x03E, "gfx1034"),
    (0x03F, "gfx90a"),
    (0x040, "gfx940"),
    (0x041, "gfx1100"),
    (0x042, "gfx1013"),
    (0x043, "gfx1150"),
    (0x044, "gfx1103"),
    (0x045, "gfx1036"),
    (0x046, "gfx1101"),
    (0x047, "gfx1102"),
    (0x048, "gfx1200"),
    (0x049, "gfx1250"),
    (0x04A, "gfx1151"),
    (0x04B, "gfx941"),
    (0x04C, "gfx942"),
    (0x04E, "gfx1201"),
    (0x04F, "gfx950"),
    (0x050, "gfx1310"),
    (0x051, "gfx9-generic"),
    (0x052, "gfx10-1-generic"),
    (0x053, "gfx10-3-generic"),
    (0x054, "gfx11-generic"),
    (0x055, "gfx1152"),
    (0x058, "gfx1153"),
    (0x059, "gfx12-generic"),
    (0x05A, "gfx1251"),
    (0x05B, "gfx12-5-generic"),
    (0x05C, "gfx1172"),
    (0x05D, "gfx1170"),
    (0x05E, "gfx1171"),
    (0x05F, "gfx9-4-generic"),
    (0x062, "gfx11-7-generic"),
)

# Feature support for ELF machine processors that are not exact build targets.
# This includes legacy decode-only processors.
ELF_MACHINE_FEATURE_SUPPORT = {
    "gfx801": (TARGET_ID_FEATURE_XNACK,),
    "gfx810": (TARGET_ID_FEATURE_XNACK,),
}

ALL_EXACT_TARGETS = object()

TARGET_FAMILIES = (
    ("all", ALL_EXACT_TARGETS),
    (
        "dcgpu-all",
        ("gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"),
    ),
    (
        "dgpu-all",
        (
            "gfx900",
            "gfx902",
            "gfx904",
            "gfx906",
            "gfx909",
            "gfx1010",
            "gfx1011",
            "gfx1012",
            "gfx1013",
            "gfx1030",
            "gfx1031",
            "gfx1032",
            "gfx1034",
            "gfx1100",
            "gfx1101",
            "gfx1102",
            "gfx1200",
            "gfx1201",
            "gfx1250",
            "gfx1251",
        ),
    ),
    ("gfx900-dgpu", ("gfx900",)),
    ("gfx906-dgpu", ("gfx906",)),
    ("gfx908-dcgpu", ("gfx908",)),
    ("gfx90a-dcgpu", ("gfx90a",)),
    ("gfx90c-igpu", ("gfx90c",)),
    ("gfx94X-all", ("gfx940", "gfx941", "gfx942")),
    ("gfx94X-dcgpu", ("gfx940", "gfx941", "gfx942")),
    ("gfx950-all", ("gfx950",)),
    ("gfx950-dcgpu", ("gfx950",)),
    ("gfx101X-all", ("gfx1010", "gfx1011", "gfx1012", "gfx1013")),
    ("gfx101X-dgpu", ("gfx1010", "gfx1011", "gfx1012", "gfx1013")),
    (
        "gfx103X-all",
        (
            "gfx1030",
            "gfx1031",
            "gfx1032",
            "gfx1033",
            "gfx1034",
            "gfx1035",
            "gfx1036",
        ),
    ),
    ("gfx103X-dgpu", ("gfx1030", "gfx1031", "gfx1032", "gfx1034")),
    ("gfx103X-igpu", ("gfx1033", "gfx1035", "gfx1036")),
    ("gfx110X-all", ("gfx1100", "gfx1101", "gfx1102", "gfx1103")),
    ("gfx110X-dgpu", ("gfx1100", "gfx1101", "gfx1102")),
    ("gfx110X-igpu", ("gfx1103",)),
    ("gfx115X-all", ("gfx1150", "gfx1151", "gfx1152", "gfx1153")),
    ("gfx115X-igpu", ("gfx1150", "gfx1151", "gfx1152", "gfx1153")),
    ("gfx117X-all", ("gfx1170", "gfx1171", "gfx1172")),
    ("gfx120X-all", ("gfx1200", "gfx1201")),
    ("gfx125X-all", ("gfx1250", "gfx1251")),
    (
        "igpu-all",
        (
            "gfx90c",
            "gfx1033",
            "gfx1035",
            "gfx1036",
            "gfx1103",
            "gfx1150",
            "gfx1151",
            "gfx1152",
            "gfx1153",
        ),
    ),
)


def find_repo_root():
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "runtime" / "src" / "iree").exists():
            return current
        current = current.parent
    print("error: could not find IREE repository root", file=sys.stderr)
    sys.exit(1)


def append_unique(values, new_values):
    for value in new_values:
        if value not in values:
            values.append(value)


def exact_targets():
    return [info.exact_processor for info in AMDGPU_EXACT_TARGET_INFOS]


def target_processor(target):
    """Returns the backend processor selected by a canonical target."""
    return _target_processor(target)


def code_object_targets():
    values = []
    for info in AMDGPU_EXACT_TARGET_INFOS:
        append_unique(values, [info.code_object_processor])
    return values


def device_binary_targets():
    values = []
    names = set()
    for info in AMDGPU_EXACT_TARGET_INFOS:
        for variant in device_binary_variants_for_exact(info.exact_processor):
            if variant.name not in names:
                values.append(variant)
                names.add(variant.name)
        if info.code_object_processor not in names:
            values.append(
                AmdgpuDeviceBinaryTarget(
                    name=info.code_object_processor,
                    processor=info.code_object_processor,
                )
            )
            names.add(info.code_object_processor)
    return values


def device_binary_target_names():
    return [target.name for target in device_binary_targets()]


def device_binary_target(name):
    for target in device_binary_targets():
        if target.name == name:
            return target
    return None


def device_binary_variants_for_exact(exact_target):
    return [
        AmdgpuDeviceBinaryTarget(
            name=overlay.target,
            processor=exact_target,
            compile_options=overlay.compile_options,
            link_options=overlay.link_options,
        )
        for overlay in AMDGPU_TARGET_OVERLAY_INFOS
        if overlay.processor == exact_target
    ]


def exact_targets_for_code_object(code_object_target):
    return [
        info.exact_processor
        for info in AMDGPU_EXACT_TARGET_INFOS
        if info.code_object_processor == code_object_target
    ]


def expand_device_binary_target_selections(selections):
    """Expands public target selectors to ordered device-binary target names."""
    exact = set(exact_targets())
    code_objects = set(code_object_targets())
    overlays = {overlay.target for overlay in AMDGPU_TARGET_OVERLAY_INFOS}
    exact_to_code_object = {
        info.exact_processor: info.code_object_processor
        for info in AMDGPU_EXACT_TARGET_INFOS
    }
    families = {family: family_targets(targets) for family, targets in TARGET_FAMILIES}
    expanded_targets = []

    def append_exact(exact_target):
        append_unique(
            expanded_targets,
            [
                variant.name
                for variant in device_binary_variants_for_exact(exact_target)
            ],
        )
        append_unique(expanded_targets, [exact_to_code_object[exact_target]])

    for selection in selections:
        if selection in overlays:
            append_unique(expanded_targets, [selection])
        elif selection in code_objects:
            for exact_target in exact_targets_for_code_object(selection):
                append_exact(exact_target)
        elif selection in exact:
            append_exact(selection)
        elif selection in families:
            for exact_target in families[selection]:
                append_exact(exact_target)
        else:
            available = sorted(overlays | code_objects | exact | set(families))
            raise ValueError(
                "unknown AMDGPU target selector '{}'. Available selectors "
                "include: {}".format(selection, ", ".join(available))
            )
    return expanded_targets


def family_targets(targets):
    if targets is ALL_EXACT_TARGETS:
        return exact_targets()
    return list(targets)


def target_family_names():
    return [family for family, _ in TARGET_FAMILIES]


def elf_machine_targets():
    values = []
    for machine, processor in ELF_MACHINE_PROCESSORS:
        features = target_id_features_for_processor(processor)
        if not features:
            features = ELF_MACHINE_FEATURE_SUPPORT.get(processor, ())
        values.append((machine, processor, features))
    return values


def validate_target_map():
    validate_target_map_data()
    exact = exact_targets()
    if len(set(exact)) != len(exact):
        raise ValueError("duplicate exact AMDGPU targets in target map")

    families = target_family_names()
    if len(set(families)) != len(families):
        raise ValueError("duplicate AMDGPU target families in target map")

    exact_set = set(exact)
    device_binary_names = device_binary_target_names()
    if len(set(device_binary_names)) != len(device_binary_names):
        raise ValueError("duplicate AMDGPU device binary targets in target map")
    elf_machine_feature_targets = set(ELF_MACHINE_FEATURE_SUPPORT)
    elf_machine_processors = [processor for _, processor in ELF_MACHINE_PROCESSORS]
    elf_machine_processor_set = set(elf_machine_processors)
    unknown_elf_machine_feature_targets = sorted(
        elf_machine_feature_targets - elf_machine_processor_set
    )
    if unknown_elf_machine_feature_targets:
        raise ValueError(
            "ELF machine feature support references unknown processors: {}".format(
                ", ".join(unknown_elf_machine_feature_targets)
            )
        )
    duplicate_elf_machine_processors = sorted(
        processor
        for processor in elf_machine_processor_set
        if elf_machine_processors.count(processor) > 1
    )
    if duplicate_elf_machine_processors:
        raise ValueError(
            "duplicate AMDGPU ELF machine processors: {}".format(
                ", ".join(duplicate_elf_machine_processors)
            )
        )
    elf_machine_values = [machine for machine, _ in ELF_MACHINE_PROCESSORS]
    duplicate_elf_machine_values = sorted(
        machine
        for machine in set(elf_machine_values)
        if elf_machine_values.count(machine) > 1
    )
    if duplicate_elf_machine_values:
        raise ValueError(
            "duplicate AMDGPU ELF machine values: {}".format(
                ", ".join(
                    "0x{:03x}".format(machine)
                    for machine in duplicate_elf_machine_values
                )
            )
        )
    missing_exact_machine_processors = sorted(exact_set - elf_machine_processor_set)
    if missing_exact_machine_processors:
        raise ValueError(
            "exact targets missing ELF machine processors: {}".format(
                ", ".join(missing_exact_machine_processors)
            )
        )
    missing_code_object_machine_processors = sorted(
        set(code_object_targets()) - elf_machine_processor_set
    )
    if missing_code_object_machine_processors:
        raise ValueError(
            "code-object targets missing ELF machine processors: {}".format(
                ", ".join(missing_code_object_machine_processors)
            )
        )

    for family, targets in TARGET_FAMILIES:
        unknown_targets = sorted(set(family_targets(targets)) - exact_set)
        if unknown_targets:
            raise ValueError(
                "target family {} references unknown exact targets: {}".format(
                    family, ", ".join(unknown_targets)
                )
            )


def import_loom_target_info(repo_root):
    repo_root_path = str(repo_root)
    if repo_root_path not in sys.path:
        sys.path.insert(0, repo_root_path)
    loom_python_path = str(repo_root / "loom/py")
    if loom_python_path not in sys.path:
        sys.path.insert(0, loom_python_path)
    from loom.target.arch.amdgpu.target_info import (
        AMDGPU_PROCESSOR_INFOS,
        kernel_descriptor_profile_supports_wavefront_size,
    )

    return AMDGPU_PROCESSOR_INFOS, kernel_descriptor_profile_supports_wavefront_size


def amdgpu_processor_infos(repo_root):
    processor_infos, _ = import_loom_target_info(repo_root)
    return {info.processor: info for info in processor_infos}


def wavefront_size_flags_expr(supports_wavefront_size, processor_info):
    flags = []
    if supports_wavefront_size(processor_info.kernel_descriptor.profile, 32):
        flags.append("IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_32")
    if supports_wavefront_size(processor_info.kernel_descriptor.profile, 64):
        flags.append("IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_64")
    if not flags:
        return "IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_NONE"
    return " | ".join(flags)


def validate_processor_table_coverage(repo_root):
    processor_infos = amdgpu_processor_infos(repo_root)
    unknown_targets = sorted(
        exact_target
        for exact_target in exact_targets()
        if exact_target not in processor_infos
    )
    if unknown_targets:
        raise ValueError(
            "AMDGPU target map references exact targets missing from the "
            "processor info table: {}".format(", ".join(unknown_targets))
        )


def generated_header(comment_prefix, output_path):
    return "\n".join(
        [
            "{} Generated by build_tools/amdgpu/target_map.py.".format(comment_prefix),
            "{} Do not edit directly; edit the Python source tables and regenerate.".format(
                comment_prefix
            ),
            "{} Output: {}".format(comment_prefix, output_path),
        ]
    )


def bzl_list(name, values):
    lines = ["{} = [".format(name)]
    lines.extend(['    "{}",'.format(value) for value in values])
    lines.append("]")
    return "\n".join(lines)


def bzl_string_dict(name, values):
    lines = [
        "# buildifier: disable=unsorted-dict-items",
        "{} = {{".format(name),
    ]
    for key, value in values:
        lines.append('    "{}": "{}",'.format(key, value))
    lines.append("}")
    return "\n".join(lines)


def bzl_string_list_dict(name, values):
    lines = [
        "# buildifier: disable=unsorted-dict-items",
        "{} = {{".format(name),
    ]
    for key, entries in values:
        if len(entries) == 1:
            lines.append('    "{}": ["{}"],'.format(key, entries[0]))
        else:
            lines.append('    "{}": ['.format(key))
            lines.extend(['        "{}",'.format(entry) for entry in entries])
            lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


def bzl_family_dict(name):
    lines = [
        "# buildifier: disable=unsorted-dict-items",
        "{} = {{".format(name),
    ]
    for family, targets in TARGET_FAMILIES:
        values = family_targets(targets)
        if targets is ALL_EXACT_TARGETS:
            lines.append('    "{}": IREE_AMDGPU_EXACT_TARGETS,'.format(family))
        elif len(values) == 1:
            lines.append('    "{}": ["{}"],'.format(family, values[0]))
        else:
            lines.append('    "{}": ['.format(family))
            lines.extend(['        "{}",'.format(value) for value in values])
            lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


def render_bzl():
    output_path = "build_tools/amdgpu/target_map.bzl"
    return (
        "\n\n".join(
            [
                generated_header("#", output_path),
                '"""Generated AMDGPU target map data."""',
                bzl_list(
                    "IREE_AMDGPU_DEFAULT_TARGET_SELECTORS",
                    DEFAULT_TARGET_SELECTIONS,
                ),
                bzl_list("IREE_AMDGPU_EXACT_TARGETS", exact_targets()),
                bzl_list("IREE_AMDGPU_CODE_OBJECT_TARGETS", code_object_targets()),
                bzl_list(
                    "IREE_AMDGPU_DEVICE_BINARY_TARGETS",
                    device_binary_target_names(),
                ),
                bzl_string_dict(
                    "IREE_AMDGPU_EXACT_TARGET_CODE_OBJECTS",
                    (
                        (info.exact_processor, info.code_object_processor)
                        for info in AMDGPU_EXACT_TARGET_INFOS
                    ),
                ),
                bzl_string_list_dict(
                    "IREE_AMDGPU_EXACT_TARGET_DEVICE_BINARY_VARIANTS",
                    [
                        (
                            exact_target,
                            [
                                variant.name
                                for variant in device_binary_variants_for_exact(
                                    exact_target
                                )
                            ],
                        )
                        for exact_target in exact_targets()
                        if device_binary_variants_for_exact(exact_target)
                    ],
                ),
                bzl_list("IREE_AMDGPU_TARGET_FAMILY_NAMES", target_family_names()),
                bzl_family_dict("IREE_AMDGPU_TARGET_FAMILIES"),
            ]
        )
        + "\n"
    )


def cmake_list(name, values):
    lines = ["set({}".format(name)]
    lines.extend(['  "{}"'.format(value) for value in values])
    lines.append(")")
    return "\n".join(lines)


def cmake_identifier(value):
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def render_cmake():
    output_path = "build_tools/amdgpu/target_map.cmake"
    lines = [
        generated_header("#", output_path),
        "",
        cmake_list("_IREE_AMDGPU_EXACT_TARGETS", exact_targets()),
        "",
        cmake_list("_IREE_AMDGPU_CODE_OBJECT_TARGETS", code_object_targets()),
        "",
        cmake_list("_IREE_AMDGPU_DEVICE_BINARY_TARGETS", device_binary_target_names()),
        "",
    ]
    for info in AMDGPU_EXACT_TARGET_INFOS:
        lines.append(
            'set(_IREE_AMDGPU_TARGET_CODE_OBJECT_{} "{}")'.format(
                info.exact_processor, info.code_object_processor
            )
        )
    for exact_target in exact_targets():
        variants = device_binary_variants_for_exact(exact_target)
        if not variants:
            continue
        lines.append(
            cmake_list(
                "_IREE_AMDGPU_TARGET_DEVICE_BINARY_VARIANTS_{}".format(exact_target),
                [variant.name for variant in variants],
            )
        )
    lines.extend(
        [
            "",
            cmake_list("_IREE_AMDGPU_TARGET_FAMILIES", target_family_names()),
            "",
        ]
    )
    for family, targets in TARGET_FAMILIES:
        var_name = "_IREE_AMDGPU_TARGET_FAMILY_{}".format(cmake_identifier(family))
        if targets is ALL_EXACT_TARGETS:
            lines.append("set({}".format(var_name))
            lines.append("  ${_IREE_AMDGPU_EXACT_TARGETS}")
            lines.append(")")
        else:
            lines.append(cmake_list(var_name, family_targets(targets)))
    lines.append("")
    return "\n".join(lines)


def render_runtime_identity_catalog_inl(repo_root):
    output_path = "runtime/src/iree/hal/drivers/amdgpu/target/identity_catalog.inl"
    lines = [
        generated_header("//", output_path),
        "//",
        "// Define IREE_AMDGPU_TARGET_MAPPING and/or",
        "// IREE_AMDGPU_PHYSICAL_TARGET before including this file.",
        "",
        "// clang-format off",
        "#if defined(IREE_AMDGPU_TARGET_MAPPING)",
    ]
    feature_flag_names = {
        TARGET_ID_FEATURE_SRAMECC: "IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_SRAMECC",
        TARGET_ID_FEATURE_XNACK: "IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_XNACK",
    }
    processor_info_rows, supports_wavefront_size = import_loom_target_info(repo_root)
    processor_infos = {info.processor: info for info in processor_info_rows}
    target_rows = (
        [
            (
                info.exact_processor,
                info.exact_processor,
                info.code_object_processor,
                info.generic_introduction_version,
                info.target_id_features,
            )
            for info in AMDGPU_EXACT_TARGET_INFOS
        ]
        + [
            (
                info.processor,
                info.processor,
                info.processor,
                0,
                info.target_id_features,
            )
            for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS
        ]
        + [
            (
                overlay.target,
                processor,
                exact_info.code_object_processor,
                exact_info.generic_introduction_version,
                exact_info.target_id_features,
            )
            for overlay in AMDGPU_TARGET_OVERLAY_INFOS
            for processor in (overlay.processor,)
            for exact_info in (
                next(
                    info
                    for info in AMDGPU_EXACT_TARGET_INFOS
                    if info.exact_processor == processor
                ),
            )
        ]
    )
    for (
        target,
        processor,
        code_object_processor,
        generic_introduction_version,
        features,
    ) in target_rows:
        processor_info = processor_infos[processor]
        feature_flags = " | ".join(feature_flag_names[feature] for feature in features)
        if not feature_flags:
            feature_flags = "IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_NONE"
        lines.append(
            'IREE_AMDGPU_TARGET_MAPPING("{}", "{}", "{}", {}, {}, {}, {})'.format(
                target,
                processor,
                code_object_processor,
                generic_introduction_version,
                feature_flags,
                processor_info.wavefront.default_size,
                wavefront_size_flags_expr(supports_wavefront_size, processor_info),
            )
        )
    lines.extend(
        [
            "#endif  // IREE_AMDGPU_TARGET_MAPPING",
            "",
            "#if defined(IREE_AMDGPU_PHYSICAL_TARGET)",
        ]
    )
    for info in AMDGPU_PHYSICAL_TARGET_INFOS:
        lines.append(
            'IREE_AMDGPU_PHYSICAL_TARGET("{}", {}, "{}")'.format(
                info.processor,
                info.asic_revision,
                info.target,
            )
        )
    lines.extend(["#endif  // IREE_AMDGPU_PHYSICAL_TARGET", ""])
    return "\n".join(lines)


def render_device_library_target_map_inl():
    output_path = (
        "runtime/src/iree/hal/drivers/amdgpu/util/device_library_target_map.inl"
    )
    lines = [
        generated_header("//", output_path),
        "//",
        "// Define IREE_AMDGPU_DEVICE_LIBRARY_TARGET_VARIANT(",
        "//     artifact, target) before including this file.",
        "",
        "// clang-format off",
        "#if defined(IREE_AMDGPU_DEVICE_LIBRARY_TARGET_VARIANT)",
    ]
    for overlay in AMDGPU_TARGET_OVERLAY_INFOS:
        lines.append(
            'IREE_AMDGPU_DEVICE_LIBRARY_TARGET_VARIANT("{}", "{}")'.format(
                overlay.target,
                overlay.target,
            )
        )
    lines.extend(
        [
            "#endif  // IREE_AMDGPU_DEVICE_LIBRARY_TARGET_VARIANT",
            "",
        ]
    )
    return "\n".join(lines)


def render_elf_machine_map_inl():
    output_path = "build_tools/amdgpu/elf_machine_map.inl"
    lines = [
        generated_header("//", output_path),
        "//",
        "// Define IREE_AMDGPU_ELF_MACHINE_TARGET(machine, processor, "
        "sramecc_supported, xnack_supported)",
        "// before including this file.",
        "",
        "// clang-format off",
        "#if defined(IREE_AMDGPU_ELF_MACHINE_TARGET)",
    ]
    for machine, processor, features in elf_machine_targets():
        sramecc = "true" if TARGET_ID_FEATURE_SRAMECC in features else "false"
        xnack = "true" if TARGET_ID_FEATURE_XNACK in features else "false"
        lines.append(
            'IREE_AMDGPU_ELF_MACHINE_TARGET(0x{:03x}u, "{}", {}, {})'.format(
                machine, processor, sramecc, xnack
            )
        )
    lines.extend(
        [
            "#endif  // IREE_AMDGPU_ELF_MACHINE_TARGET",
            "",
        ]
    )
    return "\n".join(lines)


def render_header():
    output_path = "build_tools/amdgpu/target_map.h"
    lines = [
        generated_header("//", output_path),
        "",
        "#ifndef IREE_BUILD_TOOLS_AMDGPU_TARGET_MAP_H_",
        "#define IREE_BUILD_TOOLS_AMDGPU_TARGET_MAP_H_",
        "",
        "#include <stddef.h>",
        "#include <string.h>",
        "",
        "static inline const char* iree_amdgpu_code_object_target_for_exact(",
        "    const char* exact_target) {",
        "  if (!exact_target) return NULL;",
    ]
    for info in AMDGPU_EXACT_TARGET_INFOS:
        lines.append(
            '  if (strcmp(exact_target, "{}") == 0) return "{}";'.format(
                info.exact_processor, info.code_object_processor
            )
        )
    lines.extend(
        [
            "  return NULL;",
            "}",
            "",
            "static inline int iree_amdgpu_target_label_fragment(const char* target,",
            "                                                    char* buffer,",
            "                                                    size_t capacity) {",
            "  if (!target || !buffer || capacity == 0) return 0;",
            "  size_t i = 0;",
            "  for (; target[i] != '\\0'; ++i) {",
            "    if (i + 1 >= capacity) {",
            "      buffer[0] = '\\0';",
            "      return 0;",
            "    }",
            "    const char c = target[i];",
            "    buffer[i] = (c == '-' || c == '.') ? '_' : c;",
            "  }",
            "  buffer[i] = '\\0';",
            "  return 1;",
            "}",
            "",
            "#endif  // IREE_BUILD_TOOLS_AMDGPU_TARGET_MAP_H_",
            "",
        ]
    )
    return "\n".join(lines)


def generated_outputs(repo_root):
    build_tools_output_dir = repo_root / "build_tools/amdgpu"
    runtime_target_output_dir = repo_root / "runtime/src/iree/hal/drivers/amdgpu/target"
    driver_util_output_dir = repo_root / "runtime/src/iree/hal/drivers/amdgpu/util"
    return {
        build_tools_output_dir / "target_map.bzl": render_bzl(),
        build_tools_output_dir / "target_map.cmake": render_cmake(),
        build_tools_output_dir / "elf_machine_map.inl": render_elf_machine_map_inl(),
        build_tools_output_dir / "target_map.h": render_header(),
        runtime_target_output_dir
        / "identity_catalog.inl": render_runtime_identity_catalog_inl(repo_root),
        driver_util_output_dir
        / "device_library_target_map.inl": render_device_library_target_map_inl(),
    }


def check_outputs(repo_root, outputs):
    failed = False
    for path, content in outputs.items():
        if not path.exists():
            print("error: {} does not exist".format(path), file=sys.stderr)
            failed = True
            continue
        existing = path.read_text()
        if existing == content:
            continue
        rel_path = path.relative_to(repo_root)
        print("error: {} is out of date".format(rel_path), file=sys.stderr)
        diff = difflib.unified_diff(
            existing.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=str(rel_path),
            tofile=str(rel_path) + " (generated)",
        )
        sys.stderr.writelines(diff)
        failed = True
    if failed:
        print(
            "Run 'python3 build_tools/amdgpu/target_map.py' to regenerate.",
            file=sys.stderr,
        )
        return 1
    print("AMDGPU target map generated files are up to date.")
    return 0


def write_outputs(outputs):
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print("Wrote {}".format(path))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate AMDGPU device binary target map fragments."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that generated files are up to date without modifying them.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    validate_target_map()
    validate_processor_table_coverage(repo_root)
    outputs = generated_outputs(repo_root)
    if args.check:
        return check_outputs(repo_root, outputs)
    return write_outputs(outputs)


if __name__ == "__main__":
    sys.exit(main())
