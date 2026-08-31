# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates one coherent native NPU2/Strix Halo hardware table family."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from loom.gen.support.c import c_string_literal
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header
from loom.target.arch.amd.xdna.array.model import (
    RegisterAccess,
    RegisterModule,
    StreamDirection,
    StreamPort,
    TileKind,
    validate_array_family,
)
from loom.target.arch.amd.xdna.array.npu2 import (
    AIE_RT_SOURCE_COMMIT,
    MLIR_AIE_SOURCE_COMMIT,
    NPU2_ARRAY_FAMILY,
    REGISTER_DATABASE_VERSION,
)
from loom.target.arch.amd.xdna.device.model import validate_device_profile
from loom.target.arch.amd.xdna.device.strix_halo import (
    DEVICE_PROFILES,
    XDNA_DRIVER_SOURCE_COMMIT,
)


def _header() -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator="loom.gen.target.arch.amd.xdna.hardware_tables",
        ),
    ]


_TILE_KIND_IDS = {
    TileKind.SHIM_NOC: "LOOM_XDNA_TILE_KIND_SHIM_NOC",
    TileKind.MEMORY: "LOOM_XDNA_TILE_KIND_MEMORY",
    TileKind.COMPUTE: "LOOM_XDNA_TILE_KIND_COMPUTE",
}
_REGISTER_MODULE_IDS = {
    RegisterModule.CORE: "LOOM_XDNA_REGISTER_MODULE_CORE",
    RegisterModule.COMPUTE_MEMORY: "LOOM_XDNA_REGISTER_MODULE_COMPUTE_MEMORY",
    RegisterModule.MEMORY_TILE: "LOOM_XDNA_REGISTER_MODULE_MEMORY_TILE",
    RegisterModule.SHIM_NOC: "LOOM_XDNA_REGISTER_MODULE_SHIM_NOC",
    RegisterModule.SHIM_PL: "LOOM_XDNA_REGISTER_MODULE_SHIM_PL",
}
_STREAM_DIRECTION_IDS = {
    StreamDirection.MASTER: "LOOM_XDNA_STREAM_DIRECTION_MASTER",
    StreamDirection.SLAVE: "LOOM_XDNA_STREAM_DIRECTION_SLAVE",
}
_STREAM_PORT_IDS = {
    StreamPort.CORE: "LOOM_XDNA_STREAM_PORT_CORE",
    StreamPort.DMA: "LOOM_XDNA_STREAM_PORT_DMA",
    StreamPort.TILE_CONTROL: "LOOM_XDNA_STREAM_PORT_TILE_CONTROL",
    StreamPort.FIFO: "LOOM_XDNA_STREAM_PORT_FIFO",
    StreamPort.SOUTH: "LOOM_XDNA_STREAM_PORT_SOUTH",
    StreamPort.WEST: "LOOM_XDNA_STREAM_PORT_WEST",
    StreamPort.NORTH: "LOOM_XDNA_STREAM_PORT_NORTH",
    StreamPort.EAST: "LOOM_XDNA_STREAM_PORT_EAST",
    StreamPort.TRACE: "LOOM_XDNA_STREAM_PORT_TRACE",
}
_REGISTER_ACCESS_IDS = {
    RegisterAccess.READ_WRITE: "LOOM_XDNA_REGISTER_ACCESS_READ_WRITE",
    RegisterAccess.WRITE_ONLY: "LOOM_XDNA_REGISTER_ACCESS_WRITE_ONLY",
}


def emit_array_facts() -> str:
    """Emits compact native NPU2 topology and resource facts."""
    family = NPU2_ARRAY_FAMILY
    validate_array_family(family)

    window_lines: list[str] = []
    tile_lines: list[str] = []
    for tile in family.tiles:
        window_start = len(window_lines)
        window_lines.extend(
            (
                "    {"
                f'.name = "{c_string_literal(window.name)}", '
                f".base = UINT32_C(0x{window.base:08x}), "
                f".capacity = UINT32_C(0x{window.capacity:08x}), "
                f".owner_column_delta = {window.owner_column_delta}, "
                f".owner_row_delta = {window.owner_row_delta}, "
                f".owner_kind = {_TILE_KIND_IDS[window.owner_kind]}"
                "},"
            )
            for window in tile.memory.load_windows
        )
        register_module_bits = " | ".join(f"LOOM_XDNA_REGISTER_MODULE_BIT({_REGISTER_MODULE_IDS[module]})" for module in tile.register_modules)
        dma = tile.dma
        assert dma is not None
        dma_flags = (
            int(dma.supports_compression)
            | (int(dma.supports_padding) << 1)
            | (int(dma.supports_out_of_order) << 2)
            | (int(dma.supports_tokens) << 3)
            | (int(dma.supports_repeat) << 4)
            | (int(dma.supports_tlast_suppression) << 5)
        )
        tile_lines.extend(
            [
                "    {",
                f"        .kind = {_TILE_KIND_IDS[tile.kind]},",
                f"        .first_row = {tile.first_row},",
                f"        .row_count = {tile.row_count},",
                f"        .lock_count = {tile.lock_count},",
                f"        .lock_value_minimum = {tile.lock_value_minimum},",
                f"        .lock_value_maximum = {tile.lock_value_maximum},",
                f"        .register_module_bits = {register_module_bits},",
                "        .memory = {",
                f"            .local_base = UINT32_C(0x{tile.memory.local_base:08x}),",
                f"            .local_capacity = UINT32_C(0x{tile.memory.local_capacity:08x}),",
                f"            .program_base = UINT32_C(0x{tile.memory.program_base:08x}),",
                f"            .program_capacity = UINT32_C(0x{tile.memory.program_capacity:08x}),",
                f"            .window_start = {window_start},",
                f"            .window_count = {len(tile.memory.load_windows)},",
                f"            .bank_count = {tile.memory.bank_count},",
                "        },",
                "        .dma = {",
                f"            .address_maximum = UINT64_C(0x{dma.address_maximum:016x}),",
                f"            .buffer_descriptor_count = {dma.buffer_descriptor_count},",
                f"            .channel_count_per_direction = {dma.channel_count_per_direction},",
                f"            .address_dimension_count = {dma.address_dimension_count},",
                f"            .address_alignment = {dma.address_alignment},",
                f"            .step_size_bits = {dma.step_size_bits},",
                f"            .wrap_bits = {dma.wrap_bits},",
                f"            .iteration_bits = {dma.iteration_bits},",
                f"            .task_queue_depth = {dma.task_queue_depth},",
                f"            .feature_bits = {dma_flags},",
                "        },",
                "    },",
            ]
        )

    event_lines = [f"    {{.module = {_REGISTER_MODULE_IDS[event.module]}, .event_count = {event.event_count}, .named_event_count = {event.named_event_count}}}," for event in family.events]
    stream_lines = [
        "    {"
        f".tile_kind = {_TILE_KIND_IDS[row.tile_kind]}, "
        f".direction = {_STREAM_DIRECTION_IDS[row.direction]}, "
        f".port = {_STREAM_PORT_IDS[row.port]}, "
        f".ordinal = {row.ordinal}, .count = {row.count}"
        "},"
        for row in family.stream_ports
    ]
    lines = [
        *_header(),
        "static const loom_xdna_address_window_t kLoomXdnaNpu2AddressWindows[] = {",
        *window_lines,
        "};",
        "",
        "static const loom_xdna_tile_facts_t kLoomXdnaNpu2TileFacts[] = {",
        *tile_lines,
        "};",
        "",
        "static const loom_xdna_event_module_facts_t kLoomXdnaNpu2EventFacts[] = {",
        *event_lines,
        "};",
        "",
        "static const loom_xdna_stream_port_range_t kLoomXdnaNpu2StreamPorts[] = {",
        *stream_lines,
        "};",
        "",
        "static const loom_xdna_array_family_t kLoomXdnaNpu2ArrayFamily = {",
        f'    .key = "{c_string_literal(family.key)}",',
        f"    .revision = {family.revision},",
        "    .architecture = LOOM_XDNA_ARCHITECTURE_AIE2P,",
        f"    .column_count = {family.column_count},",
        f"    .row_count = {family.row_count},",
        f"    .column_shift = {family.column_shift},",
        f"    .row_shift = {family.row_shift},",
        f"    .address_generation_granularity_bits = {family.address_generation_granularity_bits},",
        f"    .provenance_bits = UINT32_C(0x{int(family.provenance):08x}),",
        "    .address_windows = kLoomXdnaNpu2AddressWindows,",
        f"    .address_window_count = {len(window_lines)},",
        "    .tiles = kLoomXdnaNpu2TileFacts,",
        f"    .tile_count = {len(family.tiles)},",
        "    .events = kLoomXdnaNpu2EventFacts,",
        f"    .event_count = {len(family.events)},",
        "    .stream_ports = kLoomXdnaNpu2StreamPorts,",
        f"    .stream_port_count = {len(family.stream_ports)},",
        f'    .aie_rt_source_commit = "{AIE_RT_SOURCE_COMMIT}",',
        f'    .mlir_aie_source_commit = "{MLIR_AIE_SOURCE_COMMIT}",',
        f'    .register_database_version = "{REGISTER_DATABASE_VERSION}",',
        "};",
        "",
    ]
    return "\n".join(lines)


def _build_string_table(strings: Sequence[str]) -> tuple[dict[str, int], list[str]]:
    offsets: dict[str, int] = {}
    offset = 0
    lines = ["static const char kLoomXdnaRegisterStrings[] ="]
    for value in sorted(set(strings)):
        encoded = value.encode("ascii")
        if len(encoded) > 0xFF or offset > 0xFFFF:
            raise ValueError("XDNA register string table exceeds compact encoding")
        offsets[value] = offset
        lines.append(f'    "{c_string_literal(value)}\\0"')
        offset += len(encoded) + 1
    if offset > 0x10000:
        raise ValueError("XDNA register string table exceeds 64 KiB")
    lines[-1] += ";"
    return offsets, lines


def emit_register_facts() -> str:
    """Emits deduplicated native NPU2 register-pattern and field tables."""
    family = NPU2_ARRAY_FAMILY
    validate_array_family(family)
    patterns = tuple(sorted(family.registers, key=lambda row: row.key))
    pattern_ids = {pattern.key: index for index, pattern in enumerate(patterns)}
    fields = tuple(
        sorted(
            ((f"{pattern.key}.{field.name}", pattern, field) for pattern in patterns for field in pattern.fields),
            key=lambda row: row[0],
        )
    )
    if len(patterns) > 0x100 or len(fields) > 0xFFFF:
        raise ValueError("XDNA register tables exceed compact identifiers")
    strings = [key for key, _pattern_row, _field_row in fields]
    strings.extend(dimension.name for pattern in patterns for dimension in pattern.dimensions)
    string_offsets, string_lines = _build_string_table(strings)

    pattern_lines: list[str] = []
    for pattern in patterns:
        dimensions = tuple(pattern.dimensions) + (None,) * (2 - len(pattern.dimensions))
        if len(pattern.dimensions) > 2:
            raise ValueError(f"{pattern.key}: native register pattern supports two dimensions")
        dimension_values: list[str] = []
        for dimension in dimensions:
            if dimension is None:
                dimension_values.append("{0}")
            else:
                dimension_values.append(f"{{.name_offset = UINT16_C({string_offsets[dimension.name]}), .count = UINT16_C({dimension.count}), .stride = UINT32_C(0x{dimension.stride:08x})}}")
        pattern_lines.append(
            "    {"
            f".base_offset = UINT32_C(0x{pattern.base_offset:08x}), "
            f".provenance_bits = UINT32_C(0x{int(pattern.provenance):08x}), "
            f".dimensions = {{{dimension_values[0]}, {dimension_values[1]}}}, "
            f".module = {_REGISTER_MODULE_IDS[pattern.module]}, "
            f".access = {_REGISTER_ACCESS_IDS[pattern.access]}, "
            f".dimension_count = {len(pattern.dimensions)}"
            "},"
        )

    field_lines = ["    {0},"]
    for key, pattern, field in fields:
        flags = int(field.is_signed)
        field_lines.append(
            "    {"
            f".name_offset = UINT16_C({string_offsets[key]}), "
            f".pattern_id = {pattern_ids[pattern.key]}, "
            f".least_significant_bit = {field.least_significant_bit}, "
            f".bit_width = {field.bit_width}, "
            f".flags = {flags}"
            "},"
        )
    lines = [
        *_header(),
        *string_lines,
        "",
        "static const loom_xdna_register_pattern_t kLoomXdnaRegisterPatterns[] = {",
        *pattern_lines,
        "};",
        "",
        "static const loom_xdna_register_field_t kLoomXdnaRegisterFields[] = {",
        *field_lines,
        "};",
        "",
        f"static const uint16_t kLoomXdnaRegisterFieldCount = {len(fields)};",
        "",
    ]
    return "\n".join(lines)


def emit_device_profiles() -> str:
    """Emits immutable native device-profile rows."""
    profile_lines: list[str] = []
    for profile in DEVICE_PROFILES:
        validate_device_profile(profile)
        software = profile.qualified_software
        firmware = profile.firmware
        profile_lines.extend(
            [
                "    {",
                f'        .key = "{c_string_literal(profile.key)}",',
                f'        .display_name = "{c_string_literal(profile.display_name)}",',
                f"        .identity = UINT64_C(0x{profile.identity:016x}),",
                f"        .firmware_abi_identity = UINT64_C(0x{firmware.identity:016x}),",
                f"        .available_column_mask = UINT64_C(0x{profile.available_column_mask:016x}),",
                "        .array_family_resolver = loom_xdna_npu2_array_family,",
                f"        .revision = UINT32_C({profile.revision}),",
                f"        .provenance_bits = UINT32_C(0x{int(profile.provenance):08x}),",
                f"        .pci_vendor_id = UINT16_C(0x{profile.pci.vendor_id:04x}),",
                f"        .pci_device_id = UINT16_C(0x{profile.pci.device_id:04x}),",
                f"        .physical_column_origin = UINT16_C({profile.physical_column_origin}),",
                f"        .pci_revision = UINT8_C(0x{profile.pci.revision:02x}),",
                f"        .minimum_partition_column_count = {profile.minimum_partition_column_count},",
                f"        .firmware_protocol_major = {firmware.minimum_major},",
                f"        .firmware_protocol_minor = {firmware.minimum_minor},",
                f"        .firmware_device_revision = {firmware.device_revision},",
                f"        .transaction_device_generation = {firmware.transaction_device_generation},",
                f"        .native_elf_abi_major = {profile.native_elf_abi_major},",
                f"        .native_elf_abi_minor = {profile.native_elf_abi_minor},",
                "        .limits = {",
                f"            .minimum_device_memory_alignment = UINT32_C({profile.limits.minimum_device_memory_alignment}),",
                f"            .hardware_context_limit = {profile.limits.hardware_context_limit},",
                f"            .context_limit = {profile.limits.context_limit},",
                f"            .temporal_contexts_only = {str(profile.limits.temporal_contexts_only).lower()},",
                "        },",
                f'        .qualified_driver_version = "{software.driver_version}",',
                f'        .qualified_firmware_version = "{software.firmware_version}",',
                f'        .qualified_xrt_version = "{software.xrt_version}",',
                f'        .qualified_xrt_source_commit = "{software.xrt_source_commit}",',
                f'        .xdna_driver_source_commit = "{XDNA_DRIVER_SOURCE_COMMIT}",',
                "    },",
            ]
        )
    lines = [
        *_header(),
        "static const loom_xdna_device_profile_t kLoomXdnaDeviceProfiles[] = {",
        *profile_lines,
        "};",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate XDNA hardware tables.")
    parser.add_argument("--array-output", type=Path)
    parser.add_argument("--register-output", type=Path)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output_paths = (args.array_output, args.register_output, args.profile_output)
    if args.check and any(path is not None for path in output_paths):
        parser.error("--check cannot be combined with output paths")
    if not args.check and any(path is None for path in output_paths):
        parser.error("all XDNA hardware output paths are required")

    array_contents = emit_array_facts()
    register_contents = emit_register_facts()
    profile_contents = emit_device_profiles()
    if args.array_output is not None:
        write_text_file(args.array_output, array_contents)
        write_text_file(args.register_output, register_contents)
        write_text_file(args.profile_output, profile_contents)
    return 0


if __name__ == "__main__":
    sys.exit(main())
