// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical native XDNA ELF32 little-endian wire format.
//
// These declarations describe serialized bytes. The host structures below are
// decoded values and are never cast over file storage. Encoders write each
// field explicitly in little-endian order so host padding and alignment cannot
// become part of the ABI.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_ELF_FORMAT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_ELF_FORMAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Canonical ELF32 header byte length.
  LOOM_XDNA_ELF_HEADER_SIZE = 52,
  // Canonical ELF32 program-header byte length.
  LOOM_XDNA_ELF_PROGRAM_HEADER_SIZE = 32,
  // Canonical ELF32 section-header byte length.
  LOOM_XDNA_ELF_SECTION_HEADER_SIZE = 40,
  // Maximum program-header cardinality accepted by the native profile.
  LOOM_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT = 4096,
  // Maximum section-header cardinality accepted by the native profile.
  LOOM_XDNA_ELF_MAX_SECTION_HEADER_COUNT = 4096,
  // Maximum record cardinality accepted in any fixed XDNA table.
  LOOM_XDNA_ELF_MAX_TABLE_RECORD_COUNT = 65535,
  // Maximum array/control record cardinality accepted across one product.
  LOOM_XDNA_ELF_MAX_PROGRAM_RECORD_COUNT = 65535,
  // Maximum diagnostic entry-name byte length.
  LOOM_XDNA_ELF_MAX_ENTRY_NAME_LENGTH = 4096,
  // Maximum serialized byte length of a fixed metadata table.
  LOOM_XDNA_ELF_MAX_METADATA_TABLE_SIZE = 16 * 1024 * 1024,
  // Maximum serialized byte length of one array or control payload.
  LOOM_XDNA_ELF_MAX_PROGRAM_PAYLOAD_SIZE = 16 * 1024 * 1024,
  // Maximum serialized byte length of one array or control record.
  LOOM_XDNA_ELF_MAX_PROGRAM_RECORD_SIZE = 1024 * 1024,
  // Maximum serialized byte length of the ELF section-name table.
  LOOM_XDNA_ELF_MAX_SECTION_NAME_TABLE_SIZE = 1024 * 1024,
  // Maximum byte length of one ELF section name.
  LOOM_XDNA_ELF_MAX_SECTION_NAME_LENGTH = 4096,
};

// Fixed ELF identity for a canonical native XDNA product.
typedef enum loom_xdna_elf_identity_e {
  LOOM_XDNA_ELF_CLASS_32 = 1,
  LOOM_XDNA_ELF_DATA_LITTLE_ENDIAN = 1,
  LOOM_XDNA_ELF_VERSION_CURRENT = 1,
  LOOM_XDNA_ELF_OS_ABI_NONE = 0,
  LOOM_XDNA_ELF_ABI_VERSION_NONE = 0,
  LOOM_XDNA_ELF_FILE_TYPE_EXEC = 2,
  LOOM_XDNA_ELF_MACHINE_AIE = 264,
  LOOM_XDNA_ELF_AIE2P_FLAGS = 3,
} loom_xdna_elf_identity_t;

// Supported XDNA target-generation identities.
typedef enum loom_xdna_target_generation_e {
  LOOM_XDNA_TARGET_GENERATION_AIE2P = 3,
} loom_xdna_target_generation_t;

// Runtime directory roles encoded in the ELF program-header type field.
//
// XDNA types occupy one named subrange of PT_LOOS..PT_HIOS. The values are
// deliberately independent of vendor AIE processor-specific program headers.
typedef enum loom_xdna_elf_program_type_e {
  LOOM_XDNA_ELF_PROGRAM_TYPE_NOTE = 4,
  LOOM_XDNA_ELF_PROGRAM_TYPE_ENTRIES = 0x6C584401,
  LOOM_XDNA_ELF_PROGRAM_TYPE_BINDINGS = 0x6C584402,
  LOOM_XDNA_ELF_PROGRAM_TYPE_RELOCATIONS = 0x6C584403,
  LOOM_XDNA_ELF_PROGRAM_TYPE_TILE = 0x6C584404,
  LOOM_XDNA_ELF_PROGRAM_TYPE_ARRAY = 0x6C584405,
  LOOM_XDNA_ELF_PROGRAM_TYPE_CONTROL = 0x6C584406,
  LOOM_XDNA_ELF_PROGRAM_TYPE_FIRMWARE = 0x6C584407,
} loom_xdna_elf_program_type_t;

// ELF program-header permission bits used by the native profile.
typedef enum loom_xdna_elf_program_flag_bits_e {
  LOOM_XDNA_ELF_PROGRAM_FLAG_EXECUTE = 0x1,
  LOOM_XDNA_ELF_PROGRAM_FLAG_WRITE = 0x2,
  LOOM_XDNA_ELF_PROGRAM_FLAG_READ = 0x4,
} loom_xdna_elf_program_flag_bits_t;
typedef uint32_t loom_xdna_elf_program_flags_t;

// Loader capabilities that may be required by one product.
typedef enum loom_xdna_elf_capability_bits_e {
  // The loader must apply typed runtime relocation records.
  LOOM_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS = 1ull << 0,
  // The loader must execute one or more control programs.
  LOOM_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS = 1ull << 1,
  // The loader must consume one or more firmware-facing payloads.
  LOOM_XDNA_ELF_CAPABILITY_FIRMWARE_PAYLOADS = 1ull << 2,
  // The loader must zero-fill a tile placement after its file bytes.
  LOOM_XDNA_ELF_CAPABILITY_TILE_ZERO_FILL = 1ull << 3,
} loom_xdna_elf_capability_bits_t;
typedef uint64_t loom_xdna_elf_capabilities_t;

#define LOOM_XDNA_ELF_KNOWN_CAPABILITIES                                         \
  ((loom_xdna_elf_capabilities_t)(LOOM_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS | \
                                  LOOM_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS |    \
                                  LOOM_XDNA_ELF_CAPABILITY_FIRMWARE_PAYLOADS |   \
                                  LOOM_XDNA_ELF_CAPABILITY_TILE_ZERO_FILL))

// Coordinate interpretation used by tile destination descriptors.
typedef enum loom_xdna_elf_coordinate_model_e {
  // Columns and rows are relative to the partition origin in the ABI note.
  LOOM_XDNA_ELF_COORDINATE_MODEL_PARTITION_RELATIVE = 1,
} loom_xdna_elf_coordinate_model_t;

// Tile-local destination memory selected by a tile program header.
typedef enum loom_xdna_elf_tile_memory_space_e {
  LOOM_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM = 1,
  LOOM_XDNA_ELF_TILE_MEMORY_SPACE_DATA = 2,
} loom_xdna_elf_tile_memory_space_t;

enum {
  LOOM_XDNA_ELF_TILE_COLUMN_SHIFT = 0,
  LOOM_XDNA_ELF_TILE_ROW_SHIFT = 8,
  LOOM_XDNA_ELF_TILE_MEMORY_SPACE_SHIFT = 16,
  LOOM_XDNA_ELF_TILE_FLAGS_SHIFT = 24,
};

#define LOOM_XDNA_ELF_TILE_COLUMN_MASK UINT32_C(0x000000FF)
#define LOOM_XDNA_ELF_TILE_ROW_MASK UINT32_C(0x0000FF00)
#define LOOM_XDNA_ELF_TILE_MEMORY_SPACE_MASK UINT32_C(0x00FF0000)
#define LOOM_XDNA_ELF_TILE_FLAGS_MASK UINT32_C(0xFF000000)
#define LOOM_XDNA_ELF_TILE_KNOWN_FLAGS UINT8_C(0)

// Decoded tile destination carried in one ELF p_paddr field.
typedef struct loom_xdna_elf_tile_destination_t {
  // Partition-relative tile column.
  uint8_t column;
  // Partition-relative tile row.
  uint8_t row;
  // Tile-local destination memory space.
  loom_xdna_elf_tile_memory_space_t memory_space;
  // Reserved destination flags. The initial ABI requires zero.
  uint8_t flags;
} loom_xdna_elf_tile_destination_t;

// Packs |destination| into one canonical ELF p_paddr value.
iree_status_t loom_xdna_elf_pack_tile_destination(
    const loom_xdna_elf_tile_destination_t* destination,
    uint32_t* out_physical_address);

// Decodes one ELF p_paddr value without validating profile coordinates.
loom_xdna_elf_tile_destination_t loom_xdna_elf_unpack_tile_destination(
    uint32_t physical_address);

enum {
  // ELF note type scoped by the `LOOM` owner string.
  LOOM_XDNA_ELF_NOTE_TYPE_ABI = 1,
  // Serialized byte length of the complete ELF note envelope.
  LOOM_XDNA_ELF_ABI_NOTE_SIZE = 84,
  // Serialized byte length of the ABI note description.
  LOOM_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE = 64,
  // Current incompatible XDNA product ABI major version.
  LOOM_XDNA_ELF_ABI_MAJOR = 1,
  // Current backward-compatible XDNA product ABI minor version.
  LOOM_XDNA_ELF_ABI_MINOR = 0,
};

#define LOOM_XDNA_ELF_NOTE_OWNER "LOOM"
#define LOOM_XDNA_ELF_NOTE_OWNER_SIZE 5
#define LOOM_XDNA_ELF_ABI_NOTE_MAGIC UINT32_C(0x414E4458)

// Decoded XDNA ABI note description.
typedef struct loom_xdna_elf_abi_note_t {
  // XDNA product ABI major version.
  uint16_t abi_major;
  // XDNA product ABI minor version.
  uint16_t abi_minor;
  // Target-generation identity.
  loom_xdna_target_generation_t target_generation;
  // Device revision selected by the resolved device profile.
  uint32_t target_revision;
  // Stable complete device-profile identity.
  uint64_t device_profile_id;
  // Stable firmware/configuration ABI identity.
  uint64_t firmware_abi_id;
  // Stable identity of placement and product-formation policies.
  uint64_t policy_id;
  // Loader capabilities required by this product.
  loom_xdna_elf_capabilities_t required_capabilities;
  // Physical column at partition-relative column zero.
  uint16_t partition_origin_column;
  // Physical row at partition-relative row zero.
  uint16_t partition_origin_row;
  // Number of addressable partition columns.
  uint16_t partition_column_count;
  // Number of addressable partition rows.
  uint16_t partition_row_count;
  // Coordinate interpretation used by all tile destinations.
  loom_xdna_elf_coordinate_model_t coordinate_model;
} loom_xdna_elf_abi_note_t;

// Encodes one complete canonical ELF ABI note envelope.
iree_status_t loom_xdna_elf_encode_abi_note(
    const loom_xdna_elf_abi_note_t* note, iree_byte_span_t storage);

enum {
  // Serialized byte length shared by entry, binding, and relocation headers.
  LOOM_XDNA_ELF_TABLE_HEADER_SIZE = 24,
  // Current fixed-table payload ABI major version.
  LOOM_XDNA_ELF_TABLE_ABI_MAJOR = 1,
  // Current fixed-table payload ABI minor version.
  LOOM_XDNA_ELF_TABLE_ABI_MINOR = 0,
  // Serialized entry-record byte length.
  LOOM_XDNA_ELF_ENTRY_RECORD_SIZE = 40,
  // Serialized binding-record byte length.
  LOOM_XDNA_ELF_BINDING_RECORD_SIZE = 56,
  // Serialized runtime-relocation record byte length.
  LOOM_XDNA_ELF_RELOCATION_RECORD_SIZE = 48,
};

#define LOOM_XDNA_ELF_ENTRY_TABLE_MAGIC UINT32_C(0x544E4558)
#define LOOM_XDNA_ELF_BINDING_TABLE_MAGIC UINT32_C(0x444E4258)
#define LOOM_XDNA_ELF_RELOCATION_TABLE_MAGIC UINT32_C(0x4C455258)

// Decoded common header for a fixed XDNA payload table.
typedef struct loom_xdna_elf_table_header_t {
  // Payload-specific table magic.
  uint32_t magic;
  // Incompatible table ABI major version.
  uint16_t abi_major;
  // Backward-compatible table ABI minor version.
  uint16_t abi_minor;
  // Serialized table-header byte length.
  uint16_t header_size;
  // Serialized byte length of one fixed record.
  uint16_t record_size;
  // Number of fixed records following the header.
  uint32_t record_count;
  // Complete program-header payload byte length.
  uint32_t byte_length;
  // Payload-specific auxiliary byte offset, or zero when unused.
  uint32_t auxiliary_offset;
} loom_xdna_elf_table_header_t;

// Encodes one common fixed-table header.
iree_status_t loom_xdna_elf_encode_table_header(
    const loom_xdna_elf_table_header_t* header, iree_byte_span_t storage);

// Entry-record flags.
typedef enum loom_xdna_elf_entry_flag_bits_e {
  // The entry is the default when the caller omits an export ordinal.
  LOOM_XDNA_ELF_ENTRY_FLAG_DEFAULT = 1u << 0,
} loom_xdna_elf_entry_flag_bits_t;
typedef uint32_t loom_xdna_elf_entry_flags_t;

#define LOOM_XDNA_ELF_KNOWN_ENTRY_FLAGS \
  ((loom_xdna_elf_entry_flags_t)LOOM_XDNA_ELF_ENTRY_FLAG_DEFAULT)

// Decoded fixed-width executable entry record.
typedef struct loom_xdna_elf_entry_record_t {
  // Dense stable export ordinal.
  uint32_t export_ordinal;
  // Payload-relative diagnostic-name byte offset, or zero when unnamed.
  uint32_t name_offset;
  // Diagnostic-name byte length, or zero when unnamed.
  uint32_t name_length;
  // Program-header ordinal of the entry's array realization.
  uint32_t array_program_header_ordinal;
  // Program-header ordinal of initial control, or UINT32_MAX when absent.
  uint32_t control_program_header_ordinal;
  // First dense binding ordinal owned by the entry.
  uint32_t first_binding_ordinal;
  // Number of dense bindings owned by the entry.
  uint32_t binding_count;
  // Entry behavior flags.
  loom_xdna_elf_entry_flags_t flags;
  // Additional loader capabilities required by the entry.
  loom_xdna_elf_capabilities_t required_capabilities;
} loom_xdna_elf_entry_record_t;

// Encodes one fixed-width executable entry record.
iree_status_t loom_xdna_elf_encode_entry_record(
    const loom_xdna_elf_entry_record_t* record, iree_byte_span_t storage);

// Runtime resource kind supplied through one binding.
typedef enum loom_xdna_elf_binding_kind_e {
  LOOM_XDNA_ELF_BINDING_KIND_BUFFER = 1,
  LOOM_XDNA_ELF_BINDING_KIND_SCALAR = 2,
} loom_xdna_elf_binding_kind_t;

// Address space required for one runtime binding.
typedef enum loom_xdna_elf_binding_address_space_e {
  LOOM_XDNA_ELF_BINDING_ADDRESS_SPACE_NONE = 0,
  LOOM_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL = 1,
  LOOM_XDNA_ELF_BINDING_ADDRESS_SPACE_HOST = 2,
} loom_xdna_elf_binding_address_space_t;

// Runtime binding access bits.
typedef enum loom_xdna_elf_binding_access_bits_e {
  LOOM_XDNA_ELF_BINDING_ACCESS_READ = 1u << 0,
  LOOM_XDNA_ELF_BINDING_ACCESS_WRITE = 1u << 1,
} loom_xdna_elf_binding_access_bits_t;
typedef uint32_t loom_xdna_elf_binding_access_t;

#define LOOM_XDNA_ELF_KNOWN_BINDING_ACCESS                              \
  ((loom_xdna_elf_binding_access_t)(LOOM_XDNA_ELF_BINDING_ACCESS_READ | \
                                    LOOM_XDNA_ELF_BINDING_ACCESS_WRITE))

// Runtime binding visibility and cache requirement bits.
typedef enum loom_xdna_elf_binding_usage_bits_e {
  LOOM_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE = 1u << 0,
  LOOM_XDNA_ELF_BINDING_USAGE_HOST_VISIBLE = 1u << 1,
  LOOM_XDNA_ELF_BINDING_USAGE_COHERENT = 1u << 2,
  LOOM_XDNA_ELF_BINDING_USAGE_CACHED = 1u << 3,
} loom_xdna_elf_binding_usage_bits_t;
typedef uint32_t loom_xdna_elf_binding_usage_t;

#define LOOM_XDNA_ELF_KNOWN_BINDING_USAGE                                       \
  ((loom_xdna_elf_binding_usage_t)(LOOM_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE | \
                                   LOOM_XDNA_ELF_BINDING_USAGE_HOST_VISIBLE |   \
                                   LOOM_XDNA_ELF_BINDING_USAGE_COHERENT |       \
                                   LOOM_XDNA_ELF_BINDING_USAGE_CACHED))

// Decoded fixed-width runtime binding record.
typedef struct loom_xdna_elf_binding_record_t {
  // Dense stable binding ordinal.
  uint32_t binding_ordinal;
  // Dense export ordinal owning this binding.
  uint32_t entry_ordinal;
  // Runtime resource kind.
  loom_xdna_elf_binding_kind_t kind;
  // Required resource address space.
  loom_xdna_elf_binding_address_space_t address_space;
  // Required runtime access.
  loom_xdna_elf_binding_access_t access;
  // Required visibility and cache behavior.
  loom_xdna_elf_binding_usage_t usage;
  // Minimum resource byte length.
  uint64_t minimum_byte_length;
  // Minimum base-address alignment in bytes.
  uint64_t minimum_alignment;
  // Minimum legal resource-relative byte offset.
  uint64_t minimum_byte_offset;
  // Maximum legal resource-relative byte offset.
  uint64_t maximum_byte_offset;
} loom_xdna_elf_binding_record_t;

// Encodes one fixed-width runtime binding record.
iree_status_t loom_xdna_elf_encode_binding_record(
    const loom_xdna_elf_binding_record_t* record, iree_byte_span_t storage);

// Runtime relocation field interpretation.
typedef enum loom_xdna_elf_relocation_kind_e {
  LOOM_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS = 1,
  LOOM_XDNA_ELF_RELOCATION_KIND_BINDING_BYTE_LENGTH = 2,
  LOOM_XDNA_ELF_RELOCATION_KIND_BINDING_BYTE_OFFSET = 3,
  LOOM_XDNA_ELF_RELOCATION_KIND_SCALAR_VALUE = 4,
} loom_xdna_elf_relocation_kind_t;

// Decoded fixed-width runtime relocation record.
typedef struct loom_xdna_elf_relocation_record_t {
  // Program-header ordinal containing the field to patch.
  uint32_t target_program_header_ordinal;
  // Byte offset of the field within the target payload.
  uint32_t target_byte_offset;
  // Dense binding ordinal supplying the runtime value.
  uint32_t binding_ordinal;
  // Runtime field interpretation.
  loom_xdna_elf_relocation_kind_t kind;
  // Encoded field width in bytes. The initial ABI permits four or eight.
  uint8_t field_byte_width;
  // Reserved relocation flags. The initial ABI requires zero.
  uint8_t flags;
  // Signed addend applied to the supplied runtime value.
  int64_t addend;
  // Minimum permitted relocated unsigned value.
  uint64_t minimum_value;
  // Maximum permitted relocated unsigned value.
  uint64_t maximum_value;
  // Required relocated-value alignment in bytes.
  uint64_t required_alignment;
} loom_xdna_elf_relocation_record_t;

// Encodes one fixed-width runtime relocation record.
iree_status_t loom_xdna_elf_encode_relocation_record(
    const loom_xdna_elf_relocation_record_t* record, iree_byte_span_t storage);

enum {
  // Current array/control payload ABI major version.
  LOOM_XDNA_ELF_PROGRAM_ABI_MAJOR = 1,
  // Current array/control payload ABI minor version.
  LOOM_XDNA_ELF_PROGRAM_ABI_MINOR = 0,
  // Serialized array payload-header byte length.
  LOOM_XDNA_ELF_ARRAY_HEADER_SIZE = 32,
  // Serialized control payload-header byte length.
  LOOM_XDNA_ELF_CONTROL_HEADER_SIZE = 24,
  // Serialized common program-record header byte length.
  LOOM_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE = 8,
  // Required alignment of every program record.
  LOOM_XDNA_ELF_PROGRAM_RECORD_ALIGNMENT = 4,
};

#define LOOM_XDNA_ELF_ARRAY_MAGIC UINT32_C(0x52524158)
#define LOOM_XDNA_ELF_CONTROL_MAGIC UINT32_C(0x4C544358)

// Decoded array configuration payload header.
typedef struct loom_xdna_elf_array_header_t {
  // Incompatible array-program ABI major version.
  uint16_t abi_major;
  // Backward-compatible array-program ABI minor version.
  uint16_t abi_minor;
  // Number of framed configuration records.
  uint32_t record_count;
  // Complete array payload byte length.
  uint32_t byte_length;
  // Array payload behavior flags. The initial ABI requires zero.
  uint32_t flags;
  // First tile program-header ordinal used by this realization.
  uint32_t first_tile_program_header_ordinal;
  // Number of consecutive tile program headers used by this realization.
  uint32_t tile_program_header_count;
} loom_xdna_elf_array_header_t;

// Encodes one array configuration payload header.
iree_status_t loom_xdna_elf_encode_array_header(
    const loom_xdna_elf_array_header_t* header, iree_byte_span_t storage);

// Decoded control-program payload header.
typedef struct loom_xdna_elf_control_header_t {
  // Incompatible control-program ABI major version.
  uint16_t abi_major;
  // Backward-compatible control-program ABI minor version.
  uint16_t abi_minor;
  // Number of framed control records.
  uint32_t record_count;
  // Complete control payload byte length.
  uint32_t byte_length;
  // Control payload behavior flags. The initial ABI requires zero.
  uint32_t flags;
} loom_xdna_elf_control_header_t;

// Encodes one control-program payload header.
iree_status_t loom_xdna_elf_encode_control_header(
    const loom_xdna_elf_control_header_t* header, iree_byte_span_t storage);

// Decoded common header preceding one array or control program record.
typedef struct loom_xdna_elf_program_record_header_t {
  // Payload-ABI record type. Zero is invalid.
  uint16_t type;
  // Payload-ABI record flags.
  uint16_t flags;
  // Complete aligned record byte length including this header.
  uint32_t byte_length;
} loom_xdna_elf_program_record_header_t;

// Encodes one common array/control program-record header.
iree_status_t loom_xdna_elf_encode_program_record_header(
    const loom_xdna_elf_program_record_header_t* header,
    iree_byte_span_t storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_ELF_FORMAT_H_
