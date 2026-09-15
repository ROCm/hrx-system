// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical AMD XDNA executable image wire format.
//
// An image is an ELF32 little-endian executable whose program headers form the
// complete runtime directory. The structures declared here are decoded host
// values and are never cast over file storage. Codecs access every wire field
// explicitly so host padding, alignment, and endianness cannot enter the ABI.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_FORMAT_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_FORMAT_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// ELF image envelope
//===----------------------------------------------------------------------===//

enum {
  // Canonical ELF32 header byte length.
  IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE = 52,
  // Canonical ELF32 program-header byte length.
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE = 32,
  // Canonical ELF32 section-header byte length.
  IREE_HAL_AMD_XDNA_ELF_SECTION_HEADER_SIZE = 40,
  // Maximum program-header cardinality accepted by this image ABI.
  IREE_HAL_AMD_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT = 4096,
  // Maximum diagnostic section-header cardinality accepted by this image ABI.
  IREE_HAL_AMD_XDNA_ELF_MAX_SECTION_HEADER_COUNT = 4096,
  // Maximum record cardinality accepted in any fixed image table.
  IREE_HAL_AMD_XDNA_ELF_MAX_TABLE_RECORD_COUNT = 65535,
  // Maximum ARRAY/CONTROL record cardinality across one image.
  IREE_HAL_AMD_XDNA_ELF_MAX_PROGRAM_RECORD_COUNT = 65535,
  // Maximum diagnostic entry-name byte length.
  IREE_HAL_AMD_XDNA_ELF_MAX_ENTRY_NAME_LENGTH = 4096,
  // Maximum serialized byte length of one fixed metadata table.
  IREE_HAL_AMD_XDNA_ELF_MAX_METADATA_TABLE_SIZE = 16 * 1024 * 1024,
  // Maximum serialized byte length of one ARRAY or CONTROL payload.
  IREE_HAL_AMD_XDNA_ELF_MAX_PROGRAM_PAYLOAD_SIZE = 16 * 1024 * 1024,
  // Maximum serialized byte length of one ARRAY or CONTROL record.
  IREE_HAL_AMD_XDNA_ELF_MAX_PROGRAM_RECORD_SIZE = 1024 * 1024,
};

// Fixed ELF identity for a canonical XDNA executable image.
typedef enum iree_hal_amd_xdna_elf_identity_e {
  IREE_HAL_AMD_XDNA_ELF_CLASS_32 = 1,
  IREE_HAL_AMD_XDNA_ELF_DATA_LITTLE_ENDIAN = 1,
  IREE_HAL_AMD_XDNA_ELF_VERSION_CURRENT = 1,
  IREE_HAL_AMD_XDNA_ELF_OS_ABI_NONE = 0,
  IREE_HAL_AMD_XDNA_ELF_ABI_VERSION_NONE = 0,
  IREE_HAL_AMD_XDNA_ELF_FILE_TYPE_EXEC = 2,
  IREE_HAL_AMD_XDNA_ELF_MACHINE_AIE = 264,
  IREE_HAL_AMD_XDNA_ELF_AIE2P_FLAGS = 3,
} iree_hal_amd_xdna_elf_identity_t;

// Supported XDNA target-generation identities.
typedef enum iree_hal_amd_xdna_target_generation_e {
  IREE_HAL_AMD_XDNA_TARGET_GENERATION_AIE2P = 3,
} iree_hal_amd_xdna_target_generation_t;

// Runtime directory roles encoded in the ELF program-header type field.
//
// XDNA types occupy one named subrange of PT_LOOS..PT_HIOS. The values are
// independent of vendor AIE processor-specific program headers.
typedef enum iree_hal_amd_xdna_elf_program_type_e {
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_NOTE = 4,
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ENTRIES = 0x6C584401,
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_BINDINGS = 0x6C584402,
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_RELOCATIONS = 0x6C584403,
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE = 0x6C584404,
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY = 0x6C584405,
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL = 0x6C584406,
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_FIRMWARE = 0x6C584407,
} iree_hal_amd_xdna_elf_program_type_t;

// ELF program-header permission bits used by the XDNA image ABI.
typedef enum iree_hal_amd_xdna_elf_program_flag_bits_e {
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_EXECUTE = 0x1,
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_WRITE = 0x2,
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ = 0x4,
} iree_hal_amd_xdna_elf_program_flag_bits_t;
typedef uint32_t iree_hal_amd_xdna_elf_program_flags_t;

// Loader capabilities that may be required by one image.
typedef enum iree_hal_amd_xdna_elf_capability_bits_e {
  // Typed runtime relocation records must be applied.
  IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS = 1ull << 0,
  // One or more CONTROL programs must be executed.
  IREE_HAL_AMD_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS = 1ull << 1,
  // One or more firmware-facing payloads must be consumed.
  IREE_HAL_AMD_XDNA_ELF_CAPABILITY_FIRMWARE_PAYLOADS = 1ull << 2,
  // One or more TILE placements have a zero-filled tail.
  IREE_HAL_AMD_XDNA_ELF_CAPABILITY_TILE_ZERO_FILL = 1ull << 3,
} iree_hal_amd_xdna_elf_capability_bits_t;
typedef uint64_t iree_hal_amd_xdna_elf_capabilities_t;

#define IREE_HAL_AMD_XDNA_ELF_KNOWN_CAPABILITIES                                                 \
  ((iree_hal_amd_xdna_elf_capabilities_t)(IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS | \
                                          IREE_HAL_AMD_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS |    \
                                          IREE_HAL_AMD_XDNA_ELF_CAPABILITY_FIRMWARE_PAYLOADS |   \
                                          IREE_HAL_AMD_XDNA_ELF_CAPABILITY_TILE_ZERO_FILL))

// Coordinate interpretation used by TILE destination descriptors.
typedef enum iree_hal_amd_xdna_elf_coordinate_model_e {
  // Columns and rows are relative to a context whose logical origin is zero.
  IREE_HAL_AMD_XDNA_ELF_COORDINATE_MODEL_CONTEXT_RELATIVE = 1,
} iree_hal_amd_xdna_elf_coordinate_model_t;

// Tile-local destination memory selected by a TILE program header.
typedef enum iree_hal_amd_xdna_elf_tile_memory_space_e {
  IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM = 1,
  IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_DATA = 2,
} iree_hal_amd_xdna_elf_tile_memory_space_t;

enum {
  IREE_HAL_AMD_XDNA_ELF_TILE_COLUMN_SHIFT = 0,
  IREE_HAL_AMD_XDNA_ELF_TILE_ROW_SHIFT = 8,
  IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_SHIFT = 16,
  IREE_HAL_AMD_XDNA_ELF_TILE_FLAGS_SHIFT = 24,
};

#define IREE_HAL_AMD_XDNA_ELF_TILE_COLUMN_MASK UINT32_C(0x000000FF)
#define IREE_HAL_AMD_XDNA_ELF_TILE_ROW_MASK UINT32_C(0x0000FF00)
#define IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_MASK UINT32_C(0x00FF0000)
#define IREE_HAL_AMD_XDNA_ELF_TILE_FLAGS_MASK UINT32_C(0xFF000000)
#define IREE_HAL_AMD_XDNA_ELF_TILE_KNOWN_FLAGS UINT8_C(0)

// Decoded tile destination carried in one ELF p_paddr field.
typedef struct iree_hal_amd_xdna_elf_tile_destination_t {
  // Context-relative tile column.
  uint8_t column;
  // Context-relative tile row.
  uint8_t row;
  // Tile-local destination memory space.
  iree_hal_amd_xdna_elf_tile_memory_space_t memory_space;
  // Reserved destination flags. The initial ABI requires zero.
  uint8_t flags;
} iree_hal_amd_xdna_elf_tile_destination_t;

// Packs |destination| into one canonical ELF p_paddr value.
iree_status_t iree_hal_amd_xdna_elf_pack_tile_destination(
    const iree_hal_amd_xdna_elf_tile_destination_t* destination,
    uint32_t* out_physical_address);

// Decodes one ELF p_paddr value without validating target coordinates.
iree_hal_amd_xdna_elf_tile_destination_t
iree_hal_amd_xdna_elf_unpack_tile_destination(uint32_t physical_address);

//===----------------------------------------------------------------------===//
// Image ABI note
//===----------------------------------------------------------------------===//

enum {
  // ELF note type scoped by the `LOOM` owner string.
  IREE_HAL_AMD_XDNA_ELF_NOTE_TYPE_ABI = 1,
  // Serialized byte length of the complete ELF note envelope.
  IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_SIZE = 84,
  // Serialized byte length of the ABI note description.
  IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_DESCRIPTION_SIZE = 64,
  // Current incompatible XDNA image ABI major version.
  IREE_HAL_AMD_XDNA_ELF_ABI_MAJOR = 1,
  // Current backward-compatible XDNA image ABI minor version.
  IREE_HAL_AMD_XDNA_ELF_ABI_MINOR = 0,
};

#define IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER "LOOM"
#define IREE_HAL_AMD_XDNA_ELF_NOTE_OWNER_SIZE 5
#define IREE_HAL_AMD_XDNA_ELF_ABI_NOTE_MAGIC UINT32_C(0x414E4458)

// Decoded XDNA image ABI note.
typedef struct iree_hal_amd_xdna_elf_abi_note_t {
  // XDNA image ABI major version.
  uint16_t abi_major;
  // XDNA image ABI minor version.
  uint16_t abi_minor;
  // Target-generation identity.
  iree_hal_amd_xdna_target_generation_t target_generation;
  // Incompatible revision of the resolved device profile.
  uint32_t device_profile_revision;
  // Stable complete device-profile identity.
  uint64_t device_profile_id;
  // Stable firmware/configuration ABI identity.
  uint64_t firmware_abi_id;
  // Stable identity of placement and image-formation policies.
  uint64_t policy_id;
  // Loader capabilities required by this image.
  iree_hal_amd_xdna_elf_capabilities_t required_capabilities;
  // Logical context-relative column origin. The initial ABI requires zero.
  uint16_t context_origin_column;
  // Logical context-relative row origin. The initial ABI requires zero.
  uint16_t context_origin_row;
  // Number of addressable context-relative columns.
  uint16_t context_column_count;
  // Number of addressable context-relative rows.
  uint16_t context_row_count;
  // Coordinate interpretation used by all TILE destinations.
  iree_hal_amd_xdna_elf_coordinate_model_t coordinate_model;
} iree_hal_amd_xdna_elf_abi_note_t;

// Encodes one complete canonical ELF ABI note envelope.
iree_status_t iree_hal_amd_xdna_elf_encode_abi_note(
    const iree_hal_amd_xdna_elf_abi_note_t* note, iree_byte_span_t storage);

// Decodes and locally validates one complete ELF ABI note envelope.
iree_status_t iree_hal_amd_xdna_elf_decode_abi_note(
    iree_const_byte_span_t storage, iree_hal_amd_xdna_elf_abi_note_t* out_note);

//===----------------------------------------------------------------------===//
// Entry, binding, and relocation tables
//===----------------------------------------------------------------------===//

enum {
  // Serialized byte length shared by all fixed table headers.
  IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE = 24,
  // Current fixed-table payload ABI major version.
  IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MAJOR = 1,
  // Current fixed-table payload ABI minor version.
  IREE_HAL_AMD_XDNA_ELF_TABLE_ABI_MINOR = 0,
  // Serialized entry-record byte length.
  IREE_HAL_AMD_XDNA_ELF_ENTRY_RECORD_SIZE = 40,
  // Serialized binding-record byte length.
  IREE_HAL_AMD_XDNA_ELF_BINDING_RECORD_SIZE = 56,
  // Serialized runtime-relocation record byte length.
  IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE = 48,
};

#define IREE_HAL_AMD_XDNA_ELF_ENTRY_TABLE_MAGIC UINT32_C(0x544E4558)
#define IREE_HAL_AMD_XDNA_ELF_BINDING_TABLE_MAGIC UINT32_C(0x444E4258)
#define IREE_HAL_AMD_XDNA_ELF_RELOCATION_TABLE_MAGIC UINT32_C(0x4C455258)

// Decoded common header for a fixed XDNA image table.
typedef struct iree_hal_amd_xdna_elf_table_header_t {
  // Table-specific magic.
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
  // Table-specific auxiliary byte offset, or zero when unused.
  uint32_t auxiliary_offset;
} iree_hal_amd_xdna_elf_table_header_t;

// Encodes one common fixed-table header.
iree_status_t iree_hal_amd_xdna_elf_encode_table_header(
    const iree_hal_amd_xdna_elf_table_header_t* header,
    iree_byte_span_t storage);

// Decodes one common fixed-table header.
iree_status_t iree_hal_amd_xdna_elf_decode_table_header(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_table_header_t* out_header);

// Entry-record flags.
typedef enum iree_hal_amd_xdna_elf_entry_flag_bits_e {
  // The entry is selected when a caller omits an export ordinal.
  IREE_HAL_AMD_XDNA_ELF_ENTRY_FLAG_DEFAULT = 1u << 0,
} iree_hal_amd_xdna_elf_entry_flag_bits_t;
typedef uint32_t iree_hal_amd_xdna_elf_entry_flags_t;

#define IREE_HAL_AMD_XDNA_ELF_KNOWN_ENTRY_FLAGS \
  ((iree_hal_amd_xdna_elf_entry_flags_t)        \
       IREE_HAL_AMD_XDNA_ELF_ENTRY_FLAG_DEFAULT)

// Decoded fixed-width executable entry record.
typedef struct iree_hal_amd_xdna_elf_entry_record_t {
  // Dense stable export ordinal.
  uint32_t export_ordinal;
  // Payload-relative diagnostic-name byte offset, or zero when unnamed.
  uint32_t name_offset;
  // Diagnostic-name byte length, or zero when unnamed.
  uint32_t name_length;
  // Program-header ordinal of the entry's ARRAY realization.
  uint32_t array_program_header_ordinal;
  // Program-header ordinal of initial CONTROL, or UINT32_MAX when absent.
  uint32_t control_program_header_ordinal;
  // First dense binding ordinal owned by the entry.
  uint32_t first_binding_ordinal;
  // Number of dense bindings owned by the entry.
  uint32_t binding_count;
  // Entry behavior flags.
  iree_hal_amd_xdna_elf_entry_flags_t flags;
  // Additional loader capabilities required by the entry.
  iree_hal_amd_xdna_elf_capabilities_t required_capabilities;
} iree_hal_amd_xdna_elf_entry_record_t;

// Encodes one fixed-width executable entry record.
iree_status_t iree_hal_amd_xdna_elf_encode_entry_record(
    const iree_hal_amd_xdna_elf_entry_record_t* record,
    iree_byte_span_t storage);

// Decodes one fixed-width executable entry record.
iree_status_t iree_hal_amd_xdna_elf_decode_entry_record(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_entry_record_t* out_record);

// Runtime resource kind supplied through one binding.
typedef enum iree_hal_amd_xdna_elf_binding_kind_e {
  IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_BUFFER = 1,
  IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_SCALAR = 2,
} iree_hal_amd_xdna_elf_binding_kind_t;

// Address space required for one runtime binding.
typedef enum iree_hal_amd_xdna_elf_binding_address_space_e {
  IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_NONE = 0,
  IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL = 1,
  IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_HOST = 2,
} iree_hal_amd_xdna_elf_binding_address_space_t;

// Runtime binding access bits.
typedef enum iree_hal_amd_xdna_elf_binding_access_bits_e {
  IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ = 1u << 0,
  IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_WRITE = 1u << 1,
} iree_hal_amd_xdna_elf_binding_access_bits_t;
typedef uint32_t iree_hal_amd_xdna_elf_binding_access_t;

#define IREE_HAL_AMD_XDNA_ELF_KNOWN_BINDING_ACCESS                                      \
  ((iree_hal_amd_xdna_elf_binding_access_t)(IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ | \
                                            IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_WRITE))

// Runtime binding visibility and cache requirement bits.
typedef enum iree_hal_amd_xdna_elf_binding_usage_bits_e {
  IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE = 1u << 0,
  IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_HOST_VISIBLE = 1u << 1,
  IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_COHERENT = 1u << 2,
  IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_CACHED = 1u << 3,
} iree_hal_amd_xdna_elf_binding_usage_bits_t;
typedef uint32_t iree_hal_amd_xdna_elf_binding_usage_t;

#define IREE_HAL_AMD_XDNA_ELF_KNOWN_BINDING_USAGE                                               \
  ((iree_hal_amd_xdna_elf_binding_usage_t)(IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE | \
                                           IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_HOST_VISIBLE |   \
                                           IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_COHERENT |       \
                                           IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_CACHED))

// Decoded fixed-width runtime binding record.
typedef struct iree_hal_amd_xdna_elf_binding_record_t {
  // Dense stable binding ordinal.
  uint32_t binding_ordinal;
  // Dense export ordinal owning this binding.
  uint32_t entry_ordinal;
  // Runtime resource kind.
  iree_hal_amd_xdna_elf_binding_kind_t kind;
  // Required resource address space.
  iree_hal_amd_xdna_elf_binding_address_space_t address_space;
  // Required runtime access.
  iree_hal_amd_xdna_elf_binding_access_t access;
  // Required visibility and cache behavior.
  iree_hal_amd_xdna_elf_binding_usage_t usage;
  // Minimum submitted binding-range byte length.
  uint64_t minimum_byte_length;
  // Minimum base-address alignment in bytes.
  uint64_t minimum_alignment;
  // Minimum legal offset of the submitted range within its logical resource.
  uint64_t minimum_byte_offset;
  // Maximum legal offset of the submitted range within its logical resource.
  // UINT64_MAX adds no upper bound beyond the resource's actual extent.
  uint64_t maximum_byte_offset;
} iree_hal_amd_xdna_elf_binding_record_t;

// Encodes one fixed-width runtime binding record.
iree_status_t iree_hal_amd_xdna_elf_encode_binding_record(
    const iree_hal_amd_xdna_elf_binding_record_t* record,
    iree_byte_span_t storage);

// Decodes one fixed-width runtime binding record.
iree_status_t iree_hal_amd_xdna_elf_decode_binding_record(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_binding_record_t* out_record);

// Runtime relocation field interpretation.
typedef enum iree_hal_amd_xdna_elf_relocation_kind_e {
  IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS = 1,
  IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_BYTE_LENGTH = 2,
  IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_BYTE_OFFSET = 3,
  IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_SCALAR_VALUE = 4,
} iree_hal_amd_xdna_elf_relocation_kind_t;

// Decoded fixed-width runtime relocation record.
typedef struct iree_hal_amd_xdna_elf_relocation_record_t {
  // Program-header ordinal containing the field to patch.
  uint32_t target_program_header_ordinal;
  // Byte offset of the field within the target payload.
  uint32_t target_byte_offset;
  // Dense binding ordinal supplying the runtime value.
  uint32_t binding_ordinal;
  // Runtime field interpretation.
  iree_hal_amd_xdna_elf_relocation_kind_t kind;
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
} iree_hal_amd_xdna_elf_relocation_record_t;

// Encodes one fixed-width runtime relocation record.
iree_status_t iree_hal_amd_xdna_elf_encode_relocation_record(
    const iree_hal_amd_xdna_elf_relocation_record_t* record,
    iree_byte_span_t storage);

// Decodes one fixed-width runtime relocation record.
iree_status_t iree_hal_amd_xdna_elf_decode_relocation_record(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_relocation_record_t* out_record);

//===----------------------------------------------------------------------===//
// ARRAY and CONTROL program framing
//===----------------------------------------------------------------------===//

enum {
  // Current ARRAY/CONTROL payload ABI major version.
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MAJOR = 1,
  // Current ARRAY/CONTROL payload ABI minor version.
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MINOR = 0,
  // Serialized ARRAY payload-header byte length.
  IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE = 32,
  // Serialized CONTROL payload-header byte length.
  IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE = 24,
  // Serialized common program-record header byte length.
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE = 8,
  // Required alignment of every program record.
  IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_ALIGNMENT = 4,
};

#define IREE_HAL_AMD_XDNA_ELF_ARRAY_MAGIC UINT32_C(0x52524158)
#define IREE_HAL_AMD_XDNA_ELF_CONTROL_MAGIC UINT32_C(0x4C544358)

// Decoded ARRAY configuration payload header.
typedef struct iree_hal_amd_xdna_elf_array_header_t {
  // Incompatible ARRAY program ABI major version.
  uint16_t abi_major;
  // Backward-compatible ARRAY program ABI minor version.
  uint16_t abi_minor;
  // Number of framed configuration records.
  uint32_t record_count;
  // Complete ARRAY payload byte length.
  uint32_t byte_length;
  // ARRAY payload behavior flags. The initial ABI requires zero.
  uint32_t flags;
  // First TILE program-header ordinal used by this realization.
  uint32_t first_tile_program_header_ordinal;
  // Number of consecutive TILE program headers used by this realization.
  uint32_t tile_program_header_count;
} iree_hal_amd_xdna_elf_array_header_t;

// Encodes one ARRAY configuration payload header.
iree_status_t iree_hal_amd_xdna_elf_encode_array_header(
    const iree_hal_amd_xdna_elf_array_header_t* header,
    iree_byte_span_t storage);

// Decodes and locally validates one ARRAY configuration payload header.
iree_status_t iree_hal_amd_xdna_elf_decode_array_header(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_array_header_t* out_header);

// Decoded CONTROL program payload header.
typedef struct iree_hal_amd_xdna_elf_control_header_t {
  // Incompatible CONTROL program ABI major version.
  uint16_t abi_major;
  // Backward-compatible CONTROL program ABI minor version.
  uint16_t abi_minor;
  // Number of framed control records.
  uint32_t record_count;
  // Complete CONTROL payload byte length.
  uint32_t byte_length;
  // CONTROL payload behavior flags. The initial ABI requires zero.
  uint32_t flags;
} iree_hal_amd_xdna_elf_control_header_t;

// Encodes one CONTROL program payload header.
iree_status_t iree_hal_amd_xdna_elf_encode_control_header(
    const iree_hal_amd_xdna_elf_control_header_t* header,
    iree_byte_span_t storage);

// Decodes and locally validates one CONTROL program payload header.
iree_status_t iree_hal_amd_xdna_elf_decode_control_header(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_control_header_t* out_header);

// Decoded common header preceding one ARRAY or CONTROL program record.
typedef struct iree_hal_amd_xdna_elf_program_record_header_t {
  // Payload-ABI record type. Zero is invalid.
  uint16_t type;
  // Payload-ABI record flags.
  uint16_t flags;
  // Complete aligned record byte length including this header.
  uint32_t byte_length;
} iree_hal_amd_xdna_elf_program_record_header_t;

// Encodes one common ARRAY/CONTROL program-record header.
iree_status_t iree_hal_amd_xdna_elf_encode_program_record_header(
    const iree_hal_amd_xdna_elf_program_record_header_t* header,
    iree_byte_span_t storage);

// Decodes and locally validates one ARRAY/CONTROL program-record header.
iree_status_t iree_hal_amd_xdna_elf_decode_program_record_header(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_program_record_header_t* out_header);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_FORMAT_H_
