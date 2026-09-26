// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native XDNA executable storage format shared by compiler and loader.
//
// ELF32LE program headers form the load directory. Header zero is METADATA;
// remaining headers are PT_LOAD, grouped by allocation and ordered by
// destination offset. p_paddr identifies an allocation requirement and p_vaddr
// its byte offset. Exact source aliases permit one file payload to initialize
// many destinations. Only explicit p_memsz tails are zeroed; gaps remain
// undefined.
//
// Metadata contains one header, then allocation rows, uint32_t allocation uses,
// entry rows, binding rows, relocation rows, invocation rows and raw name
// bytes. Counts derive every offset. These host values are never cast over file
// storage. Fixed-width codecs require complete row storage and perform no
// validation; external admission belongs to image_create.
//
// Allocation requirements describe storage, not ownership of a runtime object.
// Each entry selects its uses; callers may share immutable DMA backing and must
// keep mutable invocation backing private. Static relocations source allocation
// addresses and run on load. Dynamic relocations source external bindings and
// run on bind, after prior users have drained. Shards and device role
// transitions are compiled payload details, not entries or loader operations.
//
// Invocation zero establishes entry state. After terminal completion, the named
// next invocation is valid while context, backing and resident state remain
// intact. Reset or replacement by another entry invalidates that continuation.
// The current finite protocol is 0 -> 1 -> 1; a self-contained range can use
// 0 -> 0. The caller owns the continuation ordinal and completion frontier.

#ifndef IREE_SCHEMAS_XDNA_EXECUTABLE_H_
#define IREE_SCHEMAS_XDNA_EXECUTABLE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  IREE_XDNA_ELF_HEADER_SIZE = 52,
  IREE_XDNA_ELF_PROGRAM_HEADER_SIZE = 32,
  IREE_XDNA_ELF_SECTION_HEADER_SIZE = 40,
  IREE_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT = 4096,
  IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT = 4096,
  IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT = 65535,
  IREE_XDNA_ELF_MAX_ENTRY_NAME_LENGTH = 4096,
  IREE_XDNA_ELF_MAX_METADATA_TABLE_SIZE = 16 * 1024 * 1024,
  IREE_XDNA_ELF_PROGRAM_TYPE_LOAD = 1,
  IREE_XDNA_ELF_PROGRAM_TYPE_METADATA = 0x6C584408,
  IREE_XDNA_ELF_METADATA_MAGIC = 0x414E4458,
  IREE_XDNA_ELF_METADATA_VERSION = 2,
  IREE_XDNA_ELF_NATIVE_TRANSACTION_0_1 = 1,
};

// Fixed ELF identity for a canonical XDNA executable image.
typedef enum iree_xdna_elf_identity_e {
  IREE_XDNA_ELF_CLASS_32 = 1,
  IREE_XDNA_ELF_DATA_LITTLE_ENDIAN = 1,
  IREE_XDNA_ELF_VERSION_CURRENT = 1,
  IREE_XDNA_ELF_OS_ABI_NONE = 0,
  IREE_XDNA_ELF_ABI_VERSION_NONE = 0,
  IREE_XDNA_ELF_FILE_TYPE_EXEC = 2,
  IREE_XDNA_ELF_MACHINE_AIE = 264,
  IREE_XDNA_ELF_AIE2P_FLAGS = 3,
} iree_xdna_elf_identity_t;

// Supported XDNA target-generation identities.
typedef enum iree_xdna_target_generation_e {
  IREE_XDNA_TARGET_GENERATION_AIE2P = 3,
} iree_xdna_target_generation_t;

// ELF program-header permission bits used by the XDNA image ABI.
typedef enum iree_xdna_elf_program_flag_bits_e {
  IREE_XDNA_ELF_PROGRAM_FLAG_EXECUTE = 0x1,
  IREE_XDNA_ELF_PROGRAM_FLAG_WRITE = 0x2,
  IREE_XDNA_ELF_PROGRAM_FLAG_READ = 0x4,
} iree_xdna_elf_program_flag_bits_t;
typedef uint32_t iree_xdna_elf_program_flags_t;

// Address domain of one caller-owned backing allocation.
typedef enum iree_xdna_elf_allocation_domain_e {
  IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND = 1,
  IREE_XDNA_ELF_ALLOCATION_DOMAIN_DMA = 2,
} iree_xdna_elf_allocation_domain_t;

// Requirements controlling safe sharing and native access to backing.
typedef enum iree_xdna_elf_allocation_flag_bits_e {
  // Contents are immutable after loading and static relocation.
  IREE_XDNA_ELF_ALLOCATION_FLAG_IMMUTABLE = 1u << 0,
  // Device code writes this allocation during execution.
  IREE_XDNA_ELF_ALLOCATION_FLAG_DEVICE_WRITE = 1u << 1,
} iree_xdna_elf_allocation_flag_bits_t;

// Field interpretation of an eight-byte address patch.
typedef enum iree_xdna_elf_relocation_kind_e {
  // Shim BD address: low word bits 31:2 and high word bits 15:0. Preserve the
  // other bits, require four-byte alignment, and accept at most 48 address
  // bits.
  IREE_XDNA_ELF_RELOCATION_KIND_SHIM_ADDRESS = 1,
} iree_xdna_elf_relocation_kind_t;

// Runtime resource kind supplied through one binding.
typedef enum iree_xdna_elf_binding_kind_e {
  // Unused ABI slot. Every other field in the binding row is zero and no
  // relocation may reference it.
  IREE_XDNA_ELF_BINDING_KIND_NONE = 0,
  // External logical buffer consumed or produced by the executable.
  IREE_XDNA_ELF_BINDING_KIND_BUFFER = 1,
} iree_xdna_elf_binding_kind_t;

// Address space required for one runtime binding.
typedef enum iree_xdna_elf_binding_address_space_e {
  IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_NONE = 0,
  IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL = 1,
  IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_HOST = 2,
} iree_xdna_elf_binding_address_space_t;

// Runtime binding access bits.
typedef enum iree_xdna_elf_binding_access_bits_e {
  IREE_XDNA_ELF_BINDING_ACCESS_READ = 1u << 0,
  IREE_XDNA_ELF_BINDING_ACCESS_WRITE = 1u << 1,
} iree_xdna_elf_binding_access_bits_t;
typedef uint32_t iree_xdna_elf_binding_access_t;

#define IREE_XDNA_ELF_KNOWN_BINDING_ACCESS                              \
  ((iree_xdna_elf_binding_access_t)(IREE_XDNA_ELF_BINDING_ACCESS_READ | \
                                    IREE_XDNA_ELF_BINDING_ACCESS_WRITE))

// Runtime binding visibility and cache requirement bits.
typedef enum iree_xdna_elf_binding_usage_bits_e {
  IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE = 1u << 0,
  IREE_XDNA_ELF_BINDING_USAGE_HOST_VISIBLE = 1u << 1,
  IREE_XDNA_ELF_BINDING_USAGE_COHERENT = 1u << 2,
  IREE_XDNA_ELF_BINDING_USAGE_CACHED = 1u << 3,
} iree_xdna_elf_binding_usage_bits_t;
typedef uint32_t iree_xdna_elf_binding_usage_t;

#define IREE_XDNA_ELF_KNOWN_BINDING_USAGE                                       \
  ((iree_xdna_elf_binding_usage_t)(IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE | \
                                   IREE_XDNA_ELF_BINDING_USAGE_HOST_VISIBLE |   \
                                   IREE_XDNA_ELF_BINDING_USAGE_COHERENT |       \
                                   IREE_XDNA_ELF_BINDING_USAGE_CACHED))

// Decoded 64-byte header row.
typedef struct iree_xdna_elf_header_record_t {
  // XDNA metadata magic.
  uint32_t magic;
  // Incompatible metadata ABI version.
  uint16_t version;
  // Native command encoding identity.
  uint16_t native_encoding;
  // Target instruction-set generation.
  uint32_t target_generation;
  // Incompatible device-profile revision.
  uint32_t device_profile_revision;
  // Execution-profile identity.
  uint64_t device_profile_id;
  // Native firmware protocol identity.
  uint64_t firmware_abi_id;
  // Required context-relative column count.
  uint16_t column_count;
  // Required context-relative row count.
  uint16_t row_count;
  // Number of allocation requirements.
  uint32_t allocation_count;
  // Number of entry-relative allocation uses.
  uint32_t allocation_use_count;
  // Number of exported entries.
  uint32_t entry_count;
  // Number of external binding contracts.
  uint32_t binding_count;
  // Number of static and dynamic address fixups.
  uint32_t relocation_count;
  // Number of native invocation ranges.
  uint32_t invocation_count;
  // Number of name bytes following all tables.
  uint32_t string_byte_length;
} iree_xdna_elf_header_record_t;

enum { IREE_XDNA_ELF_HEADER_RECORD_SIZE = 64 };

// Encodes one complete header row into caller-provided storage.
void iree_xdna_elf_encode_header(const iree_xdna_elf_header_record_t* value,
                                 uint8_t* storage);

// Decodes one complete header row without interpreting its values.
iree_xdna_elf_header_record_t iree_xdna_elf_decode_header(
    const uint8_t* storage);

// Decoded 32-byte allocation row.
typedef struct iree_xdna_elf_allocation_record_t {
  // Native-command or DMA address domain.
  uint32_t domain;
  // Immutability and device-write requirements.
  uint32_t flags;
  // Required backing capacity in bytes.
  uint64_t byte_length;
  // Power-of-two device-address alignment in bytes.
  uint64_t alignment;
  // First ELF program-header ordinal initializing this allocation.
  uint32_t first_load;
  // Number of destination-ordered load headers.
  uint32_t load_count;
} iree_xdna_elf_allocation_record_t;

enum { IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE = 32 };

// Encodes one complete allocation row into caller-provided storage.
void iree_xdna_elf_encode_allocation(
    const iree_xdna_elf_allocation_record_t* value, uint8_t* storage);

// Decodes one complete allocation row without interpreting its values.
iree_xdna_elf_allocation_record_t iree_xdna_elf_decode_allocation(
    const uint8_t* storage);

// Decoded 48-byte entry row.
typedef struct iree_xdna_elf_entry_record_t {
  // Byte offset within the trailing name table.
  uint32_t name_offset;
  // Number of name bytes, without a terminator.
  uint32_t name_length;
  // First entry-relative allocation use in the global use table.
  uint32_t first_allocation_use;
  // Number of allocations required by this entry.
  uint32_t allocation_use_count;
  // First external binding contract.
  uint32_t first_binding;
  // Number of densely indexed external bindings.
  uint32_t binding_count;
  // First fixup sourced from an allocation-use address.
  uint32_t first_static_relocation;
  // Number of fixups applied once when loading backing.
  uint32_t static_relocation_count;
  // First fixup sourced from an external binding address.
  uint32_t first_dynamic_relocation;
  // Number of fixups applied when rebinding.
  uint32_t dynamic_relocation_count;
  // First native invocation; entry-relative zero establishes state.
  uint32_t first_invocation;
  // Number of native invocation ranges.
  uint32_t invocation_count;
} iree_xdna_elf_entry_record_t;

enum { IREE_XDNA_ELF_ENTRY_RECORD_SIZE = 48 };

// Encodes one complete entry row into caller-provided storage.
void iree_xdna_elf_encode_entry(const iree_xdna_elf_entry_record_t* value,
                                uint8_t* storage);

// Decodes one complete entry row without interpreting its values.
iree_xdna_elf_entry_record_t iree_xdna_elf_decode_entry(const uint8_t* storage);

// Decoded 40-byte binding row.
typedef struct iree_xdna_elf_binding_record_t {
  // Runtime resource kind or NONE for an unused ABI slot.
  uint16_t kind;
  // Required resource address space.
  uint16_t address_space;
  // Required read/write access bits.
  uint16_t access;
  // Required visibility and cache behavior.
  uint16_t usage;
  // Minimum bound byte length.
  uint64_t minimum_byte_length;
  // Power-of-two base-address alignment in bytes.
  uint64_t minimum_alignment;
  // Minimum offset within the logical buffer.
  uint64_t minimum_byte_offset;
  // Maximum logical offset; UINT64_MAX adds no upper bound.
  uint64_t maximum_byte_offset;
} iree_xdna_elf_binding_record_t;

enum { IREE_XDNA_ELF_BINDING_RECORD_SIZE = 40 };

// Encodes one complete binding row into caller-provided storage.
void iree_xdna_elf_encode_binding(const iree_xdna_elf_binding_record_t* value,
                                  uint8_t* storage);

// Decodes one complete binding row without interpreting its values.
iree_xdna_elf_binding_record_t iree_xdna_elf_decode_binding(
    const uint8_t* storage);

// Decoded 48-byte relocation row.
typedef struct iree_xdna_elf_relocation_record_t {
  // Entry-relative destination allocation use.
  uint32_t destination_use;
  // Allocation use for static fixups, external binding for dynamic fixups.
  uint32_t source_ordinal;
  // Destination field byte offset within its allocation.
  uint32_t byte_offset;
  // Native field encoding; no command decoding is required.
  uint32_t kind;
  // Signed byte addend applied to the source address.
  int64_t addend;
  // Minimum relocated address.
  uint64_t minimum_value;
  // Maximum relocated address.
  uint64_t maximum_value;
  // Power-of-two relocated-address alignment.
  uint64_t alignment;
} iree_xdna_elf_relocation_record_t;

enum { IREE_XDNA_ELF_RELOCATION_RECORD_SIZE = 48 };

// Encodes one complete relocation row into caller-provided storage.
void iree_xdna_elf_encode_relocation(
    const iree_xdna_elf_relocation_record_t* value, uint8_t* storage);

// Decodes one complete relocation row without interpreting its values.
iree_xdna_elf_relocation_record_t iree_xdna_elf_decode_relocation(
    const uint8_t* storage);

// Decoded 16-byte invocation row.
typedef struct iree_xdna_elf_invocation_record_t {
  // Entry-relative native-command allocation use.
  uint32_t allocation_use;
  // Byte offset of the complete native command range.
  uint32_t byte_offset;
  // Number of initialized command bytes submitted.
  uint32_t byte_length;
  // Entry-relative continuation after terminal completion with state intact.
  uint32_t next_invocation;
} iree_xdna_elf_invocation_record_t;

enum { IREE_XDNA_ELF_INVOCATION_RECORD_SIZE = 16 };

// Encodes one complete invocation row into caller-provided storage.
void iree_xdna_elf_encode_invocation(
    const iree_xdna_elf_invocation_record_t* value, uint8_t* storage);

// Decodes one complete invocation row without interpreting its values.
iree_xdna_elf_invocation_record_t iree_xdna_elf_decode_invocation(
    const uint8_t* storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_SCHEMAS_XDNA_EXECUTABLE_H_
