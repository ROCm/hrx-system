// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loom-native code-object contribution assembly.
//
// Backend workers should produce immutable contribution records instead of
// mutating a shared final artifact writer. This layer stitches section byte
// fragments into final section payloads and records each contribution's offset
// in the joined section. Later symbol-table and fixup code can globalize local
// symbol/fixup offsets through those layout rows without reading arbitrary
// object files.

#ifndef LOOM_TARGET_EMIT_NATIVE_CONTRIBUTION_H_
#define LOOM_TARGET_EMIT_NATIVE_CONTRIBUTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Storage carried by a native section. A reservation occupies address space
// without contributing artifact bytes. The target's loading contract determines
// initialization, such as zero-filled process data or uninitialized tile-local
// function storage.
typedef enum loom_native_section_storage_e {
  LOOM_NATIVE_SECTION_STORAGE_CONTENTS = 0,
  LOOM_NATIVE_SECTION_STORAGE_RESERVATION = 1,
} loom_native_section_storage_t;

// Runtime access required by a section. NONE denotes nonresident information.
typedef enum loom_native_section_access_bits_e {
  LOOM_NATIVE_SECTION_ACCESS_NONE = 0,
  LOOM_NATIVE_SECTION_ACCESS_READ = 1u << 0,
  LOOM_NATIVE_SECTION_ACCESS_WRITE = 1u << 1,
  LOOM_NATIVE_SECTION_ACCESS_EXECUTE = 1u << 2,
} loom_native_section_access_bits_t;
typedef uint32_t loom_native_section_access_t;

// Assembled native storage, independent of its object-file representation.
// Product builders translate these records into format-specific section tables.
typedef struct loom_native_section_t {
  // Name shared by contributions to this section.
  iree_string_view_t name;
  // Whether the section carries bytes or reserves address space.
  loom_native_section_storage_t storage;
  // Runtime access required by the section.
  loom_native_section_access_t access;
  // Runtime address assigned by placement; zero before placement or in objects.
  uint64_t address;
  // Required power-of-two alignment in bytes.
  uint64_t alignment;
  // Arena-owned bytes for CONTENTS storage; empty for RESERVATION storage.
  iree_const_byte_span_t contents;
  // Address-space extent for RESERVATION storage; zero for CONTENTS storage.
  uint64_t reservation_length;
} loom_native_section_t;

// Returns the address-space extent of a native section.
static inline uint64_t loom_native_section_byte_length(
    const loom_native_section_t* section) {
  return section->storage == LOOM_NATIVE_SECTION_STORAGE_RESERVATION
             ? section->reservation_length
             : (uint64_t)section->contents.data_length;
}

// One worker-produced contribution to a named native section. These records
// are compiler-owned: producers establish valid names, alignment, contents,
// and matching storage/access for contributions sharing a name.
typedef struct loom_native_section_contribution_t {
  // Section name such as `.text` or `.rodata`.
  iree_string_view_t section_name;
  // Whether this contribution carries bytes or reserves address space.
  loom_native_section_storage_t storage;
  // Runtime access required by the section.
  loom_native_section_access_t access;
  // Power-of-two alignment within the joined section. Zero means one byte.
  uint64_t contribution_alignment;
  // Contribution bytes. The referenced storage must stay live for the duration
  // of assembly; the assembled output copies bytes into the caller's arena.
  iree_const_byte_span_t contents;
  // Address-space extent for RESERVATION storage; zero for CONTENTS storage.
  uint64_t reservation_length;
} loom_native_section_contribution_t;

// Resolved position of one input contribution inside the assembled sections.
typedef struct loom_native_section_contribution_layout_t {
  // Index into loom_native_section_contribution_assembly_t.sections.
  iree_host_size_t section_index;
  // Byte offset within the assembled section.
  uint64_t section_offset;
} loom_native_section_contribution_layout_t;

// Output of assembling native section contributions.
typedef struct loom_native_section_contribution_assembly_t {
  // Arena-backed native sections, with runtime addresses initially zero.
  loom_native_section_t* sections;
  // Number of entries in |sections|.
  iree_host_size_t section_count;
  // Arena-backed per-input contribution placement rows.
  loom_native_section_contribution_layout_t* contribution_layouts;
  // Number of entries in |contribution_layouts|.
  iree_host_size_t contribution_layout_count;
} loom_native_section_contribution_assembly_t;

// Assembles contribution byte fragments into joined native sections.
//
// Contributions with the same section name are concatenated in input order,
// honoring each contribution's alignment and zero-filling padding. Matching
// section names have matching storage and access. Reservations advance size
// without allocating or copying payload bytes. The resulting section alignment
// is the maximum contribution alignment observed for that section.
//
// All returned arrays, section names, and section payloads are allocated from
// |arena| and remain valid until the arena is reset or deinitialized. On
// failure |out_assembly| is left empty, though |arena| may contain abandoned
// transient allocations that will be reclaimed by the next arena reset.
iree_status_t loom_native_assemble_section_contributions(
    const loom_native_section_contribution_t* contributions,
    iree_host_size_t contribution_count,
    loom_native_section_contribution_assembly_t* out_assembly,
    iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_CONTRIBUTION_H_
