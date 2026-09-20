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

typedef enum loom_native_section_kind_e {
  LOOM_NATIVE_SECTION_KIND_NONE = 0,
  // Section stores explicit bytes in the artifact.
  LOOM_NATIVE_SECTION_KIND_BYTES = 1,
  // Section reserves zero-filled memory without artifact payload bytes.
  LOOM_NATIVE_SECTION_KIND_ZERO_FILL = 2,
} loom_native_section_kind_t;

typedef enum loom_native_section_flag_bits_e {
  // Section participates in the loaded memory image.
  LOOM_NATIVE_SECTION_FLAG_ALLOCATED = 1u << 0,
  // Loaded section contents may be modified.
  LOOM_NATIVE_SECTION_FLAG_WRITABLE = 1u << 1,
  // Loaded section contents may be executed.
  LOOM_NATIVE_SECTION_FLAG_EXECUTABLE = 1u << 2,
} loom_native_section_flag_bits_t;
typedef uint32_t loom_native_section_flags_t;

// One worker-produced byte contribution to a named native section.
typedef struct loom_native_section_contribution_t {
  // Logical section name such as `.text`, `.rodata`, or `.bss`.
  iree_string_view_t section_name;
  // Payload storage kind.
  uint32_t kind;
  // Load, access, and execution flags.
  loom_native_section_flags_t flags;
  // Alignment required before this contribution within the joined section.
  uint64_t contribution_alignment;
  // Contribution bytes. The referenced storage must stay live for the duration
  // of assembly; the assembled output copies bytes into the caller's arena.
  iree_const_byte_span_t contents;
  // Logical zero-filled byte length. This must be zero for byte contributions.
  uint64_t zero_fill_length;
} loom_native_section_contribution_t;

// One assembled native section independent of its artifact container.
typedef struct loom_native_section_t {
  // Logical section name copied into assembly-owned storage.
  iree_string_view_t name;
  // Payload storage kind.
  uint32_t kind;
  // Load, access, and execution flags.
  loom_native_section_flags_t flags;
  // Runtime address assigned by a target linker, or zero while unplaced.
  uint64_t address;
  // Required byte alignment. Zero is normalized to one byte during assembly.
  uint64_t alignment;
  // Assembled section bytes for LOOM_NATIVE_SECTION_KIND_BYTES.
  iree_const_byte_span_t contents;
  // Logical zero-filled byte length for LOOM_NATIVE_SECTION_KIND_ZERO_FILL.
  uint64_t zero_fill_length;
} loom_native_section_t;

// Returns the logical byte length of |section|.
static inline uint64_t loom_native_section_byte_length(
    const loom_native_section_t* section) {
  return section->kind == LOOM_NATIVE_SECTION_KIND_ZERO_FILL
             ? section->zero_fill_length
             : (uint64_t)section->contents.data_length;
}

// Resolved position of one input contribution inside the assembled sections.
typedef struct loom_native_section_contribution_layout_t {
  // Index into loom_native_section_contribution_assembly_t.sections.
  iree_host_size_t section_index;
  // Byte offset within the assembled section.
  uint64_t section_offset;
} loom_native_section_contribution_layout_t;

// Output of assembling contributions into format-neutral section payloads.
typedef struct loom_native_section_contribution_assembly_t {
  // Arena-backed format-neutral section descriptors.
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
// section names must have matching kind and flags. Zero-fill contributions
// advance logical section size without allocating or copying payload bytes.
// The resulting section alignment is the maximum contribution alignment
// observed for that section.
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
