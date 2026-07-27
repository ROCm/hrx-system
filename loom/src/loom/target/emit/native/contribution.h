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
#include "loom/target/emit/native/elf.h"

#ifdef __cplusplus
extern "C" {
#endif

// One worker-produced byte contribution to a named native section.
typedef struct loom_native_section_contribution_t {
  // Section-table name such as `.text`, `.rodata`, or `.note`.
  iree_string_view_t section_name;
  // ELF SHT_* section type.
  uint32_t section_type;
  // ELF SHF_* section flags.
  uint64_t section_flags;
  // Alignment required before this contribution within the joined section.
  uint64_t contribution_alignment;
  // Fixed record size for table sections, or zero when not applicable.
  uint64_t entry_size;
  // ELF sh_link field interpreted by the section type.
  uint32_t link;
  // ELF sh_info field interpreted by the section type.
  uint32_t info;
  // Contribution bytes. The referenced storage must stay live for the duration
  // of assembly; the assembled output copies bytes into the caller's arena.
  iree_const_byte_span_t contents;
} loom_native_section_contribution_t;

// Resolved position of one input contribution inside the assembled sections.
typedef struct loom_native_section_contribution_layout_t {
  // Index into loom_native_section_contribution_assembly_t.sections.
  iree_host_size_t section_index;
  // Byte offset within the assembled section.
  uint64_t section_offset;
} loom_native_section_contribution_layout_t;

// Output of assembling section contributions into ELF section payloads.
typedef struct loom_native_section_contribution_assembly_t {
  // Arena-backed ELF section descriptors suitable for
  // loom_native_elf64le_file_t.
  loom_native_elf64le_section_t* sections;
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
// section names must have matching type, flags, entry size, link, and info
// fields. The resulting section alignment is the maximum contribution alignment
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
