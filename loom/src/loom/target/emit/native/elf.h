// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tiny ELF64 little-endian writer for native target emission.
//
// This is not a linker, object reader, or general-purpose ELF library. It owns
// only the late native-emission shape Loom needs: a fixed ELF64LE envelope,
// caller-provided section payloads, caller-provided program segments over
// contiguous section ranges, and a generated section-name string table.

#ifndef LOOM_TARGET_EMIT_NATIVE_ELF_H_
#define LOOM_TARGET_EMIT_NATIVE_ELF_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_native_elf_file_type_e {
  LOOM_NATIVE_ELF_FILE_TYPE_NONE = 0,
  LOOM_NATIVE_ELF_FILE_TYPE_REL = 1,
  LOOM_NATIVE_ELF_FILE_TYPE_EXEC = 2,
  LOOM_NATIVE_ELF_FILE_TYPE_DYN = 3,
} loom_native_elf_file_type_t;

typedef enum loom_native_elf_machine_e {
  LOOM_NATIVE_ELF_MACHINE_X86_64 = 62,
  LOOM_NATIVE_ELF_MACHINE_AARCH64 = 183,
  LOOM_NATIVE_ELF_MACHINE_AMDGPU = 224,
  LOOM_NATIVE_ELF_MACHINE_RISCV = 243,
} loom_native_elf_machine_t;

typedef enum loom_native_elf_os_abi_e {
  LOOM_NATIVE_ELF_OS_ABI_NONE = 0,
  LOOM_NATIVE_ELF_OS_ABI_LINUX = 3,
  LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA = 64,
  LOOM_NATIVE_ELF_OS_ABI_STANDALONE = 255,
} loom_native_elf_os_abi_t;

typedef enum loom_native_elf_abi_version_e {
  LOOM_NATIVE_ELF_ABI_VERSION_NONE = 0,
  LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V5 = 3,
  LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V6 = 4,
} loom_native_elf_abi_version_t;

typedef enum loom_native_elf_program_type_e {
  LOOM_NATIVE_ELF_PROGRAM_TYPE_NULL = 0,
  LOOM_NATIVE_ELF_PROGRAM_TYPE_LOAD = 1,
  LOOM_NATIVE_ELF_PROGRAM_TYPE_DYNAMIC = 2,
  LOOM_NATIVE_ELF_PROGRAM_TYPE_NOTE = 4,
  LOOM_NATIVE_ELF_PROGRAM_TYPE_PHDR = 6,
  LOOM_NATIVE_ELF_PROGRAM_TYPE_GNU_STACK = 0x6474e551,
} loom_native_elf_program_type_t;

typedef enum loom_native_elf_program_flag_bits_e {
  LOOM_NATIVE_ELF_PROGRAM_FLAG_EXECUTE = 0x1,
  LOOM_NATIVE_ELF_PROGRAM_FLAG_WRITE = 0x2,
  LOOM_NATIVE_ELF_PROGRAM_FLAG_READ = 0x4,
} loom_native_elf_program_flag_bits_t;

typedef enum loom_native_elf_section_type_e {
  LOOM_NATIVE_ELF_SECTION_TYPE_NULL = 0,
  LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS = 1,
  LOOM_NATIVE_ELF_SECTION_TYPE_SYMTAB = 2,
  LOOM_NATIVE_ELF_SECTION_TYPE_STRTAB = 3,
  LOOM_NATIVE_ELF_SECTION_TYPE_HASH = 5,
  LOOM_NATIVE_ELF_SECTION_TYPE_DYNAMIC = 6,
  LOOM_NATIVE_ELF_SECTION_TYPE_NOTE = 7,
  LOOM_NATIVE_ELF_SECTION_TYPE_DYNSYM = 11,
} loom_native_elf_section_type_t;

typedef enum loom_native_elf_section_flag_bits_e {
  LOOM_NATIVE_ELF_SECTION_FLAG_WRITE = 0x1,
  LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC = 0x2,
  LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR = 0x4,
  LOOM_NATIVE_ELF_SECTION_FLAG_STRINGS = 0x20,
} loom_native_elf_section_flag_bits_t;

typedef enum loom_native_elf_amdgpu_flag_bits_e {
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_MASK = 0x0ff,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1100 = 0x041,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1150 = 0x043,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1103 = 0x044,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1101 = 0x046,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1102 = 0x047,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1200 = 0x048,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1250 = 0x049,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1151 = 0x04a,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX942 = 0x04c,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX950 = 0x04f,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX11_GENERIC = 0x054,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1152 = 0x055,
  LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1153 = 0x058,
} loom_native_elf_amdgpu_flag_bits_t;

typedef struct loom_native_elf64le_section_t {
  // Section-table name emitted into the generated `.shstrtab`.
  iree_string_view_t name;
  // ELF SHT_* section type.
  uint32_t type;
  // ELF SHF_* section flags.
  uint64_t flags;
  // Runtime virtual address for allocated sections.
  uint64_t address;
  // File alignment in bytes. Zero is normalized to one byte.
  uint64_t alignment;
  // Fixed record size for table sections, or zero when not applicable.
  uint64_t entry_size;
  // ELF sh_link field interpreted by the section type.
  uint32_t link;
  // ELF sh_info field interpreted by the section type.
  uint32_t info;
  // Section bytes written into the file.
  iree_const_byte_span_t contents;
} loom_native_elf64le_section_t;

typedef struct loom_native_elf64le_segment_t {
  // ELF PT_* program header type.
  uint32_t type;
  // ELF PF_* program header flags.
  uint32_t flags;
  // Explicit output file byte offset for sectionless segments.
  uint64_t file_offset;
  // Explicit output file byte length for sectionless segments.
  uint64_t file_size;
  // Explicit runtime memory byte length for sectionless segments.
  uint64_t memory_size;
  // First caller-provided section covered by the file image.
  iree_host_size_t first_section;
  // Number of caller-provided sections covered by the file image.
  iree_host_size_t section_count;
  // Runtime virtual address for the segment.
  uint64_t virtual_address;
  // Runtime physical address for the segment.
  uint64_t physical_address;
  // Program-header alignment in bytes. Zero is normalized to one byte.
  uint64_t alignment;
} loom_native_elf64le_segment_t;

typedef struct loom_native_elf64le_file_t {
  // ELF ET_* file type.
  uint16_t type;
  // ELF EM_* machine identifier.
  uint16_t machine;
  // ELF e_ident OS ABI identifier.
  uint8_t os_abi;
  // ELF e_ident ABI version.
  uint8_t abi_version;
  // Processor-specific ELF e_flags.
  uint32_t flags;
  // Entry point virtual address, or zero when the object has no entry.
  uint64_t entry;
  // Caller-provided sections. The writer prepends SHN_UNDEF and appends
  // `.shstrtab`; segment section indices refer only to this caller array.
  const loom_native_elf64le_section_t* sections;
  // Number of caller-provided sections.
  iree_host_size_t section_count;
  // Caller-provided program segments over contiguous ranges of |sections|.
  const loom_native_elf64le_segment_t* segments;
  // Number of caller-provided program segments.
  iree_host_size_t segment_count;
} loom_native_elf64le_file_t;

// Writes |file| as a complete ELF64 little-endian object to |stream|.
//
// The writer emits bytes sequentially. It computes all layout metadata before
// writing and does not patch or seek backward, making it suitable for ordinary
// output streams. Temporary layout and `.shstrtab` storage use
// |scratch_arena|, which must remain live until this call returns. On failure
// the arena may contain abandoned transient allocations that will be reclaimed
// by the next arena reset.
iree_status_t loom_native_elf64le_write_file(
    const loom_native_elf64le_file_t* file, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_ELF_H_
