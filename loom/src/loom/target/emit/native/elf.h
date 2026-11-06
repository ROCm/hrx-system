// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tiny little-endian ELF writer for native target emission.
//
// This is not a linker, object reader, or general-purpose ELF library. It owns
// only the late native-emission shape Loom needs: fixed ELF32LE and ELF64LE
// envelopes, caller-provided section payloads, caller-provided program segments
// over contiguous section ranges, and a generated section-name string table.

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
  LOOM_NATIVE_ELF_MACHINE_AIE = 264,
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
  LOOM_NATIVE_ELF_SECTION_TYPE_RELA = 4,
  LOOM_NATIVE_ELF_SECTION_TYPE_HASH = 5,
  LOOM_NATIVE_ELF_SECTION_TYPE_DYNAMIC = 6,
  LOOM_NATIVE_ELF_SECTION_TYPE_NOTE = 7,
  LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS = 8,
  LOOM_NATIVE_ELF_SECTION_TYPE_DYNSYM = 11,
} loom_native_elf_section_type_t;

typedef enum loom_native_elf_section_flag_bits_e {
  LOOM_NATIVE_ELF_SECTION_FLAG_WRITE = 0x1,
  LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC = 0x2,
  LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR = 0x4,
  LOOM_NATIVE_ELF_SECTION_FLAG_STRINGS = 0x20,
} loom_native_elf_section_flag_bits_t;

typedef enum loom_native_elf_dynamic_tag_e {
  LOOM_NATIVE_ELF_DYNAMIC_NULL = 0,
  LOOM_NATIVE_ELF_DYNAMIC_HASH = 4,
  LOOM_NATIVE_ELF_DYNAMIC_STRTAB = 5,
  LOOM_NATIVE_ELF_DYNAMIC_SYMTAB = 6,
  LOOM_NATIVE_ELF_DYNAMIC_RELA = 7,
  LOOM_NATIVE_ELF_DYNAMIC_RELASZ = 8,
  LOOM_NATIVE_ELF_DYNAMIC_RELAENT = 9,
  LOOM_NATIVE_ELF_DYNAMIC_STRSZ = 10,
  LOOM_NATIVE_ELF_DYNAMIC_SYMENT = 11,
} loom_native_elf_dynamic_tag_t;

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

typedef struct loom_native_elf_section_t {
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
  // Address-space extent for SHT_NOBITS sections; zero for content-backed
  // sections. The target's loading contract determines initialization.
  uint64_t zero_fill_length;
} loom_native_elf_section_t;

// Returns the logical size recorded in the section header.
static inline uint64_t loom_native_elf_section_byte_length(
    const loom_native_elf_section_t* section) {
  return section->type == LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS
             ? section->zero_fill_length
             : (uint64_t)section->contents.data_length;
}

typedef struct loom_native_elf_segment_t {
  // ELF PT_* program header type.
  uint32_t type;
  // ELF PF_* program header flags.
  uint32_t flags;
  // Explicit output file byte offset for sectionless segments.
  uint64_t file_offset;
  // Explicit output file byte length for sectionless segments.
  uint64_t file_size;
  // Runtime memory byte length. Section-backed segments normalize zero to the
  // file range and may specify a larger value for a zero-filled tail.
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
} loom_native_elf_segment_t;

typedef struct loom_native_elf32le_file_t {
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
  uint32_t entry;
  // Caller-provided sections. The writer prepends SHN_UNDEF and appends
  // `.shstrtab`; segment section indices refer only to this caller array.
  const loom_native_elf_section_t* sections;
  // Number of caller-provided sections.
  iree_host_size_t section_count;
  // Caller-provided program segments over contiguous ranges of |sections|.
  const loom_native_elf_segment_t* segments;
  // Number of caller-provided program segments.
  iree_host_size_t segment_count;
} loom_native_elf32le_file_t;

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
  const loom_native_elf_section_t* sections;
  // Number of caller-provided sections.
  iree_host_size_t section_count;
  // Caller-provided program segments over contiguous ranges of |sections|.
  const loom_native_elf_segment_t* segments;
  // Number of caller-provided program segments.
  iree_host_size_t segment_count;
} loom_native_elf64le_file_t;

typedef struct loom_native_elf_section_layout_t {
  // Section-name offset within the generated `.shstrtab`.
  uint32_t name_offset;
  // Byte offset of the section contents in the output file.
  uint64_t file_offset;
  // Byte length of the section contents in the output file; zero for NOBITS.
  uint64_t file_size;
  // Logical section byte length recorded in sh_size.
  uint64_t section_size;
  // Normalized power-of-two section alignment in bytes.
  uint64_t alignment;
} loom_native_elf_section_layout_t;

// Canonical file placement retained between payload preparation and writing.
typedef struct loom_native_elf_layout_t {
  // Arena-owned rows: null section, caller sections, and generated `.shstrtab`.
  // Caller section i has ELF index i + 1 and layout sections[i + 1].
  const loom_native_elf_section_layout_t* sections;
  // Number of entries in |sections|, including both generated sections.
  iree_host_size_t section_count;
  // Arena-owned section-name string table bytes.
  iree_string_view_t string_table;
  // Byte offset of the ELF program-header table, or zero when absent.
  uint64_t program_header_offset;
  // Byte length of the ELF program-header table.
  uint64_t program_header_size;
  // Byte offset of the ELF section-header table.
  uint64_t section_header_offset;
  // Complete serialized file byte size.
  uint64_t file_size;
} loom_native_elf_layout_t;

// Places the sections and tables of |file| without writing any bytes.
//
// Section order, names, alignments, storage kinds, and lengths, and the program
// header count are fixed by this call. The producer may then finalize
// addresses, segment ranges, entry points, and payload bytes against these
// offsets without changing that shape. The returned storage belongs to |arena|
// and remains live through writing. Failed construction leaves allocations for
// the arena reset.
iree_status_t loom_native_elf32le_build_layout(
    const loom_native_elf32le_file_t* file, iree_arena_allocator_t* arena,
    loom_native_elf_layout_t* out_layout);

// ELF64 variant of loom_native_elf32le_build_layout with the same shape and
// arena ownership contract.
iree_status_t loom_native_elf64le_build_layout(
    const loom_native_elf64le_file_t* file, iree_arena_allocator_t* arena,
    loom_native_elf_layout_t* out_layout);

// Writes a complete ELF file using its retained |layout|. The file must satisfy
// the build_layout shape contract. Writing allocates no layout storage, emits
// bytes sequentially from stream offset zero, and never seeks backward. ELF32
// field limits, including finalized addresses, are checked before writing.
iree_status_t loom_native_elf32le_write_file(
    const loom_native_elf32le_file_t* file,
    const loom_native_elf_layout_t* layout, iree_io_stream_t* stream);

// ELF64 variant of loom_native_elf32le_write_file using retained placement.
iree_status_t loom_native_elf64le_write_file(
    const loom_native_elf64le_file_t* file,
    const loom_native_elf_layout_t* layout, iree_io_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_ELF_H_
