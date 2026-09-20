// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ELF64 relocatable serialization for format-neutral native objects.

#ifndef LOOM_TARGET_EMIT_NATIVE_ELF_OBJECT_H_
#define LOOM_TARGET_EMIT_NATIVE_ELF_OBJECT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "loom/target/emit/native/elf.h"
#include "loom/target/emit/native/object.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps a target-owned native relocation kind to its ELF R_* value.
//
// The target must define every relocation kind it contributes to |object|.
// This is an infallible compiler-owned mapping; unsupported source operations
// must fail before native object serialization.
typedef uint32_t (*loom_native_elf_relocation_map_fn_t)(
    const void* user_data, uint32_t native_relocation_kind);

typedef struct loom_native_elf_relocation_mapper_t {
  // Target-owned native-to-ELF relocation mapping function.
  loom_native_elf_relocation_map_fn_t map;
  // Opaque target state passed to |map|.
  const void* user_data;
} loom_native_elf_relocation_mapper_t;

typedef struct loom_native_elf64le_relocatable_options_t {
  // ELF EM_* machine identifier.
  uint16_t machine;
  // ELF e_ident OS ABI identifier.
  uint8_t os_abi;
  // ELF e_ident ABI version.
  uint8_t abi_version;
  // Processor-specific ELF e_flags.
  uint32_t flags;
  // Target-owned relocation-kind mapper. The function is required when the
  // object contains fixups and ignored otherwise.
  loom_native_elf_relocation_mapper_t relocation_mapper;
} loom_native_elf64le_relocatable_options_t;

// Writes |object| as an ELF64 little-endian ET_REL object to |stream|.
//
// Native section contributions are assembled by name before symbols and
// fixups are globalized. The writer emits one SHT_RELA section per relocated
// assembled section, orders local symbols before non-local symbols, and remaps
// fixup symbol indices to that final order. Undefined global and weak symbols
// remain available for resolution by the final linker.
//
// All temporary assembly, symbol, relocation, and ELF layout storage uses a
// checkpoint in |scratch_arena| and is released before this call returns. The
// output stream may retain bytes independently of the scratch arena.
iree_status_t loom_native_elf64le_write_relocatable_object(
    const loom_native_object_contribution_t* object,
    const loom_native_elf64le_relocatable_options_t* options,
    iree_io_stream_t* stream, iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_ELF_OBJECT_H_
