// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ELF symbol lookup table construction over producer-owned symbol indices.

#ifndef LOOM_TARGET_EMIT_NATIVE_ELF_SYMBOLS_H_
#define LOOM_TARGET_EMIT_NATIVE_ELF_SYMBOLS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_native_elf_hash_t {
  // Arena-owned little-endian DT_HASH payload, updated directly by insertion.
  iree_byte_span_t contents;
  // Number of symbols including STN_UNDEF; also the number of hash buckets.
  uint32_t symbol_count;
} loom_native_elf_hash_t;

// Allocates an empty SysV lookup table for |symbol_count| symbols, including
// the required null symbol at index zero. Symbol indices remain those chosen by
// the producer. The serialized bucket and chain arrays are the construction
// state; no temporary host-endian copies or later finalization are needed.
iree_status_t loom_native_elf_hash_initialize(iree_host_size_t symbol_count,
                                              iree_arena_allocator_t* arena,
                                              loom_native_elf_hash_t* out_hash);

// Inserts one named symbol at its nonzero producer-owned table index. Each
// index is inserted once, after initialization and before the payload is read.
// Hash collisions form index chains within the serialized table.
void loom_native_elf_hash_insert(loom_native_elf_hash_t* hash,
                                 iree_string_view_t name,
                                 uint32_t symbol_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_ELF_SYMBOLS_H_
