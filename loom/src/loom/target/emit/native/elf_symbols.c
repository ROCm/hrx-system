// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/elf_symbols.h"

iree_status_t loom_native_elf_hash_initialize(
    iree_host_size_t symbol_count, iree_arena_allocator_t* arena,
    loom_native_elf_hash_t* out_hash) {
  *out_hash = (loom_native_elf_hash_t){0};
  iree_host_size_t byte_length = 0;
  if (symbol_count > UINT32_MAX ||
      !iree_host_size_checked_mul(symbol_count, 8, &byte_length) ||
      !iree_host_size_checked_add(byte_length, 8, &byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF symbol hash table exceeds format capacity");
  }
  uint8_t* contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, byte_length, (void**)&contents));
  memset(contents, 0, byte_length);
  iree_unaligned_store_le_u32(contents, (uint32_t)symbol_count);
  iree_unaligned_store_le_u32(contents + 4, (uint32_t)symbol_count);
  *out_hash = (loom_native_elf_hash_t){
      .contents = iree_make_byte_span(contents, byte_length),
      .symbol_count = (uint32_t)symbol_count,
  };
  return iree_ok_status();
}

void loom_native_elf_hash_insert(loom_native_elf_hash_t* hash,
                                 iree_string_view_t name,
                                 uint32_t symbol_index) {
  uint32_t value = 0;
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    value = (value << 4) + (uint8_t)name.data[i];
    const uint32_t high = value & UINT32_C(0xf0000000);
    value ^= high >> 24;
    value &= ~high;
  }
  uint8_t* bucket = hash->contents.data + 8 +
                    (iree_host_size_t)(value % hash->symbol_count) * 4;
  uint8_t* chain = hash->contents.data + 8 +
                   ((iree_host_size_t)hash->symbol_count + symbol_index) * 4;
  iree_unaligned_store_le_u32(chain, iree_unaligned_load_le_u32(bucket));
  iree_unaligned_store_le_u32(bucket, symbol_index);
}
