// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/elf_symbols.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(NativeElfSymbolsTest, SerializedHashRetainsCollisionsAndNullSymbol) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  loom_native_elf_hash_t hash;
  IREE_ASSERT_OK(loom_native_elf_hash_initialize(5, &arena, &hash));
  loom_native_elf_hash_insert(&hash, IREE_SV("A"), 1);
  loom_native_elf_hash_insert(&hash, IREE_SV("F"), 2);
  loom_native_elf_hash_insert(&hash, IREE_SV("K"), 3);
  // The SysV hash of "AAAAAAA" is 0x05555511. Unlike the single-character
  // collisions, this exercises folding high bits back into the hash.
  loom_native_elf_hash_insert(&hash, IREE_SV("AAAAAAA"), 4);
  const uint32_t expected[] = {
      5, 5,           // Bucket count and symbol count, including STN_UNDEF.
      3, 0, 4, 0, 0,  // Buckets: K -> F -> A; AAAAAAA occupies its own bucket.
      0, 0, 1, 2, 0,  // Chains indexed by the producer's symbol indices.
  };
  EXPECT_EQ(hash.contents.data_length, sizeof(expected));
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    EXPECT_EQ(iree_unaligned_load_le_u32(hash.contents.data + i * 4),
              expected[i])
        << "word " << i;
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

}  // namespace
}  // namespace loom
