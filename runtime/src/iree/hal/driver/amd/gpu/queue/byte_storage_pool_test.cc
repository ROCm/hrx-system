// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/byte_storage_pool.h"

#include "iree/testing/gtest.h"

namespace {

TEST(ByteStoragePoolTest, HostCollectionOwnsReuseAndWrapPadding) {
  iree_hal_amd_gpu_byte_storage_t generation = {.capacity = 128};
  iree_hal_amd_gpu_byte_storage_pool_t storage = {.current = &generation};
  uint64_t offset = 0, end = 0;
  ASSERT_EQ(iree_hal_amd_gpu_byte_storage_pool_reserve(&storage, 80, 8, &offset,
                                                       &end),
            IREE_HAL_AMD_GPU_BYTE_RESERVED);
  EXPECT_EQ(offset, 0u);
  EXPECT_EQ(end, 80u);
  iree_hal_amd_gpu_byte_storage_pool_commit(&storage, end, false);
  EXPECT_EQ(iree_hal_amd_gpu_byte_storage_pool_reserve(&storage, 64, 64,
                                                       &offset, &end),
            IREE_HAL_AMD_GPU_BYTE_CAPACITY);
  EXPECT_EQ(generation.end, 80u);

  // Native completion alone leaves the samples held. Once host collection
  // returns its prefix, the next aligned sample can reuse the wrapped range.
  iree_hal_amd_gpu_byte_storage_pool_retire(&storage, 80);
  ASSERT_EQ(iree_hal_amd_gpu_byte_storage_pool_reserve(&storage, 64, 64,
                                                       &offset, &end),
            IREE_HAL_AMD_GPU_BYTE_RESERVED);
  EXPECT_EQ(offset, 0u);
  EXPECT_EQ(end, 192u);
  iree_hal_amd_gpu_byte_storage_pool_commit(&storage, end, false);
  EXPECT_EQ(iree_hal_amd_gpu_byte_storage_pool_reserve(&storage, 256, 64,
                                                       &offset, &end),
            IREE_HAL_AMD_GPU_BYTE_GROW);
  EXPECT_EQ(generation.end, 192u);
}

TEST(ByteStoragePoolTest, RejectionRewindsTheGenerationBorrowedBeforeGrowth) {
  iree_hal_amd_gpu_byte_storage_t original = {.end = 16, .capacity = 64};
  iree_hal_amd_gpu_byte_storage_pool_t storage = {.current = &original};
  uint64_t offset = 0, end = 0;
  ASSERT_EQ(iree_hal_amd_gpu_byte_storage_pool_reserve(&storage, 16, 8, &offset,
                                                       &end),
            IREE_HAL_AMD_GPU_BYTE_RESERVED);
  iree_hal_amd_gpu_byte_storage_pool_commit(&storage, end, true);
  EXPECT_EQ(storage.pending, &original);
  EXPECT_EQ(original.end, 32u);
  iree_hal_amd_gpu_byte_storage_t replacement = {
      .begin = 32,
      .end = 32,
      .capacity = 128,
  };
  iree_hal_amd_gpu_byte_storage_pool_install(&storage, &replacement);
  EXPECT_EQ(storage.retired, &original);
  iree_hal_amd_gpu_byte_storage_pool_rollback(&storage);
  EXPECT_EQ(original.end, 16u);
  EXPECT_EQ(replacement.end, 32u);
  EXPECT_EQ(storage.pending, nullptr);

  ASSERT_EQ(iree_hal_amd_gpu_byte_storage_pool_reserve(&storage, 64, 8, &offset,
                                                       &end),
            IREE_HAL_AMD_GPU_BYTE_RESERVED);
  iree_hal_amd_gpu_byte_storage_pool_commit(&storage, end, true);
  iree_hal_amd_gpu_byte_storage_pool_accept(&storage);
  iree_hal_amd_gpu_byte_storage_pool_rollback(&storage);
  EXPECT_EQ(replacement.end, 96u);
}

}  // namespace
