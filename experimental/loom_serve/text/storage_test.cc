// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/storage.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/environment.h"

namespace {

class TextStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator, &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
    const iree_host_size_t lengths[] = {11 * 24, 3 * 5 * 16, 16 * 16, 16 * 16,
                                        32 + 2 * 40};
    for (size_t i = 0; i < IREE_ARRAYSIZE(lengths); ++i) {
      iree_vm_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_vm_buffer_create(lengths[i], 8, allocator, &buffer));
      IREE_ASSERT_OK(
          iree_vm_buffer_map_write(buffer, 0, lengths[i], &bytes[i]));
      results[i] = iree_vm_buffer_variant_from_ptr_move(&types, &buffer);
    }
    for (uint64_t slot : {1u, 9u}) {
      iree_unaligned_store_le_u64(bytes[0].data + slot * 24, 1 << 20);
      iree_unaligned_store_le_u64(bytes[0].data + slot * 24 + 8, 256);
      iree_unaligned_store_le_u64(bytes[0].data + slot * 24 + 16, 256);
    }
    const uint64_t geometry[] = {
        4, 256, 128, 256, 1, 256, 2, 4096, 128, 9, 512, 1, 8192, 64,
    };
    for (size_t i = 0; i < IREE_ARRAYSIZE(geometry); ++i) {
      iree_unaligned_store_le_u64(bytes[4].data + i * 8, geometry[i]);
    }
  }

  void TearDown() override {
    loom_serve_text_storage_deinitialize(&storage);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    iree_vm_environment_free(environment);
  }

  // Allocator owning source values and parsed metadata.
  const iree_allocator_t allocator = iree_allocator_system();
  // Source buffer reference-type owner.
  iree_vm_environment_t* environment = nullptr;
  // Canonical VM buffer reference handles.
  iree_vm_ref_types_t types = {};
  // Bootstrap output values transferred to the storage parser.
  iree_vm_variant_t results[5] = {};
  // Mutable source-owned bytes used to author the external record fixture.
  iree_byte_span_t bytes[5] = {};
  // Parsed geometry survives release of initialization payloads.
  loom_serve_text_storage_t storage = {};
};

TEST_F(TextStorageTest, LogicalOrderAndCarrySurviveBootstrapPayloadRelease) {
  IREE_ASSERT_OK(loom_serve_text_storage_initialize(&types, results, 3, 16, 17,
                                                    32, &storage, allocator));
  EXPECT_EQ(storage.blocks_per_row, 5u);
  EXPECT_EQ(storage.block_size, 4u);
  EXPECT_EQ(storage.region_count, 2u);
  loom_serve_text_storage_release_payloads(&storage);
  iree_hal_buffer_t* private_views[5] = {};
  const uint32_t blocks[] = {7, 2, 5};
  iree_host_size_t count = 0;
  loom_serve_snapshot_range_t* ranges = nullptr;
  IREE_ASSERT_OK(loom_serve_text_storage_plan_snapshot(
      &storage, 2, private_views, true, 3, blocks, &count, &ranges, allocator));
  ASSERT_EQ(count, 10u);
  EXPECT_EQ(ranges[0].buffer_index, 6u);
  EXPECT_EQ(ranges[0].offset, 256u);
  EXPECT_EQ(ranges[0].length, 128u);
  for (size_t plane = 0; plane < 2; ++plane) {
    for (size_t block = 0; block < 3; ++block) {
      const auto& range = ranges[1 + plane * 3 + block];
      EXPECT_EQ(range.buffer_index, 1u);
      EXPECT_EQ(range.offset, 256 + plane * 4096 + blocks[block] * 128);
      EXPECT_EQ(range.length, 128u);
    }
  }
  for (size_t block = 0; block < 3; ++block) {
    const auto& range = ranges[7 + block];
    EXPECT_EQ(range.buffer_index, 9u);
    EXPECT_EQ(range.offset, 512 + blocks[block] * 64);
    EXPECT_EQ(range.length, 64u);
  }
  iree_allocator_free(allocator, ranges);
}

TEST_F(TextStorageTest, RejectsUnrepresentablePoolGeometry) {
  iree_unaligned_store_le_u64(bytes[4].data, 3);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 16, 17, 32,
                                         &storage, allocator));
}

TEST_F(TextStorageTest, RejectsPlanesOverlappingPrivateStorage) {
  iree_unaligned_store_le_u64(bytes[4].data + 40, 128);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 16, 17, 32,
                                         &storage, allocator));
}

}  // namespace
