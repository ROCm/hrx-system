// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/storage.h"

#include "iree/hal/memory/heap_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/environment.h"

namespace {

class TextStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_environment_allocate(host_allocator, &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
    // Origin payloads have model-private layouts, including non-word extents.
    const iree_host_size_t lengths[] = {11 * 24, 3 * 4 * 16, 73, 129,
                                        64 + 2 * 40};
    for (size_t i = 0; i < IREE_ARRAYSIZE(lengths); ++i) {
      iree_vm_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(
          iree_vm_buffer_create(lengths[i], 8, host_allocator, &buffer));
      IREE_ASSERT_OK(
          iree_vm_buffer_map_write(buffer, 0, lengths[i], &bytes[i]));
      results[i] = iree_vm_buffer_variant_from_ptr_move(&types, &buffer);
    }
    for (uint64_t slot : {1u, 9u}) {
      iree_unaligned_store_le_u64(bytes[0].data + slot * 24, 1 << 20);
      iree_unaligned_store_le_u64(bytes[0].data + slot * 24 + 8, 256);
      iree_unaligned_store_le_u64(bytes[0].data + slot * 24 + 16, 256);
    }
    for (uint64_t slot : {3u, 10u}) {
      iree_unaligned_store_le_u64(bytes[0].data + slot * 24, 512);
      iree_unaligned_store_le_u64(bytes[0].data + slot * 24 + 8, 256);
    }
    for (size_t row = 0; row < 3; ++row) {
      const uint64_t views[] = {row * 64,      8, 0, 0, row * 64 + 40, 8,
                                row * 64 + 48, 8};
      for (size_t i = 0; i < IREE_ARRAYSIZE(views); ++i) {
        iree_unaligned_store_le_u64(bytes[1].data + row * 64 + i * 8, views[i]);
      }
    }
    const uint64_t geometry[] = {
        4,   256, 128,  256, 8, 64,  32, 3,    1,
        256, 2,   4096, 128, 9, 512, 1,  8192, 64,
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

  void SetPoolGeometry(uint64_t block_size, uint64_t page_count) {
    iree_unaligned_store_le_u64(bytes[4].data, block_size);
    iree_unaligned_store_le_u64(bytes[4].data + 11 * 8, page_count * 128);
    iree_unaligned_store_le_u64(bytes[4].data + 16 * 8, page_count * 64);
    iree_unaligned_store_le_u64(bytes[0].data + 1 * 24,
                                256 + 2 * page_count * 128);
    iree_unaligned_store_le_u64(bytes[0].data + 9 * 24, 512 + page_count * 64);
  }

  void MovePrivateStorage(uint64_t origin) {
    iree_unaligned_store_le_u64(bytes[0].data + 24 + 16, 0);
    iree_unaligned_store_le_u64(bytes[4].data + 32, origin + 8);
    for (size_t i = 0; i < 3 * 4; ++i) {
      uint8_t* view = bytes[1].data + i * 16;
      if (iree_unaligned_load_le_u64(view + 8)) {
        iree_unaligned_store_le_u64(view,
                                    origin + iree_unaligned_load_le_u64(view));
      }
    }
  }

  // Allocator owning source values and parsed metadata.
  const iree_allocator_t host_allocator = iree_allocator_system();
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
  IREE_ASSERT_OK(loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                                    host_allocator, &storage));
  EXPECT_EQ(storage.blocks_per_row, 5u);
  EXPECT_EQ(storage.block_size, 4u);
  EXPECT_EQ(storage.region_count, 2u);
  EXPECT_EQ(storage.recurrent.origin, 8u);
  EXPECT_EQ(storage.recurrent.stride, 64u);
  EXPECT_EQ(storage.recurrent.length, 32u);
  EXPECT_EQ(storage.recurrent.capacity, 3u);
  EXPECT_EQ(storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS].data_length,
            73u);
  EXPECT_EQ(storage.bytes[LOOM_SERVE_TEXT_STORAGE_DRAFT_ORIGINS].data_length,
            129u);
  loom_serve_text_storage_release_payloads(&storage);
  iree_hal_buffer_t* private_views[5] = {};
  const uint32_t blocks[] = {7, 2, 5};
  iree_host_size_t count = 0;
  loom_serve_snapshot_range_t* ranges = nullptr;
  IREE_ASSERT_OK(loom_serve_text_storage_plan_snapshot(
      &storage, 2, private_views, true, 3, blocks, host_allocator, &count,
      &ranges));
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
  iree_allocator_free(host_allocator, ranges);
}

TEST_F(TextStorageTest, AdjacentPagesShareRangesWithoutReorderingLogicalBytes) {
  IREE_ASSERT_OK(loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                                    host_allocator, &storage));
  const uint32_t blocks[] = {2, 3, 7, 0, 1};
  iree_hal_buffer_t* views[5] = {};
  iree_host_size_t count = 0;
  loom_serve_snapshot_range_t* ranges = nullptr;
  IREE_ASSERT_OK(loom_serve_text_storage_plan_snapshot(
      &storage, 2, views, true, IREE_ARRAYSIZE(blocks), blocks, host_allocator,
      &count, &ranges));
  const loom_serve_snapshot_range_t expected[] = {
      {6, 256, 128},  {1, 512, 256},  {1, 1152, 128}, {1, 256, 256},
      {1, 4608, 256}, {1, 5248, 128}, {1, 4352, 256}, {9, 640, 128},
      {9, 960, 64},   {9, 512, 128},
  };
  ASSERT_EQ(count, IREE_ARRAYSIZE(expected));
  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(ranges[i].buffer_index, expected[i].buffer_index);
    EXPECT_EQ(ranges[i].offset, expected[i].offset);
    EXPECT_EQ(ranges[i].length, expected[i].length);
  }
  iree_allocator_free(host_allocator, ranges);
}

TEST_F(TextStorageTest, RejectsUnrepresentablePoolGeometry) {
  iree_unaligned_store_le_u64(bytes[4].data, 3);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, FullPageIdSpacePreservesWideByteOffsets) {
  SetPoolGeometry(1, UINT32_MAX);
  IREE_ASSERT_OK(loom_serve_text_storage_initialize(
      &types, results, 3, 17, UINT32_MAX, host_allocator, &storage));
  EXPECT_EQ(storage.blocks_per_row, 17u);
  EXPECT_EQ(storage.regions[0].blocks.stride, uint64_t{UINT32_MAX} * 128);
  const uint32_t blocks[] = {UINT32_MAX - 1, 0, UINT32_MAX / 2 + 1};
  iree_hal_buffer_t* views[5] = {};
  iree_host_size_t count = 0;
  loom_serve_snapshot_range_t* ranges = nullptr;
  IREE_ASSERT_OK(loom_serve_text_storage_plan_snapshot(
      &storage, 0, views, false, IREE_ARRAYSIZE(blocks), blocks, host_allocator,
      &count, &ranges));
  ASSERT_EQ(count, 9u);
  for (size_t plane = 0; plane < 2; ++plane) {
    for (size_t block = 0; block < IREE_ARRAYSIZE(blocks); ++block) {
      EXPECT_EQ(ranges[plane * 3 + block].offset,
                256 + plane * uint64_t{UINT32_MAX} * 128 +
                    uint64_t{blocks[block]} * 128);
      EXPECT_EQ(ranges[plane * 3 + block].length, 128u);
    }
  }
  for (size_t block = 0; block < IREE_ARRAYSIZE(blocks); ++block) {
    EXPECT_EQ(ranges[6 + block].offset, 512 + uint64_t{blocks[block]} * 64);
    EXPECT_EQ(ranges[6 + block].length, 64u);
  }
  iree_allocator_free(host_allocator, ranges);
}

TEST_F(TextStorageTest, AdjacentWidePageIdsKeepWideRunOffsetsAndLengths) {
  SetPoolGeometry(1, UINT32_MAX);
  IREE_ASSERT_OK(loom_serve_text_storage_initialize(
      &types, results, 3, 17, UINT32_MAX, host_allocator, &storage));
  const uint32_t blocks[] = {UINT32_MAX - 2, UINT32_MAX - 1, 0};
  iree_hal_buffer_t* views[5] = {};
  iree_host_size_t count = 0;
  loom_serve_snapshot_range_t* ranges = nullptr;
  IREE_ASSERT_OK(loom_serve_text_storage_plan_snapshot(
      &storage, 0, views, false, IREE_ARRAYSIZE(blocks), blocks, host_allocator,
      &count, &ranges));
  ASSERT_EQ(count, 6u);
  for (size_t plane = 0; plane < 2; ++plane) {
    const uint64_t origin = 256 + plane * uint64_t{UINT32_MAX} * 128;
    EXPECT_EQ(ranges[plane * 2].offset,
              origin + uint64_t{UINT32_MAX - 2} * 128);
    EXPECT_EQ(ranges[plane * 2].length, 256u);
    EXPECT_EQ(ranges[plane * 2 + 1].offset, origin);
    EXPECT_EQ(ranges[plane * 2 + 1].length, 128u);
  }
  EXPECT_EQ(ranges[4].offset, 512 + uint64_t{UINT32_MAX - 2} * 64);
  EXPECT_EQ(ranges[4].length, 128u);
  EXPECT_EQ(ranges[5].offset, 512u);
  EXPECT_EQ(ranges[5].length, 64u);
  iree_allocator_free(host_allocator, ranges);
}

TEST_F(TextStorageTest, WidePagesHaveIndependentLogicalCapacity) {
  SetPoolGeometry(UINT32_MAX, 1);
  IREE_ASSERT_OK(loom_serve_text_storage_initialize(
      &types, results, 3, 17, UINT32_MAX, host_allocator, &storage));
  EXPECT_EQ(storage.block_size, UINT32_MAX);
  EXPECT_EQ(storage.blocks_per_row, 1u);
}

TEST_F(TextStorageTest, RejectsPoolBeyondPhysicalPageIdSpace) {
  if (sizeof(iree_host_size_t) < sizeof(uint64_t)) {
    GTEST_SKIP() << "host token counts cannot exceed the page-ID space";
  }
  const uint64_t page_count = uint64_t{UINT32_MAX} + 1;
  SetPoolGeometry(4, page_count);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(
          &types, results, 3, 17, static_cast<iree_host_size_t>(page_count * 4),
          host_allocator, &storage));
  EXPECT_EQ(storage.regions, nullptr);
}

TEST_F(TextStorageTest, DraftCacheWithoutCarryHasNoSnapshotCarryRange) {
  iree_unaligned_store_le_u64(bytes[4].data + 16, 0);
  IREE_ASSERT_OK(loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                                    host_allocator, &storage));
  loom_serve_text_storage_release_payloads(&storage);
  iree_hal_buffer_t* views[5] = {};
  const uint32_t blocks[] = {7, 2, 5};
  iree_host_size_t count = 0;
  loom_serve_snapshot_range_t* ranges = nullptr;
  IREE_ASSERT_OK(loom_serve_text_storage_plan_snapshot(
      &storage, 2, views, true, 3, blocks, host_allocator, &count, &ranges));
  ASSERT_EQ(count, 9u);
  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(ranges[i].buffer_index, i < 6 ? 1u : 9u);
    EXPECT_EQ(ranges[i].offset, i < 6
                                    ? 256 + (i / 3) * 4096 + blocks[i % 3] * 128
                                    : 512 + blocks[i % 3] * 64);
  }
  iree_allocator_free(host_allocator, ranges);
}

TEST_F(TextStorageTest, IndependentEndpointOmitsRowPrivateExecutionState) {
  // Carry has its own endpoint bank beyond the three active row entries.
  iree_unaligned_store_le_u64(bytes[0].data + 6 * 24, 5 * 128);
  iree_unaligned_store_le_u64(bytes[0].data + 6 * 24 + 8, 256);
  IREE_ASSERT_OK(loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                                    host_allocator, &storage));
  iree_hal_pool_t* heap = nullptr;
  const iree_hal_pool_host_access_t host_access = {
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .modes = IREE_HAL_MAPPING_MODE_SCOPED,
  };
  IREE_ASSERT_OK(iree_hal_heap_pool_create(host_access, host_allocator,
                                           host_allocator, &heap));
  iree_hal_buffer_t* arena = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      heap, iree_hal_make_pool_reservation_request(1 << 20, 0),
      iree_infinite_timeout(), &arena));
  iree_hal_buffer_t* views[5] = {};
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(arena, 72, 32, host_allocator, &views[1]));
  const uint32_t blocks[] = {7, 2};
  iree_host_size_t count = 0;
  loom_serve_snapshot_range_t* ranges = nullptr;
  IREE_ASSERT_OK(loom_serve_text_storage_plan_snapshot(
      &storage, 4, views, true, 2, blocks, host_allocator, &count, &ranges));
  ASSERT_EQ(count, 8u);
  EXPECT_EQ(ranges[0].buffer_index, 1u);
  EXPECT_EQ(ranges[0].offset, 72u);
  EXPECT_EQ(ranges[0].length, 32u);
  EXPECT_EQ(ranges[1].buffer_index, 6u);
  EXPECT_EQ(ranges[1].offset, 512u);
  EXPECT_EQ(ranges[1].length, 128u);
  EXPECT_EQ(ranges[2].offset, 256u + 7 * 128);
  EXPECT_EQ(ranges[3].offset, 256u + 2 * 128);
  iree_allocator_free(host_allocator, ranges);
  iree_hal_buffer_release(views[1]);
  iree_hal_buffer_release(arena);
  iree_hal_pool_release(heap);
}

TEST_F(TextStorageTest, RejectsPlanesOverlappingPrivateStorage) {
  iree_unaligned_store_le_u64(bytes[4].data + 72, 128);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, CacheCanPrecedeDemandInitializedPrivateStorage) {
  MovePrivateStorage(8192);
  IREE_ASSERT_OK(loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                                    host_allocator, &storage));
  EXPECT_EQ(storage.recurrent.origin, 8200u);
  EXPECT_EQ(storage.regions[0].blocks.origin, 256u);
}

TEST_F(TextStorageTest, UninitializedPrivateStorageCannotOverlapCache) {
  MovePrivateStorage(4608);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, RejectsOverlappingRecurrentSlots) {
  iree_unaligned_store_le_u64(bytes[4].data + 40, 31);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, RejectsRecurrentSlotsBeyondPrivateStorage) {
  iree_unaligned_store_le_u64(bytes[4].data + 56, 5);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, RejectsMissingRecurrentHeadroom) {
  iree_unaligned_store_le_u64(bytes[4].data + 56, 2);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, RejectsPrivateViewInsideRecurrentSlot) {
  iree_unaligned_store_le_u64(bytes[1].data, 8);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, RejectsPrivateViewCrossingNextRecurrentSlot) {
  // The view starts in a valid gap but crosses the next slot at byte 72.
  iree_unaligned_store_le_u64(bytes[1].data, 40);
  iree_unaligned_store_le_u64(bytes[1].data + 8, 33);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, RejectsOriginPayloadOverlappingPageMaps) {
  // The target payload still fits; the larger draft payload crosses the map.
  iree_unaligned_store_le_u64(bytes[4].data + 8, 128);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, RejectsOriginPayloadBeyondItsAllocation) {
  iree_unaligned_store_le_u64(bytes[0].data + 3 * 24, 72);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, RejectsTruncatedPageMapAllocation) {
  // Three rows with five pages each need 60 map bytes after byte 256.
  iree_unaligned_store_le_u64(bytes[0].data + 10 * 24, 315);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, RejectsNullOriginPayload) {
  iree_vm_variant_reset(&results[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS]);
  results[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS] =
      iree_vm_buffer_variant_from_ptr_borrowed(&types, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                         host_allocator, &storage));
}

TEST_F(TextStorageTest, AcceptsEmptyOriginPayload) {
  iree_vm_variant_reset(&results[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS]);
  iree_vm_buffer_t* empty = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(0, 8, host_allocator, &empty));
  results[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS] =
      iree_vm_buffer_variant_from_ptr_move(&types, &empty);
  IREE_ASSERT_OK(loom_serve_text_storage_initialize(&types, results, 3, 17, 32,
                                                    host_allocator, &storage));
  EXPECT_EQ(storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS].data_length,
            0u);
}

}  // namespace
