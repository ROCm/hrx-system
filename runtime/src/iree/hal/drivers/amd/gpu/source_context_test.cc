// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/source_context.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

std::array<uint8_t, 96> SiteTable() {
  std::array<uint8_t, 96> data = {};
  iree_unaligned_store_le_u32(data.data(), 0x5449534C);
  data[4] = 1;
  data[5] = 32;
  iree_unaligned_store_le_u16(data.data() + 6, 48);
  iree_unaligned_store_le_u32(data.data() + 8, 1);
  iree_unaligned_store_le_u32(data.data() + 12, 80);
  iree_unaligned_store_le_u32(data.data() + 16, 12);
  iree_unaligned_store_le_u32(data.data() + 20, 92);
  iree_unaligned_store_le_u32(data.data() + 24, 4);
  uint8_t* row = data.data() + 32;
  iree_unaligned_store_le_u32(row + 4, 0x1029);
  iree_unaligned_store_le_u32(row + 8, 3);
  iree_unaligned_store_le_u32(row + 16, 4);
  iree_unaligned_store_le_u32(row + 24, 11);
  iree_unaligned_store_le_u32(row + 28, 7);
  iree_unaligned_store_le_u32(row + 32, 3);
  iree_unaligned_store_le_u32(row + 36, 7);
  iree_unaligned_store_le_u32(row + 40, 19);
  iree_unaligned_store_le_u16(row + 44, 1);
  memcpy(data.data() + 80, "kernel.loom", 12);
  memcpy(data.data() + 92, "test", 4);
  return data;
}

class SourceContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_amd_gpu_source_context_registry_initialize(iree_allocator_system(),
                                                        &registry_);
  }
  void TearDown() override {
    iree_hal_amd_gpu_source_context_registry_deinitialize(&registry_);
  }
  // Real device-lifetime source registry under test.
  iree_hal_amd_gpu_source_context_registry_t registry_ = {};
};

TEST_F(SourceContextTest, RetainsSourceAfterExecutableBytesAreReleased) {
  auto bytes = SiteTable();
  const iree_hal_amd_gpu_source_context_t* context = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_source_context_registry_register(
      &registry_, iree_make_const_byte_span(bytes.data(), bytes.size()),
      &context));
  bytes.fill(0);
  EXPECT_EQ(iree_hal_amd_gpu_source_context_executable_id(context), 1u);
  iree_hal_device_event_site_t site;
  ASSERT_TRUE(iree_hal_amd_gpu_source_context_resolve(context, 0, &site));
  EXPECT_TRUE(
      iree_string_view_equal(site.operation_name, IREE_SV("kernel.assert")));
  EXPECT_TRUE(iree_string_view_equal(site.source_file, IREE_SV("kernel.loom")));
  EXPECT_EQ(site.start_line, 7u);
  EXPECT_EQ(site.start_column, 3u);
  EXPECT_EQ(site.end_line, 7u);
  EXPECT_EQ(site.end_column, 19u);
  ASSERT_EQ(site.producer_payload.data_length, 4u);
  EXPECT_EQ(memcmp(site.producer_payload.data, "test", 4), 0);
  EXPECT_FALSE(
      iree_hal_amd_gpu_source_context_resolve(context, UINT64_MAX, &site));
}

TEST_F(SourceContextTest, StrippedExecutablesRetainDistinctIdentities) {
  const iree_hal_amd_gpu_source_context_t* first = nullptr;
  const iree_hal_amd_gpu_source_context_t* second = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_source_context_registry_register(
      &registry_, iree_const_byte_span_empty(), &first));
  IREE_ASSERT_OK(iree_hal_amd_gpu_source_context_registry_register(
      &registry_, iree_const_byte_span_empty(), &second));
  EXPECT_NE(iree_hal_amd_gpu_source_context_executable_id(first),
            iree_hal_amd_gpu_source_context_executable_id(second));
  iree_hal_device_event_site_t site;
  EXPECT_FALSE(iree_hal_amd_gpu_source_context_resolve(first, 0, &site));
}

TEST_F(SourceContextTest, RejectsIncompleteAndInvalidTableRecords) {
  auto bytes = SiteTable();
  const iree_hal_amd_gpu_source_context_t* context = nullptr;
  for (size_t length = 1; length < bytes.size(); ++length) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_amd_gpu_source_context_registry_register(
            &registry_, iree_make_const_byte_span(bytes.data(), length),
            &context));
    EXPECT_EQ(context, nullptr);
  }
  for (size_t offset :
       {size_t{0}, size_t{4}, size_t{5}, size_t{6}, size_t{8}, size_t{12},
        size_t{20}, size_t{32}, size_t{40}, size_t{48}, size_t{52}, size_t{56},
        size_t{76}, size_t{91}}) {
    auto invalid = bytes;
    invalid[offset] = 0xFF;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_amd_gpu_source_context_registry_register(
            &registry_,
            iree_make_const_byte_span(invalid.data(), invalid.size()),
            &context));
    EXPECT_EQ(context, nullptr);
  }
}

}  // namespace
}  // namespace iree::hal::amd::gpu
