// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/xdna/direct_binding.h"

#include <algorithm>
#include <array>
#include <vector>

#include "iree/hal/drivers/amd/xdna/image/testing/image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::StatusCode;
using iree::hal::amd::xdna::testing::ImageFixture;
using iree::hal::amd::xdna::testing::MakeImageTarget;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;

class DirectBindingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto source = MakeOwnedByteSequence(ImageFixture().Build());
    const auto target = MakeImageTarget();
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
        source.get(), &target, iree_allocator_system(), &image_));
    for (size_t i = 0; i < storage_.size(); ++i) {
      const auto allocation = iree_hal_amd_xdna_image_tables_allocation(
          iree_hal_amd_xdna_image_tables(image_), i);
      bytes_[i].assign(allocation.byte_length, 0xCC);
      storage_[i].mapping =
          iree_make_byte_span(bytes_[i].data(), bytes_[i].size());
      storage_[i].memory = reinterpret_cast<amdf_memory_t*>(uintptr_t{16} + i);
      storage_[i].access_ordinal = 3;
      storage_[i].memory_byte_offset = 65536;
      storage_[i].device_address = UINT64_C(0x123456780000) + i * 32768;
    }
    ASSERT_NO_FATAL_FAILURE(
        WrapBinding(IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                    IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
                    IREE_HAL_BUFFER_USAGE_STORAGE));
    IREE_ASSERT_OK(iree_hal_amd_xdna_executable_storage_load(
        image_, 0, storage_.size(), storage_.data()));
  }

  void TearDown() override {
    iree_hal_buffer_release(binding_.buffer_ref.buffer);
    iree_hal_amd_xdna_image_destroy(image_);
  }

  void WrapBinding(iree_hal_memory_type_t type, iree_hal_memory_access_t access,
                   iree_hal_buffer_usage_t usage) {
    iree_hal_buffer_release(binding_.buffer_ref.buffer);
    binding_ = {};
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(), type, access, usage,
        buffer_.size(), iree_make_byte_span(buffer_.data(), buffer_.size()),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &binding_.buffer_ref.buffer));
    binding_.buffer_ref.length = 16;
    binding_.byte_length = 16;
    binding_.device_address = UINT64_C(0xABCD12340000);
  }

  iree_status_t Bind() {
    return iree_xdna_executable_storage_bind(image_, 0, storage_.size(),
                                             storage_.data(), 1, &binding_);
  }

  iree_hal_amd_xdna_image_t* image_ = nullptr;
  std::array<std::vector<uint8_t>, 2> bytes_;
  std::array<iree_hal_amd_xdna_executable_storage_t, 2> storage_ = {};
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) std::array<uint8_t, 128> buffer_ = {};
  iree_hal_amd_xdna_executable_binding_t binding_ = {};
};

TEST_F(DirectBindingTest, RebindsOnlyDynamicAddresses) {
  for (uint64_t address :
       {UINT64_C(0xABCD12340000), UINT64_C(0x123456789000)}) {
    const auto before = bytes_;
    binding_.device_address = address;
    IREE_ASSERT_OK(Bind());
    EXPECT_TRUE(std::equal(bytes_[0].begin(), bytes_[0].begin() + 8,
                           before[0].begin()));
    EXPECT_TRUE(std::equal(bytes_[0].begin() + 16, bytes_[0].end(),
                           before[0].begin() + 16));
    EXPECT_EQ(bytes_[1], before[1]);
    EXPECT_EQ(iree_unaligned_load_le_u32(bytes_[0].data() + 8),
              (uint32_t)(address + 4) | 1u);
    EXPECT_EQ(iree_unaligned_load_le_u32(bytes_[0].data() + 12),
              UINT32_C(0xA5A50000) | (uint32_t)((address + 4) >> 32));
  }
}

TEST_F(DirectBindingTest, RejectsInvalidLogicalRangesBeforePatching) {
  const auto before = bytes_;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_xdna_executable_storage_bind(image_, 0, storage_.size(),
                                        storage_.data(), 0, nullptr));
  binding_.buffer_ref.offset = buffer_.size() - 8;
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, Bind());
  binding_.buffer_ref.offset = 0;
  binding_.buffer_ref.length = 8;
  binding_.byte_length = 8;
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, Bind());
  binding_.buffer_ref.offset = UINT64_MAX;
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, Bind());
  EXPECT_EQ(bytes_, before);
}

TEST_F(DirectBindingTest, RejectsInvalidAddressesBeforePatching) {
  const auto before = bytes_;
  for (uint64_t address :
       {UINT64_MAX - 3, UINT64_C(0x1000000000000), UINT64_C(3)}) {
    binding_.device_address = address;
    IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange, Bind());
    EXPECT_EQ(bytes_, before);
  }
}

TEST_F(DirectBindingTest, EnforcesLogicalBufferContract) {
  const auto before = bytes_;
  ASSERT_NO_FATAL_FAILURE(WrapBinding(IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                                      IREE_HAL_MEMORY_ACCESS_WRITE,
                                      IREE_HAL_BUFFER_USAGE_STORAGE));
  IREE_EXPECT_STATUS_IS(StatusCode::kPermissionDenied, Bind());
  ASSERT_NO_FATAL_FAILURE(WrapBinding(IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                                      IREE_HAL_MEMORY_ACCESS_READ,
                                      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_EXPECT_STATUS_IS(StatusCode::kPermissionDenied, Bind());
  ASSERT_NO_FATAL_FAILURE(WrapBinding(IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
                                      IREE_HAL_MEMORY_ACCESS_READ,
                                      IREE_HAL_BUFFER_USAGE_STORAGE));
  IREE_EXPECT_STATUS_IS(StatusCode::kPermissionDenied, Bind());
  EXPECT_EQ(bytes_, before);
}

}  // namespace
