// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/vm/buffer.h"

#include "iree/hal/testing/mapped_transfer_buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

using iree::hal::testing::MakeMappedTransferBuffer;

class VmBufferTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_vm_buffer_release(view_);
    iree_vm_buffer_release(imported_);
    iree_hal_buffer_release(exported_);
    iree_hal_buffer_release(source_);
  }

  // Owned HAL view used as an import source.
  iree_hal_buffer_t* source_ = nullptr;
  // Owned imported or VM-created root.
  iree_vm_buffer_t* imported_ = nullptr;
  // Owned nested VM view.
  iree_vm_buffer_t* view_ = nullptr;
  // Owned exported HAL view.
  iree_hal_buffer_t* exported_ = nullptr;
};

TEST_F(VmBufferTest, NoncoherentAliasPreservesRangeAndNativeCacheOperations) {
  auto native =
      MakeMappedTransferBuffer(0, IREE_HAL_BUFFER_USAGE_MAPPING |
                                      IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT);
  for (size_t i = 0; i < native->device_contents.size(); ++i) {
    native->device_contents[i] = static_cast<uint8_t>(i);
  }
  IREE_ASSERT_OK(iree_hal_buffer_subspan(native.get(), 8, 40,
                                         iree_allocator_system(), &source_));
  IREE_ASSERT_OK(loom_vm_buffer_import(source_, 4, 24, iree_allocator_system(),
                                       &imported_));
  IREE_ASSERT_OK(loom_vm_buffer_invalidate(imported_));
  EXPECT_EQ(1u, native->invalidate_count);
  EXPECT_EQ(12u, native->maintenance_range.offset);
  EXPECT_EQ(24u, native->maintenance_range.length);
  IREE_ASSERT_OK(iree_vm_buffer_subspan(
      imported_, 3, 10,
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_allocator_system(), &view_));
  auto* data = static_cast<uint8_t*>(iree_vm_buffer_data(view_));
  ASSERT_NE(nullptr, data);
  EXPECT_EQ(15, data[0]);
  EXPECT_EQ(24, data[9]);
  data[1] = 99;
  IREE_ASSERT_OK(loom_vm_buffer_flush(view_));
  EXPECT_EQ(1u, native->flush_count);
  EXPECT_EQ(15u, native->maintenance_range.offset);
  EXPECT_EQ(10u, native->maintenance_range.length);
  EXPECT_EQ(99, native->device_contents[16]);

  iree_vm_buffer_release(imported_);
  imported_ = nullptr;
  iree_hal_buffer_release(source_);
  source_ = nullptr;
  IREE_ASSERT_OK(
      loom_vm_buffer_export(view_, iree_allocator_system(), &exported_));
  EXPECT_EQ(native.get(), iree_hal_buffer_allocated_buffer(exported_));
  EXPECT_EQ(15u, iree_hal_buffer_byte_offset(exported_));
  EXPECT_EQ(10u, iree_hal_buffer_byte_length(exported_));
  EXPECT_FALSE(iree_any_bit_set(iree_hal_buffer_memory_type(exported_),
                                IREE_HAL_MEMORY_TYPE_HOST_COHERENT));

  // Only the exported HAL view now owns the native allocation. Its mapping
  // must still call the native cache operation instead of a heap no-op.
  iree_vm_buffer_release(view_);
  view_ = nullptr;
  native->device_contents[15] = 73;
  native.reset();
  uint8_t value = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(exported_, 0, &value, sizeof(value)));
  EXPECT_EQ(73, value);
}

TEST_F(VmBufferTest, CoherentImportElidesCacheOperations) {
  auto native = MakeMappedTransferBuffer(
      IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT);
  IREE_ASSERT_OK(loom_vm_buffer_import(native.get(), 0, 16,
                                       iree_allocator_system(), &imported_));
  IREE_ASSERT_OK(loom_vm_buffer_invalidate(imported_));
  static_cast<uint8_t*>(iree_vm_buffer_data(imported_))[0] = 27;
  IREE_ASSERT_OK(loom_vm_buffer_flush(imported_));
  EXPECT_EQ(27, native->device_contents[0]);
  EXPECT_EQ(0u, native->invalidate_count);
  EXPECT_EQ(0u, native->flush_count);
}

TEST_F(VmBufferTest, ReadOnlyAliasDoesNotFlush) {
  auto native =
      MakeMappedTransferBuffer(0, IREE_HAL_BUFFER_USAGE_MAPPING |
                                      IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT);
  IREE_ASSERT_OK(loom_vm_buffer_import(native.get(), 0, 16,
                                       iree_allocator_system(), &imported_));
  IREE_ASSERT_OK(iree_vm_buffer_subspan(imported_, 4, 8,
                                        IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                        iree_allocator_system(), &view_));
  IREE_ASSERT_OK(loom_vm_buffer_invalidate(view_));
  IREE_ASSERT_OK(loom_vm_buffer_flush(view_));
  EXPECT_EQ(1u, native->invalidate_count);
  EXPECT_EQ(4u, native->maintenance_range.offset);
  EXPECT_EQ(8u, native->maintenance_range.length);
  EXPECT_EQ(0u, native->flush_count);
}

TEST_F(VmBufferTest, NativeCacheErrorsPropagate) {
  auto native =
      MakeMappedTransferBuffer(0, IREE_HAL_BUFFER_USAGE_MAPPING |
                                      IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT);
  IREE_ASSERT_OK(loom_vm_buffer_import(native.get(), 0, 16,
                                       iree_allocator_system(), &imported_));
  native->invalidate_code = IREE_STATUS_DATA_LOSS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        loom_vm_buffer_invalidate(imported_));
  native->flush_code = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        loom_vm_buffer_flush(imported_));
}

TEST_F(VmBufferTest, VmOwnedAliasRetainsCoherentStorage) {
  IREE_ASSERT_OK(
      iree_vm_buffer_create(32, 16, iree_allocator_system(), &imported_));
  IREE_ASSERT_OK(iree_vm_buffer_subspan(
      imported_, 8, 16,
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_allocator_system(), &view_));
  static_cast<uint8_t*>(iree_vm_buffer_data(view_))[0] = 47;
  IREE_ASSERT_OK(loom_vm_buffer_invalidate(view_));
  IREE_ASSERT_OK(loom_vm_buffer_flush(view_));
  IREE_ASSERT_OK(
      loom_vm_buffer_export(view_, iree_allocator_system(), &exported_));
  iree_vm_buffer_release(imported_);
  imported_ = nullptr;
  iree_vm_buffer_release(view_);
  view_ = nullptr;
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_memory_type(exported_),
                                IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
  uint8_t value = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(exported_, 0, &value, sizeof(value)));
  EXPECT_EQ(47, value);
}

TEST_F(VmBufferTest, EmptyImportedViewPreservesNativeOffset) {
  auto native =
      MakeMappedTransferBuffer(0, IREE_HAL_BUFFER_USAGE_MAPPING |
                                      IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT);
  IREE_ASSERT_OK(loom_vm_buffer_import(native.get(), 8, 16,
                                       iree_allocator_system(), &imported_));
  IREE_ASSERT_OK(iree_vm_buffer_subspan(imported_, 16, 0,
                                        IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                        iree_allocator_system(), &view_));
  IREE_ASSERT_OK(
      loom_vm_buffer_export(view_, iree_allocator_system(), &exported_));
  EXPECT_EQ(24u, iree_hal_buffer_byte_offset(exported_));
  EXPECT_EQ(0u, iree_hal_buffer_byte_length(exported_));
}

}  // namespace
}  // namespace loom
