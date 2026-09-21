// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_buffer.h"

#include <array>

#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ReplayRecorderBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_io_file_handle_t* file_handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(recording_.data(), recording_.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &file_handle));
    iree_status_t status = iree_hal_replay_recorder_create(
        file_handle, nullptr, iree_allocator_system(), &recorder_);
    iree_io_file_handle_release(file_handle);
    IREE_ASSERT_OK(status);

    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
        sizeof(storage_), iree_make_byte_span(storage_, sizeof(storage_)),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &native_buffer_));
  }

  void TearDown() override {
    iree_hal_buffer_release(native_buffer_);
    if (recorder_) {
      IREE_EXPECT_OK(iree_hal_replay_recorder_close(recorder_));
      iree_hal_replay_recorder_release(recorder_);
    }
  }

  // Destination for the real recorder's capture stream.
  std::array<uint8_t, 4096> recording_ = {};
  // Recorder retained until the wrapped buffers have been released.
  iree_hal_replay_recorder_t* recorder_ = nullptr;
  // Native host backing exposed by the wrapped buffer.
  alignas(64) uint8_t storage_[128] = {};
  // Whole native allocation from which the recorded view is created.
  iree_hal_buffer_t* native_buffer_ = nullptr;
};

TEST_F(ReplayRecorderBufferTest, ExportsNestedSubspansOfANativeView) {
  iree_hal_buffer_t* native_view = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      native_buffer_, 32, 64, iree_allocator_system(), &native_view));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_recorder_buffer_create_proxy(
      recorder_, /*device_id=*/1, /*buffer_id=*/2,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, /*placement_device=*/nullptr, native_view,
      iree_allocator_system(), &buffer));
  iree_hal_buffer_release(native_view);

  iree_hal_external_buffer_t external_buffer = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(
      buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(storage_ + 32, external_buffer.handle.host_allocation.ptr);
  EXPECT_EQ(64u, external_buffer.size);

  iree_hal_buffer_t* whole_view = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 0, IREE_HAL_WHOLE_BUFFER,
                                         iree_allocator_system(), &whole_view));
  EXPECT_EQ(buffer, whole_view);
  iree_hal_buffer_release(whole_view);

  iree_hal_buffer_t* subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 14, 16,
                                         iree_allocator_system(), &subspan));
  iree_hal_buffer_t* nested_subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(subspan, 4, 8, iree_allocator_system(),
                                         &nested_subspan));
  iree_hal_buffer_release(subspan);
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_buffer_export(
      nested_subspan, IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(storage_ + 50, external_buffer.handle.host_allocation.ptr);
  EXPECT_EQ(8u, external_buffer.size);
  IREE_ASSERT_OK(iree_hal_buffer_export(
      nested_subspan, IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(storage_ + 50),
            external_buffer.handle.device_allocation.ptr);
  EXPECT_EQ(8u, external_buffer.size);
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      nested_subspan, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      0, IREE_HAL_WHOLE_BUFFER, &mapping));
  EXPECT_EQ(storage_ + 50, mapping.contents.data);
  EXPECT_EQ(8u, mapping.contents.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  iree_hal_buffer_release(nested_subspan);
}

}  // namespace
