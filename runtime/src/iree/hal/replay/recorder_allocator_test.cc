// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_allocator.h"

#include <array>

#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct ImportAllocatorState {
  // Whether new allocations fail while frees continue to reach the system.
  bool fail_allocations = false;
  // Number of allocation attempts, including rejected attempts.
  iree_host_size_t allocation_attempt_count = 0;
  // Number of successful allocations not yet freed.
  iree_host_size_t live_allocation_count = 0;
};

static iree_status_t ImportAllocatorCtl(void* self,
                                        iree_allocator_command_t command,
                                        const void* params, void** inout_ptr) {
  auto* state = static_cast<ImportAllocatorState*>(self);
  if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
      command == IREE_ALLOCATOR_COMMAND_CALLOC) {
    ++state->allocation_attempt_count;
    if (state->fail_allocations) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
  }
  iree_allocator_t system_allocator = iree_allocator_system();
  iree_status_t status =
      system_allocator.ctl(system_allocator.self, command, params, inout_ptr);
  if (iree_status_is_ok(status)) {
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC) {
      ++state->live_allocation_count;
    } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      --state->live_allocation_count;
    }
  }
  return status;
}

static iree_allocator_t ImportAllocator(ImportAllocatorState* state) {
  return iree_allocator_t{state, ImportAllocatorCtl};
}

class ReplayRecorderAllocatorTest : public ::testing::Test {
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
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("test"), iree_allocator_system(),
        ImportAllocator(&native_state_), &native_allocator_));
    IREE_ASSERT_OK(iree_hal_replay_recorder_wrap_allocator(
        recorder_, /*device_id=*/1, /*placement_device=*/nullptr,
        native_allocator_, ImportAllocator(&proxy_state_), &allocator_));
  }

  void TearDown() override {
    iree_hal_allocator_release(allocator_);
    iree_hal_allocator_release(native_allocator_);
    if (recorder_) {
      IREE_EXPECT_OK(iree_hal_replay_recorder_close(recorder_));
      iree_hal_replay_recorder_release(recorder_);
    }
    EXPECT_EQ(0u, native_state_.live_allocation_count);
    EXPECT_EQ(0u, proxy_state_.live_allocation_count);
  }

  iree_status_t Import(iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    params.usage = IREE_HAL_BUFFER_USAGE_MAPPING;
    iree_hal_external_buffer_t external_buffer = {};
    external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
    external_buffer.size = sizeof(storage_);
    external_buffer.handle.host_allocation.ptr = storage_;
    iree_hal_buffer_release_callback_t release_callback = {
        [](void* user_data, iree_hal_buffer_t*) {
          ++*static_cast<int*>(user_data);
        },
        &release_count_,
    };
    return iree_hal_allocator_import_buffer(
        allocator_, params, &external_buffer, release_callback, out_buffer);
  }

  void VerifySuccessfulImport() {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(Import(&buffer));
    EXPECT_EQ(0, release_count_);
    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE, 0,
        IREE_HAL_WHOLE_BUFFER, &mapping));
    EXPECT_EQ(storage_, mapping.contents.data);
    EXPECT_EQ(sizeof(storage_), mapping.contents.data_length);
    mapping.contents.data[7] = 0xA7;
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    EXPECT_EQ(0xA7, storage_[7]);
    EXPECT_EQ(0, release_count_);
    iree_hal_buffer_release(buffer);
    EXPECT_EQ(1, release_count_);
  }

  // Storage owned by the test caller, never freed by the release counter.
  alignas(64) uint8_t storage_[64] = {};
  // Number of times the imported buffer relinquished the caller's storage.
  int release_count_ = 0;
  // Host allocation control for the real native heap allocator.
  ImportAllocatorState native_state_;
  // Host allocation control for the recording allocator and buffer wrappers.
  ImportAllocatorState proxy_state_;
  // Destination for the actual capture stream.
  std::array<uint8_t, 16384> recording_ = {};
  // Recorder retained until its wrappers have been released.
  iree_hal_replay_recorder_t* recorder_ = nullptr;
  // Real native heap allocator wrapped by the recorder.
  iree_hal_allocator_t* native_allocator_ = nullptr;
  // Recording allocator exercised through the public import API.
  iree_hal_allocator_t* allocator_ = nullptr;
};

TEST_F(ReplayRecorderAllocatorTest, ProxyFailurePreservesCallerOwnership) {
  const auto native_attempt_count = native_state_.allocation_attempt_count;
  const auto proxy_live_count = proxy_state_.live_allocation_count;
  proxy_state_.fail_allocations = true;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Import(&buffer));
  EXPECT_EQ(nullptr, buffer);
  EXPECT_EQ(0, release_count_);
  EXPECT_EQ(native_attempt_count, native_state_.allocation_attempt_count);
  EXPECT_EQ(proxy_live_count, proxy_state_.live_allocation_count);
  proxy_state_.fail_allocations = false;
  VerifySuccessfulImport();
}

TEST_F(ReplayRecorderAllocatorTest, NativeFailureReleasesTentativeProxy) {
  const auto proxy_live_count = proxy_state_.live_allocation_count;
  native_state_.fail_allocations = true;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Import(&buffer));
  EXPECT_EQ(nullptr, buffer);
  EXPECT_EQ(0, release_count_);
  EXPECT_EQ(proxy_live_count, proxy_state_.live_allocation_count);
  native_state_.fail_allocations = false;
  VerifySuccessfulImport();
}

TEST_F(ReplayRecorderAllocatorTest, SuccessfulImportReleasesCallerOnce) {
  VerifySuccessfulImport();
}

}  // namespace
