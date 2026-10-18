// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/file_transfer.h"

#include <array>
#include <cstring>
#include <thread>

#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/semaphore.h"
#include "iree/hal/testing/mapped_transfer_buffer.h"
#include "iree/hal/utils/host_semaphore.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// The queue dependency completes inside submit, as native work is allowed to
// do. Transfers, host waits, captured ownership, and public completion all use
// production code. Hardware execution is covered by the driver CTS.
struct NativeQueue : iree_hal_queue_t {
  // Public point must remain pending until submit relinquishes its arguments.
  iree_hal_semaphore_t* completion = nullptr;
  // Counts the native copy and final cache boundary separately.
  uint32_t submissions = 0;
  // Selected native rejection, or zero when every operation is accepted.
  uint32_t reject_submission = 0;
  // Completion may occur on a different CPU before submit returns.
  bool concurrent_completion = false;
  // A nested submit would reuse the operation while its producer still reads
  // it.
  bool submitting = false;

  // Retains native work after submit, including after a failure notification.
  bool defer_retirement = false;
  // Borrowed file owner held until the simulated native work retires.
  iree_hal_amd_gpu_retirement_callback_t pending_callback = {};
  // Captured native operands whose bytes remain borrowed until retirement.
  iree_hal_transfer_operation_t pending_operation = {};

  static void Copy(const iree_hal_transfer_operation_t* operation) {
    if (operation) {
      if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD) {
        const auto& copy = operation->upload;
        auto* target = static_cast<iree::hal::testing::MappedTransferBuffer*>(
            copy.target_buffer);
        std::memcpy(target->device_contents.data() + copy.target_offset,
                    copy.source, copy.length);
      } else {
        EXPECT_EQ(operation->type, IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD);
        const auto& copy = operation->download;
        auto* source = static_cast<iree::hal::testing::MappedTransferBuffer*>(
            copy.source_buffer);
        std::memcpy(copy.target,
                    source->device_contents.data() + copy.source_offset,
                    copy.length);
      }
    }
  }

  static iree_status_t Submit(iree_hal_queue_t* base,
                              const iree_hal_transfer_operation_t* operation,
                              const void* before, const void* after,
                              iree_hal_amd_gpu_retirement_callback_t callback) {
    auto* queue = static_cast<NativeQueue*>(base);
    EXPECT_FALSE(queue->submitting);
    queue->submitting = true;
    ++queue->submissions;
    if (queue->submissions == queue->reject_submission) {
      queue->submitting = false;
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "native submit rejected");
    }
    if (queue->defer_retirement) {
      queue->pending_callback = callback;
      if (operation) {
        queue->pending_operation = *operation;
      }
      queue->submitting = false;
      return iree_ok_status();
    }
    Copy(operation);
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, 7, queue->submissions);
    if (queue->concurrent_completion) {
      std::thread completion([callback, &frontier] {
        callback.fn(callback.user_data, iree_ok_status(), &frontier, true);
      });
      completion.join();
    } else {
      callback.fn(callback.user_data, iree_ok_status(), &frontier, true);
    }
    // Native completion has fired, but these borrowed operands still belong to
    // this submit call. Advancing or freeing the file capture here is a bug.
    if (before) {
      EXPECT_EQ(*static_cast<const uint64_t*>(before), 17u);
    }
    if (after) {
      EXPECT_EQ(*static_cast<const uint64_t*>(after), 29u);
    }
    uint64_t public_value = UINT64_MAX;
    IREE_CHECK_OK(iree_hal_semaphore_query(queue->completion, &public_value));
    EXPECT_EQ(public_value, 0u);
    queue->submitting = false;
    return iree_ok_status();
  }
};

struct Device {
  // HAL identity retained by the production capture.
  iree_hal_resource_t resource;
  // Last release observation, independent of the captured storage.
  bool destroyed = false;

  Device() {
    static const iree_hal_device_vtable_t vtable = {
        .destroy =
            [](iree_hal_device_t* base) {
              reinterpret_cast<Device*>(base)->destroyed = true;
            },
    };
    iree_hal_resource_initialize(&vtable, &resource);
  }
  iree_hal_device_t* get() {
    return reinterpret_cast<iree_hal_device_t*>(this);
  }
};

class FileTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_semaphore_create(
        proactor_, 0, 1, iree_allocator_system(), &completion_));
    iree_async_semaphore_wait_context_initialize(&waits_);
    iree_hal_amd_gpu_capture_pool_initialize(4096, iree_allocator_system(),
                                             &captures_);
    for (size_t i = 0; i < contents_.size(); ++i) {
      contents_[i] = uint8_t(i * 7);
    }
    iree_io_file_handle_t* handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(contents_.data(), contents_.size()),
        {[](void* data, iree_io_file_handle_primitive_t) {
           *static_cast<bool*>(data) = true;
         },
         &file_released_},
        iree_allocator_system(), &handle));
    IREE_ASSERT_OK(iree_hal_memory_file_wrap(
        nullptr, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        handle, IREE_HAL_MEMORY_FILE_FLAG_NONE, iree_allocator_system(),
        &file_));
    iree_io_file_handle_release(handle);
  }
  void TearDown() override {
    iree_hal_file_release(file_);
    iree_hal_amd_gpu_capture_pool_deinitialize(&captures_);
    iree_async_semaphore_wait_context_deinitialize(&waits_);
    iree_hal_semaphore_release(completion_);
    iree_async_proactor_release(proactor_);
  }

  void Transfer(iree_hal_memory_access_t access, bool concurrent,
                uint32_t reject = 0, bool deferred = false) {
    Device device;
    iree_hal_queue_family_t family = {.device = device.get()};
    NativeQueue queue;
    queue.queue_family = &family;
    queue.completion = completion_;
    queue.concurrent_completion = concurrent;
    queue.reject_submission = reject;
    iree_hal_amd_gpu_context_t context = {};
    uint32_t failures = 0;
    context.event_sink = {
        .fn =
            [](void* user_data, const iree_hal_device_event_t* event) {
              EXPECT_EQ(event->type, IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE);
              ++*static_cast<uint32_t*>(user_data);
            },
        .user_data = &failures,
    };
    const iree_hal_amd_gpu_file_executor_t executor = {
        .queue = &queue,
        .context = &context,
        .captures = &captures_,
        .waits = &waits_,
        .barrier_size = sizeof(uint64_t),
        .submit = NativeQueue::Submit,
    };
    auto buffer = iree::hal::testing::MakeMappedTransferBuffer();
    std::array<uint8_t, 64> expected = contents_;
    if (access == IREE_HAL_MEMORY_ACCESS_WRITE) {
      buffer->device_contents = expected;
      contents_.fill(0);
    }
    iree_hal_semaphore_t* dependency = nullptr;
    IREE_ASSERT_OK(iree_hal_host_semaphore_create(
        proactor_, deferred ? 0 : 1, iree_allocator_system(), &dependency));
    uint64_t value = 1;
    const uint64_t before = 17, after = 29;
    IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit(
        &executor, {1, &dependency, &value}, {1, &completion_, &value}, file_,
        access, 5, buffer.get(), 5, 50, &before, &after));
    iree_hal_device_release(device.get());
    if (deferred) {
      EXPECT_EQ(queue.submissions, 0u);
      EXPECT_FALSE(device.destroyed);
      IREE_ASSERT_OK(iree_hal_semaphore_signal(dependency, 1, nullptr));
    }
    EXPECT_TRUE(device.destroyed);
    EXPECT_EQ(failures, reject ? 1u : 0u);
    uint64_t completed_value = 0;
    if (reject) {
      EXPECT_THAT(
          iree::Status(iree_hal_semaphore_query(completion_, &completed_value)),
          iree::testing::status::StatusIs(iree::StatusCode::kUnavailable));
      EXPECT_EQ(queue.submissions, reject);
    } else {
      IREE_ASSERT_OK(iree_hal_semaphore_query(completion_, &completed_value));
      EXPECT_EQ(completed_value, 1u);
      EXPECT_EQ(queue.submissions, 2u);
      iree_async_single_frontier_t frontier = {};
      EXPECT_EQ(iree_async_semaphore_query_frontier(
                    reinterpret_cast<iree_async_semaphore_t*>(completion_),
                    iree_async_single_frontier_as_frontier(&frontier), 1),
                1u);
      EXPECT_EQ(frontier.entries[0].axis, 7u);
      EXPECT_EQ(frontier.entries[0].epoch, 2u);
      const auto& actual = access == IREE_HAL_MEMORY_ACCESS_READ
                               ? buffer->device_contents
                               : contents_;
      for (size_t i = 5; i < 55; ++i) {
        EXPECT_EQ(actual[i], expected[i]);
      }
    }
    iree_hal_semaphore_release(dependency);
  }

  // Real host semaphore backend; no poll thread is necessary.
  iree_async_proactor_t* proactor_ = nullptr;
  // Retained public completion through the assertion boundary.
  iree_hal_semaphore_t* completion_ = nullptr;
  // Production tracker associations.
  iree_async_semaphore_wait_context_t waits_;
  // Production capture recycling, including immediate return after completion.
  iree_hal_amd_gpu_capture_pool_t captures_;
  // Imported host file bytes.
  std::array<uint8_t, 64> contents_;
  // Retained file wrapper shared with the captured operation.
  iree_hal_file_t* file_ = nullptr;
  // Final backing release observed independently of the retained wrapper.
  bool file_released_ = false;
};

TEST_F(FileTransferTest, InlineReadJoinsSubmitBeforeAdvancing) {
  Transfer(IREE_HAL_MEMORY_ACCESS_READ, false);
}
TEST_F(FileTransferTest, InlineWriteJoinsSubmitBeforeAdvancing) {
  Transfer(IREE_HAL_MEMORY_ACCESS_WRITE, false);
}
TEST_F(FileTransferTest, ConcurrentReadJoinsSubmitBeforeAdvancing) {
  Transfer(IREE_HAL_MEMORY_ACCESS_READ, true);
}
TEST_F(FileTransferTest, ConcurrentWriteJoinsSubmitBeforeAdvancing) {
  Transfer(IREE_HAL_MEMORY_ACCESS_WRITE, true);
}
TEST_F(FileTransferTest, RejectedCopyFailsPublicCompletion) {
  Transfer(IREE_HAL_MEMORY_ACCESS_READ, false, 1);
}
TEST_F(FileTransferTest, RejectedReturnBoundaryFailsPublicCompletion) {
  Transfer(IREE_HAL_MEMORY_ACCESS_READ, false, 2);
}
TEST_F(FileTransferTest, DeferredDependencyRetainsParentThroughCompletion) {
  Transfer(IREE_HAL_MEMORY_ACCESS_READ, true, 0, true);
}

TEST_F(FileTransferTest, FailedNativeWorkRetainsFileUntilRetirement) {
  Device device;
  iree_hal_queue_family_t family = {.device = device.get()};
  NativeQueue queue;
  queue.queue_family = &family;
  queue.defer_retirement = true;
  iree_hal_amd_gpu_context_t context = {};
  uint32_t failures = 0;
  context.event_sink = {
      .fn =
          [](void* user_data, const iree_hal_device_event_t* event) {
            EXPECT_EQ(event->type, IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE);
            ++*static_cast<uint32_t*>(user_data);
          },
      .user_data = &failures,
  };
  const iree_hal_amd_gpu_file_executor_t executor = {
      .queue = &queue,
      .context = &context,
      .captures = &captures_,
      .waits = &waits_,
      .barrier_size = sizeof(uint64_t),
      .submit = NativeQueue::Submit,
  };
  auto buffer = iree::hal::testing::MakeMappedTransferBuffer();
  const uint64_t before = 17, after = 29;
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit(
      &executor, {}, {1, &completion_, &value}, file_,
      IREE_HAL_MEMORY_ACCESS_READ, 5, buffer.get(), 5, 50, &before, &after));
  iree_hal_file_release(file_);
  file_ = nullptr;
  iree_hal_device_release(device.get());
  EXPECT_EQ(queue.submissions, 1u);
  EXPECT_FALSE(file_released_);
  EXPECT_FALSE(device.destroyed);
  auto callback = queue.pending_callback;
  callback.fn(callback.user_data, iree_status_from_code(IREE_STATUS_ABORTED),
              nullptr, false);
  uint64_t completed_value = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_query(completion_, &completed_value));
  EXPECT_FALSE(file_released_);
  EXPECT_FALSE(device.destroyed);
  // The accepted copy may still access the file bytes after public failure.
  NativeQueue::Copy(&queue.pending_operation);
  for (size_t i = 5; i < 55; ++i) {
    EXPECT_EQ(buffer->device_contents[i], contents_[i]);
  }
  callback.fn(callback.user_data, iree_status_from_code(IREE_STATUS_ABORTED),
              nullptr, true);
  EXPECT_TRUE(file_released_);
  EXPECT_TRUE(device.destroyed);
  EXPECT_EQ(queue.submissions, 1u);
  EXPECT_EQ(failures, 1u);
}

}  // namespace
