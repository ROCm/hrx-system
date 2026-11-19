// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/file_transfer.h"

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/driver/amd/gpu/queue/semaphore.h"
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
  // Accepted pipeline owners, independent of native packet completion.
  std::atomic<int> host_owners{0};
  static void Begin(iree_hal_queue_t* queue) {
    ++static_cast<NativeQueue*>(queue)->host_owners;
  }
  static void End(iree_hal_queue_t* queue) {
    --static_cast<NativeQueue*>(queue)->host_owners;
  }

  // Public point must remain pending until submit relinquishes its arguments.
  iree_hal_semaphore_t* completion = nullptr;
  // Sequential native submission ordinal used as the completion epoch.
  uint32_t submissions = 0;
  // Submission carrying the caller's consumer boundary, or zero if absent.
  uint32_t consumer_submission = 0;
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
  std::vector<iree_hal_amd_gpu_transfer_operation_t> pending_operations;

  static void Copy(const iree_hal_amd_gpu_transfer_operation_t* operation) {
    if (operation) {
      if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD) {
        const auto& copy = operation->target.device;
        auto* target =
            static_cast<iree::hal::testing::MappedTransferBuffer*>(copy.buffer);
        std::memcpy(target->device_contents.data() + copy.offset,
                    operation->source.host, operation->length);
      } else if (operation->type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY) {
        const auto& read = operation->source.device;
        const auto& write = operation->target.device;
        auto* source =
            static_cast<iree::hal::testing::MappedTransferBuffer*>(read.buffer);
        auto* target = static_cast<iree::hal::testing::MappedTransferBuffer*>(
            write.buffer);
        std::memcpy(target->device_contents.data() + write.offset,
                    source->device_contents.data() + read.offset,
                    operation->length);
      } else {
        EXPECT_EQ(operation->type, IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD);
        const auto& copy = operation->source.device;
        auto* source =
            static_cast<iree::hal::testing::MappedTransferBuffer*>(copy.buffer);
        std::memcpy(operation->target.host,
                    source->device_contents.data() + copy.offset,
                    operation->length);
      }
    }
  }

  static iree_status_t Submit(
      iree_hal_queue_t* base, iree_host_size_t count,
      const iree_hal_amd_gpu_transfer_operation_t* operations,
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
    if (after) {
      EXPECT_EQ(queue->consumer_submission, 0u);
      queue->consumer_submission = queue->submissions;
    }
    if (queue->defer_retirement) {
      queue->pending_callback = callback;
      queue->pending_operations.clear();
      for (iree_host_size_t i = 0; i < count; ++i) {
        queue->pending_operations.push_back(operations[i]);
      }
      queue->submitting = false;
      return iree_ok_status();
    }
    for (iree_host_size_t i = 0; i < count; ++i) {
      Copy(&operations[i]);
    }
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

// A file exposing only execution storage. No host read/write implementation
// exists, so the production owner must preserve the native buffer path.
struct DeviceFile {
  // HAL file identity retained independently of this observation storage.
  iree_hal_resource_t resource;
  // Owned execution storage, released by final file destruction.
  iree_hal_buffer_t* buffer;
  // Observed final release after the accepted native copy retires.
  bool destroyed = false;

  explicit DeviceFile(iree_hal_buffer_t* storage) : buffer(storage) {
    static const iree_hal_file_vtable_t vtable = {
        .destroy =
            [](iree_hal_file_t* base) {
              auto* file = reinterpret_cast<DeviceFile*>(base);
              iree_hal_buffer_release(file->buffer);
              file->destroyed = true;
            },
        .allowed_access =
            [](iree_hal_file_t*) {
              return iree_hal_memory_access_t{IREE_HAL_MEMORY_ACCESS_ALL};
            },
        .length = [](iree_hal_file_t*) { return uint64_t{64}; },
        .storage_buffer =
            [](iree_hal_file_t* base) {
              return reinterpret_cast<DeviceFile*>(base)->buffer;
            },
    };
    iree_hal_resource_initialize(&vtable, &resource);
    iree_hal_buffer_retain(buffer);
  }

  iree_hal_file_t* get() { return reinterpret_cast<iree_hal_file_t*>(this); }
};

class FileTransferTest : public ::testing::Test {
 protected:
  // Qualifies the device-address dependency consumed by the real file capture.
  // Its storage stays live with this fixture until all native uses have joined.
  void QualifyBuffer(iree::hal::testing::MappedTransferBuffer* buffer) {
    device_scope_.bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
    device_scope_.interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
    device_scope_.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    device_contract_->scopes = &device_scope_;
    device_contract_->scope_count = 1;
    buffer->host_binding.device_address =
        reinterpret_cast<uintptr_t>(buffer->device_contents.data());
    buffer->memory.contract = device_contract_.get();
    buffer->memory.bindings = &buffer->host_binding;
    buffer->host_binding_index = IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE;
    buffer->memory_type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    buffer->allowed_usage |= IREE_HAL_BUFFER_USAGE_TRANSFER;
  }
  // Queue's immutable buffer-access contract.
  iree_hal_memory_scope_access_t device_scope_ = {};
  // Native binding layout shared by the one buffer in each operation.
  std::unique_ptr<iree_hal_memory_contract_t> device_contract_ =
      std::make_unique<iree_hal_memory_contract_t>();
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
        proactor_, nullptr, 0, 1, iree_allocator_system(), &completion_));
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
                uint32_t reject = 0, bool deferred = false,
                amdf_queue_publication_modes_t mode =
                    AMDF_QUEUE_PUBLICATION_MODE_KERNEL) {
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
        .begin = NativeQueue::Begin,
        .end = NativeQueue::End,
        .waits = &waits_,
        .publication_mode = mode,
        .transfer_site = IREE_HAL_MEMORY_SITE_QUEUE,
        .barrier_size = sizeof(uint64_t),
        .submit = NativeQueue::Submit,
    };
    auto buffer = iree::hal::testing::MakeMappedTransferBuffer();
    QualifyBuffer(buffer.get());
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
    const iree_hal_file_binding_t file_binding = {file_, 5, 50};
    const iree_hal_buffer_binding_t buffer_binding = {buffer.get(), 5, 50};
    const iree_hal_file_transfer_t transfer = {
        .row_length = 50,
        .row_count = 1,
        .slice_count = 1,
    };
    const iree_hal_file_transfer_list_t list = {
        .host = {1, &transfer},
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit_batch(
        &executor, {1, &dependency, &value}, {1, &completion_, &value}, access,
        {1, &file_binding}, {1, &buffer_binding}, &list, &before, &after));
    if (deferred) {
      EXPECT_EQ(queue.submissions, 0u);
      EXPECT_FALSE(device.destroyed);
      IREE_ASSERT_OK(iree_hal_semaphore_signal(dependency, 1, nullptr));
    }
    EXPECT_EQ(queue.host_owners.load(), 0);
    iree_hal_device_release(device.get());
    EXPECT_TRUE(device.destroyed);
    EXPECT_EQ(failures, 0u);
    uint64_t completed_value = 0;
    if (reject) {
      EXPECT_THAT(
          iree::Status(iree_hal_semaphore_query(completion_, &completed_value)),
          iree::testing::status::StatusIs(iree::StatusCode::kUnavailable));
      EXPECT_EQ(queue.submissions, reject);
    } else {
      IREE_ASSERT_OK(iree_hal_semaphore_query(completion_, &completed_value));
      EXPECT_EQ(completed_value, 1u);
      const uint32_t expected_submissions =
          mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL &&
                  access == IREE_HAL_MEMORY_ACCESS_READ
              ? 1u
              : 2u;
      EXPECT_EQ(queue.submissions, expected_submissions);
      EXPECT_EQ(queue.consumer_submission, expected_submissions);
      iree_async_single_frontier_t frontier = {};
      EXPECT_EQ(iree_async_semaphore_query_frontier(
                    reinterpret_cast<iree_async_semaphore_t*>(completion_), 1,
                    iree_async_single_frontier_as_frontier(&frontier)),
                1u);
      EXPECT_EQ(frontier.entries[0].axis, 7u);
      EXPECT_EQ(frontier.entries[0].epoch, expected_submissions);
      const auto& actual = access == IREE_HAL_MEMORY_ACCESS_READ
                               ? buffer->device_contents
                               : contents_;
      for (size_t i = 5; i < 55; ++i) {
        EXPECT_EQ(actual[i], expected[i]);
      }
    }
    iree_hal_semaphore_release(dependency);
  }

  void FailedNativeWork(iree_hal_memory_access_t access);
  void IndirectCountWithoutRecords(uint64_t count);
  void DeviceStorage(iree_hal_memory_access_t access, bool allow_access = true);

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

TEST_F(FileTransferTest, UserQueuePublishesConsumerBoundaryAfterCopy) {
  Transfer(IREE_HAL_MEMORY_ACCESS_READ, false, 0, false,
           AMDF_QUEUE_PUBLICATION_MODE_USER);
}

void FileTransferTest::DeviceStorage(iree_hal_memory_access_t access,
                                     bool allow_access) {
  Device device;
  iree_hal_queue_family_t family = {.device = device.get()};
  NativeQueue queue;
  queue.queue_family = &family;
  queue.completion = completion_;
  queue.defer_retirement = true;
  iree_hal_amd_gpu_context_t context = {};
  const iree_hal_amd_gpu_file_executor_t executor = {
      .queue = &queue,
      .context = &context,
      .captures = &captures_,
      .begin = NativeQueue::Begin,
      .end = NativeQueue::End,
      .waits = &waits_,
      .publication_mode = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
      .transfer_site = IREE_HAL_MEMORY_SITE_QUEUE,
      .barrier_size = sizeof(uint64_t),
      .submit = NativeQueue::Submit,
  };
  auto storage = iree::hal::testing::MakeMappedTransferBuffer();
  auto payload = iree::hal::testing::MakeMappedTransferBuffer();
  QualifyBuffer(storage.get());
  QualifyBuffer(payload.get());
  if (!allow_access) {
    storage->allowed_access = access == IREE_HAL_MEMORY_ACCESS_READ
                                  ? IREE_HAL_MEMORY_ACCESS_WRITE
                                  : IREE_HAL_MEMORY_ACCESS_READ;
  }
  for (size_t i = 0; i < 64; ++i) {
    storage->device_contents[i] = static_cast<uint8_t>(i * 7);
    payload->device_contents[i] = static_cast<uint8_t>(i * 11);
  }
  DeviceFile file(storage.get());
  const auto expected_storage = storage->device_contents;
  const auto expected_payload = payload->device_contents;
  const iree_hal_file_binding_t file_binding = {file.get(), 17, 32};
  const iree_hal_buffer_binding_t buffer_binding = {payload.get(), 13, 32};
  const iree_hal_file_transfer_t transfer = {
      .row_length = 7,
      .row_count = 2,
      .slice_count = 1,
      .file_strides = {11, 0},
      .buffer_strides = {13, 0},
  };
  const iree_hal_file_transfer_list_t list = {.host = {1, &transfer}};
  const uint64_t before = 17, after = 29;
  uint64_t value = 1;
  iree_status_t status = iree_hal_amd_gpu_file_transfer_submit_batch(
      &executor, {}, {1, &completion_, &value}, access, {1, &file_binding},
      {1, &buffer_binding}, &list, &before, &after);
  iree_hal_file_release(file.get());
  if (!allow_access) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED, status);
    EXPECT_EQ(queue.submissions, 0u);
    EXPECT_TRUE(file.destroyed);
    IREE_ASSERT_OK(iree_hal_semaphore_query(completion_, &value));
    EXPECT_EQ(value, 0u);
    return;
  }
  IREE_ASSERT_OK(status);
  EXPECT_FALSE(file.destroyed);
  EXPECT_EQ(queue.pending_operations.size(), 2u);
  for (const auto& copy : queue.pending_operations) {
    EXPECT_EQ(copy.type, IREE_HAL_TRANSFER_OPERATION_TYPE_COPY);
    NativeQueue::Copy(&copy);
  }
  // This device-to-device copy owns both the file and its consumer boundary.
  EXPECT_EQ(queue.consumer_submission, 1u);
  queue.pending_callback.fn(queue.pending_callback.user_data, iree_ok_status(),
                            nullptr, true);
  EXPECT_EQ(queue.submissions, 1u);
  EXPECT_TRUE(file.destroyed);
  EXPECT_EQ(queue.host_owners.load(), 0);
  IREE_ASSERT_OK(iree_hal_semaphore_query(completion_, &value));
  EXPECT_EQ(value, 1u);
  auto expected = access == IREE_HAL_MEMORY_ACCESS_READ ? expected_payload
                                                        : expected_storage;
  for (size_t row = 0; row < 2; ++row) {
    if (access == IREE_HAL_MEMORY_ACCESS_READ) {
      std::memcpy(expected.data() + 13 + row * 13,
                  expected_storage.data() + 17 + row * 11, 7);
    } else {
      std::memcpy(expected.data() + 17 + row * 11,
                  expected_payload.data() + 13 + row * 13, 7);
    }
  }
  EXPECT_EQ(access == IREE_HAL_MEMORY_ACCESS_READ ? payload->device_contents
                                                  : storage->device_contents,
            expected);
}

TEST_F(FileTransferTest, DeviceFileReadCopiesNativeStorageAndRetainsOwner) {
  DeviceStorage(IREE_HAL_MEMORY_ACCESS_READ);
}

TEST_F(FileTransferTest, DeviceFileWriteCopiesNativeStorageAndRetainsOwner) {
  DeviceStorage(IREE_HAL_MEMORY_ACCESS_WRITE);
}

TEST_F(FileTransferTest, DeviceFileRejectsUnreadableStorageBeforeSubmission) {
  DeviceStorage(IREE_HAL_MEMORY_ACCESS_READ, false);
}

TEST_F(FileTransferTest, DeviceFileRejectsUnwritableStorageBeforeSubmission) {
  DeviceStorage(IREE_HAL_MEMORY_ACCESS_WRITE, false);
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
  Transfer(IREE_HAL_MEMORY_ACCESS_WRITE, false, 2);
}
TEST_F(FileTransferTest, DeferredDependencyRetainsParentThroughCompletion) {
  Transfer(IREE_HAL_MEMORY_ACCESS_READ, true, 0, true);
}

void FileTransferTest::FailedNativeWork(iree_hal_memory_access_t access) {
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
      .begin = NativeQueue::Begin,
      .end = NativeQueue::End,
      .waits = &waits_,
      .publication_mode = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
      .transfer_site = IREE_HAL_MEMORY_SITE_QUEUE,
      .barrier_size = sizeof(uint64_t),
      .submit = NativeQueue::Submit,
  };
  auto buffer = iree::hal::testing::MakeMappedTransferBuffer();
  QualifyBuffer(buffer.get());
  const std::array<uint8_t, 64> expected = contents_;
  if (access == IREE_HAL_MEMORY_ACCESS_WRITE) {
    buffer->device_contents = expected;
    contents_.fill(0);
  }
  const uint64_t before = 17, after = 29;
  uint64_t value = 1;
  const iree_hal_file_binding_t file_binding = {file_, 5, 50};
  const iree_hal_buffer_binding_t buffer_binding = {buffer.get(), 5, 50};
  const iree_hal_file_transfer_t transfer = {
      .row_length = 50,
      .row_count = 1,
      .slice_count = 1,
  };
  const iree_hal_file_transfer_list_t list = {.host = {1, &transfer}};
  IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit_batch(
      &executor, {}, {1, &completion_, &value}, access, {1, &file_binding},
      {1, &buffer_binding}, &list, &before, &after));
  iree_hal_file_release(file_);
  file_ = nullptr;
  EXPECT_EQ(queue.submissions, 1u);
  EXPECT_FALSE(file_released_);
  EXPECT_FALSE(device.destroyed);
  auto callback = queue.pending_callback;
  callback.fn(callback.user_data, iree_status_from_code(IREE_STATUS_ABORTED),
              nullptr, false);
  uint64_t completed_value = 0;
  IREE_EXPECT_OK(iree_hal_semaphore_query(completion_, &completed_value));
  EXPECT_EQ(completed_value, 0u);
  EXPECT_FALSE(file_released_);
  EXPECT_FALSE(device.destroyed);
  // The accepted copy can still access file bytes after an internal error.
  for (const auto& operation : queue.pending_operations) {
    NativeQueue::Copy(&operation);
  }
  const auto& actual = access == IREE_HAL_MEMORY_ACCESS_READ
                           ? buffer->device_contents
                           : contents_;
  for (size_t i = 5; i < 55; ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
  callback.fn(callback.user_data, iree_status_from_code(IREE_STATUS_ABORTED),
              nullptr, true);
  EXPECT_TRUE(file_released_);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_query(completion_, &completed_value));
  EXPECT_EQ(queue.host_owners.load(), 0);
  iree_hal_device_release(device.get());
  EXPECT_TRUE(device.destroyed);
  EXPECT_EQ(queue.submissions, 1u);
  // The native dependency owns fault reporting. Propagating its status through
  // the file operation must not report it a second time as a driver fault.
  EXPECT_EQ(failures, 0u);
}

TEST_F(FileTransferTest, FailedNativeReadPublishesOnlyAfterRetirement) {
  FailedNativeWork(IREE_HAL_MEMORY_ACCESS_READ);
}

TEST_F(FileTransferTest, FailedNativeWritePublishesOnlyAfterRetirement) {
  FailedNativeWork(IREE_HAL_MEMORY_ACCESS_WRITE);
}

TEST_F(FileTransferTest, UnqualifiedMemoryFileFailsBeforeNativeSubmission) {
  Device device;
  iree_hal_queue_family_t family = {.device = device.get()};
  NativeQueue queue;
  queue.queue_family = &family;
  iree_hal_amd_gpu_context_t context = {
      .event_sink = iree_hal_device_event_sink_discard(),
  };
  const iree_hal_amd_gpu_file_executor_t executor = {
      .queue = &queue,
      .context = &context,
      .captures = &captures_,
      .begin = NativeQueue::Begin,
      .end = NativeQueue::End,
      .waits = &waits_,
      .publication_mode = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
      .transfer_site = IREE_HAL_MEMORY_SITE_QUEUE,
      .barrier_size = sizeof(uint64_t),
      .submit = NativeQueue::Submit,
  };
  auto buffer = iree::hal::testing::MakeMappedTransferBuffer(
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, IREE_HAL_BUFFER_USAGE_TRANSFER);
  const uint64_t barrier = 0;
  uint64_t value = 1;
  const iree_hal_file_binding_t file_binding = {file_, 0, contents_.size()};
  const iree_hal_buffer_binding_t buffer_binding = {
      buffer.get(),
      0,
      contents_.size(),
  };
  const iree_hal_file_transfer_t transfer = {
      .row_length = contents_.size(),
      .row_count = 1,
      .slice_count = 1,
  };
  const iree_hal_file_transfer_list_t list = {.host = {1, &transfer}};
  IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit_batch(
      &executor, {}, {1, &completion_, &value}, IREE_HAL_MEMORY_ACCESS_READ,
      {1, &file_binding}, {1, &buffer_binding}, &list, &barrier, &barrier));
  uint64_t observed = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_semaphore_query(completion_, &observed));
  EXPECT_EQ(queue.submissions, 0u);
  EXPECT_EQ(queue.host_owners, 0);
}

void FileTransferTest::IndirectCountWithoutRecords(uint64_t count) {
  Device device;
  iree_hal_queue_family_t family = {.device = device.get()};
  NativeQueue queue;
  queue.queue_family = &family;
  queue.completion = completion_;
  iree_hal_amd_gpu_context_t context = {};
  const iree_hal_amd_gpu_file_executor_t executor = {
      .queue = &queue,
      .context = &context,
      .captures = &captures_,
      .begin = NativeQueue::Begin,
      .end = NativeQueue::End,
      .waits = &waits_,
      .publication_mode = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
      .transfer_site = IREE_HAL_MEMORY_SITE_QUEUE,
      .barrier_size = sizeof(uint64_t),
      .submit = NativeQueue::Submit,
  };
  auto count_buffer = iree::hal::testing::MakeMappedTransferBuffer();
  QualifyBuffer(count_buffer.get());
  iree_unaligned_store_le_u64(count_buffer->device_contents.data(), count);
  // This record buffer cannot be accessed by the native executor. Neither an
  // empty prefix nor a rejected count may try to resolve or read its bytes.
  auto records = iree::hal::testing::MakeMappedTransferBuffer(
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, IREE_HAL_BUFFER_USAGE_TRANSFER);
  records->host_binding_index = IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE;
  const iree_hal_file_transfer_list_t list = {
      .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT,
      .indirect = {{records.get(), 0, 64}, count_buffer.get(), 0},
  };
  const uint64_t before = 17, after = 29;
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_amd_gpu_file_transfer_submit_batch(
      &executor, {}, {1, &completion_, &value}, IREE_HAL_MEMORY_ACCESS_READ, {},
      {}, &list, &before, &after));
  uint64_t observed = 0;
  if (count) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          iree_hal_semaphore_query(completion_, &observed));
    EXPECT_EQ(queue.submissions, 1u);
  } else {
    IREE_ASSERT_OK(iree_hal_semaphore_query(completion_, &observed));
    EXPECT_EQ(observed, 1u);
    EXPECT_EQ(queue.submissions, 2u);
  }
  EXPECT_EQ(queue.host_owners, 0);
  iree_hal_device_release(device.get());
}

TEST_F(FileTransferTest, EmptyIndirectCountDoesNotAcquireRecords) {
  IndirectCountWithoutRecords(0);
}

TEST_F(FileTransferTest, OversizedIndirectCountFailsBeforeAcquiringRecords) {
  IndirectCountWithoutRecords(2);
}

}  // namespace
