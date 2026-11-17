// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/hal/driver/amd/gpu/queue/membership.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {
using iree::hal::cts::Ref;

class FileLifecycleTest : public iree::hal::amd::gpu::cts::GpuNativeTest {
 protected:
  iree_hal_device_event_sink_t event_sink() override {
    return {
        .fn =
            [](void* user_data, const iree_hal_device_event_t* event) {
              auto* self = static_cast<FileLifecycleTest*>(user_data);
              if (event->type == IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE) {
                ++self->driver_failures_;
              }
              iree_hal_device_event_sink_publish(
                  iree_hal_device_event_sink_stderr(), event);
            },
        .user_data = this,
    };
  }

  // Use the production compute family: PM4 on RDNA, AQL on CDNA. Transfer
  // families are exercised on the same device below.
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_UNKNOWN;
  }

  iree_status_t Import(iree_byte_span_t contents,
                       const iree::testing::TempFilePath* path,
                       iree_hal_file_t** out_file) {
    auto allocator = iree_allocator_system();
    iree_io_file_handle_t* handle = nullptr;
    if (path) {
      IREE_RETURN_IF_ERROR(iree_io_file_contents_write(
          path->path_view(),
          iree_make_const_byte_span(contents.data, contents.data_length),
          allocator));
      IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
          IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
              IREE_IO_FILE_MODE_ASYNC | IREE_IO_FILE_MODE_SHARE_READ |
              IREE_IO_FILE_MODE_SHARE_WRITE,
          path->path_view(), allocator, &handle));
    } else {
      IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
          IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE, contents, {},
          allocator, &handle));
    }
    iree_status_t status = iree_hal_file_import(
        device_, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_ALL, handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE,
        out_file);
    iree_io_file_handle_release(handle);
    return status;
  }

  void RoundTrip(const iree_hal_queue_family_t* family, bool platform_file) {
    constexpr size_t kLength = 8197;
    std::vector<uint8_t> source(kLength);
    std::vector<uint8_t> target(kLength + 2, 0xCD);
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = static_cast<uint8_t>((i * 29) ^ (i >> 7));
    }
    iree::testing::TempFilePath source_path("iree_file_close_source");
    iree::testing::TempFilePath target_path("iree_file_close_target");
    Ref<iree_hal_file_t> source_file, target_file;
    IREE_ASSERT_OK(Import(iree_make_byte_span(source.data(), source.size()),
                          platform_file ? &source_path : nullptr,
                          source_file.out()));
    IREE_ASSERT_OK(Import(iree_make_byte_span(target.data(), target.size()),
                          platform_file ? &target_path : nullptr,
                          target_file.out()));
    Ref<iree_hal_pool_t> pool;
    IREE_ASSERT_OK(iree::hal::cts::CreateQueueSlabPool(
        group_, family, IREE_HAL_BUFFER_USAGE_TRANSFER, {},
        iree_allocator_system(), pool.out()));
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(kLength + 3, 4),
        iree_infinite_timeout(), buffer.out()));
    Ref<iree_hal_queue_t> queue;
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    IREE_ASSERT_OK(iree_hal_queue_acquire(family, &params, queue.out()));
    Ref<iree_hal_semaphore_t> gate, timeline;
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(gate.out()));
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(timeline.out()));
    auto* gate_handle = gate.get();
    auto* timeline_handle = timeline.get();
    uint64_t one = 1, two = 2;

    IREE_ASSERT_OK(iree_hal_queue_read(queue, {1, &gate_handle, &one},
                                       {1, &timeline_handle, &one}, source_file,
                                       0, buffer, 3, kLength, nullptr, 0));
    iree_status_t write_status = iree_hal_queue_write(
        queue, {1, &timeline_handle, &one}, {1, &timeline_handle, &two}, buffer,
        3, target_file, 1, kLength, nullptr, 0);

    // Close exactly the production admission boundary after both public calls
    // return. Neither dependency is ready, so all native continuations must
    // run after closure. The family owns the exact protocol's lifecycle.
    const auto* native_family =
        reinterpret_cast<const iree_hal_amd_gpu_queue_family_t*>(family);
    native_family->lifecycle->close(queue);
    EXPECT_FALSE(native_family->lifecycle->is_host_idle(queue));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                          iree_hal_queue_read(queue, {}, {}, source_file, 0,
                                              buffer, 3, kLength, nullptr, 0));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        iree_hal_queue_write(queue, {}, {}, buffer, 3, target_file, 1, kLength,
                             nullptr, 0));
    source_file.reset();
    target_file.reset();
    buffer.reset();

    // Release the gate even when write admission failed, and join any accepted
    // read before returning. Stack file spans outlive every captured borrower.
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, one, nullptr));
    const uint64_t completed = iree_status_is_ok(write_status) ? two : one;
    iree_status_t status = iree_status_join(
        write_status, iree_hal_semaphore_wait(timeline, completed,
                                              iree_infinite_timeout(), 0));
    queue.reset();
    IREE_ASSERT_OK(status);

    if (platform_file) {
      iree_io_file_contents_t* contents = nullptr;
      IREE_ASSERT_OK(iree_io_file_contents_read(
          target_path.path_view(), iree_allocator_system(), &contents));
      target.assign(contents->buffer.data,
                    contents->buffer.data + contents->buffer.data_length);
      iree_io_file_contents_free(contents);
    }
    ASSERT_EQ(target.size(), kLength + 2);
    EXPECT_EQ(target.front(), 0xCD);
    EXPECT_EQ(target.back(), 0xCD);
    EXPECT_EQ(std::vector<uint8_t>(target.begin() + 1, target.end() - 1),
              source);
  }

  void FailureStaysWithOperation(const iree_hal_queue_family_t* family) {
    const uint64_t failures_before = driver_failures_.load();
    std::vector<uint8_t> source(8192);
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = static_cast<uint8_t>((i * 29) ^ (i >> 7));
    }
    iree::testing::TempFilePath path("iree_file_failure");
    Ref<iree_hal_file_t> file;
    IREE_ASSERT_OK(Import(iree_make_byte_span(source.data(), source.size()),
                          &path, file.out()));
    Ref<iree_hal_pool_t> pool;
    IREE_ASSERT_OK(iree::hal::cts::CreateQueueSlabPool(
        group_, family, IREE_HAL_BUFFER_USAGE_TRANSFER, {},
        iree_allocator_system(), pool.out()));
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(source.size(), 4),
        iree_infinite_timeout(), buffer.out()));
    Ref<iree_hal_queue_t> queue;
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    IREE_ASSERT_OK(iree_hal_queue_acquire(family, &params, queue.out()));
    Ref<iree_hal_semaphore_t> gate, failed, progress;
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(gate.out()));
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(failed.out()));
    ASSERT_NO_FATAL_FAILURE(CreateSemaphore(progress.out()));
    auto* gate_handle = gate.get();
    auto* failed_handle = failed.get();
    auto* progress_handle = progress.get();
    uint64_t one = 1, two = 2;
    IREE_ASSERT_OK(iree_hal_queue_read(queue, {1, &gate_handle, &one},
                                       {1, &failed_handle, &one}, file, 0,
                                       buffer, 0, source.size(), nullptr, 0));
    // Truncate after capture, before the wait permits access. The accepted
    // read must retire its prefix and return the ordinary file error.
    source.resize(4096);
    iree_status_t truncate_status = iree_io_file_contents_write(
        path.path_view(),
        iree_make_const_byte_span(source.data(), source.size()),
        iree_allocator_system());
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, one, nullptr));
    iree::Status read_status(
        iree_hal_semaphore_wait(failed, one, iree_infinite_timeout(), 0));
    IREE_ASSERT_OK(truncate_status);
    EXPECT_THAT(read_status,
                iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));

    // Independent work on the same queue and file remains usable. Retention
    // and publication use the same production path as the failed operation.
    IREE_ASSERT_OK(iree_hal_queue_read(queue, {}, {1, &progress_handle, &one},
                                       file, 0, buffer, 0, source.size(),
                                       nullptr, 0));
    std::vector<uint8_t> target(source.size(), 0xCD);
    iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffer,
                .target = target.data(),
                .length = target.size(),
            },
    };
    IREE_ASSERT_OK(iree_hal_queue_transfer(queue, {1, &progress_handle, &one},
                                           {1, &progress_handle, &two}, 1,
                                           &download, nullptr));
    file.reset();
    buffer.reset();
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(progress, two, iree_infinite_timeout(), 0));
    EXPECT_EQ(target, source);
    EXPECT_EQ(driver_failures_.load(), failures_before);
  }

  // Device diagnostics are independent of an individual file's completion.
  std::atomic<uint64_t> driver_failures_{0};
};

TEST_F(FileLifecycleTest, CapturesSurviveClosureAndIsolateFileFailures) {
  ASSERT_NO_FATAL_FAILURE(Create());
  const auto* families =
      iree_hal_device_spec_queues(iree_hal_device_spec(device_));
  size_t tested = 0;
  for (uint32_t i = 0; i < families->family_count; ++i) {
    if (!iree_any_bit_set(families->families[i].role_flags,
                          IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
      continue;
    }
    SCOPED_TRACE(i);
    const auto* family = iree_hal_device_queue_family(device_, i);
    for (bool platform_file : {false, true}) {
      SCOPED_TRACE(platform_file ? "platform file" : "memory file");
      ASSERT_NO_FATAL_FAILURE(RoundTrip(family, platform_file));
    }
    ASSERT_NO_FATAL_FAILURE(FailureStaysWithOperation(family));
    ++tested;
  }
  EXPECT_GT(tested, 0u);
  RecordProperty("queue_family_count", static_cast<int>(tested));
}

}  // namespace
