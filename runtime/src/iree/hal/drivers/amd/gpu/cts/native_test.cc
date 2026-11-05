// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <semaphore>
#include <string>
#include <utility>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/fabric/fabric.h"
#include "iree/hal/drivers/amd/gpu/cts/native_transfer_test_base.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/drivers/amd/gpu/queue/semaphore.h"
#include "iree/hal/drivers/amd/gpu/queue/signal.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory_backend.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::amd::gpu::cts::GpuNativeTest;
using iree::hal::amd::gpu::cts::GpuNativeTransferTest;

TEST_F(GpuNativeTest, TrimReleasesUnusedCaptureCache) {
  auto* context = context_;
  Create();
  if (HasFatalFailure()) {
    return;
  }
  // Construction may still be releasing builtin-upload metadata. Join those
  // callbacks before measuring host allocations owned by capture-cache trim.
  iree_hal_amd_gpu_context_join_memory(context);
  const size_t before = allocator_.live_count.load();
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  const size_t after = allocator_.live_count.load();
  // Each queue releases its preallocated capture backing; multiple captures
  // may share one slab. Repeating trim must not retain another idle batch.
  EXPECT_LE(after, before - options_.queue_count);
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_EQ(allocator_.live_count.load(), after);
  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&done);
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[0], iree_hal_semaphore_list_empty(), {1, &done, &value}, nullptr,
      IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, value, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_semaphore_release(done);
}

class GpuNativeRecordedTransferTest : public GpuNativeTransferTest {
 protected:
  iree_hal_buffer_ref_t Reference(size_t index, uint64_t offset,
                                  uint64_t length) {
    return GetParam().second
               ? iree_hal_make_indirect_buffer_ref(index + 1, offset, length)
               : iree_hal_make_buffer_ref(buffers_[index], offset, length);
  }
};

TEST_P(GpuNativeRecordedTransferTest,
       ReusableResidentTransfersCaptureBytesAndBindings) {
  std::vector<uint8_t> source(kLength);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = (i * 17 + 7) & 0xff;
  }
  auto expected = source;
  iree_hal_command_buffer_t* recording = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, GetParam().second ? 3 : 0,
      &recording));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      recording, source.data(), 0, Reference(0, 0, kLength), 0));
  std::fill(source.begin(), source.end(), 0xAB);
  const iree_hal_memory_barrier_t memory = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
  };
  const iree_hal_barrier_t dependency = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .memory_barrier_count = 1,
      .memory_barriers = &memory,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording, Reference(0, 0, IREE_HAL_WHOLE_BUFFER),
      Reference(1, 0, IREE_HAL_WHOLE_BUFFER), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
  for (uint32_t i = 0; i < 48; ++i) {
    const uint32_t pattern_length = 1u << (i % 3);
    const uint32_t pattern = 0x71B36DA5u + i;
    const size_t offset = i * 512 + pattern_length;
    const size_t length = 65 * pattern_length;
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        recording, Reference(1, offset, length), &pattern, pattern_length, 0));
    for (size_t j = 0; j < length; ++j) {
      expected[offset + j] = (uint8_t)(pattern >> ((j % pattern_length) * 8));
    }
  }
  // Empty remaining ranges execute no memory access, including when their
  // extents are resolved from an invocation binding table.
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording, Reference(0, kLength, IREE_HAL_WHOLE_BUFFER),
      Reference(1, kLength, IREE_HAL_WHOLE_BUFFER), 0));
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  uint64_t one = 1, two = 2, three = 3;
  std::array<iree_hal_buffer_binding_t, 3> bindings = {};
  bindings[1] = {buffers_[0], 0, IREE_HAL_WHOLE_BUFFER};
  bindings[2] = {buffers_[1], 0, IREE_HAL_WHOLE_BUFFER};
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queues_[0], {1, &gate_, &one}, {1, &timeline_, &one}, recording,
      {GetParam().second ? bindings.size() : 0, bindings.data()}, 0));
  std::swap(bindings[1], bindings[2]);
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queues_[1], {1, &timeline_, &one}, {1, &timeline_, &two}, recording,
      {GetParam().second ? bindings.size() : 0, bindings.data()}, 0));
  bindings.fill({});
  iree_hal_command_buffer_release(recording);
  std::vector<uint8_t> downloaded(kLength, 0xCC);
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = buffers_[GetParam().second ? 0 : 1],
              .target = downloaded.data(),
              .length = kLength,
          },
  };
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {1, &timeline_, &two},
                                         {1, &timeline_, &three}, 1, &download,
                                         nullptr));
  ReleaseBuffers();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, one, nullptr));
  Wait(three);
  EXPECT_EQ(downloaded, expected);
}

TEST_P(GpuNativeRecordedTransferTest,
       InterleavedDirectAndResidentTransfersPreserveEveryResult) {
  constexpr uint32_t kCount = 128;
  iree_hal_command_buffer_t* recording = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, GetParam().second ? 3 : 0,
      &recording));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording, Reference(0, 0, 4), Reference(1, 0, 4), 0));
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  std::array<iree_hal_buffer_binding_t, 3> bindings = {};
  bindings[1] = {buffers_[0], 0, IREE_HAL_WHOLE_BUFFER};
  bindings[2] = {buffers_[1], 0, IREE_HAL_WHOLE_BUFFER};
  // One caller publishes ahead without a host wait. Small native capacities
  // force wrap and retirement reuse while direct and GPU-produced packets
  // share the FIFO. Each direct copy preserves its preceding resident result.
  for (uint32_t i = 0; i < kCount; ++i) {
    const uint32_t pattern = 0xCAFE0000u + i;
    uint64_t before = i * 3, filled = before + 1, executed = before + 2,
             copied = before + 3;
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[0], {i ? 1u : 0u, &timeline_, &before},
        {1, &timeline_, &filled}, buffers_[0], 0, 4, &pattern, 4, nullptr, 0));
    IREE_ASSERT_OK(iree_hal_queue_execute(
        queues_[0], {1, &timeline_, &filled}, {1, &timeline_, &executed},
        recording, {GetParam().second ? bindings.size() : 0, bindings.data()},
        0));
    IREE_ASSERT_OK(iree_hal_queue_copy(
        queues_[0], {1, &timeline_, &executed}, {1, &timeline_, &copied},
        buffers_[1], 0, buffers_[1], (i + 1) * 4, 4, nullptr, 0));
  }
  iree_hal_command_buffer_release(recording);
  std::array<uint32_t, kCount> output = {};
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = buffers_[1],
              .source_offset = 4,
              .target = output.data(),
              .length = sizeof(output),
          },
  };
  uint64_t before = kCount * 3, downloaded = before + 1;
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {1, &timeline_, &before},
                                         {1, &timeline_, &downloaded}, 1,
                                         &download, nullptr));
  ReleaseBuffers();
  Wait(downloaded);
  for (uint32_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(output[i], 0xCAFE0000u + i) << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    BindingMode, GpuNativeRecordedTransferTest,
    ::testing::Values(std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, false),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, true),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, false),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, true)),
    [](const auto& info) {
      return std::string(info.param.first == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                             ? "Aql"
                             : "Pm4") +
             (info.param.second ? "Table" : "Direct");
    });

TEST_P(GpuNativeTransferTest, CapturedUpdatesAndDeferredHostTransfers) {
  std::vector<uint8_t> update(kLength), upload(kLength, 0);
  for (size_t i = 0; i < kLength; ++i) {
    update[i] = (i * 17 + 7) & 0xff;
  }
  const auto expected_update = update;
  std::vector<uint8_t> downloaded_update(kLength, 0xCC);
  std::vector<uint8_t> downloaded_upload(kLength, 0xCC);
  uint64_t ready = 1, produced = 1, downloaded = 2;
  const iree_hal_transfer_operation_t downloads[] = {
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = buffers_[0],
                  .target = downloaded_update.data(),
                  .length = kLength,
              },
      },
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = buffers_[1],
                  .target = downloaded_upload.data(),
                  .length = kLength,
              },
      },
  };
  // Publish the consumer first: it must leave the native FIFO available for
  // the producer and must not report completion before its host copies finish.
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queues_[1], {1, &timeline_, &produced}, {1, &timeline_, &downloaded},
      IREE_ARRAYSIZE(downloads), downloads, Barriers()));
  iree_hal_transfer_operation_t uploads[] = {
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
          .update =
              {
                  .source_buffer = update.data(),
                  .target_buffer = buffers_[0],
                  .length = kLength,
              },
      },
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload =
              {
                  .source = upload.data(),
                  .target_buffer = buffers_[1],
                  .length = kLength,
              },
      },
  };
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queues_[0], {1, &gate_, &ready}, {1, &timeline_, &produced},
      IREE_ARRAYSIZE(uploads), uploads, Barriers()));
  std::memset(uploads, 0, sizeof(uploads));
  std::fill(update.begin(), update.end(), 0xAB);
  for (size_t i = 0; i < kLength; ++i) {
    upload[i] = (i * 29 + 13) & 0xff;
  }
  ReleaseBuffers();
  uint64_t observed = 99;
  IREE_ASSERT_OK(iree_hal_semaphore_query(timeline_, &observed));
  EXPECT_EQ(observed, 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, ready, nullptr));
  Wait(downloaded);
  EXPECT_EQ(downloaded_update, expected_update);
  EXPECT_EQ(downloaded_upload, upload);
}

TEST_P(GpuNativeTransferTest,
       BulkTransfersWrapStagingAndRetainDeferredBuffers) {
  using iree::hal::cts::Ref;
  constexpr size_t kBulkLength = 17 * 1024 * 1024 + 37;
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(kBulkLength + 3, 4),
      iree_infinite_timeout(), buffer.out()));
  std::vector<uint8_t> source(kBulkLength, 0), target(kBulkLength + 2, 0xCC);
  for (uint64_t iteration = 0; iteration < 2; ++iteration) {
    SCOPED_TRACE(iteration);
    uint64_t ready = iteration + 1;
    uint64_t produced = iteration * 2 + 1;
    uint64_t downloaded = produced + 1;
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffer,
                .source_offset = 3,
                .target = target.data() + 1,
                .length = kBulkLength,
            },
    };
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        queues_[1], {1, &timeline_, &produced}, {1, &timeline_, &downloaded}, 1,
        &download, Barriers()));
    const iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload =
            {
                .source = source.data(),
                .target_buffer = buffer,
                .target_offset = 3,
                .length = kBulkLength,
            },
    };
    IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {1, &gate_, &ready},
                                           {1, &timeline_, &produced}, 1,
                                           &upload, Barriers()));
    // Upload reads these borrowed bytes only after the host dependency. The
    // second iteration starts at the previous short tail in the staging ring.
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = static_cast<uint8_t>((i * 29) ^ (i >> 9) ^ iteration);
    }
    std::fill(target.begin(), target.end(), 0xCC);
    if (iteration == 1) {
      buffer.reset();
    }
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, ready, nullptr));
    Wait(downloaded);
    EXPECT_EQ(target.front(), 0xCC);
    EXPECT_EQ(target.back(), 0xCC);
    EXPECT_TRUE(std::equal(source.begin(), source.end(), target.begin() + 1));
  }
}

TEST_P(GpuNativeTransferTest, SmallDownloadsRetainEveryHostDestination) {
  constexpr size_t kDownloadCount = 37;
  constexpr size_t kDownloadLength = 17;
  std::array<uint8_t, kDownloadCount * kDownloadLength> source;
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i * 29 + 7);
  }
  std::array<std::array<uint8_t, kDownloadLength + 2>, kDownloadCount> targets;
  std::array<iree_hal_transfer_operation_t, kDownloadCount> downloads = {};
  for (size_t i = 0; i < downloads.size(); ++i) {
    targets[i].fill(0xCC);
    downloads[i] = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffers_[0],
                .source_offset = 3 + i * kDownloadLength,
                .target = targets[i].data() + 1,
                .length = kDownloadLength,
            },
    };
  }
  uint64_t ready = 1, produced = 1, downloaded = 2;
  // The small argument ring bounds each download run. Distinct row identities
  // and guarded destinations prove each retirement scatters the correct bytes,
  // and completion covers every host copy across partial rows and ring reuse.
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queues_[1], {1, &timeline_, &produced}, {1, &timeline_, &downloaded},
      downloads.size(), downloads.data(), Barriers()));
  const iree_hal_transfer_operation_t update = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
      .update =
          {
              .source_buffer = source.data(),
              .target_buffer = buffers_[0],
              .target_offset = 3,
              .length = source.size(),
          },
  };
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {1, &gate_, &ready},
                                         {1, &timeline_, &produced}, 1, &update,
                                         Barriers()));
  ReleaseBuffers();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, ready, nullptr));
  Wait(downloaded);
  for (size_t i = 0; i < targets.size(); ++i) {
    const auto& target = targets[i];
    EXPECT_EQ(target.front(), 0xCC);
    EXPECT_EQ(target.back(), 0xCC);
    const auto first = source.begin() + i * kDownloadLength;
    EXPECT_TRUE(std::equal(first, first + kDownloadLength, target.begin() + 1));
  }
}

TEST_P(GpuNativeTransferTest,
       HostStagingVisibilityIsIndependentOfPayloadBoundaries) {
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_queue_barriers_t upload_barriers = {
      GetParam().second ? &empty : nullptr, &after_};
  const iree_hal_queue_barriers_t download_barriers = {
      &before_, GetParam().second ? &empty : nullptr};
  std::vector<uint8_t> source(kLength), target(kLength, 0xCC);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>((i * 31) ^ (i >> 8));
  }
  uint64_t uploaded = 1, downloaded = 2;
  // Optional payload boundaries cover only the device-to-device handoff.
  // Upload still acquires its host staging, and download still publishes its
  // staging before the retirement worker reads it.
  IREE_ASSERT_OK(iree_hal_queue_upload(
      queues_[0], {}, {1, &timeline_, &uploaded}, source.data(), buffers_[0], 0,
      source.size(), &upload_barriers));
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[1], {1, &timeline_, &uploaded}, {1, &timeline_, &downloaded},
      buffers_[0], 0, target.data(), target.size(), &download_barriers));
  ReleaseBuffers();
  Wait(downloaded);
  EXPECT_EQ(target, source);
}

TEST_P(GpuNativeTransferTest, DeferredHostCallForwardsOutputsToNestedTransfer) {
  std::vector<uint8_t> source(kLength), snapshot(kLength), output(kLength);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>((i * 31) ^ (i >> 8));
  }
  uint64_t uploaded = 1, downloaded = 2, completed = 3, returned = 4;
  IREE_ASSERT_OK(iree_hal_queue_upload(
      queues_[0], {}, {1, &timeline_, &uploaded}, source.data(), buffers_[0], 0,
      source.size(), Barriers()));
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[1], {1, &timeline_, &uploaded}, {1, &timeline_, &downloaded},
      buffers_[0], 0, snapshot.data(), snapshot.size(), Barriers()));
  struct Callback {
    // Downloaded input, kept live through the dependent final download.
    std::vector<uint8_t>* snapshot;
    // Known input used to verify download visibility before modification.
    const std::vector<uint8_t>* expected;
    // Nested transfer destination, retained by the enclosing test.
    iree_hal_buffer_t* target;
    // Original public output timeline, passed through without substitution.
    iree_hal_semaphore_t* completion;
    // Explicit queried transition or conservative defaults for nested work.
    const iree_hal_queue_barriers_t* barriers;
  } callback = {&snapshot, &source, buffers_[1], timeline_, Barriers()};
  const auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t arguments[4],
          iree_hal_host_call_context_t* context) {
        auto* callback = static_cast<Callback*>(user_data);
        EXPECT_EQ(*callback->snapshot, *callback->expected);
        EXPECT_EQ(context->signal_semaphore_list.count, 1u);
        EXPECT_EQ(context->signal_semaphore_list.semaphores[0],
                  callback->completion);
        EXPECT_EQ(context->signal_semaphore_list.payload_values[0], 3u);
        for (auto& value : *callback->snapshot) {
          value ^= 0xA5;
        }
        IREE_RETURN_IF_ERROR(iree_hal_queue_upload(
            context->queue, {}, context->signal_semaphore_list,
            callback->snapshot->data(), callback->target, 0,
            callback->snapshot->size(), callback->barriers));
        return iree_status_from_code(IREE_STATUS_DEFERRED);
      },
      &callback);
  const uint64_t arguments[4] = {0};
  IREE_ASSERT_OK(iree_hal_queue_host_call(
      queues_[0], {1, &timeline_, &downloaded}, {1, &timeline_, &completed},
      call, arguments, IREE_HAL_HOST_CALL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[1], {1, &timeline_, &completed}, {1, &timeline_, &returned},
      buffers_[1], 0, output.data(), output.size(), Barriers()));
  Wait(returned);
  for (auto& value : source) {
    value ^= 0xA5;
  }
  EXPECT_EQ(output, source);
}

TEST_P(GpuNativeTransferTest, BulkCopiesAndFillsCrossNativePacketBoundaries) {
  using iree::hal::cts::Ref;
  constexpr size_t kBulkLength = 32 * 1024 * 1024 + 37;
  Ref<iree_hal_buffer_t> source_buffer, target_buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(kBulkLength + 1, 4),
      iree_infinite_timeout(), source_buffer.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(kBulkLength + 8, 4),
      iree_infinite_timeout(), target_buffer.out()));
  std::vector<uint8_t> source(kBulkLength), target(kBulkLength + 8, 0);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>((i * 29) ^ (i >> 9) ^ (i >> 24));
  }
  const uint8_t guard = 0xCC;
  uint64_t completed = 0, uploaded = 1, filled = 2, copied = 3, downloaded = 4;
  iree_status_t status = iree_hal_queue_upload(
      queues_[0], {}, {1, &timeline_, &uploaded}, source.data(), source_buffer,
      1, source.size(), Barriers());
  if (iree_status_is_ok(status)) {
    completed = uploaded;
    status = iree_hal_queue_fill(
        queues_[1], {1, &timeline_, &uploaded}, {1, &timeline_, &filled},
        target_buffer, 0, target.size(), &guard, sizeof(guard), Barriers(), 0);
  }
  if (iree_status_is_ok(status)) {
    completed = filled;
    status = iree_hal_queue_copy(
        queues_[0], {1, &timeline_, &filled}, {1, &timeline_, &copied},
        source_buffer, 1, target_buffer, 3, source.size(), Barriers(), 0);
  }
  if (iree_status_is_ok(status)) {
    completed = copied;
    status = iree_hal_queue_download(
        queues_[1], {1, &timeline_, &copied}, {1, &timeline_, &downloaded},
        target_buffer, 0, target.data(), target.size(), Barriers());
  }
  if (iree_status_is_ok(status)) {
    completed = downloaded;
  }
  // Captures own both buffers through the last packet and any host copy.
  // Borrowed host bytes also survive an error after accepting a prefix.
  source_buffer.reset();
  target_buffer.reset();
  if (completed) {
    status = iree_status_join(
        status, iree_hal_semaphore_wait(timeline_, completed,
                                        iree_infinite_timeout(), 0));
  }
  IREE_ASSERT_OK(status);
  EXPECT_TRUE(std::equal(source.begin(), source.end(), target.begin() + 3));
  EXPECT_TRUE(std::all_of(target.begin(), target.begin() + 3,
                          [guard](uint8_t value) { return value == guard; }));
  EXPECT_TRUE(std::all_of(target.end() - 5, target.end(),
                          [guard](uint8_t value) { return value == guard; }));
}

TEST_P(GpuNativeTransferTest,
       ConcurrentTransfersResumeAfterStagingCapacityReturns) {
  // One batch proves capacity-return progress without a later submission.
  // Two lanes cover both passive and active waits; sustained contention lives
  // in the manual native transfer stress test.
  RunConcurrentTransfers(2, 1);
}

TEST_P(GpuNativeTransferTest, ManyFillsAndUnalignedCopiesRetireInPieces) {
  const uint8_t zero = 0, guard = 0xCC;
  uint64_t initialized = 1, filled = 2, copied = 3, downloaded = 4;
  const iree_hal_transfer_operation_t initialize[] = {
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill =
              {
                  .target_buffer = buffers_[0],
                  .length = kLength,
                  .pattern = &zero,
                  .pattern_length = 1,
              },
      },
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill =
              {
                  .target_buffer = buffers_[1],
                  .length = kLength,
                  .pattern = &guard,
                  .pattern_length = 1,
              },
      },
  };
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queues_[0], {}, {1, &timeline_, &initialized}, IREE_ARRAYSIZE(initialize),
      initialize, Barriers()));
  uint8_t pattern1 = 0x37;
  uint16_t pattern2 = 0x8A12;
  uint32_t pattern4 = 0xFEDCBA98;
  std::vector<iree_hal_transfer_operation_t> operations;
  std::vector<uint8_t> source(kLength, 0);
  for (size_t i = 0; i < 512; ++i) {
    const size_t offsets[] = {i * 64 + 1, i * 64 + 16, i * 64 + 36};
    const size_t lengths[] = {11, 14, 20};
    const void* patterns[] = {&pattern1, &pattern2, &pattern4};
    for (size_t p = 0; p < 3; ++p) {
      operations.push_back({
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill =
              {
                  .target_buffer = buffers_[0],
                  .target_offset = offsets[p],
                  .length = lengths[p],
                  .pattern = patterns[p],
                  .pattern_length = size_t{1} << p,
              },
      });
      for (size_t j = 0; j < lengths[p]; ++j) {
        source[offsets[p] + j] =
            static_cast<const uint8_t*>(patterns[p])[j % (size_t{1} << p)];
      }
    }
  }
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queues_[0], {1, &timeline_, &initialized}, {1, &timeline_, &filled},
      operations.size(), operations.data(), Barriers()));
  pattern1 = pattern2 = pattern4 = 0;
  operations.clear();
  IREE_ASSERT_OK(iree_hal_queue_copy(queues_[1], {1, &timeline_, &filled},
                                     {1, &timeline_, &copied}, buffers_[0], 1,
                                     buffers_[1], 3, kLength - 4, Barriers(),
                                     IREE_HAL_COPY_FLAG_NONE));
  std::vector<uint8_t> output(kLength, 0);
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = buffers_[1],
              .target = output.data(),
              .length = kLength,
          },
  };
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[1], {1, &timeline_, &copied},
                                         {1, &timeline_, &downloaded}, 1,
                                         &download, Barriers()));
  ReleaseBuffers();
  Wait(downloaded);
  std::vector<uint8_t> expected(kLength, guard);
  std::memcpy(expected.data() + 3, source.data() + 1, kLength - 4);
  EXPECT_EQ(output, expected);
}

TEST_P(GpuNativeTransferTest, CopiesPreserveEveryVectorAlignmentAndTail) {
  std::array<uint8_t, 32> source;
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i * 31 + 7);
  }
  const uint8_t guard = 0xCC;
  std::vector<uint8_t> expected(kLength, guard), output(kLength, 0);
  std::vector<iree_hal_transfer_operation_t> copies;
  for (size_t source_alignment = 0; source_alignment < 16; ++source_alignment) {
    for (size_t target_alignment = 0; target_alignment < 16;
         ++target_alignment) {
      for (size_t length : {size_t{16}, size_t{17}}) {
        const size_t target_offset = copies.size() * 64 + target_alignment;
        copies.push_back({
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
            .copy =
                {
                    .source_buffer = buffers_[0],
                    .source_offset = source_alignment,
                    .target_buffer = buffers_[1],
                    .target_offset = target_offset,
                    .length = length,
                },
        });
        std::memcpy(expected.data() + target_offset,
                    source.data() + source_alignment, length);
      }
    }
  }
  uint64_t uploaded = 1, filled = 2, primed = 3, copied = 4, downloaded = 5;
  IREE_ASSERT_OK(iree_hal_queue_upload(
      queues_[0], {}, {1, &timeline_, &uploaded}, source.data(), buffers_[0], 0,
      source.size(), Barriers()));
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], {1, &timeline_, &uploaded}, {1, &timeline_, &filled},
      buffers_[1], 0, kLength, &guard, sizeof(guard), Barriers(), 0));
  // A byte upload can leave less than one argument record at the end of the
  // staging allocation. The following batch must wrap and keep progressing.
  std::array<uint8_t, 1000> staging_prefix;
  std::fill(staging_prefix.begin(), staging_prefix.end(), guard);
  IREE_ASSERT_OK(iree_hal_queue_upload(queues_[1], {1, &timeline_, &filled},
                                       {1, &timeline_, &primed},
                                       staging_prefix.data(), buffers_[1], 0,
                                       staging_prefix.size(), Barriers()));
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queues_[1], {1, &timeline_, &primed}, {1, &timeline_, &copied},
      copies.size(), copies.data(), Barriers()));
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[1], {1, &timeline_, &copied}, {1, &timeline_, &downloaded},
      buffers_[1], 0, output.data(), output.size(), Barriers()));
  ReleaseBuffers();
  Wait(downloaded);
  EXPECT_EQ(output, expected);
}

TEST_P(GpuNativeTransferTest, ZeroLengthOperationsAreOnlyABarrier) {
  const iree_hal_transfer_operation_t operations[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL, .fill = {}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE, .update = {}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY, .copy = {}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD, .upload = {}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD, .download = {}},
  };
  uint64_t value = 1;
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queues_[0], {1, &gate_, &value}, {1, &timeline_, &value},
      IREE_ARRAYSIZE(operations), operations,
      GetParam().second ? &barriers : nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, value, nullptr));
  Wait(value);
}

INSTANTIATE_TEST_SUITE_P(
    Boundaries, GpuNativeTransferTest,
    ::testing::Values(std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, false),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, true),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, false),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, true)),
    [](const auto& info) {
      return std::string(info.param.first == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                             ? "Aql"
                             : "Pm4") +
             (info.param.second ? "Explicit" : "Default");
    });

TEST_F(GpuNativeTest, AdmissionFailureReachesSignalsAndPreservesQueueProgress) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  for (bool fail_before_capture : {false, true}) {
    SCOPED_TRACE(fail_before_capture);
    iree_hal_semaphore_t* gate = nullptr;
    iree_hal_semaphore_t* failed = nullptr;
    iree_hal_semaphore_t* done = nullptr;
    CreateSemaphore(&gate);
    CreateSemaphore(&failed);
    CreateSemaphore(&done);
    if (fail_before_capture) {
      iree_hal_semaphore_fail(
          gate, iree_make_status(IREE_STATUS_ABORTED, "producer failure"));
    }
    uint64_t value = 1;
    IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {1, &gate, &value},
                                          {1, &failed, &value}, nullptr, 0));
    IREE_ASSERT_OK(
        iree_hal_queue_barrier(queues_[0], {}, {1, &done, &value}, nullptr, 0));
    // Unresolved or failed work leaves later independent operations runnable.
    IREE_ASSERT_OK(iree_hal_semaphore_wait(done, value, iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    if (!fail_before_capture) {
      iree_hal_semaphore_fail(
          gate, iree_make_status(IREE_STATUS_ABORTED, "producer failure"));
    }
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_hal_semaphore_wait(failed, value, iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE));
    iree_hal_semaphore_release(gate);
    iree_hal_semaphore_release(failed);
    iree_hal_semaphore_release(done);
  }
}

TEST_F(GpuNativeTest, ManyFutureDependenciesUseOwnedWaitStorage) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  // The copied edge array exceeds the ordinary capture block, so admission
  // owns a separate wait record until the future producers have all arrived.
  std::array<iree_hal_semaphore_t*, 64> gates = {};
  for (auto& gate : gates) {
    CreateSemaphore(&gate);
  }
  std::array<uint64_t, 64> values;
  values.fill(1);
  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&done);
  uint64_t done_value = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[0], {gates.size(), gates.data(), values.data()},
      {1, &done, &done_value}, nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  values.fill(99);
  for (auto* gate : gates) {
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
    iree_hal_semaphore_release(gate);
  }
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      done, done_value, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_semaphore_release(done);
}

TEST_F(GpuNativeTest, FinalReleaseJoinsCompletedWorkAndCleanup) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&done);
  uint64_t value = 1;
  IREE_ASSERT_OK(
      iree_hal_queue_barrier(queues_[0], {}, {1, &done, &value}, nullptr, 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  iree_hal_semaphore_release(done);
  iree_hal_device_release(device_);
  device_ = nullptr;
  iree_hal_device_group_release(group_);
  group_ = nullptr;
  EXPECT_EQ(allocator_.live_count.load(), 0u);
}

TEST_F(GpuNativeTest, FinalReleaseAfterFailedAcceptedWait) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  iree_hal_semaphore_t* done = nullptr;
  iree_hal_semaphore_t* gate = nullptr;
  CreateSemaphore(&done);
  CreateSemaphore(&gate);
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {1, &gate, &value},
                                        {1, &done, &value}, nullptr, 0));
  iree_hal_semaphore_fail(gate, iree_make_status(IREE_STATUS_ABORTED,
                                                 "accepted dependency failed"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  iree_hal_semaphore_release(gate);
  iree_hal_semaphore_release(done);
  iree_hal_device_release(device_);
  device_ = nullptr;
  iree_hal_device_group_release(group_);
  group_ = nullptr;
  EXPECT_EQ(allocator_.live_count.load(), 0u);
}

TEST_F(GpuNativeTest, FailedQueueConstructionReleasesItsOwners) {
  options_.queue.argument_capacity = 0;
  auto params = iree_hal_device_create_params_default();
  params.proactor_pool = proactors_;
  auto* context = context_;
  context_ = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_device_create(context, IREE_SV("rejected native queue"),
                                     &options_, &params, allocator_.allocator(),
                                     &device_));
  EXPECT_EQ(device_, nullptr);
  EXPECT_EQ(allocator_.live_count.load(), 0u);
}

using iree::hal::amd::gpu::cts::CompletionPause;

class GpuNativeCompletionTest
    : public GpuNativeTest,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {
 protected:
  amdf_queue_command_type_t command_type() const override { return GetParam(); }
};

TEST_P(GpuNativeCompletionTest, IndependentCompletionDuringHostCallback) {
  const auto& family = context_->families[options_.queue_family_ordinal];
  if (!(family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) ||
      !(family.user_queue_capabilities &
        AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER)) {
    GTEST_SKIP() << "Caller retirement assistance requires USER publication";
  }
  const void* native_domain = context_;
  Create();
  if (HasFatalFailure()) {
    return;
  }
  using iree::hal::cts::Ref;
  Ref<iree_hal_semaphore_t> callback, independent, marker;
  CreateSemaphore(callback.out());
  CreateSemaphore(independent.out());
  CreateSemaphore(marker.out());
  iree_hal_semaphore_t* callback_handle = callback;
  iree_hal_semaphore_t* independent_handle = independent;
  iree_hal_semaphore_t* marker_handle = marker;
  uint64_t one = 1;
  CompletionPause pause{
      reinterpret_cast<iree_async_semaphore_t*>(callback_handle)};
  IREE_ASSERT_OK(pause.Arm(one));
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[0], {}, {1, &callback_handle, &one}, nullptr, 0));
  pause.entered.acquire();
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[1], {}, {1, &independent_handle, &one}, nullptr, 0));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[1],
                                        {1, &independent_handle, &one},
                                        {1, &marker_handle, &one}, nullptr, 0));
  // The native semaphore point precedes the PM4 host retirement record. Join
  // a later point on the same FIFO before testing the single retirement-assist
  // attempt, so this tests source ownership rather than that publication gap.
  iree_hal_amd_gpu_semaphore_wait_t producer = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_acquire_wait(
      marker, one, native_domain, &producer));
  std::unique_ptr<iree_hal_amd_gpu_signal_t,
                  decltype(&iree_hal_amd_gpu_signal_release)>
      signal(producer.signal, iree_hal_amd_gpu_signal_release);
  ASSERT_EQ(producer.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
  const int64_t pending = GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL ? 1 : 0;
  while (iree_atomic_load(
             reinterpret_cast<iree_atomic_int64_t*>(signal->host_value),
             iree_memory_order_acquire) == pending) {
    std::this_thread::yield();
  }
  // The active waiter retires its own independent native completion while
  // the event worker is still inside the first queue's detached notification.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      independent, one, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_ACTIVE));
  uint64_t value = 0;
  IREE_EXPECT_OK(iree_hal_semaphore_query(independent, &value));
  EXPECT_EQ(value, one);
  pause.Join();
}

TEST_P(GpuNativeCompletionTest, HostCallbackCanAcquireAndReleaseAnIdleQueue) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  using iree::hal::cts::Ref;
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  iree_hal_semaphore_t* done_handle = done;
  uint64_t one = 1;
  const auto call = iree_hal_make_host_call(
      +[](void*, const uint64_t[4], iree_hal_host_call_context_t* context) {
        iree_hal_queue_params_t params;
        iree_hal_queue_params_initialize(&params);
        iree_hal_queue_t* acquired = nullptr;
        IREE_RETURN_IF_ERROR(iree_hal_queue_acquire(
            iree_hal_queue_family(context->queue), &params, &acquired));
        iree_hal_queue_release(acquired);
        return iree_ok_status();
      },
      nullptr);
  const uint64_t arguments[4] = {0};
  IREE_ASSERT_OK(
      iree_hal_queue_host_call(queues_[0], {}, {1, &done_handle, &one}, call,
                               arguments, IREE_HAL_HOST_CALL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, one, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
}

INSTANTIATE_TEST_SUITE_P(Protocol, GpuNativeCompletionTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
                         [](const auto& info) {
                           return info.param == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                                      ? "Aql"
                                      : "Pm4";
                         });

TEST_F(GpuNativeTest, NativeConsumerPreservesProducerRetirementFailure) {
  const void* native_domain = context_;
  Create();
  if (HasFatalFailure()) {
    return;
  }
  using iree::hal::cts::Ref;
  Ref<iree_hal_semaphore_t> paused, producer, consumer;
  CreateSemaphore(paused.out());
  CreateSemaphore(producer.out());
  CreateSemaphore(consumer.out());
  iree_hal_semaphore_t* paused_handle = paused;
  iree_hal_semaphore_t* producer_handle = producer;
  iree_hal_semaphore_t* consumer_handle = consumer;
  uint64_t one = 1;

  // A public completion callback holds the device's single completion owner.
  // Native submission remains live, so both operations below can reach the
  // hardware before any host observer retires the producer.
  CompletionPause pause{
      reinterpret_cast<iree_async_semaphore_t*>(paused_handle)};
  IREE_ASSERT_OK(pause.Arm(one));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {},
                                        {1, &paused_handle, &one}, nullptr, 0));
  pause.entered.acquire();
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[1], {}, {1, &producer_handle, &one}, nullptr, 0));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {1, &producer_handle, &one},
                                        {1, &consumer_handle, &one}, nullptr,
                                        0));
  // A published native consumer proves the dependency was admitted without
  // waiting on the producer's host timeline, which is still held above.
  iree_hal_amd_gpu_semaphore_wait_t published = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_acquire_wait(
      consumer, one, native_domain, &published));
  EXPECT_EQ(published.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
  iree_hal_amd_gpu_signal_release(published.signal);
  iree_hal_semaphore_fail(
      producer, iree_make_status(IREE_STATUS_ABORTED, "producer failed"));
  pause.Join();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_wait(consumer, one, iree_infinite_timeout(), 0));
  // Fixture teardown joins device ownership and checks complete reclamation.
}

TEST_F(GpuNativeTest, NonfinalReleaseFromCompletionCallback) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&gate);
  CreateSemaphore(&done);
  uint64_t value = 1;
  struct Callback {
    // Explicit application reference, subordinate to the outer device owner.
    iree_hal_device_t* device;
    // Joins the application's callback before returning its outer references.
    std::binary_semaphore finished{0};
  } callback{device_};
  iree_hal_device_retain(device_);
  iree_async_semaphore_timepoint_t timepoint = {
      .callback =
          [](void* user_data, iree_async_semaphore_timepoint_t*,
             iree_status_t status) {
            IREE_CHECK_OK(status);
            auto* callback = static_cast<Callback*>(user_data);
            iree_hal_device_release(callback->device);
            callback->finished.release();
          },
      .user_data = &callback,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(done), value, &timepoint));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {1, &gate, &value},
                                        {1, &done, &value}, nullptr, 0));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, value, nullptr));
  iree_hal_semaphore_release(gate);
  callback.finished.acquire();
  iree_status_free(iree_hal_semaphore_wait(done, value, iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_semaphore_release(done);
  iree_hal_device_release(device_);
  device_ = nullptr;
  iree_hal_device_group_release(group_);
  group_ = nullptr;
  EXPECT_EQ(allocator_.live_count.load(), 0u);
}

TEST_F(GpuNativeTest, NonfinalReleaseFromFailedDependencyCallback) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&gate);
  CreateSemaphore(&done);
  uint64_t value = 1;
  struct Callback {
    // Explicit application reference, subordinate to the outer device owner.
    iree_hal_device_t* device;
    // Joins the application's callback before returning its outer references.
    std::binary_semaphore finished{0};
  } callback{device_};
  iree_hal_device_retain(device_);
  iree_async_semaphore_timepoint_t timepoint = {
      .callback =
          [](void* user_data, iree_async_semaphore_timepoint_t*,
             iree_status_t status) {
            IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
            auto* callback = static_cast<Callback*>(user_data);
            iree_hal_device_release(callback->device);
            callback->finished.release();
          },
      .user_data = &callback,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(done), value, &timepoint));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {1, &gate, &value},
                                        {1, &done, &value}, nullptr, 0));
  // A callback may return an application reference while its outer owner
  // keeps the device alive through cleanup.
  iree_hal_semaphore_fail(gate, iree_make_status(IREE_STATUS_ABORTED,
                                                 "accepted dependency failed"));
  iree_hal_semaphore_release(gate);
  callback.finished.acquire();
  iree_status_free(iree_hal_semaphore_wait(done, value, iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_semaphore_release(done);
  iree_hal_device_release(device_);
  device_ = nullptr;
  iree_hal_device_group_release(group_);
  group_ = nullptr;
  EXPECT_EQ(allocator_.live_count.load(), 0u);
}

class GpuNativeAtomicTest : public GpuNativeTest,
                            public ::testing::WithParamInterface<
                                std::pair<amdf_queue_command_type_t, bool>> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return GetParam().first;
  }
};

TEST_P(GpuNativeAtomicTest, RecordedAtomicsFollowReplacementBacking) {
  using iree::hal::cts::Ref;
  Create();
  if (HasFatalFailure()) {
    return;
  }
  iree_hal_pool_family_access_t access = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
      .atomics =
          {
              .queue =
                  {
                      .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
                  },
          },
  };
  const iree_hal_slab_pool_options_t pool_options = {};
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(
      iree_hal_slab_pool_create(group_, {1, &access, {}}, &pool_options,
                                iree_allocator_system(), source.out()));
  iree_hal_pool_t* source_pool = source.get();
  access.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  const iree_hal_pool_scope_t scope = {1, &access, {}};
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, iree_allocator_system(),
      &capabilities));
  ASSERT_NE(capabilities.max_byte_length, 0u);
  const auto unit = capabilities.reservation_granularity;
  const iree_hal_virtual_buffer_options_t options = {.byte_length = unit};
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(
      iree_hal_virtual_buffer_create(group_, scope, {1, &source_pool}, &options,
                                     iree_allocator_system(), target.out()));
  iree_hal_memory_scope_t queue_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_QUEUE, access.family}, &queue_scope));
  const auto atomic_operations =
      iree_hal_buffer_query_atomic_operations(target, queue_scope);
  ASSERT_TRUE(iree_all_bits_set(atomic_operations.device_scope_32,
                                IREE_HAL_ATOMIC_OPERATION_FLAG_STORE));
  const auto transitions = iree_hal_buffer_transition_table(target);
  iree_hal_memory_transition_pair_t pair;
  IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
      transitions, queue_scope, queue_scope, IREE_HAL_MEMORY_TRANSITION_RELEASE,
      &pair));
  const auto pair_info =
      iree_hal_memory_transition_query_info(transitions, pair);
  EXPECT_EQ(pair_info.producer_atomics.device_scope_32,
            atomic_operations.device_scope_32);
  EXPECT_EQ(pair_info.consumer_atomics.device_scope_32,
            atomic_operations.device_scope_32);
  const bool supports_add =
      iree_all_bits_set(atomic_operations.device_scope_32,
                        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD);
  RecordProperty("atomic_operations", supports_add ? "STORE+ADD" : "STORE");
  Ref<iree_hal_command_buffer_t> commands;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      access.family, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_ATOMIC, GetParam().second ? 1 : 0,
      commands.out()));
  const auto reference =
      GetParam().second
          ? iree_hal_make_indirect_buffer_ref(0, 0, sizeof(uint32_t))
          : iree_hal_make_buffer_ref(target, 0, sizeof(uint32_t));
  const iree_hal_buffer_binding_t binding = {target, 0, sizeof(uint32_t)};
  const iree_hal_buffer_binding_table_t table = {GetParam().second ? 1u : 0u,
                                                 &binding};
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(commands));
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
      commands, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, reference,
      {
          .value = 99,
          .flags = IREE_HAL_ATOMIC_FLAG_RELEASE,
          .width = IREE_HAL_ATOMIC_WIDTH_32,
      }));
  if (supports_add) {
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
        commands, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, reference,
        {
            .operand = 1,
            .flags =
                IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
            .width = IREE_HAL_ATOMIC_WIDTH_32,
            .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
        }));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(commands));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(source, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  std::array<Ref<iree_hal_buffer_t>, 2> backings;
  Ref<iree_hal_semaphore_t> completion;
  CreateSemaphore(completion.out());
  auto* timeline = completion.get();
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};
  for (size_t i = 0; i < backings.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        source, iree_hal_make_pool_reservation_request(unit, 0),
        iree_infinite_timeout(), backings[i].out()));
    uint64_t initialized = i * 4 + 1, mapped = initialized + 1,
             executed = initialized + 2, read = initialized + 3;
    const uint32_t initial = 40 + i;
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[0], {}, {1, &timeline, &initialized}, backings[i], 0,
        sizeof(initial), &initial, sizeof(initial), nullptr, 0));
    const iree_hal_memory_mapping_t mapping = {
        .target = target,
        .scope = all,
        .target_range = {0, unit},
        .source = backings[i],
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
    };
    IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {1, &timeline, &initialized},
                                      {1, &timeline, &mapped}, 1, &mapping, 0));
    IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {1, &timeline, &mapped},
                                          {1, &timeline, &executed}, commands,
                                          table, 0));
    uint32_t result = 0;
    IREE_ASSERT_OK(iree_hal_queue_download(
        queues_[0], {1, &timeline, &executed}, {1, &timeline, &read},
        backings[i], 0, &result, sizeof(result), nullptr));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(timeline, read, iree_infinite_timeout(), 0));
    EXPECT_EQ(result, supports_add ? 100u : 99u);
  }
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, all));
}

INSTANTIATE_TEST_SUITE_P(
    Protocol, GpuNativeAtomicTest,
    ::testing::Values(std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, false),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, true),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, false),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, true)),
    [](const auto& info) {
      return std::string(info.param.first == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                             ? "Aql"
                             : "Pm4") +
             (info.param.second ? "Indirect" : "Direct");
    });

class GpuNativeVirtualBufferTest : public GpuNativeTest {
 protected:
  // Exercise the production protocol selected for each physical target.
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_UNKNOWN;
  }

  // Qualify the complete source contract without acquiring physical backing.
  // The public constructor must still succeed when a factory supplies a plan.
  iree_status_t QueryPoolSupport(iree_hal_pool_scope_t scope,
                                 const iree_hal_slab_pool_options_t& options,
                                 bool* out_supported) {
    *out_supported = false;
    const iree_hal_slab_pool_plan_callback_t callback = {
        .fn =
            +[](void* user_data, iree_hal_slab_pool_plan_t* plan) {
              *static_cast<bool*>(user_data) = true;
              iree_hal_slab_pool_plan_destroy(plan);
              return iree_ok_status();
            },
        .user_data = out_supported,
    };
    const auto* backend = iree_hal_device_memory_backend(device_);
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < backend->factory_count && iree_status_is_ok(status); ++i) {
      const auto* factory = backend->factories[i];
      status = factory->query(factory->self, group_, scope, &options, callback,
                              iree_allocator_system());
    }
    return status;
  }
};

TEST_F(GpuNativeVirtualBufferTest,
       BatchedMappingsReplaceProtectAndReleaseNativeBacking) {
  using iree::hal::cts::Ref;
  Create();
  if (HasFatalFailure()) {
    return;
  }
  const iree_hal_pool_family_access_t access = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS) |
                    (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE),
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, {1, &access, {}}, &options, iree_allocator_system(), pool.out()));
  const auto source = iree_hal_pool_query_mapping_requirements(pool);
  auto target_access = access;
  target_access.interfaces = UINT64_C(1)
                             << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  const iree_hal_pool_scope_t scope = {1, &target_access, {}};
  iree_hal_pool_t* source_pool = pool.get();
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, iree_allocator_system(),
      &capabilities));
  ASSERT_NE(capabilities.max_byte_length, 0u);
  EXPECT_EQ(capabilities.operations, IREE_HAL_MEMORY_MAPPING_OPERATIONS_ALL);
  const uint64_t unit =
      iree_max(source.minimum_granularity,
               iree_max(capabilities.mapping_granularity,
                        capabilities.reservation_granularity));
  const iree_hal_virtual_buffer_options_t reservation_options = {
      .byte_length = 3 * unit,
      .mapping_granularity = unit,
  };
  Ref<iree_hal_buffer_t> reservation;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, scope, {1, &source_pool}, &reservation_options,
      iree_allocator_system(), reservation.out()));
  iree_hal_memory_scope_t program_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, access.family}, &program_scope));
  std::array<Ref<iree_hal_buffer_t>, 2> backing;
  for (auto& buffer : backing) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(5 * unit, 0),
        iree_infinite_timeout(), buffer.out()));
  }
  Ref<iree_hal_semaphore_t> progress;
  CreateSemaphore(progress.out());
  iree_hal_semaphore_t* timeline = progress;
  uint64_t value = 0;
  std::array<std::vector<uint32_t>, 2> expected;
  for (size_t i = 0; i < backing.size(); ++i) {
    const uint32_t poison = 0xEFEF0000u + i;
    expected[i].assign(5 * unit / sizeof(uint32_t), poison);
    uint64_t before = value++;
    IREE_ASSERT_OK(
        iree_hal_queue_fill(queues_[0], {before ? 1u : 0u, &timeline, &before},
                            {1, &timeline, &value}, backing[i], 0, 5 * unit,
                            &poison, sizeof(poison), nullptr, 0));
  }
  std::array<iree_hal_memory_mapping_t, 3> mappings;
  for (uint32_t epoch = 0; epoch < 3; ++epoch) {
    SCOPED_TRACE(epoch);
    std::vector<uint32_t> expected_mapped(3 * unit / sizeof(uint32_t));
    std::array<Ref<iree_hal_buffer_t>, 3> source_views;
    for (uint32_t i = 0; i < mappings.size(); ++i) {
      const uint32_t index = (epoch + i) % 2;
      IREE_ASSERT_OK(iree_hal_buffer_subspan(backing[index], (1 + i) * unit,
                                             unit, iree_allocator_system(),
                                             source_views[i].out()));
      mappings[i] = {
          .target = reservation,
          .scope = program_scope,
          .target_range = {i * unit, unit},
          .source = source_views[i],
          .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
          .operation = epoch ? IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE
                             : IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
          .previous_source =
              epoch ? backing[(epoch - 1 + i) % 2].get() : nullptr,
          .previous_source_offset = epoch ? (1 + i) * unit : 0,
      };
    }
    uint64_t before = value++;
    IREE_ASSERT_OK(iree_hal_queue_map(
        queues_[0], {1, &timeline, &before}, {1, &timeline, &value},
        mappings.size(), mappings.data(),
        epoch == 1 ? IREE_HAL_QUEUE_MAP_FLAG_BORROW_BUFFERS
                   : IREE_HAL_QUEUE_MAP_FLAG_NONE));
    for (uint32_t i = 0; i < mappings.size(); ++i) {
      if (epoch != 1) {
        source_views[i].reset();
      }
      // Later protection supplies the original source coordinates itself.
      mappings[i].source = backing[(epoch + i) % 2];
      mappings[i].source_offset = (1 + i) * unit;
      mappings[i].previous_source = nullptr;
      mappings[i].previous_source_offset = 0;
      const uint32_t pattern = 0x01010000u + epoch * 16 + i;
      uint64_t filled = value++;
      IREE_ASSERT_OK(iree_hal_queue_fill(
          queues_[1], {1, &timeline, &filled}, {1, &timeline, &value},
          reservation, i * unit, unit, &pattern, sizeof(pattern), nullptr, 0));
      auto& words = expected[(epoch + i) % 2];
      std::fill(words.begin() + (1 + i) * unit / sizeof(uint32_t),
                words.begin() + (2 + i) * unit / sizeof(uint32_t), pattern);
      std::fill(expected_mapped.begin() + i * unit / sizeof(uint32_t),
                expected_mapped.begin() + (i + 1) * unit / sizeof(uint32_t),
                pattern);
    }
    if (!epoch) {
      Ref<iree_hal_semaphore_t> gate, rejected;
      CreateSemaphore(gate.out());
      CreateSemaphore(rejected.out());
      iree_hal_semaphore_t* gate_handle = gate;
      iree_hal_semaphore_t* rejected_handle = rejected;
      uint64_t one = 1;
      auto removals = mappings;
      for (auto& removal : removals) {
        removal.previous_source = removal.source;
        removal.previous_source_offset = removal.source_offset;
        removal.source = nullptr;
        removal.source_offset = 0;
        removal.access = 0;
        removal.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
      }
      IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {1, &gate_handle, &one},
                                        {1, &rejected_handle, &one},
                                        removals.size(), removals.data(), 0));
      iree_hal_semaphore_fail(gate,
                              iree_status_from_code(IREE_STATUS_CANCELLED));
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_CANCELLED,
          iree_hal_semaphore_wait(rejected, one, iree_infinite_timeout(), 0));
    }
    for (auto& mapping : mappings) {
      mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT;
      mapping.access = IREE_HAL_MEMORY_ACCESS_READ;
    }
    uint64_t protected_before = value++;
    IREE_ASSERT_OK(iree_hal_queue_map(
        queues_[0], {1, &timeline, &protected_before}, {1, &timeline, &value},
        mappings.size(), mappings.data(), 0));
    std::vector<uint32_t> observed_mapped(expected_mapped.size());
    uint64_t read = value++;
    IREE_ASSERT_OK(iree_hal_queue_download(
        queues_[1], {1, &timeline, &read}, {1, &timeline, &value}, reservation,
        0, observed_mapped.data(), 3 * unit, nullptr));
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(progress, value, iree_infinite_timeout(), 0));
    EXPECT_EQ(observed_mapped, expected_mapped);
    std::vector<uint32_t> observed(5 * unit / sizeof(uint32_t));
    for (size_t i = 0; i < backing.size(); ++i) {
      uint64_t copied = value++;
      IREE_ASSERT_OK(iree_hal_queue_download(
          queues_[1], {1, &timeline, &copied}, {1, &timeline, &value},
          backing[i], 0, observed.data(), 5 * unit, nullptr));
      IREE_ASSERT_OK(
          iree_hal_semaphore_wait(progress, value, iree_infinite_timeout(), 0));
      EXPECT_EQ(observed, expected[i]);
    }
  }
  for (auto& mapping : mappings) {
    mapping.previous_source = mapping.source;
    mapping.previous_source_offset = mapping.source_offset;
    mapping.source = nullptr;
    mapping.source_offset = 0;
    mapping.access = 0;
    mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
  }
  uint64_t before = value++;
  IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {1, &timeline, &before},
                                    {1, &timeline, &value}, mappings.size(),
                                    mappings.data(), 0));
  // The captured target owns its addresses through native removal. Physical
  // backing remains explicitly owned until that checked removal completes.
  reservation.reset();
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(progress, value, iree_infinite_timeout(), 0));
  for (auto& buffer : backing) {
    buffer.reset();
  }
  iree_hal_pool_stats_t statistics;
  iree_hal_pool_query_stats(pool, &statistics);
  EXPECT_EQ(statistics.reservation_count, 0u);
}

TEST_F(GpuNativeVirtualBufferTest,
       MixedBackingPreservesCommonHostAndGpuAddresses) {
  using iree::hal::cts::Ref;
  Create();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  uint32_t physical_node = IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID;
  const auto* topology = iree_hal_device_group_topology(group_);
  for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
       ++i) {
    const auto* node = iree_hal_topology_node_at(topology, i);
    if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
        node->device_ordinal ==
            iree_hal_device_topology_info(device_)->topology_index) {
      physical_node = node->ordinal;
      break;
    }
  }
  ASSERT_NE(physical_node, IREE_HAL_TOPOLOGY_NODE_ORDINAL_INVALID);
  iree_hal_pool_family_access_t family = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
  };
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  pool_options.placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, physical_node};
  bool supported = false;
  IREE_ASSERT_OK(QueryPoolSupport({1, &family, {}}, pool_options, &supported));
  if (!supported) {
    GTEST_SKIP() << "no device-local remappable backing route";
  }
  Ref<iree_hal_pool_t> local_pool;
  IREE_ASSERT_OK(
      iree_hal_slab_pool_create(group_, {1, &family, {}}, &pool_options,
                                iree_allocator_system(), local_pool.out()));
  pool_options.placement = {};
  // The CPU-accessible half requires cached host backing. Public visibility
  // alone can select a device-local allocation with a fixed CPU aperture.
  const iree_hal_pool_scope_t scope = {
      1,
      &family,
      {IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_MAPPING_MODE_PERSISTENT,
       IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK}};
  IREE_ASSERT_OK(QueryPoolSupport(scope, pool_options, &supported));
  if (!supported) {
    GTEST_SKIP() << "no cached host-accessible remappable backing route";
  }
  Ref<iree_hal_pool_t> shared_pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options,
                                           iree_allocator_system(),
                                           shared_pool.out()));
  family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  iree_hal_pool_t* sources[] = {local_pool, shared_pool};
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, {IREE_ARRAYSIZE(sources), sources},
      iree_allocator_system(), &capabilities));
  if (!capabilities.max_byte_length ||
      !iree_all_bits_set(capabilities.supported_flags,
                         IREE_HAL_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS)) {
    GTEST_SKIP() << "mixed backing does not support a common host/GPU address";
  }
  const auto unit = std::max(capabilities.reservation_granularity,
                             capabilities.mapping_granularity);
  // Retain both generations until all mappings are removed. The virtual buffer
  // owns addresses, while the caller owns the physical backing and its
  // lifetime.
  std::array<std::array<Ref<iree_hal_buffer_t>, 2>, 2> backings;
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = 2 * unit,
      .mapping_granularity = unit,
      .flags = IREE_HAL_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
  };
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      group_, scope, {IREE_ARRAYSIZE(sources), sources}, &options,
      iree_allocator_system(), target.out()));
  iree_hal_memory_scope_t host, program;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_HOST, nullptr}, &host));
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family.family}, &program));
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};
  EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(target, program,
                                                               local_pool));
  EXPECT_FALSE(iree_hal_virtual_buffer_query_pool_compatibility(target, host,
                                                                local_pool));
  EXPECT_TRUE(iree_hal_virtual_buffer_query_pool_compatibility(target, all,
                                                               shared_pool));
  const iree_hal_buffer_native_binding_slot_t slot = {
      .index = target.get()
                   ->memory.contract->scopes[program.id]
                   .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS],
      .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const auto address =
      iree_hal_buffer_native_binding(target, slot).device_address;
  Ref<iree_hal_semaphore_t> completion;
  CreateSemaphore(completion.out());
  iree_hal_semaphore_t* timeline = completion;
  for (size_t epoch = 0; epoch < backings.size(); ++epoch) {
    for (size_t i = 0; i < backings[epoch].size(); ++i) {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          sources[i], iree_hal_make_pool_reservation_request(unit, unit),
          iree_infinite_timeout(), backings[epoch][i].out()));
    }
    std::array<iree_hal_memory_mapping_t, 2> mappings = {{
        {
            .target = target,
            .scope = program,
            .target_range = {0, unit},
            .source = backings[epoch][0],
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
        },
        {
            .target = target,
            .scope = all,
            .target_range = {unit, unit},
            .source = backings[epoch][1],
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
        },
    }};
    auto invalid = mappings[0];
    invalid.scope = host;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_PERMISSION_DENIED,
        iree_hal_queue_map(queues_[0], {}, {}, 1, &invalid, 0));
    uint64_t mapped = epoch * 3 + 1, copied = mapped + 1, returned = mapped + 2;
    IREE_ASSERT_OK(iree_hal_queue_map(queues_[0], {}, {1, &timeline, &mapped},
                                      mappings.size(), mappings.data(), 0));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        timeline, mapped, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    iree_byte_span_t host_view;
    IREE_ASSERT_OK(
        iree_hal_buffer_native_host_span(target, unit, 512, &host_view));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(host_view.data), address + unit);
    EXPECT_EQ(iree_hal_buffer_native_binding(target, slot).device_address,
              address);
    auto* words = reinterpret_cast<uint32_t*>(host_view.data);
    for (uint32_t i = 0; i < 128; ++i) {
      words[i] = i < 64 ? 0x1000u * (epoch + 1) + i : 0;
    }
    IREE_ASSERT_OK(iree_hal_buffer_flush_host_range(target, unit, 512));
    IREE_ASSERT_OK(iree_hal_queue_copy(queues_[0], {1, &timeline, &mapped},
                                       {1, &timeline, &copied}, target, unit,
                                       target, 0, 256, nullptr, 0));
    IREE_ASSERT_OK(iree_hal_queue_copy(queues_[1], {1, &timeline, &copied},
                                       {1, &timeline, &returned}, target, 0,
                                       target, unit + 256, 256, nullptr, 0));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(timeline, returned,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_buffer_invalidate_host_range(target, unit, 512));
    for (uint32_t i = 0; i < 64; ++i) {
      EXPECT_EQ(words[i], 0x1000u * (epoch + 1) + i);
      EXPECT_EQ(words[64 + i], words[i]);
    }
  }
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, all));
}

}  // namespace
