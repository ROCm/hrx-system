// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <semaphore>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/fabric/fabric.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/drivers/amd/gpu/queue/semaphore.h"
#include "iree/hal/drivers/amd/gpu/queue/signal.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::amd::gpu::cts::GpuNativeTest;

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
  EXPECT_EQ(allocator_.live_count.load(),
            before - options_.queue_count * options_.queue.capture_block_count);
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

class GpuNativeTransferTest : public GpuNativeTest,
                              public ::testing::WithParamInterface<
                                  std::pair<amdf_queue_command_type_t, bool>> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return GetParam().first;
  }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    // Start with small staging, retirement, and primary-ring capacities. Bulk
    // transfers must grow staging and reuse it through native completion
    // events.
    options_.queue.ring_byte_length = 4096;
    options_.queue.retirement_capacity = 4;
    options_.queue.argument_capacity = 1024;
    Create();
    if (HasFatalFailure()) {
      return;
    }
    const iree_hal_pool_family_access_t access = {
        .family = iree_hal_queue_family(queues_[0]),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &access,
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                             iree_allocator_system(), &pool_));
    iree_hal_memory_scope_t queue_scope;
    IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
        group_, {IREE_HAL_MEMORY_SITE_QUEUE, access.family}, &queue_scope));
    const auto transition =
        iree_hal_pool_query_transition(pool_, queue_scope, queue_scope);
    ASSERT_TRUE(iree_hal_memory_effects_is_supported(transition.release));
    ASSERT_TRUE(iree_hal_memory_effects_is_supported(transition.acquire));
    ASSERT_FALSE(
        iree_hal_memory_effects_requires_resources(transition.release));
    ASSERT_FALSE(
        iree_hal_memory_effects_requires_resources(transition.acquire));
    acquire_.effects = transition.acquire;
    release_.effects = transition.release;
    for (auto& buffer : buffers_) {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          pool_, iree_hal_make_pool_reservation_request(kLength, 4),
          iree_infinite_timeout(), &buffer));
    }
    CreateSemaphore(&timeline_);
    CreateSemaphore(&gate_);
  }

  void TearDown() override {
    iree_hal_semaphore_release(gate_);
    iree_hal_semaphore_release(timeline_);
    ReleaseBuffers();
    iree_hal_pool_release(pool_);
    GpuNativeTest::TearDown();
  }

  void ReleaseBuffers() {
    for (auto& buffer : buffers_) {
      iree_hal_buffer_release(buffer);
      buffer = nullptr;
    }
  }

  const iree_hal_queue_barriers_t* Barriers() {
    return GetParam().second ? &barriers_ : nullptr;
  }

  void Wait(uint64_t value) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        timeline_, value, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Odd extent exercises byte copies and fill tails beyond the packet chunk.
  static constexpr size_t kLength = 32781;
  // Explicit source of buffers usable by the device's transfer family.
  iree_hal_pool_t* pool_ = nullptr;
  // Independent payload allocations retained by accepted transactions.
  std::array<iree_hal_buffer_t*, 2> buffers_ = {};
  // User-visible completion sequence, including host downloads.
  iree_hal_semaphore_t* timeline_ = nullptr;
  // Host-controlled readiness for borrowed uploads and captured updates.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Exact pool transitions between shader or copy-engine accesses. An empty
  // payload boundary is valid only when the queried transition is empty.
  iree_hal_barrier_t acquire_ = {};
  // Publishes every transfer in the batch before its completion semaphore.
  iree_hal_barrier_t release_ = {};
  // Stable descriptors shared by concurrent submitting threads.
  iree_hal_barrier_list_t before_ = {1, &acquire_};
  // Stable producer boundary paired with before_.
  iree_hal_barrier_list_t after_ = {1, &release_};
  // Explicit qualified alternatives to conservative default boundaries.
  iree_hal_queue_barriers_t barriers_ = {&before_, &after_};
};

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
  constexpr size_t kBulkLength = 3 * 1024 * 1024 + 37;
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

TEST_P(GpuNativeTransferTest,
       ConcurrentTransfersResumeAfterStagingCapacityReturns) {
  using iree::hal::cts::Ref;
  constexpr size_t kLaneCount = 8;
  constexpr size_t kIterations = 16;
  constexpr size_t kTransferLength = 3 * 1024 * 1024 + 37;
  std::array<Ref<iree_hal_buffer_t>, kLaneCount> buffers;
  std::array<Ref<iree_hal_semaphore_t>, kLaneCount> timelines;
  for (size_t lane = 0; lane < kLaneCount; ++lane) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(kTransferLength + 3, 4),
        iree_infinite_timeout(), buffers[lane].out()));
    CreateSemaphore(timelines[lane].out());
  }
  const auto* barriers = Barriers();
  std::array<std::thread, kLaneCount> threads;
  for (size_t lane = 0; lane < kLaneCount; ++lane) {
    threads[lane] = std::thread([&, lane] {
      std::vector<uint8_t> source(kTransferLength);
      std::vector<uint8_t> target(kTransferLength + 2, 0xCC);
      for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>((i * 29) ^ (i >> 9) ^ lane);
      }
      auto* timeline = timelines[lane].get();
      for (size_t iteration = 0; iteration < kIterations; ++iteration) {
        SCOPED_TRACE(::testing::Message()
                     << "lane " << lane << ", iteration " << iteration);
        uint64_t before = iteration * 2;
        uint64_t uploaded = before + 1;
        uint64_t downloaded = before + 2;
        source.front() = static_cast<uint8_t>(iteration);
        source.back() = static_cast<uint8_t>(lane ^ iteration);
        // Independent lanes share one bounded native staging ring. Both the
        // calling threads and deferred publisher must resume when retirement
        // returns space; no later submission is needed to wake stalled work.
        IREE_ASSERT_OK(iree_hal_queue_upload(
            queues_[0], {iteration ? 1u : 0u, &timeline, &before},
            {1, &timeline, &uploaded}, source.data(), buffers[lane], 3,
            kTransferLength, barriers));
        IREE_ASSERT_OK(iree_hal_queue_download(
            queues_[0], {1, &timeline, &uploaded}, {1, &timeline, &downloaded},
            buffers[lane], 3, target.data() + 1, kTransferLength, barriers));
        IREE_ASSERT_OK(iree_hal_semaphore_wait(
            timeline, downloaded, iree_infinite_timeout(),
            (lane + iteration) & 1 ? IREE_ASYNC_WAIT_FLAG_ACTIVE
                                   : IREE_ASYNC_WAIT_FLAG_NONE));
        EXPECT_EQ(target.front(), 0xCC);
        EXPECT_EQ(target.back(), 0xCC);
        EXPECT_EQ(
            std::memcmp(source.data(), target.data() + 1, kTransferLength), 0);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
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

TEST_F(GpuNativeTest, BatchedMappingsReplaceProtectAndReleaseNativeBacking) {
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
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      group_, scope, iree_allocator_system(), &capabilities));
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
      group_, scope, &reservation_options, iree_allocator_system(),
      reservation.out()));
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

}  // namespace
