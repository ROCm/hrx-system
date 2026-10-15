// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstring>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/fabric/memory_backend.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/drivers/amd/registration/driver_module.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

IREE_ASYNC_FIXED_FRONTIER_TYPE(CompletedFrontier, 2);

using iree::hal::amd::gpu::cts::DeviceAllocator;
using iree::hal::amd::gpu::cts::GpuNativeTest;

TEST_F(GpuNativeTest, TrimReleasesUnusedCaptureCache) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  const size_t before = allocator_.live_count.load();
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_EQ(allocator_.live_count.load(),
            before - options_.queue_count * options_.pm4.capture_block_count);
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

class GpuNativeDispatchTest : public GpuNativeTest,
                              public ::testing::WithParamInterface<uint32_t> {};

TEST_P(GpuNativeDispatchTest, DirectDispatchThroughExplicitPoolAndSealedGroup) {
  constexpr uint32_t kThreads = 4;
  constexpr uint32_t kSteps = 256;
  const uint32_t queue_count = GetParam();
  const uint32_t gfx_major = context_->target_info.gfx_ip.major;
  options_.pm4.argument_capacity = 8;
  options_.pm4.retirement_capacity = 8;
  Create();
  if (HasFatalFailure()) {
    return;
  }

  const auto* family = iree_hal_queue_family(queues_[0]);
  const iree_hal_pool_family_access_t access = {
      .family = family,
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &access,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_READ,
              .modes = IREE_HAL_MAPPING_MODE_SCOPED,
          },
  };
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &pool_options,
                                           iree_allocator_system(), &pool));
  iree_hal_buffer_t* output = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(kThreads * kSteps * 4, 4),
      iree_infinite_timeout(), &output));
  iree_hal_memory_scope_t program_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &program_scope));
  iree_hal_buffer_native_binding_slot_t address_slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool, program_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
      &address_slot));
  const uint64_t address =
      iree_hal_buffer_native_binding(output, address_slot).device_address;

  const auto* files = iree_hal_amd_gpu_hip_images_create();
  iree_const_byte_span_t artifact = iree_const_byte_span_empty();
  const char* selector = gfx_major == 11 ? "gfx11_generic" : "gfx12_generic";
  for (size_t i = 0; i < iree_hal_amd_gpu_hip_images_size(); ++i) {
    if (std::strstr(files[i].name, selector)) {
      artifact = iree_make_const_byte_span(files[i].data, files[i].size);
    }
  }
  ASSERT_NE(artifact.data, nullptr);
  auto* target =
      &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
           ->targets[0];
  iree_hal_executable_load_params_t load;
  iree_hal_executable_load_params_initialize(&load);
  load.executable_data = artifact;
  std::array<iree_hal_executable_t*, kThreads> executables = {};
  std::array<iree_hal_executable_function_t, kThreads> functions = {};
  std::array<iree_hal_semaphore_t*, kThreads> timelines = {};
  for (uint32_t t = 0; t < kThreads; ++t) {
    IREE_ASSERT_OK(
        iree_hal_executable_load(family, target, &load, &executables[t]));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executables[t], IREE_SV("image_increment"), &functions[t]));
    CreateSemaphore(&timelines[t]);
  }
  std::array<std::thread, kThreads> threads;
  for (uint32_t t = 0; t < kThreads; ++t) {
    threads[t] = std::thread([&, t] {
      auto submit = [&](uint32_t i) {
        struct {
          uint64_t output;
          uint32_t value;
        } arguments = {address + (t * kSteps + i) * 4, 1};
        uint64_t before = i, after = i + 1;
        auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
        config.workgroup_size[0] = 32;
        config.workgroup_size[1] = config.workgroup_size[2] = 1;
        auto binding =
            iree_hal_make_buffer_ref(output, (t * kSteps + i) * 4, 4);
        IREE_ASSERT_OK(iree_hal_queue_dispatch(
            queues_[i % queue_count], {i ? 1u : 0u, &timelines[t], &before},
            {1, &timelines[t], &after}, executables[t], functions[t], config,
            t & 1 ? iree_make_const_byte_span(&arguments, 12)
                  : iree_make_const_byte_span(&arguments.value, 4),
            t & 1 ? iree_hal_buffer_ref_list_empty()
                  : iree_hal_buffer_ref_list_t{1, &binding},
            nullptr,
            t & 1 ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS : 0));
        std::memset(&arguments, 0xCD, sizeof(arguments));
      };
      // A consumer cannot occupy the native FIFO until its producer has been
      // published, including when both operations target the same queue.
      for (uint32_t i = 0; i < kSteps; i += 2) {
        submit(i + 1);
        submit(i);
      }
      iree_hal_executable_release(executables[t]);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  // Trim can overlap native execution and retirement without taking storage
  // away from publishers or peer queues retaining binary-signal readers.
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  for (auto* timeline : timelines) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        timeline, kSteps, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    CompletedFrontier frontier;
    const uint8_t count = iree_async_semaphore_query_frontier(
        reinterpret_cast<iree_async_semaphore_t*>(timeline),
        iree_async_fixed_frontier_as_frontier(&frontier), 2);
    ASSERT_EQ(count, queue_count);
    for (uint8_t i = 0; i < count; ++i) {
      EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(
          tracker_, frontier.entries[i].axis, frontier.entries[i].epoch));
    }
    iree_hal_semaphore_release(timeline);
  }
  std::array<uint32_t, kThreads * kSteps> values;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(output, 0, values.data(), sizeof(values)));
  for (uint32_t t = 0; t < kThreads; ++t) {
    for (uint32_t i = 0; i < kSteps; ++i) {
      EXPECT_EQ(values[t * kSteps + i], 20 + i) << t << ":" << i;
    }
  }
  iree_hal_buffer_release(output);
  iree_hal_pool_release(pool);
}

INSTANTIATE_TEST_SUITE_P(QueuePlacement, GpuNativeDispatchTest,
                         ::testing::Values(1u, 2u),
                         [](const ::testing::TestParamInfo<uint32_t>& info) {
                           return info.param == 1 ? "SameQueue" : "CrossQueue";
                         });

class GpuNativeTransferTest : public GpuNativeTest,
                              public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    // Every real transaction crosses staging, retirement, and primary-ring
    // capacity boundaries. Progress must come from native completion events.
    options_.pm4.ring_byte_length = 4096;
    options_.pm4.retirement_capacity = 4;
    options_.pm4.argument_capacity = 1024;
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
    static const iree_hal_barrier_list_t empty = {};
    static const iree_hal_queue_barriers_t barriers = {&empty, &empty};
    return GetParam() ? &barriers : nullptr;
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
};

class GpuNativeRecordedTransferTest : public GpuNativeTransferTest {
 protected:
  iree_hal_buffer_ref_t Reference(size_t index, uint64_t offset,
                                  uint64_t length) {
    return GetParam()
               ? iree_hal_make_indirect_buffer_ref(index + 1, offset, length)
               : iree_hal_make_buffer_ref(buffers_[index], offset, length);
  }
};

TEST_P(GpuNativeRecordedTransferTest,
       ReusableTransfersCaptureBytesAndRetireBoundedPrefixes) {
  std::vector<uint8_t> source(kLength);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = (i * 17 + 7) & 0xff;
  }
  auto expected = source;
  iree_hal_command_buffer_t* recording = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, GetParam() ? 3 : 0, &recording));
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
  // Empty remaining ranges still advance the invocation without requiring a
  // native buffer access or an unbounded search for the next nonempty record.
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
      {GetParam() ? bindings.size() : 0, bindings.data()}, 0));
  std::swap(bindings[1], bindings[2]);
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queues_[1], {1, &timeline_, &one}, {1, &timeline_, &two}, recording,
      {GetParam() ? bindings.size() : 0, bindings.data()}, 0));
  bindings.fill({});
  iree_hal_command_buffer_release(recording);
  std::vector<uint8_t> downloaded(kLength, 0xCC);
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = buffers_[GetParam() ? 0 : 1],
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

INSTANTIATE_TEST_SUITE_P(BindingMode, GpuNativeRecordedTransferTest,
                         ::testing::Values(false, true));

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
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queues_[0], {1, &gate_, &value}, {1, &timeline_, &value},
      IREE_ARRAYSIZE(operations), operations, Barriers()));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, value, nullptr));
  Wait(value);
}

INSTANTIATE_TEST_SUITE_P(Boundaries, GpuNativeTransferTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "ExplicitEmpty" : "Default";
                         });

class GpuNativeAllocationTest : public GpuNativeTest {
 protected:
  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
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
    iree_hal_memory_scope_t memory_scope;
    IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
        group_, {.kind = IREE_HAL_MEMORY_SITE_PROGRAM, .family = access.family},
        &memory_scope));
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool_, memory_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot_));
    for (auto& semaphore : semaphores_) {
      CreateSemaphore(&semaphore);
    }
  }

  void TearDown() override {
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_hal_pool_release(pool_);
    GpuNativeTest::TearDown();
  }

  void Wait(iree_hal_semaphore_t* semaphore, uint64_t value = 1) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        semaphore, value, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  void CreateFinitePool(iree_hal_pool_t** out_pool, uint32_t block_count = 1) {
    iree_hal_fixed_block_pool_options_t options;
    iree_hal_fixed_block_pool_options_initialize(&options);
    options.block_size = 4096;
    options.blocks_per_slab = block_count;
    iree_hal_pool_reservation_request_t request;
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_query_backing_request(
        pool_, &options, &request));
    iree_hal_buffer_t* backing = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, request, iree_infinite_timeout(), &backing));
    options.blocks_per_slab = 0;
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
        backing, 0, request.allocation_size, &options, iree_allocator_system(),
        out_pool));
    iree_hal_buffer_release(backing);
  }

  // Explicit native storage owner, kept alive through all borrowed roots.
  iree_hal_pool_t* pool_ = nullptr;
  // Resolved once from the pool's immutable access contract.
  iree_hal_buffer_native_binding_slot_t slot_ = {};
  // Independent allocation, transfer, release, and host-gate timelines.
  std::array<iree_hal_semaphore_t*, 4> semaphores_ = {};
};

TEST_F(GpuNativeAllocationTest, PendingSubspansSurviveCrossQueueTransfers) {
  uint64_t one = 1, two = 2;
  std::array<iree_hal_pool_reservation_request_t, 2> requests;
  requests.fill(iree_hal_make_pool_reservation_request(4096, 4));
  std::array<iree_hal_buffer_t*, 2> roots = {}, spans = {};
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      queues_[0], {1, &semaphores_[3], &one}, {1, &semaphores_[0], &one}, pool_,
      requests.size(), requests.data(), roots.data()));
  requests.fill({});
  for (size_t i = 0; i < roots.size(); ++i) {
    EXPECT_EQ(iree_hal_buffer_native_binding(roots[i], slot_).device_address,
              0u);
    IREE_ASSERT_OK(iree_hal_buffer_subspan(roots[i], 17, 257,
                                           iree_allocator_system(), &spans[i]));
  }
  // Failed capture must undo its prefix mark, even before backing exists.
  iree_hal_buffer_t* duplicates[] = {roots[0], roots[0]};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_queue_dealloca(queues_[0], {1, &semaphores_[0], &one}, {},
                              IREE_ARRAYSIZE(duplicates), duplicates));

  std::array<std::array<uint8_t, 257>, 2> actual = {};
  const std::array<uint8_t, 2> patterns = {0xA7, 0x53};
  std::array<iree_hal_transfer_operation_t, 2> fills = {}, downloads = {};
  for (size_t i = 0; i < roots.size(); ++i) {
    fills[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    fills[i].fill.target_buffer = spans[i];
    fills[i].fill.length = actual[i].size();
    fills[i].fill.pattern = &patterns[i];
    fills[i].fill.pattern_length = 1;
    downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    downloads[i].download.source_buffer = spans[i];
    downloads[i].download.target = actual[i].data();
    downloads[i].download.length = actual[i].size();
  }
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queues_[1], {1, &semaphores_[1], &one}, {1, &semaphores_[1], &two},
      downloads.size(), downloads.data(), nullptr));
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {1, &semaphores_[0], &one},
                                         {1, &semaphores_[1], &one},
                                         fills.size(), fills.data(), nullptr));
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &semaphores_[1], &two},
                                         {1, &semaphores_[2], &one},
                                         roots.size(), roots.data()));
  for (auto* span : spans) {
    iree_hal_buffer_release(span);
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores_[3], one, nullptr));
  Wait(semaphores_[2]);
  for (size_t i = 0; i < roots.size(); ++i) {
    for (uint8_t byte : actual[i]) {
      EXPECT_EQ(byte, i ? 0x53 : 0xA7);
    }
    EXPECT_EQ(iree_hal_buffer_native_binding(roots[i], slot_).device_address,
              0u);
    iree_hal_buffer_release(roots[i]);
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
}

TEST_F(GpuNativeAllocationTest, CapacityWaitDoesNotBlockReleasingItsStorage) {
  iree_hal_pool_t* finite_pool = nullptr;
  CreateFinitePool(&finite_pool);
  if (HasFatalFailure()) {
    return;
  }
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  iree_hal_buffer_t* first = nullptr;
  iree_hal_buffer_t* second = nullptr;
  uint64_t one = 1, two = 2;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {},
                                       {1, &semaphores_[0], &one}, finite_pool,
                                       1, &request, &first));
  Wait(semaphores_[0]);
  const uint64_t address =
      iree_hal_buffer_native_binding(first, slot_).device_address;
  ASSERT_NE(address, 0u);
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[1], {},
                                       {1, &semaphores_[1], &one}, finite_pool,
                                       1, &request, &second));
  // Join the initial service attempt, proving the next allocation is actually
  // waiting on capacity before its producer is admitted to that same service.
  iree_hal_memory_maintenance_call(
      finite_pool->maintenance, [](void*) {}, nullptr);
  uint64_t observed = 99;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[1], &observed));
  EXPECT_EQ(observed, 0u);
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], {1, &semaphores_[0], &one},
                                         {1, &semaphores_[2], &one}, 1,
                                         &first));
  Wait(semaphores_[1]);
  EXPECT_EQ(iree_hal_buffer_native_binding(second, slot_).device_address,
            address);
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &semaphores_[1], &one},
                                         {1, &semaphores_[2], &two}, 1,
                                         &second));
  iree_hal_buffer_release(first);
  iree_hal_buffer_release(second);
  Wait(semaphores_[2], two);
  iree_hal_pool_release(finite_pool);
}

TEST_F(GpuNativeAllocationTest, RecycledBatchJoinsIndependentOwnerFrontiers) {
  iree_hal_pool_t* finite_pool = nullptr;
  CreateFinitePool(&finite_pool, 2);
  if (HasFatalFailure()) {
    return;
  }
  std::array<iree_hal_pool_reservation_request_t, 2> requests;
  requests.fill(iree_hal_make_pool_reservation_request(4096, 4));
  std::array<iree_hal_pool_reservation_t, 2> reservations;
  std::array<iree_hal_pool_acquire_info_t, 2> infos;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      finite_pool, requests.size(), requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations.data(), infos.data(),
      &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  // Independent host participants return logical ownership before their
  // outstanding accesses complete. The next queue acquisition must join both.
  std::array<iree_async_axis_t, 2> axes;
  for (size_t i = 0; i < axes.size(); ++i) {
    axes[i] = iree_async_axis_make(1, 0, IREE_ASYNC_CAUSAL_DOMAIN_HOST, i);
    IREE_ASSERT_OK(
        iree_async_frontier_tracker_register_axis(tracker_, axes[i], nullptr));
    CompletedFrontier returned = {};
    iree_async_frontier_initialize(
        iree_async_fixed_frontier_as_frontier(&returned), 2);
    returned.entry_count = 1;
    returned.entries[0] = {axes[i], 1};
    iree_hal_pool_release_reservations(
        finite_pool, 1, &reservations[i],
        iree_async_fixed_frontier_as_frontier(&returned));
  }
  std::array<iree_hal_buffer_t*, 2> roots = {};
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      queues_[0], {}, {1, &semaphores_[0], &one}, finite_pool, requests.size(),
      requests.data(), roots.data()));
  iree_hal_memory_maintenance_call(
      finite_pool->maintenance, [](void*) {}, nullptr);
  uint64_t observed = 99;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[0], &observed));
  EXPECT_EQ(observed, 0u);
  iree_async_frontier_tracker_advance(tracker_, axes[1], 1);
  iree_hal_memory_maintenance_call(
      finite_pool->maintenance, [](void*) {}, nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[0], &observed));
  EXPECT_EQ(observed, 0u);
  iree_async_frontier_tracker_advance(tracker_, axes[0], 1);
  Wait(semaphores_[0]);
  for (auto* root : roots) {
    EXPECT_NE(iree_hal_buffer_native_binding(root, slot_).device_address, 0u);
  }
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &semaphores_[0], &one},
                                         {1, &semaphores_[1], &one},
                                         roots.size(), roots.data()));
  for (auto* root : roots) {
    iree_hal_buffer_release(root);
  }
  Wait(semaphores_[1]);
  iree_hal_pool_release(finite_pool);
  for (auto axis : axes) {
    iree_async_frontier_tracker_retire_axis(
        tracker_, axis, iree_status_from_code(IREE_STATUS_CANCELLED));
  }
}

TEST_F(GpuNativeAllocationTest, ReleaseTransactionSpansIndependentPools) {
  std::array<iree_hal_pool_t*, 2> pools = {};
  for (auto& pool : pools) {
    CreateFinitePool(&pool);
  }
  if (HasFatalFailure()) {
    return;
  }
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  std::array<iree_hal_buffer_t*, 2> roots = {};
  std::array<uint64_t, 2> values = {1, 1};
  for (size_t i = 0; i < roots.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[i], {},
                                         {1, &semaphores_[i], &values[i]},
                                         pools[i], 1, &request, &roots[i]));
  }
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(
      queues_[0], {2, semaphores_.data(), values.data()},
      {1, &semaphores_[2], &one}, roots.size(), roots.data()));
  for (auto* root : roots) {
    iree_hal_buffer_release(root);
  }
  Wait(semaphores_[2]);
  for (auto* pool : pools) {
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool, &stats);
    EXPECT_EQ(stats.reservation_count, 0u);
    iree_hal_pool_release(pool);
  }
}

TEST_F(GpuNativeAllocationTest, RejectedBatchLeavesOutputsAndPoolUntouched) {
  iree_hal_pool_t* finite_pool = nullptr;
  CreateFinitePool(&finite_pool);
  if (HasFatalFailure()) {
    return;
  }
  const iree_hal_pool_reservation_request_t requests[] = {
      iree_hal_make_pool_reservation_request(4096, 4),
      iree_hal_make_pool_reservation_request(4097, 4),
  };
  // Use existing live objects as sentinels; no output ownership transfers on
  // synchronous rejection of a request the finite pool can never satisfy.
  iree_hal_buffer_t* sentinel = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, requests[0], iree_infinite_timeout(), &sentinel));
  std::array<iree_hal_buffer_t*, 2> outputs = {sentinel, sentinel};
  uint64_t one = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_queue_alloca(queues_[0], {}, {1, &semaphores_[0], &one},
                            finite_pool, IREE_ARRAYSIZE(requests), requests,
                            outputs.data()));
  for (auto* output : outputs) {
    EXPECT_EQ(output, sentinel);
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(finite_pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  uint64_t observed = 99;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphores_[0], &observed));
  EXPECT_EQ(observed, 0u);
  iree_hal_buffer_release(sentinel);
  iree_hal_pool_release(finite_pool);
}

TEST_F(GpuNativeAllocationTest, FailedReleaseDependencyPreservesReservation) {
  const auto request = iree_hal_make_pool_reservation_request(4096, 4);
  iree_hal_buffer_t* root = nullptr;
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      queues_[0], {}, {1, &semaphores_[0], &one}, pool_, 1, &request, &root));
  Wait(semaphores_[0]);
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[0], {1, &semaphores_[3], &one},
                                         {1, &semaphores_[1], &one}, 1, &root));
  iree_hal_semaphore_fail(
      semaphores_[3], iree_make_status(IREE_STATUS_ABORTED, "producer failed"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_wait(semaphores_[1], one, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_NE(iree_hal_buffer_native_binding(root, slot_).device_address, 0u);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &semaphores_[0], &one},
                                         {1, &semaphores_[2], &one}, 1, &root));
  iree_hal_buffer_release(root);
  Wait(semaphores_[2]);
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
}

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

TEST_F(GpuNativeTest, LastReleaseFromCallerWithAcceptedWork) {
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
  iree_hal_semaphore_release(done);
  iree_hal_device_group_release(group_);
  group_ = nullptr;
  iree_hal_device_release(device_);
  device_ = nullptr;
  // Closing the device preserves the normal dependency of accepted work.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, value, nullptr));
  iree_hal_semaphore_release(gate);
  allocator_.reclaimed.acquire();
}

TEST_F(GpuNativeTest, FailedQueueConstructionReleasesWorkersAndKeepsContext) {
  options_.pm4.argument_capacity = 0;
  auto params = iree_hal_device_create_params_default();
  params.proactor_pool = proactors_;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_device_create(context_, IREE_SV("rejected native queue"),
                                     &options_, &params, allocator_.allocator(),
                                     &device_));
  EXPECT_EQ(device_, nullptr);
  EXPECT_EQ(allocator_.live_count.load(), 0u);
  EXPECT_FALSE(context_->cleanup_blocked);
  options_.pm4.argument_capacity = 256;
  Create();
}

TEST_F(GpuNativeTest, LastReleaseFromCompletionCallback) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&gate);
  CreateSemaphore(&done);
  uint64_t value = 1;
  iree_async_semaphore_timepoint_t timepoint = {
      .callback =
          [](void* user_data, iree_async_semaphore_timepoint_t*,
             iree_status_t status) {
            IREE_CHECK_OK(status);
            iree_hal_device_release(static_cast<iree_hal_device_t*>(user_data));
          },
      .user_data = device_,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(done), value, &timepoint));
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {1, &gate, &value},
                                        {1, &done, &value}, nullptr, 0));
  iree_hal_device_group_release(group_);
  group_ = nullptr;
  device_ = nullptr;
  iree_hal_semaphore_release(done);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, value, nullptr));
  iree_hal_semaphore_release(gate);
  allocator_.reclaimed.acquire();
}

TEST_F(GpuNativeTest, LastReleaseFromSharedService) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  const auto* backend =
      reinterpret_cast<const iree_hal_amd_fabric_memory_backend_t*>(
          iree_hal_device_memory_backend(device_));
  auto* service = backend->maintenance;
  iree_hal_device_group_release(group_);
  group_ = nullptr;
  auto* device = device_;
  device_ = nullptr;
  iree_hal_memory_maintenance_call(
      service,
      [](void* data) {
        iree_hal_device_release(static_cast<iree_hal_device_t*>(data));
      },
      device);
  allocator_.reclaimed.acquire();
}

TEST_F(GpuNativeTest, HostCallbacksProgressWhileColdServiceIsOccupied) {
  using iree::hal::cts::Ref;
  Create();
  if (HasFatalFailure()) {
    return;
  }
  const iree_hal_pool_family_access_t family = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, {1, &family, {}}, &options, iree_allocator_system(), pool.out()));
  ASSERT_NE(pool.get()->maintenance, nullptr);
  for (auto flags :
       {IREE_HAL_HOST_CALL_FLAG_NONE, IREE_HAL_HOST_CALL_FLAG_RELAXED}) {
    SCOPED_TRACE(flags);
    Ref<iree_hal_semaphore_t> done;
    CreateSemaphore(done.out());
    {
      struct ServiceCall {
        // First-field maintenance entry, borrowed until finished.
        iree_hal_memory_maintenance_entry_t entry;
        // Signals that the existing service is occupied.
        std::binary_semaphore started{0};
        // Explicit release of the simulated slow native operation.
        std::binary_semaphore proceed{0};
        // Joins the last service access, including assertion-exit cleanup.
        std::binary_semaphore finished{0};
        // Identity of the actual cold-memory worker.
        std::thread::id thread;
        ~ServiceCall() {
          proceed.release();
          finished.acquire();
        }
      } service;
      service.entry = {
          .fn =
              +[](iree_hal_memory_maintenance_entry_t* entry) {
                auto* service = reinterpret_cast<ServiceCall*>(entry);
                service->thread = std::this_thread::get_id();
                service->started.release();
                service->proceed.acquire();
                service->finished.release();
              },
      };
      struct Callback {
        // Exact other native queue used for a reentrant asynchronous callback.
        iree_hal_queue_t* next_queue;
        // Actual callback executor identity, published through called.
        std::thread::id thread;
        // Joins callback entry without depending on the return cache boundary.
        std::binary_semaphore called{0};
      } callback = {.next_queue = queues_[1]};
      const auto call = iree_hal_make_host_call(
          +[](void* user_data, const uint64_t arguments[4],
              iree_hal_host_call_context_t* context) {
            auto* callback = static_cast<Callback*>(user_data);
            callback->thread = std::this_thread::get_id();
            iree_status_t status = iree_hal_queue_barrier(
                callback->next_queue, {}, context->signal_semaphore_list,
                nullptr, 0);
            callback->called.release();
            return iree_status_is_ok(status)
                       ? iree_status_from_code(IREE_STATUS_DEFERRED)
                       : status;
          },
          &callback);
      iree_hal_memory_maintenance_enqueue(pool.get()->maintenance,
                                          &service.entry);
      service.started.acquire();
      uint64_t value = 1;
      iree_hal_semaphore_t* done_handle = done;
      const uint64_t arguments[4] = {0};
      IREE_ASSERT_OK(iree_hal_queue_host_call(
          queues_[0], {}, {1, &done_handle, &value}, call, arguments, flags));
      callback.called.acquire();
      EXPECT_NE(callback.thread, service.thread);
      EXPECT_NE(callback.thread, std::this_thread::get_id());
      if (flags == IREE_HAL_HOST_CALL_FLAG_RELAXED) {
        IREE_ASSERT_OK(
            iree_hal_semaphore_wait(done, 1, iree_infinite_timeout(), 0));
      }
      // Join service before the ordinary return boundary can retire. The scoped
      // blocker releases service on assertion exits as well.
    }
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(done, 1, iree_infinite_timeout(), 0));
  }
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
        pool, iree_hal_make_pool_reservation_request(5 * unit, unit),
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

class DriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_hal_driver_registry_allocate(iree_allocator_system(), &registry_));
    IREE_ASSERT_OK(iree_hal_amd_driver_module_register(registry_));
    IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
        registry_, IREE_SV("rdna"), &driver_));
    IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
        driver_, IREE_SV("rdna"), iree_allocator_system(), &count_, &infos_));
    if (!count_) {
      GTEST_SKIP() << "No RDNA endpoint";
    }
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactors_));
  }

  void TearDown() override {
    for (size_t i = 0; i < devices_.size(); ++i) {
      iree_hal_semaphore_release(semaphores_[i]);
      if (devices_[i]) {
        iree_hal_device_release(devices_[i]);
        allocators_[i].reclaimed.acquire();
      }
      EXPECT_EQ(allocators_[i].live_count.load(), 0u);
    }
    iree_async_proactor_pool_release(proactors_);
    iree_allocator_free(iree_allocator_system(), infos_);
    iree_hal_driver_release(driver_);
    iree_hal_driver_registry_free(registry_);
  }

  void Create(size_t index, iree_string_view_t path) {
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    const iree_string_pair_t options[] = {
        {.key = IREE_SV("queue_count"), .value = IREE_SV("2")}};
    iree::Status status = iree_hal_driver_create_device_by_path(
        driver_, IREE_SV("rdna"), path, IREE_ARRAYSIZE(options), options,
        &params, allocators_[index].allocator(), &devices_[index]);
    if (status.code() == iree::StatusCode::kUnimplemented ||
        status.code() == iree::StatusCode::kUnavailable) {
      GTEST_SKIP() << status.ToString();
    }
    IREE_ASSERT_OK(status);
    allocators_[index].armed.store(true);
    EXPECT_TRUE(iree_string_view_equal(iree_hal_device_id(devices_[index]),
                                       IREE_SV("rdna")));
    ASSERT_NE(iree_hal_device_queue(devices_[index], 0, 0), nullptr);
    ASSERT_NE(iree_hal_device_queue(devices_[index], 0, 1), nullptr);
  }

  void Run(size_t index) {
    iree_hal_device_t* device = devices_[index];
    iree_hal_semaphore_t** semaphore = &semaphores_[index];
    IREE_ASSERT_OK(
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
    uint64_t first = 1, second = 2;
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        iree_hal_device_queue(device, 0, 1), {1, semaphore, &first},
        {1, semaphore, &second}, nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        iree_hal_device_queue(device, 0, 0), iree_hal_semaphore_list_empty(),
        {1, semaphore, &first}, nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(*semaphore, second,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Public registry retaining the canonical AMD driver until explicitly freed.
  iree_hal_driver_registry_t* registry_ = nullptr;
  // Independent reference acquired through the RDNA scheme.
  iree_hal_driver_t* driver_ = nullptr;
  // Public discovery rows, independent of native execution resources.
  iree_hal_device_info_t* infos_ = nullptr;
  // Number of architecture-filtered discovery rows.
  iree_host_size_t count_ = 0;
  // Process progress service supplied to public device construction.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // Device ownership includes asynchronous final worker cleanup.
  std::array<DeviceAllocator, 2> allocators_;
  // Independently provisioned devices sharing one native fabric.
  std::array<iree_hal_device_t*, 2> devices_ = {};
  // Cross-queue dependencies and final completions observed by this test.
  std::array<iree_hal_semaphore_t*, 2> semaphores_ = {};
};

TEST_F(DriverTest, SchemesShareOneDriverAndFabric) {
  iree_hal_driver_t* xdna_driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
      registry_, IREE_SV("xdna"), &xdna_driver));
  EXPECT_EQ(driver_, xdna_driver);
  iree_hal_driver_release(xdna_driver);
}

TEST_F(DriverTest, OrdinalAndNativeIdentityOutliveDriver) {
  // The registry selects PROCESS native lifetime for host registration. Both
  // devices must share that owner: KFD cannot reacquire a destroyed primary VM.
  Create(0, IREE_SV("0"));
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  Create(1, infos_[0].path);
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  iree_hal_driver_release(driver_);
  driver_ = nullptr;
  iree_hal_driver_registry_free(registry_);
  registry_ = nullptr;
  Run(0);
  Run(1);
}

TEST_F(DriverTest, RejectsUnknownDeviceParametersBeforeActivation) {
  auto params = iree_hal_device_create_params_default();
  params.proactor_pool = proactors_;
  const iree_string_pair_t invalid[] = {
      {.key = IREE_SV("columns"), .value = IREE_SV("1")}};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_driver_create_device_by_id(
          driver_, infos_[0].device_id, IREE_ARRAYSIZE(invalid), invalid,
          &params, allocators_[0].allocator(), &devices_[0]));
  EXPECT_EQ(devices_[0], nullptr);
}

}  // namespace
