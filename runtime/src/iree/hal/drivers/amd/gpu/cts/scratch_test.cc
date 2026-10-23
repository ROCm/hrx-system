// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <thread>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/gpu/cts/scratch_images.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/queue.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/queue.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

using iree::hal::cts::Ref;

class GpuScratchTest : public GpuNativeTest {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(memory_gate_.CreateFabric(&fabric_));
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const amdf_queue_format_features_t scratch_feature =
        command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
            ? uint64_t{AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE}
            : uint64_t{AMDF_GPU_PM4_FORMAT_FEATURE_DISPATCH_SCRATCH};
    bool available = false;
    for (uint32_t i = 0; i < context_->endpoint_info.queue_family_count; ++i) {
      const auto& family = context_->families[i];
      available |= family.command_type == command_type() &&
                   (family.format_features & scratch_feature);
    }
    if (!available) {
      GTEST_SKIP() << "Native private-memory service is unavailable";
    }
    const auto gfx = context_->target_info.gfx_ip;
    options_.queue.ring_byte_length = 4096;
    options_.queue.retirement_capacity = 128;
    Create();
    if (HasFatalFailure()) {
      return;
    }
    const auto* family = iree_hal_queue_family(queues_[0]);
    IREE_ASSERT_OK(iree::hal::cts::CreateQueueSlabPool(
        group_, family, IREE_HAL_BUFFER_USAGE_STORAGE,
        {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MAPPING_MODE_SCOPED},
        iree_allocator_system(), pool_.out()));
    const auto* files = iree_hal_amd_gpu_scratch_images_create();
    const char* selector = gfx.major == 9
                               ? (gfx.minor == 4 ? "gfx9-4-generic" : "gfx90a")
                           : gfx.major == 11 ? "gfx11-generic"
                                             : "gfx12-generic";
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    for (size_t i = 0; i < iree_hal_amd_gpu_scratch_images_size(); ++i) {
      if (std::strstr(files[i].name, selector)) {
        load.executable_data =
            iree_make_const_byte_span(files[i].data, files[i].size);
      }
    }
    ASSERT_NE(load.executable_data.data, nullptr);
    const auto* target =
        &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
             ->targets[0];
    IREE_ASSERT_OK(
        iree_hal_executable_load(family, target, &load, executable_.out()));
    for (size_t i = 0; i < functions_.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
          executable_, i ? IREE_SV("private_large") : IREE_SV("private_small"),
          &functions_[i]));
    }
  }

  void TearDown() override {
    executable_.reset();
    pool_.reset();
    GpuNativeTest::TearDown();
  }

  static iree_hal_dispatch_config_t Config() {
    auto config = iree_hal_make_static_dispatch_config(kGroups, 1, 1);
    config.workgroup_size[0] = 64;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    return config;
  }

  void Allocate(iree_hal_buffer_t** out_buffer) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(kOutputBytes, 4),
        iree_infinite_timeout(), out_buffer));
  }

  void Dispatch(uint32_t queue, uint32_t kernel, iree_hal_buffer_t* output,
                uint32_t seed, uint32_t rotation, iree_hal_semaphore_t* done,
                uint64_t value) {
    const uint32_t constants[] = {seed, rotation};
    const auto reference = iree_hal_make_buffer_ref(output, 0, kOutputBytes);
    uint64_t prior = value - 1;
    const iree_hal_semaphore_list_t waits =
        prior ? iree_hal_semaphore_list_t{1, &done, &prior}
              : iree_hal_semaphore_list_empty();
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queues_[queue], waits, {1, &done, &value}, executable_,
        functions_[kernel], Config(),
        iree_make_const_byte_span(constants, sizeof(constants)),
        {1, &reference}, nullptr, 0));
  }

  void Wait(iree_hal_semaphore_t* done, uint64_t value) {
    IREE_ASSERT_OK(
        iree_hal_semaphore_wait(done, value, iree_infinite_timeout(), 0));
  }

  // Public completion can precede the final retirement callback return. AQL
  // trim leaves busy queues bound, so full reclamation requires actual idle.
  void AwaitIdle(uint32_t queue) {
    while (!(command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                 ? iree_hal_amd_gpu_aql_queue_is_idle(queues_[queue])
                 : iree_hal_amd_gpu_pm4_queue_is_idle(queues_[queue]))) {
      std::this_thread::yield();
    }
  }

  void Drain(uint32_t queue, iree_hal_semaphore_t* done, uint64_t value) {
    IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[queue], {},
                                          {1, &done, &value}, nullptr, 0));
    Wait(done, value);
  }

  void Check(iree_hal_buffer_t* output, uint32_t kernel, uint32_t seed,
             uint32_t rotation, uint64_t offset = 0) {
    const uint32_t count = kernel ? 64 : 16;
    std::vector<uint32_t> values(kGroups * 64 * count);
    IREE_ASSERT_OK(iree_hal_buffer_map_read(output, offset, values.data(),
                                            values.size() * sizeof(uint32_t)));
    for (uint32_t global = 0; global < kGroups * 64; ++global) {
      for (uint32_t slot = 0; slot < count; ++slot) {
        const uint32_t selected = (slot + global % 64 + rotation) % count;
        ASSERT_EQ(values[global * count + slot],
                  seed + global * 16909063u + selected * 4129u)
            << "workitem=" << global << " slot=" << slot;
      }
    }
  }

  // Enough workgroups to exercise different resident waves and workitems.
  static constexpr uint32_t kGroups = 33;
  // One maximum-sized output, shared by direct and late-bound entry points.
  static constexpr uint64_t kOutputBytes = kGroups * 64 * 64 * sizeof(uint32_t);
  // Real native allocation dependency, joined before fixture destruction.
  NativeMemoryGate memory_gate_;
  // Explicit output storage permitting host observation after GPU completion.
  Ref<iree_hal_pool_t> pool_;
  // Source-built private-memory kernels with different fixed extents.
  Ref<iree_hal_executable_t> executable_;
  // Small and large private-memory entry points, respectively.
  std::array<iree_hal_executable_function_t, 2> functions_ = {};
};

class GpuProtocolScratchTest
    : public GpuScratchTest,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {
 protected:
  amdf_queue_command_type_t command_type() const override { return GetParam(); }
};

TEST_P(GpuProtocolScratchTest, DirectGrowthReuseTrimAndRegrowth) {
  Ref<iree_hal_buffer_t> output;
  Allocate(output.out());
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  ASSERT_FALSE(HasFatalFailure());
  const auto before = memory_gate_.ResetPeak();
  uint64_t value = 0;
  for (uint32_t kernel : {0u, 1u, 0u, 1u}) {
    SCOPED_TRACE(kernel);
    const uint32_t seed = 37 + value;
    Dispatch(0, kernel, output, seed, 5, done, ++value);
    ASSERT_FALSE(HasFatalFailure());
    Wait(done, value);
    Check(output, kernel, seed, 5);
    ASSERT_FALSE(HasFatalFailure());
  }
  Drain(0, done, ++value);
  ASSERT_FALSE(HasFatalFailure());
  const auto warm = memory_gate_.QueryUsage();
  EXPECT_GT(warm.live_bytes, before.live_bytes);
  for (uint32_t i = 0; i < 8; ++i) {
    Dispatch(0, i % 2, output, i, 7, done, ++value);
    ASSERT_FALSE(HasFatalFailure());
    Wait(done, value);
    Check(output, i % 2, i, 7);
  }
  Drain(0, done, ++value);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(memory_gate_.QueryUsage().allocation_count, warm.allocation_count);
  AwaitIdle(0);
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  const auto trimmed = memory_gate_.WaitForBytes(before.live_bytes);
  EXPECT_EQ(trimmed.destroy_failures, 0u);
  EXPECT_EQ(trimmed.live_bytes, before.live_bytes);
  Dispatch(0, 0, output, 123, 11, done, ++value);
  ASSERT_FALSE(HasFatalFailure());
  Wait(done, value);
  Check(output, 0, 123, 11);
  EXPECT_GT(memory_gate_.QueryUsage().allocation_count, warm.allocation_count);
}

INSTANTIATE_TEST_SUITE_P(Protocol, GpuProtocolScratchTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
                         [](const auto& info) {
                           return info.param == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                                      ? "Aql"
                                      : "Pm4";
                         });

class GpuAqlScratchTest : public GpuScratchTest {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  }
};

TEST_F(GpuAqlScratchTest, TrimRetainsScratchUntilRetirementCompletes) {
  Ref<iree_hal_buffer_t> output;
  Allocate(output.out());
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  ASSERT_FALSE(HasFatalFailure());
  const auto before = memory_gate_.QueryUsage();
  Dispatch(0, 0, output, 37, 5, done, 1);
  ASSERT_FALSE(HasFatalFailure());
  Wait(done, 1);
  Check(output, 0, 37, 5);
  ASSERT_FALSE(HasFatalFailure());
  const auto warm = memory_gate_.QueryUsage();
  ASSERT_GT(warm.live_bytes, before.live_bytes);

  // Hold the real completion callback after signaling, before its final queue
  // access. A cold trim must neither wait for that callback nor detach scratch
  // while the queue still owns retirement. No timing assumption is required.
  CompletionPause pause{reinterpret_cast<iree_async_semaphore_t*>(done.get())};
  uint64_t value = 2;
  IREE_ASSERT_OK(pause.Arm(value));
  auto* semaphore = done.get();
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[0], {}, {1, &semaphore, &value},
                                        nullptr, 0));
  pause.entered.acquire();
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_EQ(memory_gate_.QueryUsage().live_bytes, warm.live_bytes);
  pause.Join();

  AwaitIdle(0);
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  const auto trimmed = memory_gate_.WaitForBytes(before.live_bytes);
  EXPECT_EQ(trimmed.destroy_failures, 0u);
  EXPECT_EQ(trimmed.live_bytes, before.live_bytes);
  Dispatch(0, 0, output, 123, 11, done, 3);
  ASSERT_FALSE(HasFatalFailure());
  Wait(done, 3);
  Check(output, 0, 123, 11);
}

TEST_F(GpuAqlScratchTest, PausedFullRingResumesWhileAnotherQueueProgresses) {
  Ref<iree_hal_buffer_t> small, large;
  Allocate(small.out());
  Allocate(large.out());
  Ref<iree_hal_semaphore_t> ready, grown;
  CreateSemaphore(ready.out());
  CreateSemaphore(grown.out());
  ASSERT_FALSE(HasFatalFailure());
  Dispatch(1, 0, small, 3, 0, ready, 1);
  ASSERT_FALSE(HasFatalFailure());
  Wait(ready, 1);
  Check(small, 0, 3, 0);
  {
    NativeMemoryGate::Hold hold(&memory_gate_);
    Dispatch(0, 1, large, 5, 1, grown, 1);
    ASSERT_FALSE(HasFatalFailure());
    hold.WaitEntered();
    // Firmware has paused queue 0 for its first private allocation. Fill its
    // 64-packet ring, then park further submissions for capacity. The completed
    // allocation must bypass those submissions to resume the paused consumer.
    for (uint64_t value = 2; value <= 65; ++value) {
      Dispatch(0, 1, large, value, 1, grown, value);
      ASSERT_FALSE(HasFatalFailure());
    }
    // Queue 1 already has suitable scratch. A kernel memory call held for
    // queue 0 cannot block caller publication or retirement on this queue.
    Dispatch(1, 0, small, 7, 2, ready, 2);
    ASSERT_FALSE(HasFatalFailure());
    Wait(ready, 2);
    Check(small, 0, 7, 2);
  }
  Wait(grown, 65);
  Check(large, 1, 65, 1);
}

TEST_F(GpuScratchTest, ExistingScratchProgressesWhileGrowthIsHeld) {
  Ref<iree_hal_buffer_t> small, large;
  Allocate(small.out());
  Allocate(large.out());
  Ref<iree_hal_semaphore_t> ready, grown;
  CreateSemaphore(ready.out());
  CreateSemaphore(grown.out());
  ASSERT_FALSE(HasFatalFailure());
  Dispatch(0, 0, small, 3, 0, ready, 1);
  ASSERT_FALSE(HasFatalFailure());
  Wait(ready, 1);
  Check(small, 0, 3, 0);
  {
    NativeMemoryGate::Hold hold(&memory_gate_);
    Dispatch(0, 1, large, 5, 1, grown, 1);
    ASSERT_FALSE(HasFatalFailure());
    hold.WaitEntered();
    Dispatch(0, 0, small, 7, 2, ready, 2);
    ASSERT_FALSE(HasFatalFailure());
    Wait(ready, 2);
    Check(small, 0, 7, 2);
    // This is the same public queue. Only the larger demand is parked; the
    // native memory call remains held until all ready work has completed.
    Drain(0, ready, 3);
  }
  Wait(grown, 1);
  Check(large, 1, 5, 1);
}

TEST_P(GpuProtocolScratchTest, RecordedMixedExtentsReplayOnIndependentQueues) {
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      1, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  for (uint32_t kernel = 0; kernel < 2; ++kernel) {
    const uint32_t constants[] = {11 + kernel, 3 + kernel};
    const auto reference = iree_hal_make_indirect_buffer_ref(
        0, kernel * kOutputBytes, kOutputBytes);
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        recording, executable_, functions_[kernel], Config(),
        iree_make_const_byte_span(constants, sizeof(constants)),
        {1, &reference}, 0));
  }
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  std::array<Ref<iree_hal_buffer_t>, 2> outputs;
  std::array<Ref<iree_hal_semaphore_t>, 2> done;
  for (uint32_t queue = 0; queue < 2; ++queue) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(kOutputBytes * 2, 4),
        iree_infinite_timeout(), outputs[queue].out()));
    CreateSemaphore(done[queue].out());
  }
  ASSERT_FALSE(HasFatalFailure());
  for (uint64_t value = 1; value <= 8; ++value) {
    for (uint32_t queue = 0; queue < 2; ++queue) {
      auto* semaphore = done[queue].get();
      uint64_t prior = value - 1;
      const iree_hal_buffer_binding_t binding = {
          .buffer = outputs[queue],
          .length = kOutputBytes * 2,
      };
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queues_[queue], {1, &semaphore, &prior}, {1, &semaphore, &value},
          recording, {1, &binding}, 0));
    }
  }
  recording.reset();
  for (uint32_t queue = 0; queue < 2; ++queue) {
    Wait(done[queue], 8);
    for (uint32_t kernel = 0; kernel < 2; ++kernel) {
      Check(outputs[queue], kernel, 11 + kernel, 3 + kernel,
            kernel * kOutputBytes);
    }
  }
}

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
