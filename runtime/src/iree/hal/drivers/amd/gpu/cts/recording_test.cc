// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <memory>
#include <thread>
#include <tuple>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {
using iree::hal::amd::gpu::cts::GpuNativeTest;

class GpuNativeRecordingTest : public GpuNativeTest {
 protected:
  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const uint32_t gfx_major = context_->target_info.gfx_ip.major;
    options_.pm4.ring_byte_length = 4096;
    options_.pm4.retirement_capacity = 4;
    options_.pm4.argument_capacity = 32;
    Create();
    if (HasFatalFailure()) {
      return;
    }
    const auto* family = iree_hal_queue_family(queues_[0]);
    const iree_hal_pool_family_access_t access = {
        .family = family,
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                 IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
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
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                             iree_allocator_system(), &pool_));
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
    IREE_ASSERT_OK(
        iree_hal_executable_load(family, target, &load, &executable_));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("image_accumulate"), &function_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    iree_hal_pool_release(pool_);
    GpuNativeTest::TearDown();
  }

  void Record(iree_hal_buffer_ref_t reference, uint32_t count,
              iree_hal_command_buffer_mode_t mode,
              iree_hal_command_buffer_t** out_buffer) {
    IREE_ASSERT_OK(
        iree_hal_command_buffer_create(iree_hal_queue_family(queues_[0]), mode,
                                       IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                       reference.buffer ? 0 : 3, out_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(*out_buffer));
    const iree_hal_memory_barrier_t memory = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                        IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
    };
    const iree_hal_barrier_t dependency = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .memory_barrier_count = 1,
        .memory_barriers = &memory,
    };
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    for (uint32_t i = 0; i < count; ++i) {
      if (i) {
        IREE_ASSERT_OK(
            iree_hal_command_buffer_barrier(*out_buffer, &dependency));
      }
      uint32_t value = 1;
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          *out_buffer, executable_, function_, config,
          iree_make_const_byte_span(&value, sizeof(value)), {1, &reference},
          0));
      value = 0xCDCDCDCD;
    }
    const iree_hal_barrier_t release = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(*out_buffer, &release));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(*out_buffer));
  }

  // Explicit shared backing contract for payloads and queued reservations.
  iree_hal_pool_t* pool_ = nullptr;
  // Real HIP-produced executable shared by immutable recordings.
  iree_hal_executable_t* executable_ = nullptr;
  // Kernel whose only mutable state belongs to its invocation binding.
  iree_hal_executable_function_t function_ = {};
};

TEST_F(GpuNativeRecordingTest,
       BorrowedLifetimesStillCaptureBindingTableEntries) {
  iree_hal_buffer_t* output = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(64, 4),
      iree_infinite_timeout(), &output));
  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&gate);
  CreateSemaphore(&done);
  iree_hal_command_buffer_t* recording = nullptr;
  Record(iree_hal_make_indirect_buffer_ref(2, 4, 4), 1000,
         IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED |
             IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
         &recording);
  ASSERT_FALSE(HasFatalFailure());
  uint64_t one = 1, two = 2;
  uint32_t zero = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], {1, &gate, &one}, {1, &done, &one}, output, 0, 64, &zero,
      sizeof(zero), nullptr, IREE_HAL_FILL_FLAG_NONE));
  std::array<iree_hal_buffer_binding_t, 3> table = {};
  table[2] = {.buffer = output, .offset = 16, .length = 16};
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queues_[1], {1, &done, &one}, {1, &done, &two}, recording,
      {table.size(), table.data()},
      IREE_HAL_QUEUE_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME));
  table.fill({});
  iree_hal_command_buffer_release(recording);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, one, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, two, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  uint32_t value = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output, 20, &value, sizeof(value)));
  EXPECT_EQ(value, 1000u);
  iree_hal_buffer_release(output);
  iree_hal_semaphore_release(done);
  iree_hal_semaphore_release(gate);
}

TEST_F(GpuNativeRecordingTest, HostCallPublishesShaderAndHostWrites) {
  const iree_hal_pool_family_access_t access = {
      .family = iree_hal_queue_family(queues_[0]),
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &access,
      .host =
          {
              .access =
                  IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
              .modes = IREE_HAL_MAPPING_MODE_SCOPED,
          },
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                           iree_allocator_system(), &pool));
  iree_hal_buffer_t* output = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), &output));
  iree_hal_semaphore_t* progress = nullptr;
  CreateSemaphore(&progress);
  uint64_t initialized = 1, produced = 2, called = 3, consumed = 4;
  uint32_t initial = 31, first_addend = 7, last_addend = 11;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {},
                                     {1, &progress, &initialized}, output, 0, 4,
                                     &initial, 4, nullptr, 0));
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_queue_barriers_t producer_barriers = {
      .before = nullptr,
      .after = &empty,
  };
  const iree_hal_queue_barriers_t consumer_barriers = {
      .before = &empty,
      .after = nullptr,
  };
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  const auto binding = iree_hal_make_buffer_ref(output, 0, 4);
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queues_[0], {1, &progress, &initialized}, {1, &progress, &produced},
      executable_, function_, config,
      iree_make_const_byte_span(&first_addend, 4), {1, &binding},
      &producer_barriers, 0));
  const auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t arguments[4],
          iree_hal_host_call_context_t* context) {
        auto* output = static_cast<iree_hal_buffer_t*>(user_data);
        uint32_t value = 0;
        IREE_RETURN_IF_ERROR(
            iree_hal_buffer_map_read(output, 0, &value, sizeof(value)));
        if (value != arguments[0]) {
          return iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "shader writes were not visible to host");
        }
        value *= 2;
        return iree_hal_buffer_map_write(output, 0, &value, sizeof(value));
      },
      output);
  const uint64_t arguments[4] = {initial + first_addend};
  IREE_ASSERT_OK(iree_hal_queue_host_call(
      queues_[0], {1, &progress, &produced}, {1, &progress, &called}, call,
      arguments, IREE_HAL_HOST_CALL_FLAG_WAIT_ACTIVE));
  // Neither neighboring dispatch supplies the host-call cache boundary.
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queues_[0], {1, &progress, &called}, {1, &progress, &consumed},
      executable_, function_, config,
      iree_make_const_byte_span(&last_addend, 4), {1, &binding},
      &consumer_barriers, 0));
  IREE_ASSERT_OK(
      iree_hal_semaphore_wait(progress, consumed, iree_infinite_timeout(), 0));
  uint32_t observed = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output, 0, &observed, 4));
  EXPECT_EQ(observed, (initial + first_addend) * 2 + last_addend);
  iree_hal_semaphore_release(progress);
  iree_hal_buffer_release(output);
  iree_hal_pool_release(pool);
}

TEST_F(GpuNativeRecordingTest,
       NativeArgumentsRequireOnlyIndirectParameterUsage) {
  const auto* family = iree_hal_queue_family(queues_[0]);
  const iree_hal_pool_family_access_t access = {
      .family = family,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_pool_scope_t scope = {.family_count = 1, .families = &access};
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  iree_hal_pool_t* argument_pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &options, iree_allocator_system(), &argument_pool));
  iree_hal_buffer_t* arguments = nullptr;
  iree_hal_buffer_t* output = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      argument_pool, iree_hal_make_pool_reservation_request(32, 8),
      iree_infinite_timeout(), &arguments));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(4, 4),
      iree_infinite_timeout(), &output));
  iree_hal_memory_scope_t program_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &program_scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool_, program_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  struct {
    // Native result address consumed by the HIP kernel.
    uint64_t output;
    // Added to the result by each direct or recorded dispatch.
    uint32_t value;
  } native = {iree_hal_buffer_native_binding(output, slot).device_address, 7};
  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&done);
  uint64_t value = 1;
  const uint32_t zero = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &done, &value}, output,
                                     0, 4, &zero, 4, nullptr, 0));
  uint64_t previous = value++;
  IREE_ASSERT_OK(iree_hal_queue_update(queues_[0], {1, &done, &previous},
                                       {1, &done, &value}, &native, 0,
                                       arguments, 0, 12, nullptr, 0));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  const std::array<uint32_t, 3> counts = {1, 1, 1};
  previous = value++;
  IREE_ASSERT_OK(iree_hal_queue_update(
      queues_[0], {1, &done, &previous}, {1, &done, &value}, counts.data(), 0,
      arguments, 16, sizeof(counts), nullptr, 0));
  for (auto argument_flags :
       {IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_ARGUMENTS,
        IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS}) {
    for (auto parameter_flags :
         {IREE_HAL_DISPATCH_FLAG_NONE,
          IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
          IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
      const auto flags = argument_flags | parameter_flags;
      const auto direct = iree_hal_make_buffer_ref(arguments, 0, 12);
      config.workgroup_count_ref = iree_hal_make_buffer_ref(arguments, 16, 12);
      previous = value++;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queues_[0], {1, &done, &previous}, {1, &done, &value}, executable_,
          function_, config, {}, {1, &direct}, nullptr, flags));
      for (bool table_binding : {false, true}) {
        iree_hal_command_buffer_t* recording = nullptr;
        IREE_ASSERT_OK(iree_hal_command_buffer_create(
            family, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
            IREE_HAL_COMMAND_CATEGORY_DISPATCH, table_binding ? 1 : 0,
            &recording));
        IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
        const iree_hal_barrier_t acquire = {
            .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
            .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
                                 IREE_HAL_EXECUTION_STAGE_DISPATCH,
            .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
        };
        IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
        const auto reference = table_binding
                                   ? iree_hal_make_indirect_buffer_ref(0, 0, 12)
                                   : direct;
        config.workgroup_count_ref =
            table_binding ? iree_hal_make_indirect_buffer_ref(0, 16, 12)
                          : iree_hal_make_buffer_ref(arguments, 16, 12);
        IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
            recording, executable_, function_, config, {}, {1, &reference},
            flags));
        const iree_hal_barrier_t release = {
            .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
            .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
            .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
        };
        IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
        IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
        const iree_hal_buffer_binding_t table = {
            .buffer = arguments,
            .length = 32,
        };
        previous = value++;
        IREE_ASSERT_OK(iree_hal_queue_execute(
            queues_[0], {1, &done, &previous}, {1, &done, &value}, recording,
            {table_binding ? 1u : 0u, &table}, 0));
        iree_hal_command_buffer_release(recording);
      }
    }
  }
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, value, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  uint32_t result = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output, 0, &result, sizeof(result)));
  EXPECT_EQ(result, 126u);
  iree_hal_semaphore_release(done);
  iree_hal_buffer_release(output);
  iree_hal_buffer_release(arguments);
  iree_hal_pool_release(argument_pool);
}

class GpuNativeRecordingSizeTest
    : public GpuNativeRecordingTest,
      public ::testing::WithParamInterface<uint32_t> {};

TEST_P(GpuNativeRecordingSizeTest,
       ConcurrentTablesRetainIndependentResidentInstances) {
  const uint32_t count = GetParam();
  iree_hal_command_buffer_t* recording = nullptr;
  Record(iree_hal_make_indirect_buffer_ref(2, 4, 4), count,
         IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, &recording);
  ASSERT_FALSE(HasFatalFailure());
  std::array<iree_hal_buffer_t*, 2> outputs = {};
  std::array<iree_hal_semaphore_t*, 2> done = {};
  iree_hal_semaphore_t* gate = nullptr;
  CreateSemaphore(&gate);
  uint64_t one = 1;
  const uint32_t zero = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(64, 4),
        iree_infinite_timeout(), &outputs[i]));
    CreateSemaphore(&done[i]);
    IREE_ASSERT_OK(iree_hal_queue_fill(
        queues_[i], {1, &gate, &one}, {1, &done[i], &one}, outputs[i], 0, 64,
        &zero, sizeof(zero), nullptr, IREE_HAL_FILL_FLAG_NONE));
  }
  std::array<iree_hal_buffer_binding_t, 3> table = {};
  table[2] = {outputs[0], 0, 64};
  std::array<iree_hal_amd_gpu_pm4_command_binding_t, 3> captured;
  iree_hal_amd_gpu_pm4_command_invocation_t invocation;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_command_invocation_initialize(
      recording, {table.size(), table.data()}, captured.data(), 0,
      &invocation));
  // One buffer input, plus at most one executable base, serves every dispatch.
  // The same bound holds across native block boundaries in large recordings.
  EXPECT_LE(invocation.program->value_count, 5u);
  EXPECT_LE(invocation.program->argument_length, 104u);
  EXPECT_EQ(invocation.program->dispatch_count, count);
  EXPECT_LE(invocation.program->entry.word_count, 0xfffffu);
  if (count == 65536) {
    EXPECT_GT(invocation.program->resident.byte_length, 0xfffffu * 4);
  }
  iree_hal_amd_gpu_pm4_command_invocation_deinitialize(&invocation);
  std::array<std::thread, 2> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      for (uint64_t value = 2; value <= 3; ++value) {
        uint64_t previous = value - 1;
        iree_hal_buffer_t* view = nullptr;
        IREE_ASSERT_OK(iree_hal_buffer_subspan(outputs[i], 8, 32,
                                               iree_allocator_system(), &view));
        std::array<iree_hal_buffer_binding_t, 3> table = {};
        table[2] = {.buffer = view, .offset = 8, .length = 16};
        IREE_ASSERT_OK(iree_hal_queue_execute(
            queues_[i], {1, &done[i], &previous}, {1, &done[i], &value},
            recording, {table.size(), table.data()}, 0));
        table.fill({});
        iree_hal_buffer_release(view);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  iree_hal_command_buffer_release(recording);
  iree_hal_executable_release(executable_);
  executable_ = nullptr;
  // Captures are the only owners of their distinct subspan handles, recording,
  // and executable after the callers' arrays and references disappear.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  for (size_t i = 0; i < outputs.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(done[i], 3, iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    std::array<uint32_t, 16> values;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(outputs[i], 0, values.data(), sizeof(values)));
    for (size_t j = 0; j < values.size(); ++j) {
      EXPECT_EQ(values[j], j == 5 ? count * 2 : 0) << i << ":" << j;
    }
    iree_hal_buffer_release(outputs[i]);
    iree_hal_semaphore_release(done[i]);
  }
  iree_hal_semaphore_release(gate);
}

TEST_P(GpuNativeRecordingSizeTest,
       DirectRecordingPrecedesQueuedBackingCommitment) {
  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&gate);
  CreateSemaphore(&done);
  uint64_t one = 1, two = 2, three = 3, four = 4, five = 5;
  const auto request = iree_hal_make_pool_reservation_request(64, 4);
  iree_hal_buffer_t* output = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {1, &gate, &one},
                                       {1, &done, &one}, pool_, 1, &request,
                                       &output));
  iree_hal_command_buffer_t* recording = nullptr;
  Record(iree_hal_make_buffer_ref(output, 4, 4), GetParam(),
         IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, &recording);
  ASSERT_FALSE(HasFatalFailure());
  uint32_t value = 0;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], {1, &done, &one}, {1, &done, &two}, output, 0, 64, &value,
      sizeof(value), nullptr, IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_execute(queues_[1], {1, &done, &two},
                                        {1, &done, &three}, recording, {}, 0));
  iree_hal_command_buffer_release(recording);
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = output,
              .source_offset = 4,
              .target = &value,
              .length = sizeof(value),
          },
  };
  IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {1, &done, &three},
                                         {1, &done, &four}, 1, &download,
                                         nullptr));
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queues_[1], {1, &done, &four},
                                         {1, &done, &five}, 1, &output));
  iree_hal_buffer_release(output);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, five, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(value, GetParam());
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  iree_hal_semaphore_release(done);
  iree_hal_semaphore_release(gate);
}

INSTANTIATE_TEST_SUITE_P(ProgramSize, GpuNativeRecordingSizeTest,
                         ::testing::Values(1u, 1000u, 8000u, 65536u));

TEST_F(GpuNativeRecordingTest, TransfersAndDispatchSharePendingNativeBindings) {
  for (bool indirect : {false, true}) {
    SCOPED_TRACE(indirect);
    iree_hal_semaphore_t* gate = nullptr;
    iree_hal_semaphore_t* done = nullptr;
    CreateSemaphore(&gate);
    CreateSemaphore(&done);
    uint64_t one = 1, two = 2, three = 3;
    const auto request = iree_hal_make_pool_reservation_request(64, 4);
    iree_hal_buffer_t* output = nullptr;
    IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {1, &gate, &one},
                                         {1, &done, &one}, pool_, 1, &request,
                                         &output));
    iree_hal_command_buffer_t* recording = nullptr;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queues_[0]),
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_TRANSFER,
        indirect ? 3 : 0, &recording));
    const auto reference = [&](uint64_t offset, uint64_t length) {
      return indirect ? iree_hal_make_indirect_buffer_ref(2, offset, length)
                      : iree_hal_make_buffer_ref(output, 8 + offset, length);
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
    const uint32_t zero = 0, initial = 10, increment = 17;
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        recording, reference(0, IREE_HAL_WHOLE_BUFFER), &zero, sizeof(zero),
        0));
    const iree_hal_barrier_t transfer_dependency = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
    };
    IREE_ASSERT_OK(
        iree_hal_command_buffer_barrier(recording, &transfer_dependency));
    IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(recording, &initial, 0,
                                                         reference(8, 4), 0));
    const iree_hal_memory_barrier_t to_shader_memory = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                        IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
    };
    const iree_hal_barrier_t to_shader = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .memory_barrier_count = 1,
        .memory_barriers = &to_shader_memory,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &to_shader));
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 32;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    const auto binding = reference(8, 4);
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        recording, executable_, function_, config,
        iree_make_const_byte_span(&increment, sizeof(increment)), {1, &binding},
        0));
    const iree_hal_memory_barrier_t to_transfer_memory = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
    };
    const iree_hal_barrier_t to_transfer = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .memory_barrier_count = 1,
        .memory_barriers = &to_transfer_memory,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &to_transfer));
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
        recording, reference(8, 4), reference(16, 4), 0));
    const iree_hal_barrier_t release = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
    std::array<iree_hal_buffer_binding_t, 3> table = {};
    table[2] = {output, 8, 32};
    IREE_ASSERT_OK(iree_hal_queue_execute(
        queues_[1], {1, &done, &one}, {1, &done, &two}, recording,
        {indirect ? table.size() : 0, table.data()}, 0));
    table.fill({});
    iree_hal_command_buffer_release(recording);
    std::array<uint32_t, 8> actual;
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = output,
                .source_offset = 8,
                .target = actual.data(),
                .length = sizeof(actual),
            },
    };
    IREE_ASSERT_OK(iree_hal_queue_transfer(queues_[0], {1, &done, &two},
                                           {1, &done, &three}, 1, &download,
                                           nullptr));
    iree_hal_buffer_release(output);
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, one, nullptr));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(done, three, iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(actual, (std::array<uint32_t, 8>{0, 0, 27, 0, 27, 0, 0, 0}));
    iree_hal_pool_stats_t statistics;
    iree_hal_pool_query_stats(pool_, &statistics);
    EXPECT_EQ(statistics.reservation_count, 0u);
    iree_hal_semaphore_release(done);
    iree_hal_semaphore_release(gate);
  }
}

// Public direct dispatch, direct-buffer recording, and reusable table recording
// share the same device-produced tuple and hidden native ABI expectations.
class GpuNativeInvocationTest : public GpuNativeRecordingTest {
 protected:
  void TearDown() override {
    if (HasFatalFailure()) {
      if (gate_) {
        IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
      }
      // A synchronous admission failure leaves earlier accepted operations
      // valid. Join their actual last signals before returning pool storage.
      for (size_t i = 0; i < done_.size(); ++i) {
        if (submitted_values_[i]) {
          IREE_EXPECT_OK(iree_hal_semaphore_wait(done_[i], submitted_values_[i],
                                                 iree_infinite_timeout(), 0));
        }
      }
    }
    recordings_.clear();
    buffers_.clear();
    for (auto* semaphore : done_) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_hal_semaphore_release(gate_);
    // Readback destinations remain members until every native use has joined.
    GpuNativeRecordingTest::TearDown();
  }
  // Storage survives assertion failure and the joined device teardown.
  std::array<std::array<uint32_t, 64 * 8>, 6> actual_ = {};
  // Captured timeline owners for each independent invocation.
  std::array<iree_hal_semaphore_t*, 6> done_ = {};
  // Last successfully accepted signal; teardown joins only submitted work.
  std::array<uint64_t, 6> submitted_values_ = {};
  // Holds asynchronous submissions until every caller capture is complete.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Caller resources released early after their last submission captures them.
  std::vector<
      std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>>
      buffers_;
  // Immutable recordings retained until all intended invocations are submitted.
  std::vector<std::unique_ptr<iree_hal_command_buffer_t,
                              decltype(&iree_hal_command_buffer_release)>>
      recordings_;
};

TEST_F(GpuNativeInvocationTest, WholeCopyConstraintsCoverTransitiveBindings) {
  std::array<iree_hal_buffer_t*, 4> buffers;
  const uint64_t lengths[] = {64, 80, 96, 128};
  for (size_t i = 0; i < buffers.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(lengths[i], 4),
        iree_infinite_timeout(), &buffers[i]));
    buffers_.emplace_back(buffers[i], iree_hal_buffer_release);
  }
  iree_hal_command_buffer_t* recording = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 2, &recording));
  recordings_.emplace_back(recording, iree_hal_command_buffer_release);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  std::array<uint8_t, 64> source;
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = (uint8_t)(i * 7 + 3);
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      recording, source.data(), 0, iree_hal_make_buffer_ref(buffers[0], 0, 64),
      0));
  const iree_hal_memory_barrier_t memory = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
  };
  const iree_hal_barrier_t dependency = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .memory_barrier_count = 1,
      .memory_barriers = &memory,
  };
  const iree_hal_buffer_ref_t references[] = {
      iree_hal_make_buffer_ref(buffers[0], 7, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_indirect_buffer_ref(0, 23, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_indirect_buffer_ref(1, 39, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_buffer_ref(buffers[3], 71, IREE_HAL_WHOLE_BUFFER),
  };
  for (size_t i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(recording, references[i],
                                                       references[i + 1], 0));
  }
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  std::array<iree_hal_buffer_binding_t, 2> table = {
      {{buffers[1], 0, 80}, {buffers[2], 0, 92}}};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_execute(queues_[0], {}, {}, recording,
                             {table.size(), table.data()}, 0));
  table[1].length = 96;
  CreateSemaphore(&done_[0]);
  uint64_t one = 1;
  IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {}, {1, &done_[0], &one},
                                        recording, {table.size(), table.data()},
                                        0));
  submitted_values_[0] = one;
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done_[0], one, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  std::array<uint8_t, 57> actual;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(buffers[3], 71, actual.data(), actual.size()));
  EXPECT_EQ(memcmp(actual.data(), source.data() + 7, actual.size()), 0);
}

TEST_F(GpuNativeInvocationTest,
       InconsistentRecordingCannotSealPartialCommands) {
  iree_hal_command_buffer_t* recording = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, 2, &recording));
  recordings_.emplace_back(recording, iree_hal_command_buffer_release);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording, iree_hal_make_indirect_buffer_ref(0, 0, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_indirect_buffer_ref(1, 8, IREE_HAL_WHOLE_BUFFER), 0));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_command_buffer_copy_buffer(
          recording,
          iree_hal_make_indirect_buffer_ref(1, 0, IREE_HAL_WHOLE_BUFFER),
          iree_hal_make_indirect_buffer_ref(0, 0, IREE_HAL_WHOLE_BUFFER), 0));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_command_buffer_end(recording));
  std::array<iree_hal_amd_gpu_pm4_command_binding_t, 2> captured;
  iree_hal_amd_gpu_pm4_command_invocation_t invocation;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_gpu_pm4_command_invocation_initialize(
                            recording, {}, captured.data(), 0, &invocation));
  iree_hal_amd_gpu_pm4_command_invocation_deinitialize(&invocation);
}

class GpuNativeIndirectTest
    : public GpuNativeInvocationTest,
      public ::testing::WithParamInterface<std::tuple<int, bool, bool>> {};

TEST_P(GpuNativeIndirectTest,
       DeviceCountsAndHiddenGeometryShareInvocationLifetime) {
  const auto [mode, dynamic, native] = GetParam();
  iree_hal_executable_function_t producer, consumer;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("image_counts"), &producer));
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("image_geometry"), &consumer));
  const auto flags =
      (dynamic ? IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS
               : IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS) |
      (native ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS : 0);
  constexpr size_t kLength = 64 * 8 * sizeof(uint32_t);
  constexpr uint32_t kMarker = 0xC1247A93;
  const uint32_t zero = 0;
  const std::array<std::array<uint32_t, 3>, 6> shapes = {
      {{3, 2, 2}, {2, 1, 1}, {2, 3, 1}, {0, 2, 2}, {2, 0, 2}, {2, 2, 0}}};
  CreateSemaphore(&gate_);
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_command_buffer_t* shared_recording = nullptr;
  for (size_t invocation = 0; invocation < shapes.size(); ++invocation) {
    SCOPED_TRACE(invocation);
    const auto* family = iree_hal_queue_family(queues_[0]);
    auto* queue = queues_[invocation % queues_.size()];
    CreateSemaphore(&done_[invocation]);
    ASSERT_FALSE(HasFatalFailure());
    auto* timeline = done_[invocation];
    uint64_t one = 1, two = 2, three = 3, four = 4, five = 5;
    std::array<iree_hal_buffer_t*, 2> buffers = {};
    const auto request = iree_hal_make_pool_reservation_request(4096, 8);
    if (native) {
      // Native pointer bytes are authored by the caller and need known backing.
      for (auto& buffer : buffers) {
        IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
            pool_, request, iree_infinite_timeout(), &buffer));
        buffers_.emplace_back(buffer, iree_hal_buffer_release);
      }
      IREE_ASSERT_OK(iree_hal_queue_barrier(queue, {1, &gate_, &one},
                                            {1, &timeline, &one}, nullptr, 0));
    } else {
      const std::array<iree_hal_pool_reservation_request_t, 2> requests = {
          request, request};
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          queue, {1, &gate_, &one}, {1, &timeline, &one}, pool_,
          requests.size(), requests.data(), buffers.data()));
      for (auto* buffer : buffers) {
        buffers_.emplace_back(buffer, iree_hal_buffer_release);
      }
    }
    submitted_values_[invocation] = one;
    const auto reference = [&](size_t index, uint64_t offset, uint64_t length) {
      return mode == 2
                 ? iree_hal_make_indirect_buffer_ref(index, offset, length)
                 : iree_hal_make_buffer_ref(buffers[index], 4 + offset, length);
    };
    auto config = iree_hal_make_static_dispatch_config(UINT32_MAX, UINT32_MAX,
                                                       UINT32_MAX);
    config.workgroup_size[0] = 8;
    config.workgroup_size[1] = config.workgroup_size[2] = 2;
    config.workgroup_count_ref = reference(0, 4080, 12);
    const auto output_ref = reference(1, 0, kLength);
    struct {
      // Caller-authored explicit native argument pointer.
      uint64_t output;
      // Caller-authored explicit native scalar.
      uint32_t marker;
    } native_arguments = {0, kMarker};
    if (native) {
      iree_hal_memory_scope_t scope;
      IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
          group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &scope));
      iree_hal_buffer_native_binding_slot_t slot;
      IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
          pool_, scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
      native_arguments.output =
          iree_hal_buffer_native_binding(buffers[1], slot).device_address + 4;
    }
    const auto constants =
        native ? iree_make_const_byte_span(&native_arguments, 12)
               : iree_make_const_byte_span(&kMarker, 4);
    const iree_hal_buffer_ref_list_t bindings =
        native ? iree_hal_buffer_ref_list_empty()
               : iree_hal_buffer_ref_list_t{1, &output_ref};
    auto producer_config = iree_hal_make_static_dispatch_config(1, 1, 1);
    producer_config.workgroup_size[0] = 32;
    producer_config.workgroup_size[1] = producer_config.workgroup_size[2] = 1;
    const std::array<iree_hal_buffer_ref_t, 2> producer_refs = {
        reference(0, 4080, 12), reference(0, 0, 12)};
    iree_hal_command_buffer_t* recording = shared_recording;
    if (mode && !recording) {
      IREE_ASSERT_OK(iree_hal_command_buffer_create(
          family, 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH, mode == 2 ? 2 : 0,
          &recording));
      recordings_.emplace_back(recording, iree_hal_command_buffer_release);
      IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
      const iree_hal_barrier_t acquire = {
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
          .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
                               IREE_HAL_EXECUTION_STAGE_DISPATCH,
          .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
      };
      IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
      if (dynamic) {
        IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
            recording, executable_, producer, producer_config, {},
            {producer_refs.size(), producer_refs.data()}, 0));
        const iree_hal_memory_barrier_t memory = {
            .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
            .target_scope = IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ,
        };
        const iree_hal_barrier_t dependency = {
            .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
            .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
            .memory_barrier_count = 1,
            .memory_barriers = &memory,
        };
        IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
      }
      IREE_ASSERT_OK(
          iree_hal_command_buffer_dispatch(recording, executable_, consumer,
                                           config, constants, bindings, flags));
      const iree_hal_barrier_t release = {
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
          .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
          .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
      };
      IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
      IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
      if (mode == 2 && !native) {
        shared_recording = recording;
      }
    }
    const std::array<iree_hal_transfer_operation_t, 2> initialize = {{
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
            .fill =
                {
                    .target_buffer = buffers[1],
                    .length = 4096,
                    .pattern = &zero,
                    .pattern_length = 4,
                },
        },
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
            .update =
                {
                    .source_buffer = shapes[invocation].data(),
                    .target_buffer = buffers[0],
                    .target_offset = 4,
                    .length = 12,
                },
        },
    }};
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        queue, {1, &timeline, &one}, {1, &timeline, &two}, initialize.size(),
        initialize.data(), nullptr));
    submitted_values_[invocation] = two;
    uint64_t preceding = two;
    if (!mode || !dynamic) {
      const std::array<iree_hal_buffer_ref_t, 2> direct_refs = {
          iree_hal_make_buffer_ref(buffers[0], 4084, 12),
          iree_hal_make_buffer_ref(buffers[0], 4, 12)};
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queue, {1, &timeline, &two}, {1, &timeline, &three}, executable_,
          producer, producer_config, {},
          {direct_refs.size(), direct_refs.data()}, nullptr, 0));
      submitted_values_[invocation] = three;
      preceding = three;
    }
    if (mode) {
      std::array<iree_hal_buffer_binding_t, 2> table = {
          {{buffers[0], 4, 4092}, {buffers[1], 4, kLength}}};
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queue, {1, &timeline, &preceding}, {1, &timeline, &four}, recording,
          {mode == 2 ? table.size() : 0, table.data()}, 0));
      table.fill({});
      if (recording != shared_recording) {
        recordings_.pop_back();
      }
    } else {
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queue, {1, &timeline, &preceding}, {1, &timeline, &four}, executable_,
          consumer, config, constants, bindings, nullptr, flags));
    }
    submitted_values_[invocation] = four;
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffers[1],
                .source_offset = 4,
                .target = actual_[invocation].data(),
                .length = kLength,
            },
    };
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        queues_[(invocation + 1) % queues_.size()], {1, &timeline, &four},
        {1, &timeline, &five}, 1, &download, nullptr));
    submitted_values_[invocation] = five;
    buffers_.clear();
  }
  recordings_.clear();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  for (size_t invocation = 0; invocation < shapes.size(); ++invocation) {
    SCOPED_TRACE(invocation);
    IREE_ASSERT_OK(iree_hal_semaphore_wait(done_[invocation], 5,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    for (uint32_t z = 0; z < 4; ++z) {
      for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
          const bool launched = x < shapes[invocation][0] &&
                                y < shapes[invocation][1] &&
                                z < shapes[invocation][2];
          const std::array<uint32_t, 8> expected =
              launched ? std::array<uint32_t, 8>{shapes[invocation][0],
                                                 shapes[invocation][1],
                                                 shapes[invocation][2],
                                                 8,
                                                 2,
                                                 2,
                                                 kMarker,
                                                 3}
                       : std::array<uint32_t, 8>{};
          const size_t offset = (x + 4 * y + 16 * z) * 8;
          for (size_t field = 0; field < expected.size(); ++field) {
            EXPECT_EQ(actual_[invocation][offset + field], expected[field])
                << offset << ":" << field;
          }
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(LaunchPaths, GpuNativeIndirectTest,
                         ::testing::Combine(::testing::Values(0, 1, 2),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

class GpuNativePacketTest
    : public GpuNativeInvocationTest,
      public ::testing::WithParamInterface<std::tuple<int, int, int>> {};

TEST_P(GpuNativePacketTest, ShaderObservesCompleteInvocationOwnedPacket) {
  const auto [mode, count_mode, argument_mode] = GetParam();
  const bool indirect_counts = count_mode >= 2;
  const bool exact = count_mode == 1;
  const bool indirect_arguments = argument_mode >= 2;
  const auto* family = iree_hal_queue_family(queues_[0]);
  iree_hal_executable_function_t consumer, producer;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("image_dispatch_packet"), &consumer));
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("image_counts"), &producer));
  iree_hal_memory_scope_t scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool_, scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  const iree_hal_dispatch_flags_t flags =
      (count_mode == 1 ? IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT : 0) |
      (count_mode == 2 ? IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS
                       : 0) |
      (count_mode >= 3 ? IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS
                       : 0) |
      (argument_mode == 1 ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS
                          : 0) |
      (argument_mode == 2 ? IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_ARGUMENTS
                          : 0) |
      (argument_mode == 3 ? IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS
                          : 0);
  const std::array<std::array<uint32_t, 3>, 6> shapes = {
      {{3, 2, 2}, {2, 1, 1}, {2, 3, 1}, {0, 2, 2}, {2, 0, 2}, {2, 2, 0}}};
  const size_t invocation_count = indirect_counts ? shapes.size() : 3;
  constexpr uint32_t kMarker = 0xA319D457;
  constexpr size_t kResultLength = 22 * sizeof(uint32_t);
  std::array<uint64_t, 6> external_addresses = {};
  const uint32_t zero = 0;
  CreateSemaphore(&gate_);
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_command_buffer_t* shared_recording = nullptr;
  for (size_t invocation = 0; invocation < invocation_count; ++invocation) {
    SCOPED_TRACE(invocation);
    auto* queue = queues_[invocation % queues_.size()];
    CreateSemaphore(&done_[invocation]);
    ASSERT_FALSE(HasFatalFailure());
    auto* timeline = done_[invocation];
    uint64_t one = 1, two = 2, three = 3, four = 4;
    std::array<iree_hal_buffer_t*, 2> buffers = {};
    for (auto& buffer : buffers) {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          pool_, iree_hal_make_pool_reservation_request(4096, 64),
          iree_infinite_timeout(), &buffer));
      buffers_.emplace_back(buffer, iree_hal_buffer_release);
    }
    const auto reference = [&](size_t index, uint64_t offset, uint64_t length) {
      return mode == 2
                 ? iree_hal_make_indirect_buffer_ref(index, offset, length)
                 : iree_hal_make_buffer_ref(buffers[index], offset, length);
    };
    struct {
      // Caller-supplied native shader pointer.
      uint64_t output;
      // Changing scalar proves external native arguments are bound per use.
      uint32_t marker;
    } native = {
        iree_hal_buffer_native_binding(buffers[1], slot).device_address,
        kMarker + (indirect_arguments ? (uint32_t)invocation : 0),
    };
    external_addresses[invocation] =
        iree_hal_buffer_native_binding(buffers[0], slot).device_address + 64;
    auto config = iree_hal_make_static_dispatch_config(
        shapes[invocation][0], shapes[invocation][1], shapes[invocation][2]);
    config.workgroup_size[0] = 8;
    config.workgroup_size[1] = 2;
    config.workgroup_size[2] = 1;
    config.dynamic_workgroup_local_memory = 512;
    if (exact) {
      config.workitem_count[0] = shapes[invocation][0] * 8 - 1;
      config.workitem_count[1] = shapes[invocation][1] * 2 - 1;
      config.workitem_count[2] = shapes[invocation][2];
    }
    if (indirect_counts) {
      config.workgroup_count_ref = reference(0, 4084, 12);
    }
    const auto binding = indirect_arguments ? reference(0, 64, 12)
                                            : reference(1, 0, kResultLength);
    const auto constants =
        argument_mode == 1   ? iree_make_const_byte_span(&native, 12)
        : indirect_arguments ? iree_const_byte_span_empty()
                             : iree_make_const_byte_span(&kMarker, 4);
    const iree_hal_buffer_ref_list_t bindings =
        argument_mode == 1 ? iree_hal_buffer_ref_list_empty()
                           : iree_hal_buffer_ref_list_t{1, &binding};
    const std::array<iree_hal_transfer_operation_t, 3> initialize = {{
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
            .fill =
                {
                    .target_buffer = buffers[1],
                    .length = kResultLength,
                    .pattern = &zero,
                    .pattern_length = 4,
                },
        },
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
            .update =
                {
                    .source_buffer = shapes[invocation].data(),
                    .target_buffer = buffers[0],
                    .length = 12,
                },
        },
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
            .update =
                {
                    .source_buffer = &native,
                    .target_buffer = buffers[0],
                    .target_offset = 64,
                    .length = 12,
                },
        },
    }};
    IREE_ASSERT_OK(
        iree_hal_queue_transfer(queue, {1, &gate_, &one}, {1, &timeline, &one},
                                initialize.size(), initialize.data(), nullptr));
    submitted_values_[invocation] = one;
    uint64_t preceding = one;
    if (indirect_counts && (!mode || count_mode == 2)) {
      const std::array<iree_hal_buffer_ref_t, 2> references = {
          iree_hal_make_buffer_ref(buffers[0], 4084, 12),
          iree_hal_make_buffer_ref(buffers[0], 0, 12)};
      auto produce_config = iree_hal_make_static_dispatch_config(1, 1, 1);
      produce_config.workgroup_size[0] = 32;
      produce_config.workgroup_size[1] = produce_config.workgroup_size[2] = 1;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queue, {1, &timeline, &one}, {1, &timeline, &two}, executable_,
          producer, produce_config, {}, {references.size(), references.data()},
          nullptr, 0));
      submitted_values_[invocation] = two;
      preceding = two;
    }
    if (mode) {
      auto* recording = shared_recording;
      if (!recording) {
        IREE_ASSERT_OK(iree_hal_command_buffer_create(
            family, 0,
            IREE_HAL_COMMAND_CATEGORY_DISPATCH |
                IREE_HAL_COMMAND_CATEGORY_TRANSFER,
            mode == 2 ? 2 : 0, &recording));
        recordings_.emplace_back(recording, iree_hal_command_buffer_release);
        IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
        const iree_hal_barrier_t acquire = {
            .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
            .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
                                 IREE_HAL_EXECUTION_STAGE_DISPATCH,
            .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
        };
        IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
        if (count_mode >= 3) {
          const bool copy = count_mode == 4;
          if (copy) {
            IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
                recording, reference(0, 0, 12), reference(0, 4084, 12), 0));
          } else {
            const std::array<iree_hal_buffer_ref_t, 2> references = {
                reference(0, 4084, 12), reference(0, 0, 12)};
            auto produce_config = iree_hal_make_static_dispatch_config(1, 1, 1);
            produce_config.workgroup_size[0] = 32;
            produce_config.workgroup_size[1] =
                produce_config.workgroup_size[2] = 1;
            IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
                recording, executable_, producer, produce_config, {},
                {references.size(), references.data()}, 0));
          }
          const iree_hal_memory_barrier_t memory = {
              .source_scope = static_cast<iree_hal_access_scope_t>(
                  copy ? IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE
                       : IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE),
              .target_scope = IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ,
          };
          const iree_hal_barrier_t dependency = {
              .source_stage_mask = static_cast<iree_hal_execution_stage_t>(
                  copy ? IREE_HAL_EXECUTION_STAGE_TRANSFER
                       : IREE_HAL_EXECUTION_STAGE_DISPATCH),
              .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
              .memory_barrier_count = 1,
              .memory_barriers = &memory,
          };
          IREE_ASSERT_OK(
              iree_hal_command_buffer_barrier(recording, &dependency));
        }
        IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
            recording, executable_, consumer, config, constants, bindings,
            flags));
        const iree_hal_barrier_t release = {
            .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
            .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
            .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
        };
        IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
        IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
        if (mode == 2 && indirect_counts && argument_mode != 1) {
          shared_recording = recording;
        }
      }
      std::array<iree_hal_buffer_binding_t, 2> table = {
          {{buffers[0], 0, 4096}, {buffers[1], 0, kResultLength}}};
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queue, {1, &timeline, &preceding}, {1, &timeline, &three}, recording,
          {mode == 2 ? table.size() : 0, table.data()}, 0));
      table.fill({});
      if (recording != shared_recording) {
        recordings_.pop_back();
      }
    } else {
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queue, {1, &timeline, &preceding}, {1, &timeline, &three},
          executable_, consumer, config, constants, bindings, nullptr, flags));
    }
    submitted_values_[invocation] = three;
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffers[1],
                .target = actual_[invocation].data(),
                .length = kResultLength,
            },
    };
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        queues_[(invocation + 1) % queues_.size()], {1, &timeline, &three},
        {1, &timeline, &four}, 1, &download, nullptr));
    submitted_values_[invocation] = four;
    buffers_.clear();
  }
  recordings_.clear();
  iree_hal_executable_release(executable_);
  executable_ = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  uint64_t kernel_object = 0;
  for (size_t invocation = 0; invocation < invocation_count; ++invocation) {
    SCOPED_TRACE(invocation);
    IREE_ASSERT_OK(iree_hal_semaphore_wait(done_[invocation], 4,
                                           iree_infinite_timeout(), 0));
    const auto& actual = actual_[invocation];
    if (!shapes[invocation][0] || !shapes[invocation][1] ||
        !shapes[invocation][2]) {
      for (size_t i = 0; i < 22; ++i) {
        EXPECT_EQ(actual[i], 0u);
      }
      continue;
    }
    const auto address = [&](size_t word) {
      return uint64_t{actual[word]} | (uint64_t{actual[word + 1]} << 32);
    };
    const uint16_t rank = indirect_counts || shapes[invocation][2] != 1 ? 3 : 2;
    EXPECT_EQ(actual[0], (uint32_t{rank} << 16) | 2u);
    EXPECT_EQ(actual[1], (2u << 16) | 8u);
    EXPECT_EQ(actual[2], 1u);
    EXPECT_EQ(actual[3], shapes[invocation][0] * 8 - (exact ? 1 : 0));
    EXPECT_EQ(actual[4], shapes[invocation][1] * 2 - (exact ? 1 : 0));
    EXPECT_EQ(actual[5], shapes[invocation][2]);
    EXPECT_EQ(actual[6], 0u);
    EXPECT_EQ(actual[7], 512u);
    if (!invocation) {
      kernel_object = address(8);
    }
    EXPECT_NE(kernel_object, 0u);
    EXPECT_EQ(kernel_object & 63u, 0u);
    EXPECT_EQ(address(8), kernel_object);
    EXPECT_NE(address(16), 0u);
    EXPECT_EQ(address(16) & 63u, 0u);
    EXPECT_EQ(address(10), indirect_arguments ? external_addresses[invocation]
                                              : address(16) - 64);
    for (size_t i = 12; i < 16; ++i) {
      EXPECT_EQ(actual[i], 0u);
    }
    EXPECT_EQ(actual[18], kMarker + (indirect_arguments ? invocation : 0));
    EXPECT_EQ(actual[19], 0u);
    EXPECT_EQ(actual[20], 0u);
    EXPECT_EQ(actual[21], 12u);
  }
}

INSTANTIATE_TEST_SUITE_P(PacketPaths, GpuNativePacketTest,
                         ::testing::Combine(::testing::Values(0, 1, 2),
                                            ::testing::Values(0, 1, 2, 3, 4),
                                            ::testing::Values(0, 1, 2, 3)));

class GpuCommandBufferTest : public GpuNativeTest,
                             public ::testing::WithParamInterface<bool> {};

TEST_P(GpuCommandBufferTest, SealedBlockPipelinesWithoutHostReplay) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  iree_hal_command_buffer_t* recording = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, &recording));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  if (GetParam()) {
    const iree_hal_barrier_t barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    for (uint32_t i = 0; i < 10000; ++i) {
      IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &barrier));
    }
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  iree_hal_amd_gpu_pm4_command_invocation_t invocation;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_command_invocation_initialize(
      recording, {}, nullptr, 0, &invocation));
  EXPECT_LE(invocation.program->entry.word_count, 24u);
  EXPECT_EQ(invocation.program->dispatch_count, 0u);
  EXPECT_EQ(invocation.program->operation_count, GetParam() ? 1u : 0u);
  EXPECT_EQ(invocation.program->value_count, 3u);
  EXPECT_EQ(invocation.program->argument_length, 88u);
  iree_hal_amd_gpu_pm4_command_invocation_deinitialize(&invocation);

  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&done);
  constexpr uint64_t kExecutions = 64;
  for (uint64_t value = 1; value <= kExecutions; ++value) {
    IREE_ASSERT_OK(iree_hal_queue_execute(
        queues_[0], iree_hal_semaphore_list_empty(), {1, &done, &value},
        recording, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  }
  iree_hal_command_buffer_release(recording);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      done, kExecutions, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_semaphore_release(done);
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
}

INSTANTIATE_TEST_SUITE_P(Recordings, GpuCommandBufferTest, ::testing::Bool());

}  // namespace
