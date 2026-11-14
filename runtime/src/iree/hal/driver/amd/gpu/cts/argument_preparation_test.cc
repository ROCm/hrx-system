// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/queue/aql/command_buffer.h"
#include "iree/hal/driver/amd/gpu/queue/aql/publication.h"
#include "iree/hal/driver/amd/gpu/queue/aql/state.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

using iree::hal::cts::Ref;

class ArgumentPreparationTest : public GpuNativeTest {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    options_.queue.ring_byte_length = 8192;
    options_.queue.argument_capacity = 64;
    options_.queue.retirement_capacity = 4;
    allocator_.backing = {this, Allocate};
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    IREE_ASSERT_OK(LookupExecutableData(
        "amd_gpu_argument_preparation", context_->target_info,
        IREE_SV("argument_preparation.bin"), &load.executable_data));
    Create();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const auto* family = iree_hal_queue_family(queues_[0]);
    const auto* queue =
        reinterpret_cast<iree_hal_amd_gpu_aql_queue_t*>(queues_[0]);
    const bool local =
        (queue->storage.capabilities &
         AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION) != 0;
    EXPECT_EQ(queue->producer.argument_streaming, local);
    EXPECT_EQ(queue->producer.arguments->memory.info.memory_class,
              local ? AMDF_MEMORY_CLASS_LOCAL : AMDF_MEMORY_CLASS_SYSTEM);
    RecordProperty("kernarg_memory_class", local ? "LOCAL" : "SYSTEM");
    const iree_hal_pool_family_access_t access = {
        .family = family,
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &access,
        .host = {IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
                 IREE_HAL_MAPPING_MODE_SCOPED},
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        group_, scope, &options, iree_allocator_system(), pool_.out()));
    const auto* target =
        &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
             ->targets[0];
    IREE_ASSERT_OK(
        iree_hal_executable_load(family, target, &load, executable_.out()));
  }

  void TearDown() override {
    executable_.reset();
    pool_.reset();
    GpuNativeTest::TearDown();
  }

  static iree_status_t Allocate(void* self_pointer,
                                iree_allocator_command_t command,
                                const void* params, void** inout_pointer) {
    auto* self = static_cast<ArgumentPreparationTest*>(self_pointer);
    void* failed_pointer = self->failed_reallocation_.load();
    if (command == IREE_ALLOCATOR_COMMAND_REALLOC && failed_pointer &&
        *inout_pointer == failed_pointer) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, params, inout_pointer);
  }

  void Record(iree_hal_executable_function_t function,
              iree_const_byte_span_t constants,
              iree_hal_buffer_ref_list_t bindings,
              iree_hal_dispatch_flags_t flags,
              iree_hal_command_buffer_t** out_recording) {
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queues_[0]),
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, 1, out_recording));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(*out_recording));
    auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
    config.workgroup_size[0] = 64;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(*out_recording, executable_,
                                                    function, config, constants,
                                                    bindings, flags));
    // A stream establishes payload ordering after every recorded operation.
    const iree_hal_memory_barrier_t memory = {
        IREE_HAL_ACCESS_SCOPE_MEMORY_READ | IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
        IREE_HAL_ACCESS_SCOPE_MEMORY_READ | IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
    };
    const iree_hal_barrier_t barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        .memory_barrier_count = 1,
        .memory_barriers = &memory,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(*out_recording, &barrier));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(*out_recording));
  }

  iree_hal_amd_gpu_aql_queue_t* NativeQueue(uint32_t ordinal) {
    return reinterpret_cast<iree_hal_amd_gpu_aql_queue_t*>(queues_[ordinal]);
  }

  // Real allocator rejects only growth of the explicitly selected workspace.
  std::atomic<void*> failed_reallocation_ = nullptr;
  // Host-visible output allocations with the program's native address contract.
  Ref<iree_hal_pool_t> pool_;
  // Loom-produced complete argument layouts for the selected physical device.
  Ref<iree_hal_executable_t> executable_;
};

TEST_F(ArgumentPreparationTest, OneShotArgumentSizesPreserveCpuPreparation) {
  constexpr size_t kLaunchCount = 18;
  constexpr size_t kOutputBytes = 64 * sizeof(double);
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_,
      iree_hal_make_pool_reservation_request(kLaunchCount * kOutputBytes,
                                             alignof(double)),
      iree_infinite_timeout(), output.out()));
  iree_hal_memory_scope_t program_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, iree_hal_queue_family(queues_[0])},
      &program_scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool_, program_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  const uint64_t address =
      iree_hal_buffer_native_binding(output, slot).device_address;
  std::array<Ref<iree_hal_semaphore_t>, 2> done;
  for (auto& semaphore : done) {
    CreateSemaphore(semaphore.out());
  }
  ASSERT_FALSE(HasFatalFailure());
  std::array<uint64_t, 2> submitted = {};
  enum class Arguments { kNative, kConcrete, kDeferred };
  for (uint32_t epoch = 0; epoch < 2; ++epoch) {
    for (size_t byte_length :
         {64u, 256u, 376u, 384u, 392u, 504u, 512u, 520u, 1024u, 2048u, 4096u,
          8192u, 8200u, 16384u, 32768u}) {
      SCOPED_TRACE(byte_length);
      const std::string name = "k" + std::to_string(byte_length);
      iree_hal_executable_function_t function;
      IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
          executable_, iree_make_string_view(name.data(), name.size()),
          &function));
      for (Arguments kind :
           {Arguments::kNative, Arguments::kConcrete, Arguments::kDeferred}) {
        SCOPED_TRACE(static_cast<int>(kind));
        std::vector<double> observed(kLaunchCount * 64, -1.0);
        IREE_ASSERT_OK(iree_hal_buffer_map_write(
            output, 0, observed.data(), observed.size() * sizeof(double)));
        std::vector<uint8_t> arguments(byte_length);
        for (uint32_t i = 0; i < kLaunchCount; ++i) {
          const uint64_t launch_address = address + i * kOutputBytes;
          const double first = epoch + byte_length + 0.5;
          const double last = i + 1.0;
          std::memcpy(arguments.data(), &launch_address,
                      sizeof(launch_address));
          std::memcpy(arguments.data() + 8, &first, sizeof(first));
          std::memcpy(arguments.data() + byte_length - 8, &last, sizeof(last));
          const bool native = kind == Arguments::kNative;
          auto binding =
              kind == Arguments::kDeferred
                  ? iree_hal_make_indirect_buffer_ref(0, 0, kOutputBytes)
                  : iree_hal_make_buffer_ref(output, i * kOutputBytes,
                                             kOutputBytes);
          Ref<iree_hal_command_buffer_t> recording;
          Record(function,
                 iree_make_const_byte_span(arguments.data() + (native ? 0 : 8),
                                           byte_length - (native ? 0 : 8)),
                 native ? iree_hal_buffer_ref_list_t{}
                        : iree_hal_buffer_ref_list_t{1, &binding},
                 native ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS
                        : IREE_HAL_DISPATCH_FLAG_NONE,
                 recording.out());
          ASSERT_FALSE(HasFatalFailure());
          auto* program =
              iree_hal_amd_gpu_aql_command_buffer_program(recording);
          ASSERT_NE(program->host_template, nullptr);
          EXPECT_EQ(program->resident.pool, nullptr);
          EXPECT_EQ(program->blocks[0].packet_count, 2u);
          const iree_hal_amd_gpu_aql_command_invocation_t invocation = {
              .program = program,
          };
          EXPECT_EQ(iree_hal_amd_gpu_aql_execute_argument_length(&invocation),
                    byte_length);
          std::fill(arguments.begin(), arguments.end(), 0xCD);
          iree_hal_buffer_binding_t table = {
              .buffer = output,
              .offset = i * kOutputBytes,
              .length = kOutputBytes,
          };
          const uint32_t queue_index = i % 2;
          auto* semaphore = done[queue_index].get();
          uint64_t next = submitted[queue_index] + 1;
          // Stream order is explicit: native allocation may defer an earlier
          // submission while another ready operation publishes on this queue.
          const iree_hal_semaphore_list_t waits = {
              submitted[queue_index] ? 1u : 0u, &semaphore,
              &submitted[queue_index]};
          IREE_ASSERT_OK(iree_hal_queue_execute(
              queues_[queue_index], waits, {1, &semaphore, &next}, recording,
              kind == Arguments::kDeferred
                  ? iree_hal_buffer_binding_table_t{1, &table}
                  : iree_hal_buffer_binding_table_t{},
              0));
          submitted[queue_index] = next;
          table = {};
        }
        for (uint32_t i = 0; i < done.size(); ++i) {
          IREE_ASSERT_OK(iree_hal_semaphore_wait(done[i], submitted[i],
                                                 iree_infinite_timeout(),
                                                 IREE_ASYNC_WAIT_FLAG_NONE));
          EXPECT_GE(NativeQueue(i)->producer.preparation.data_length,
                    byte_length + 128u);
        }
        EXPECT_NE(NativeQueue(0)->producer.preparation.data,
                  NativeQueue(1)->producer.preparation.data);
        IREE_ASSERT_OK(iree_hal_buffer_map_read(
            output, 0, observed.data(), observed.size() * sizeof(double)));
        for (size_t i = 0; i < observed.size(); ++i) {
          EXPECT_EQ(observed[i], epoch + byte_length + i / 64 + 1.5) << i;
        }
      }
    }
    IREE_ASSERT_OK(iree_hal_device_trim(device_));
    for (uint32_t i = 0; i < done.size(); ++i) {
      EXPECT_EQ(NativeQueue(i)->producer.preparation.data, nullptr);
      EXPECT_EQ(NativeQueue(i)->producer.preparation.data_length, 0u);
    }
  }
}

TEST_F(ArgumentPreparationTest, DirectRunsAcquireChangingHostInputs) {
  constexpr uint32_t kLaunchCount = 65;
  constexpr uint32_t kBindingCount = 9;
  Ref<iree_hal_buffer_t> data;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_,
      iree_hal_make_pool_reservation_request(
          kLaunchCount * kBindingCount * sizeof(uint32_t), 8),
      iree_infinite_timeout(), data.out()));
  const auto* family = iree_hal_queue_family(queues_[0]);
  iree_hal_memory_scope_t program_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &program_scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool_, program_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  const uint64_t address =
      iree_hal_buffer_native_binding(data, slot).device_address;
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("many_bindings"), &function));

  // Exercise both the backend default and explicit coherent HOST arguments.
  Ref<iree_hal_queue_t> host_queue;
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.placement = IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL;
  IREE_ASSERT_OK(iree_hal_queue_acquire(family, &params, host_queue.out()));
  const auto* native_host_queue =
      reinterpret_cast<iree_hal_amd_gpu_aql_queue_t*>(host_queue.get());
  EXPECT_EQ(native_host_queue->producer.arguments->memory.info.memory_class,
            AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_FALSE(native_host_queue->producer.argument_streaming);
  const std::array<iree_hal_queue_t*, 2> queues = {queues_[0],
                                                   host_queue.get()};
  std::array<Ref<iree_hal_semaphore_t>, 2> done;
  for (auto& semaphore : done) {
    CreateSemaphore(semaphore.out());
  }
  ASSERT_FALSE(HasFatalFailure());
  std::array<uint64_t, 2> submitted = {};
  const iree_hal_memory_barrier_t acquire_memory = {
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
  };
  const iree_hal_barrier_t acquire = {
      .source_stage_mask =
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .memory_barrier_count = 1,
      .memory_barriers = &acquire_memory,
  };
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  const iree_hal_barrier_list_t before = {1, &acquire};
  const iree_hal_barrier_list_t after = {1, &release};
  const iree_hal_queue_barriers_t barriers = {&before, &after};
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = config.workgroup_size[1] =
      config.workgroup_size[2] = 1;
  std::array<uint32_t, kLaunchCount * kBindingCount> words;
  for (uint32_t epoch = 1; epoch <= 8; ++epoch) {
    for (uint32_t i = 0; i < words.size(); ++i) {
      words[i] = epoch * 10000 + i;
    }
    // Reuse warm payload caches after each CPU rewrite. Native publication
    // supplies the host-to-device acquisition; payload scopes are device-only.
    IREE_ASSERT_OK(
        iree_hal_buffer_map_write(data, 0, words.data(), sizeof(words)));
    for (uint32_t i = 0; i < kLaunchCount; ++i) {
      std::array<uint8_t, kBindingCount * sizeof(uint64_t) + sizeof(uint32_t)>
          arguments;
      for (uint32_t j = 0; j < kBindingCount; ++j) {
        const uint64_t input_address =
            address + (i * kBindingCount + j) * sizeof(uint32_t);
        std::memcpy(arguments.data() + j * sizeof(input_address),
                    &input_address, sizeof(input_address));
      }
      std::memcpy(arguments.data() + kBindingCount * sizeof(uint64_t), &i,
                  sizeof(i));
      const uint32_t queue_index = i % queues.size();
      auto* semaphore = done[queue_index].get();
      uint64_t next = submitted[queue_index] + 1;
      const iree_hal_semaphore_list_t waits = {submitted[queue_index] ? 1u : 0u,
                                               &semaphore,
                                               &submitted[queue_index]};
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queues[queue_index], waits, {1, &semaphore, &next}, executable_,
          function, config,
          iree_make_const_byte_span(arguments.data(), arguments.size()), {},
          &barriers, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS));
      submitted[queue_index] = next;
      arguments.fill(0xCD);
    }
    for (uint32_t i = 0; i < done.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(done[i], submitted[i],
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(data, 0, words.data(), sizeof(words)));
    for (uint32_t i = 0; i < kLaunchCount; ++i) {
      const uint32_t first = epoch * 10000 + i * kBindingCount;
      EXPECT_EQ(words[i * kBindingCount + 8], 8 * first + 28 + i);
    }
    IREE_ASSERT_OK(iree_hal_device_trim(device_));
  }
}

TEST_F(ArgumentPreparationTest, ReplaysCaptureDenseBindingValuesAcrossQueues) {
  // The 128-packet ring publishes at most 64 packets per native block. Distinct
  // bindings in every dispatch force captured values to survive interleaved
  // chunks, including the final partial block.
  constexpr uint32_t kDispatchCount = 129;
  constexpr uint32_t kBindingCount = kDispatchCount * 9;
  constexpr uint32_t kInvocationCount = 4;
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_,
      iree_hal_make_pool_reservation_request(
          kInvocationCount * kBindingCount * sizeof(uint32_t), 4),
      iree_infinite_timeout(), output.out()));
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("many_bindings"), &function));
  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(queues_[0]), 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      kBindingCount, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = config.workgroup_size[1] =
      config.workgroup_size[2] = 1;
  for (uint32_t i = 0; i < kDispatchCount; ++i) {
    std::array<iree_hal_buffer_ref_t, 9> bindings;
    for (uint32_t j = 0; j < bindings.size(); ++j) {
      bindings[j] = iree_hal_make_indirect_buffer_ref(i * 9 + j, 0, 4);
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        recording, executable_, function, config,
        iree_make_const_byte_span(&i, sizeof(i)),
        {bindings.size(), bindings.data()}, IREE_HAL_DISPATCH_FLAG_NONE));
  }
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  const auto* program = iree_hal_amd_gpu_aql_command_buffer_program(recording);
  ASSERT_NE(program->host_template, nullptr);
  EXPECT_EQ(program->resident.pool, nullptr);
  EXPECT_EQ(program->value_count, kBindingCount + 1);
  EXPECT_GT(program->host_template->fixup_count, 256u);
  ASSERT_EQ(program->block_count, 3u);
  EXPECT_LT(program->host_template->preparation_byte_length,
            program->host_template->byte_length);
  std::array<Ref<iree_hal_semaphore_t>, kInvocationCount> done;
  for (auto& semaphore : done) {
    CreateSemaphore(semaphore.out());
  }
  iree_hal_semaphore_t* gate = nullptr;
  CreateSemaphore(&gate);
  const auto abort_gate = +[](iree_hal_semaphore_t* semaphore) {
    iree_hal_semaphore_fail(semaphore,
                            iree_status_from_code(IREE_STATUS_ABORTED));
    iree_hal_semaphore_release(semaphore);
  };
  std::unique_ptr<iree_hal_semaphore_t, decltype(abort_gate)> gate_owner(
      gate, abort_gate);
  ASSERT_FALSE(HasFatalFailure());
  std::vector<uint32_t> words(kInvocationCount * kBindingCount);
  for (uint64_t epoch = 1; epoch <= 2; ++epoch) {
    for (uint32_t i = 0; i < words.size(); ++i) {
      words[i] = epoch * 10000 + i;
    }
    IREE_ASSERT_OK(iree_hal_buffer_map_write(output, 0, words.data(),
                                             words.size() * sizeof(uint32_t)));
    for (uint32_t i = 0; i < kInvocationCount; ++i) {
      std::array<iree_hal_buffer_binding_t, kBindingCount> table;
      for (uint32_t j = 0; j < table.size(); ++j) {
        table[j] = {
            .buffer = output,
            .offset = (i * kBindingCount + j) * sizeof(uint32_t),
            .length = sizeof(uint32_t),
        };
      }
      auto* semaphore = done[i].get();
      IREE_ASSERT_OK(iree_hal_queue_execute(queues_[i % 2], {1, &gate, &epoch},
                                            {1, &semaphore, &epoch}, recording,
                                            {table.size(), table.data()}, 0));
      table.fill({});
    }
    if (epoch == 2) {
      recording.reset();
    }
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, epoch, nullptr));
    for (const auto& semaphore : done) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, epoch,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    IREE_ASSERT_OK(iree_hal_buffer_map_read(output, 0, words.data(),
                                            words.size() * sizeof(uint32_t)));
    for (uint32_t i = 0; i < kInvocationCount; ++i) {
      for (uint32_t j = 0; j < kDispatchCount; ++j) {
        const uint32_t first = epoch * 10000 + i * kBindingCount + j * 9;
        EXPECT_EQ(words[i * kBindingCount + j * 9 + 8], 8 * first + 28 + j);
      }
    }
    IREE_ASSERT_OK(iree_hal_device_trim(device_));
  }
}

TEST_F(ArgumentPreparationTest, FailedWorkspaceGrowthPreservesOwnership) {
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(64 * sizeof(double), 8),
      iree_infinite_timeout(), output.out()));
  const auto binding = iree_hal_make_buffer_ref(output, 0, 64 * sizeof(double));
  Ref<iree_hal_semaphore_t> done;
  CreateSemaphore(done.out());
  ASSERT_FALSE(HasFatalFailure());
  auto* semaphore = done.get();
  void* original_workspace = nullptr;
  iree_host_size_t original_capacity = 0;
  for (size_t byte_length : {64u, 32768u}) {
    const std::string name = "k" + std::to_string(byte_length);
    iree_hal_executable_function_t function;
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, iree_make_string_view(name.data(), name.size()),
        &function));
    std::vector<double> arguments((byte_length - 8) / sizeof(double), 1.0);
    Ref<iree_hal_command_buffer_t> recording;
    Record(function,
           iree_make_const_byte_span(arguments.data(), byte_length - 8),
           {1, &binding}, IREE_HAL_DISPATCH_FLAG_NONE, recording.out());
    ASSERT_FALSE(HasFatalFailure());
    if (original_workspace) {
      failed_reallocation_.store(original_workspace);
    }
    uint64_t value = byte_length;
    iree_status_t status = iree_hal_queue_execute(
        queues_[0], {}, {1, &semaphore, &value}, recording, {}, 0);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(
          semaphore, value, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    }
    if (original_workspace) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(NativeQueue(0)->producer.preparation.data, original_workspace);
      EXPECT_EQ(NativeQueue(0)->producer.preparation.data_length,
                original_capacity);
    } else {
      IREE_ASSERT_OK(status);
      original_workspace = NativeQueue(0)->producer.preparation.data;
      original_capacity = NativeQueue(0)->producer.preparation.data_length;
      ASSERT_NE(original_workspace, nullptr);
      ASSERT_LT(original_capacity, 32768u);
    }
  }
  // Joined queue destruction returns the preserved allocation to its owner;
  // the fixture's device allocator checks that every allocation is released.
}

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
