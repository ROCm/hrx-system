// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/base/byte_sequence.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/gpu/kernarg.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/command_buffer.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

IREE_ASYNC_FIXED_FRONTIER_TYPE(CompletedFrontier, 2);
using iree::hal::amd::gpu::cts::GpuNativeTest;
using iree::hal::cts::Ref;

struct DispatchPlacement {
  // Packet protocol selected independently of the physical target.
  amdf_queue_command_type_t command_type;
  // Number of queues participating in each workload.
  uint32_t queue_count;
  // Require the real cold-copy path even when the native CPU aperture works.
  bool staged;
};

class GpuNativeDispatchTest
    : public GpuNativeTest,
      public ::testing::WithParamInterface<DispatchPlacement> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return GetParam().command_type;
  }

  void SetUp() override {
    if (GetParam().staged) {
      IREE_ASSERT_OK(memory_gate_.CreateFabric(&fabric_));
      memory_gate_.RequireCodeUpload();
    }
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const auto gfx = context_->target_info.gfx_ip;
    const char* selector = gfx.major == 9
                               ? (gfx.minor == 4       ? "gfx9_4_generic"
                                  : gfx.stepping == 10 ? "gfx90a"
                                                       : "gfx9_generic")
                           : gfx.major == 11 ? "gfx11_generic"
                                             : "gfx12_generic";
    const auto* files = iree_hal_amd_gpu_hip_images_create();
    for (size_t i = 0; i < iree_hal_amd_gpu_hip_images_size(); ++i) {
      if (std::strstr(files[i].name, selector)) {
        artifact_ = iree_make_const_byte_span(files[i].data, files[i].size);
      }
    }
    ASSERT_NE(artifact_.data, nullptr);
  }

  iree_status_t Load(iree_hal_executable_t** out_executable) {
    const auto* target =
        &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
             ->targets[0];
    iree_hal_executable_load_params_t load;
    iree_hal_executable_load_params_initialize(&load);
    load.executable_data = artifact_;
    return iree_hal_executable_load(iree_hal_queue_family(queues_[0]), target,
                                    &load, out_executable);
  }

  void RunIndirect(bool recorded);

  // Native capability restriction; allocation and execution remain real.
  iree::hal::amd::gpu::cts::NativeMemoryGate memory_gate_;

  // Complete compiler-produced image matching the selected physical target.
  iree_const_byte_span_t artifact_ = {};
};

TEST_P(GpuNativeDispatchTest, ExecutableGlobalsPreservePlacementAndVisibility) {
  Create();
  if (HasFatalFailure()) {
    return;
  }
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(Load(executable.out()));
  iree_hal_executable_global_t token;
  IREE_ASSERT_OK(iree_hal_executable_lookup_global_by_name(
      executable, IREE_SV("image_value"), &token));
  iree_hal_buffer_t* global = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(executable, token, &global));
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("image_increment"), &function));
  Ref<iree_hal_semaphore_t> timeline;
  CreateSemaphore(timeline.out());
  if (HasFatalFailure()) {
    return;
  }
  auto* semaphore = timeline.get();
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  const auto binding = iree_hal_make_buffer_ref(global, 0, sizeof(uint32_t));
  const uint32_t increment = 9;
  EXPECT_NE(
      iree_hal_buffer_memory_type(global) & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      0u);
  const bool mapped =
      iree_hal_buffer_memory_type(global) & IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  if (GetParam().staged) {
    EXPECT_FALSE(mapped);
  }
  const uint32_t values[] = {31, 47, 83, 109};
  uint64_t completed = 0;
  for (uint64_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    auto* queue = queues_[i % GetParam().queue_count];
    if (mapped) {
      IREE_ASSERT_OK(
          iree_hal_buffer_map_write(global, 0, &values[i], sizeof(values[i])));
    } else {
      const iree_hal_transfer_operation_t update = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
          .update =
              {
                  .source_buffer = &values[i],
                  .target_buffer = global,
                  .length = sizeof(values[i]),
              },
      };
      uint64_t ready = completed + 1;
      IREE_ASSERT_OK(iree_hal_queue_transfer(
          queue, {completed ? 1u : 0u, &semaphore, &completed},
          {1, &semaphore, &ready}, 1, &update, nullptr));
      completed = ready;
    }
    uint64_t after = completed + 1;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue, {completed ? 1u : 0u, &semaphore, &completed},
        {1, &semaphore, &after}, executable, function, config,
        iree_make_const_byte_span(&increment, sizeof(increment)), {1, &binding},
        nullptr, IREE_HAL_DISPATCH_FLAG_NONE));
    uint32_t observed = 0;
    if (mapped) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, after,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
      IREE_ASSERT_OK(
          iree_hal_buffer_map_read(global, 0, &observed, sizeof(observed)));
      completed = after;
    } else {
      const iree_hal_transfer_operation_t download = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = global,
                  .target = &observed,
                  .length = sizeof(observed),
              },
      };
      completed = after + 1;
      IREE_ASSERT_OK(iree_hal_queue_transfer(queue, {1, &semaphore, &after},
                                             {1, &semaphore, &completed}, 1,
                                             &download, nullptr));
      IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, completed,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    EXPECT_EQ(observed, values[i] + increment);
  }
}

TEST_P(GpuNativeDispatchTest, DirectDispatchThroughExplicitPoolAndSealedGroup) {
  constexpr uint32_t kThreads = 4;
  constexpr uint32_t kSteps = 256;
  const uint32_t queue_count = GetParam().queue_count;
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

  std::array<iree_hal_executable_t*, kThreads> executables = {};
  std::array<iree_hal_executable_function_t, kThreads> functions = {};
  std::array<iree_hal_semaphore_t*, kThreads> timelines = {};
  for (uint32_t t = 0; t < kThreads; ++t) {
    IREE_ASSERT_OK(Load(&executables[t]));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executables[t], IREE_SV("image_increment"), &functions[t]));
    CreateSemaphore(&timelines[t]);
  }
  // An idle caller can discover failure before creating any capture. It still
  // completes the rejected operation through its output timeline and leaves
  // later independent dispatches runnable. Executing this dispatch by mistake
  // would also change image_increment's global and corrupt the checks below.
  iree_hal_semaphore_t* failed_input = nullptr;
  iree_hal_semaphore_t* failed_output = nullptr;
  CreateSemaphore(&failed_input);
  CreateSemaphore(&failed_output);
  iree_hal_semaphore_fail(
      failed_input, iree_make_status(IREE_STATUS_ABORTED, "producer failed"));
  uint64_t failed_value = 1;
  uint32_t increment = 1;
  auto failed_binding = iree_hal_make_buffer_ref(output, 0, sizeof(uint32_t));
  auto failed_config = iree_hal_make_static_dispatch_config(1, 1, 1);
  failed_config.workgroup_size[0] = 32;
  failed_config.workgroup_size[1] = failed_config.workgroup_size[2] = 1;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queues_[0], {1, &failed_input, &failed_value},
      {1, &failed_output, &failed_value}, executables[0], functions[0],
      failed_config, iree_make_const_byte_span(&increment, sizeof(increment)),
      {1, &failed_binding}, nullptr, 0));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_semaphore_wait(failed_output, failed_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_semaphore_release(failed_output);
  iree_hal_semaphore_release(failed_input);
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

TEST_P(GpuNativeDispatchTest, DirectDispatchThroughExplicitPoolAndSealedGroup) {
  constexpr uint32_t kThreads = 4;
  constexpr uint32_t kSteps = 256;
  const uint32_t queue_count = GetParam().queue_count;
  options_.queue.ring_byte_length = 4096;
  options_.queue.argument_capacity = 8;
  options_.queue.retirement_capacity = 8;
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

  std::array<iree_hal_executable_t*, kThreads> executables = {};
  std::array<iree_hal_executable_function_t, kThreads> functions = {};
  std::array<iree_hal_semaphore_t*, kThreads> timelines = {};
  for (uint32_t t = 0; t < kThreads; ++t) {
    IREE_ASSERT_OK(Load(&executables[t]));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executables[t], IREE_SV("image_increment"), &functions[t]));
    CreateSemaphore(&timelines[t]);
  }
  // An idle caller can discover failure before creating any capture. It still
  // completes the rejected operation through its output timeline and leaves
  // later independent dispatches runnable. Executing this dispatch by mistake
  // would also change image_increment's global and corrupt the checks below.
  iree_hal_semaphore_t* failed_input = nullptr;
  iree_hal_semaphore_t* failed_output = nullptr;
  CreateSemaphore(&failed_input);
  CreateSemaphore(&failed_output);
  iree_hal_semaphore_fail(
      failed_input, iree_make_status(IREE_STATUS_ABORTED, "producer failed"));
  uint64_t failed_value = 1;
  uint32_t increment = 1;
  auto failed_binding = iree_hal_make_buffer_ref(output, 0, sizeof(uint32_t));
  auto failed_config = iree_hal_make_static_dispatch_config(1, 1, 1);
  failed_config.workgroup_size[0] = 32;
  failed_config.workgroup_size[1] = failed_config.workgroup_size[2] = 1;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queues_[0], {1, &failed_input, &failed_value},
      {1, &failed_output, &failed_value}, executables[0], functions[0],
      failed_config, iree_make_const_byte_span(&increment, sizeof(increment)),
      {1, &failed_binding}, nullptr, 0));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_semaphore_wait(failed_output, failed_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_semaphore_release(failed_output);
  iree_hal_semaphore_release(failed_input);
  std::array<std::thread, kThreads> threads;
  for (uint32_t t = 0; t < kThreads; ++t) {
    threads[t] = std::thread([&, t] {
      auto submit = [&](uint32_t i) {
        struct {
          // Caller-authored pointer to this invocation's output.
          uint64_t output;
          // Explicit scalar following the native pointer.
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

void GpuNativeDispatchTest::RunIndirect(bool recorded) {
  constexpr uint32_t kInvocations = 96;
  constexpr uint32_t kOutputLength = 64 * 8 * sizeof(uint32_t);
  constexpr uint32_t kMarker = 0xA3725901;
  const std::array<std::array<uint32_t, 3>, 6> shapes = {
      {{3, 2, 2}, {2, 1, 1}, {2, 3, 1}, {0, 2, 2}, {2, 0, 2}, {2, 2, 0}}};
  options_.queue.ring_byte_length = 4096;
  options_.queue.argument_capacity = 8;
  options_.queue.retirement_capacity = 4;
  Create();
  if (HasFatalFailure()) {
    return;
  }

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(Load(&executable));
  Ref<iree_hal_executable_t> executable_owner(executable);
  iree_hal_executable_function_t producer, consumer;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("image_counts"), &producer));
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("image_geometry"), &consumer));

  // This caller authors complete native ABI objects as a HIP frontend does.
  // Deliberately different hidden counts detect any unwanted runtime writes.
  iree_byte_span_t bytes = {nullptr, artifact_.data_length};
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                       bytes.data_length,
                                       reinterpret_cast<void**>(&bytes.data)));
  memcpy(bytes.data, artifact_.data, bytes.data_length);
  iree_byte_sequence_t* sequence = nullptr;
  iree_status_t status = iree_byte_sequence_create_from_span_move(
      &bytes, iree_allocator_system(), &sequence);
  iree_allocator_free(iree_allocator_system(), bytes.data);
  IREE_ASSERT_OK(status);
  iree_hal_amd_gpu_image_t image = {};
  status = iree_hal_amd_gpu_hsaco_image_initialize(
      sequence, iree_allocator_system(), &image);
  iree_byte_sequence_release(sequence);
  IREE_ASSERT_OK(status);
  iree_hal_amd_gpu_kernarg_layout_t* layout = nullptr;
  for (size_t i = 0; i < image.metadata.kernel_count; ++i) {
    const auto& metadata = image.metadata.kernels[i];
    if (iree_string_view_equal(metadata.reflection_name,
                               IREE_SV("image_geometry"))) {
      IREE_ASSERT_OK(iree_hal_amd_gpu_kernarg_layout_allocate(
          &metadata, iree_allocator_system(), &layout));
    }
  }
  ASSERT_NE(layout, nullptr);
  iree_hal_amd_gpu_image_deinitialize(&image);
  const uint64_t argument_stride =
      iree_align_uint64(layout->byte_length, layout->alignment);
  const auto* family = iree_hal_queue_family(queues_[0]);
  const iree_hal_pool_family_access_t access = {
      .family = family,
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE |
               IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
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
  Ref<iree_hal_pool_t> pool_owner(pool);
  std::array<Ref<iree_hal_buffer_t>, 3> buffer_owners;
  std::array<iree_hal_buffer_t*, 3> buffers = {};
  const std::array<uint64_t, 3> lengths = {kInvocations * 32,
                                           kInvocations * argument_stride,
                                           kInvocations * kOutputLength};
  for (size_t i = 0; i < buffers.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool, iree_hal_make_pool_reservation_request(lengths[i], 64),
        iree_infinite_timeout(), &buffers[i]));
    buffer_owners[i].reset(buffers[i]);
  }
  iree_hal_memory_scope_t program_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      group_, {IREE_HAL_MEMORY_SITE_PROGRAM, family}, &program_scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      pool, program_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS, &slot));
  const uint64_t output_address =
      iree_hal_buffer_native_binding(buffers[2], slot).device_address;
  std::vector<uint32_t> control(kInvocations * 8, 0xCDCDCDCD);
  std::vector<uint8_t> native(lengths[1]);
  std::vector<uint32_t> output(lengths[2] / 4, 0);
  for (uint32_t i = 0; i < kInvocations; ++i) {
    const auto& shape = shapes[(i / 3) % shapes.size()];
    memcpy(control.data() + i * 8 + 4, shape.data(), 12);
    const uint64_t address = output_address + i * kOutputLength;
    const iree_hal_amd_gpu_kernarg_inputs_t inputs = {
        .block_count = {7, 9, 11},
        .group_size = {8, 2, 2},
        .grid_dimensions = 3,
    };
    iree_hal_amd_gpu_kernarg_layout_emplace(
        layout, &kMarker, &address, &inputs,
        native.data() + i * argument_stride);
  }
  iree_hal_amd_gpu_kernarg_layout_free(layout);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(buffers[0], 0, control.data(),
                                           control.size() * sizeof(uint32_t)));
  IREE_ASSERT_OK(
      iree_hal_buffer_map_write(buffers[1], 0, native.data(), native.size()));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(buffers[2], 0, output.data(),
                                           output.size() * sizeof(uint32_t)));
  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_semaphore_t* done = nullptr;
  CreateSemaphore(&gate);
  CreateSemaphore(&done);
  const auto release_gate = +[](iree_hal_semaphore_t* semaphore) {
    iree_hal_semaphore_fail(semaphore,
                            iree_status_from_code(IREE_STATUS_ABORTED));
    iree_hal_semaphore_release(semaphore);
  };
  std::unique_ptr<iree_hal_semaphore_t, decltype(release_gate)> gate_owner(
      gate, release_gate);
  Ref<iree_hal_semaphore_t> done_owner(done);
  Ref<iree_hal_command_buffer_t> recording;
  const iree_hal_barrier_t boundary = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE |
               IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  if (recorded) {
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        family, 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, recording.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &boundary));
  }
  uint64_t one = 1;
  for (uint32_t i = 0; i < kInvocations; ++i) {
    const uint32_t mode = i % 3;
    auto config = iree_hal_make_static_dispatch_config(UINT32_MAX, UINT32_MAX,
                                                       UINT32_MAX);
    config.workgroup_size[0] = 8;
    config.workgroup_size[1] = config.workgroup_size[2] = 2;
    config.workgroup_count_ref =
        iree_hal_make_buffer_ref(buffers[0], i * 32, 12);
    const auto binding =
        mode == 2 ? iree_hal_make_buffer_ref(buffers[1], i * argument_stride,
                                             argument_stride)
                  : iree_hal_make_buffer_ref(buffers[2], i * kOutputLength,
                                             kOutputLength);
    const struct {
      // Caller-owned explicit pointer; the output stays alive through all uses.
      uint64_t output;
      // Scalar following the native pointer, with no implicit host padding.
      uint32_t marker;
    } arguments = {output_address + i * kOutputLength, kMarker};
    const auto constants = mode == 0 ? iree_make_const_byte_span(&kMarker, 4)
                           : mode == 1
                               ? iree_make_const_byte_span(&arguments, 12)
                               : iree_const_byte_span_empty();
    const iree_hal_buffer_ref_list_t bindings =
        mode == 1 ? iree_hal_buffer_ref_list_empty()
                  : iree_hal_buffer_ref_list_t{1, &binding};
    const auto flags =
        (i & 1 ? IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS
               : IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS) |
        (mode == 1 ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS : 0) |
        (mode == 2 ? IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS : 0);
    uint64_t before = 2 * i, produced = before + 1, consumed = before + 2;
    if (!recorded) {
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queues_[i % GetParam().queue_count], {1, &done, &produced},
          {1, &done, &consumed}, executable, consumer, config, constants,
          bindings, nullptr, flags));
    }
    // Consumer-before-producer is legal even on the same HAL queue. Native
    // publication must wait for the edge without occupying its producer's FIFO.
    auto producer_config = iree_hal_make_static_dispatch_config(1, 1, 1);
    producer_config.workgroup_size[0] = 32;
    producer_config.workgroup_size[1] = producer_config.workgroup_size[2] = 1;
    const std::array<iree_hal_buffer_ref_t, 2> references = {
        config.workgroup_count_ref,
        iree_hal_make_buffer_ref(buffers[0], i * 32 + 16, 12)};
    if (recorded) {
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          recording, executable, producer, producer_config, {},
          {references.size(), references.data()}, 0));
      IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &boundary));
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          recording, executable, consumer, config, constants, bindings, flags));
    } else {
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queues_[0], {1, i ? &done : &gate, i ? &before : &one},
          {1, &done, &produced}, executable, producer, producer_config, {},
          {references.size(), references.data()}, nullptr, 0));
    }
  }
  if (recorded) {
    uint64_t completed = 2 * kInvocations;
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &boundary));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
    IREE_ASSERT_OK(
        iree_hal_queue_execute(queues_[GetParam().queue_count - 1], {1, &gate, &one},
                               {1, &done, &completed}, recording, {}, 0));
    recording.reset();
  }
  executable_owner.reset();
  buffer_owners[0].reset();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done, 2 * kInvocations,
                                         iree_infinite_timeout(), 0));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffers[2], 0, output.data(),
                                          output.size() * sizeof(uint32_t)));
  for (uint32_t i = 0; i < kInvocations; ++i) {
    const auto& shape = shapes[(i / 3) % shapes.size()];
    for (uint32_t z = 0; z < 4; ++z) {
      for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
          const bool active = x < shape[0] && y < shape[1] && z < shape[2];
          const auto counts =
              i % 3 == 2 ? std::array<uint32_t, 3>{7, 9, 11} : shape;
          const std::array<uint32_t, 8> expected =
              active
                  ? std::array<uint32_t, 8>{counts[0], counts[1], counts[2], 8,
                                            2,         2,         kMarker,   3}
                  : std::array<uint32_t, 8>{};
          const size_t base = i * kOutputLength / 4 + (x + 4 * y + 16 * z) * 8;
          for (size_t field = 0; field < expected.size(); ++field) {
            EXPECT_EQ(output[base + field], expected[field])
                << i << ":" << x << ":" << y << ":" << z << ":" << field;
          }
        }
      }
    }
  }
  std::vector<uint8_t> actual_native(native.size());
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffers[1], 0, actual_native.data(),
                                          actual_native.size()));
  EXPECT_EQ(actual_native, native);
}

TEST_P(GpuNativeDispatchTest, DeviceCountsPublishArgumentsAndPacketBeforeUse) {
  RunIndirect(false);
}

TEST_P(GpuNativeDispatchTest,
       ResidentIndirectPairsCrossChunksWithDeviceCounts) {
  RunIndirect(true);
}

TEST_P(GpuNativeDispatchTest,
       ResidentProgramsRetainBindingsAcrossChunksAndQueues) {
  options_.queue.ring_byte_length = 4096;
  options_.queue.argument_capacity = 8;
  options_.queue.retirement_capacity = 4;
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
  Ref<iree_hal_pool_t> pool_owner(pool);
  constexpr uint32_t kInvocations = 16;
  iree_hal_buffer_t* output = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(kInvocations * 4, 4),
      iree_infinite_timeout(), &output));
  Ref<iree_hal_buffer_t> output_owner(output);
  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(Load(&executable));
  Ref<iree_hal_executable_t> executable_owner(executable);
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("image_accumulate"), &function));
  iree_hal_semaphore_t* gate = nullptr;
  CreateSemaphore(&gate);
  const auto release_gate = +[](iree_hal_semaphore_t* semaphore) {
    iree_hal_semaphore_fail(semaphore,
                            iree_status_from_code(IREE_STATUS_ABORTED));
    iree_hal_semaphore_release(semaphore);
  };
  std::unique_ptr<iree_hal_semaphore_t, decltype(release_gate)> gate_owner(
      gate, release_gate);
  std::vector<Ref<iree_hal_semaphore_t>> done_owners;
  std::array<iree_hal_semaphore_t*, kInvocations> done = {};
  for (auto& semaphore : done) {
    CreateSemaphore(&semaphore);
    done_owners.emplace_back(semaphore);
  }
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  const iree_hal_memory_barrier_t dispatch_memory = {
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
          IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
  };
  const iree_hal_barrier_t dependency = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .memory_barrier_count = 1,
      .memory_barriers = &dispatch_memory,
  };
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  // Small and multi-wrap recordings share exactly the same public lifecycle.
  // Repeated invocations contend for instances on two FIFOs, then idle trim
  // drops spare instances before the next epoch reuses the immutable program.
  for (uint32_t count : {1u, 97u}) {
    iree_hal_command_buffer_t* recording = nullptr;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        family, 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH, 3, &recording));
    Ref<iree_hal_command_buffer_t> recording_owner(recording);
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
    const auto binding = iree_hal_make_indirect_buffer_ref(2, 0, 4);
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
    for (uint32_t i = 0; i < count; ++i) {
      if (i) {
        IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
      }
      uint32_t value = i + 1;
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          recording, executable, function, config,
          iree_make_const_byte_span(&value, 4), {1, &binding}, 0));
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
    if (command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
      const auto* program =
          iree_hal_amd_gpu_aql_command_buffer_program(recording);
      uint32_t dispatch_index = 0;
      for (uint32_t i = 0; i < program->block_count; ++i) {
        const auto& block = program->blocks[i];
        const auto* packets =
            reinterpret_cast<const iree_hal_amd_gpu_aql_packet_t*>(
                program->resident.contents.data + block.offset);
        for (uint32_t j = 0; j < block.packet_count; ++j, ++dispatch_index) {
          const uint32_t acquire_scope =
              dispatch_index == 0 ? IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM
                                  : IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT;
          const uint32_t release_scope =
              dispatch_index + 1 == count
                  ? IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM
                  : IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT;
          EXPECT_EQ(packets[j].words[0] & 0xFFFFu,
                    IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
                        IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | acquire_scope |
                        release_scope);
        }
      }
      EXPECT_EQ(dispatch_index, count);
    }
    for (uint64_t epoch = 1; epoch <= 2; ++epoch) {
      std::array<uint32_t, kInvocations> initial;
      initial.fill(7);
      IREE_ASSERT_OK(iree_hal_buffer_map_write(output, 0, initial.data(),
                                               sizeof(initial)));
      uint64_t gate_value = count * 2 + epoch;
      for (uint32_t i = 0; i < kInvocations; ++i) {
        std::array<iree_hal_buffer_binding_t, 3> table = {};
        table[2] = {.buffer = output, .offset = i * 4, .length = 4};
        IREE_ASSERT_OK(iree_hal_queue_execute(
            queues_[i % GetParam().queue_count], {1, &gate, &gate_value},
            {1, &done[i], &gate_value}, recording, {table.size(), table.data()},
            0));
        table.fill({});
      }
      IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, gate_value, nullptr));
      for (auto semaphore : done) {
        IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, gate_value,
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE));
      }
      std::array<uint32_t, kInvocations> actual = {};
      IREE_ASSERT_OK(
          iree_hal_buffer_map_read(output, 0, actual.data(), sizeof(actual)));
      for (auto value : actual) {
        EXPECT_EQ(value, 7 + count * (count + 1) / 2);
      }
      IREE_ASSERT_OK(iree_hal_device_trim(device_));
    }
    recording_owner.reset();
  }
  // Capture retains both the command buffer and its executable after the
  // caller releases them while the first dependency is still unavailable.
  iree_hal_command_buffer_t* recording = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      family, 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH, 2, &recording));
  Ref<iree_hal_command_buffer_t> recording_owner(recording);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
  // Independent producers may overlap. Their release cannot be folded into
  // either dispatch, so the dependency must join the complete cohort.
  for (uint32_t i = 0; i < 2; ++i) {
    auto binding = iree_hal_make_indirect_buffer_ref(i, 0, 4);
    uint32_t value = i + 1;
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        recording, executable, function, config,
        iree_make_const_byte_span(&value, 4), {1, &binding}, 0));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
  auto binding = iree_hal_make_indirect_buffer_ref(0, 0, 4);
  uint32_t value = 11;
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      recording, executable, function, config,
      iree_make_const_byte_span(&value, 4), {1, &binding}, 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  if (command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
    const auto* program =
        iree_hal_amd_gpu_aql_command_buffer_program(recording);
    ASSERT_EQ(program->block_count, 1u);
    EXPECT_EQ(program->blocks[0].packet_count, 4u);
    const auto* packets =
        reinterpret_cast<const iree_hal_amd_gpu_aql_packet_t*>(
            program->resident.contents.data + program->blocks[0].offset);
    const uint32_t expected[] = {
        IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
            IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
            IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM,
        IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH,
        IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
            IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
            IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT |
            IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT,
        IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
            IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
            IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM,
    };
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
      EXPECT_EQ(packets[i].words[0] & 0xFFFFu, expected[i]);
    }
  }
  uint64_t final_value = 1000;
  const iree_hal_buffer_binding_t table[] = {
      {.buffer = output, .offset = 0, .length = 4},
      {.buffer = output, .offset = 4, .length = 4},
  };
  IREE_ASSERT_OK(iree_hal_queue_execute(queues_[0], {1, &gate, &final_value},
                                        {1, &done[0], &final_value}, recording,
                                        {IREE_ARRAYSIZE(table), table}, 0));
  recording_owner.reset();
  executable_owner.reset();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, final_value, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done[0], final_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  uint32_t actual[2] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output, 0, actual, sizeof(actual)));
  EXPECT_EQ(actual[0], 7u + 97u * 98u / 2 + 12u);
  EXPECT_EQ(actual[1], 7u + 97u * 98u / 2 + 2u);
}

INSTANTIATE_TEST_SUITE_P(
    QueuePlacement, GpuNativeDispatchTest,
    ::testing::Values(
        DispatchPlacement{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 1, false},
        DispatchPlacement{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 2, false},
        DispatchPlacement{AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, 1, false},
        DispatchPlacement{AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, 2, false},
        DispatchPlacement{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 1, true}),
    [](const auto& info) {
      return std::string(info.param.command_type ==
                                 AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                             ? "Aql"
                             : "Pm4") +
             (info.param.queue_count == 1 ? "SameQueue" : "CrossQueue") +
             (info.param.staged ? "StagedCode" : "");
    });

}  // namespace
