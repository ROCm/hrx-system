// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/task/queue/queue.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <future>
#include <thread>
#include <utility>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/device_group.h"
#include "iree/hal/driver/task/device.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

static iree_status_t SubmitFileTransfer(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_memory_access_t access) {
  return access == IREE_HAL_MEMORY_ACCESS_READ
             ? iree_hal_queue_read(&queue->base, waits, signals, files, buffers,
                                   transfers, barriers, 0)
             : iree_hal_queue_write(&queue->base, waits, signals, buffers,
                                    files, transfers, barriers, 0);
}

class TaskQueueTest : public ::testing::TestWithParam<iree_host_size_t> {
 protected:
  void SetUp() override {
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(GetParam(), &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    iree_status_t executor_status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(executor_status);

    IREE_ASSERT_OK(iree_hal_allocator_create_heap(iree_allocator_system(),
                                                  &device_allocator_));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        /*node_count=*/1, /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool_));
  }

  void TearDown() override {
    iree_async_proactor_pool_release(proactor_pool_);
    iree_hal_allocator_release(device_allocator_);
    iree_task_executor_release(executor_);
  }

  iree_status_t CreateDeviceGroup(iree_hal_device_group_t** out_device_group) {
    return CreateDeviceGroup(/*abort_on_failure=*/false, out_device_group);
  }

  iree_status_t CreateDeviceGroup(bool abort_on_failure,
                                  iree_hal_device_group_t** out_device_group) {
    *out_device_group = nullptr;
    iree_hal_task_device_params_t device_params;
    iree_hal_task_device_params_initialize(&device_params);
    device_params.abort_on_failure = abort_on_failure;
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool_;
    iree_hal_device_t* device = nullptr;
    iree_status_t status = iree_hal_task_device_create(
        IREE_SV("task_queue_test"), &device_params, /*queue_count=*/1,
        &executor_, /*loader_count=*/0, /*loaders=*/nullptr, device_allocator_,
        &create_params, iree_allocator_system(), &device);
    iree_async_frontier_tracker_t* frontier_tracker = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_async_frontier_tracker_create(
          iree_async_frontier_tracker_options_default(),
          iree_allocator_system(), &frontier_tracker);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_create_from_device(
          device, frontier_tracker, iree_allocator_system(), out_device_group);
    }
    iree_async_frontier_tracker_release(frontier_tracker);
    iree_hal_device_release(device);
    return status;
  }

  iree_status_t WrapTransferBuffer(iree_byte_span_t bytes,
                                   iree_hal_buffer_release_callback_t release,
                                   iree_hal_buffer_t** out_buffer) {
    return iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(),
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
        bytes.data_length, bytes, release, iree_allocator_system(), out_buffer);
  }

  iree_status_t OpenTransferFile(iree_hal_device_t* device,
                                 iree_string_view_t path,
                                 iree_byte_span_t bytes, bool native,
                                 iree_hal_file_t** out_file) {
    iree_io_file_handle_t* handle = nullptr;
    if (native) {
      IREE_RETURN_IF_ERROR(iree_io_file_contents_write(
          path, iree_const_cast_byte_span(bytes), iree_allocator_system()));
      IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
          IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
              IREE_IO_FILE_MODE_SHARE_READ | IREE_IO_FILE_MODE_SHARE_WRITE |
              IREE_IO_FILE_MODE_ASYNC,
          path, iree_allocator_system(), &handle));
    } else {
      IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
          IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE, bytes,
          iree_io_file_handle_release_callback_null(), iree_allocator_system(),
          &handle));
    }
    iree_status_t status = iree_hal_file_import(
        device, nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_ALL, handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE,
        out_file);
    iree_io_file_handle_release(handle);
    return status;
  }

  // Executor shared by each short-lived device in the test.
  iree_task_executor_t* executor_ = nullptr;

  // Heap allocator shared by each short-lived device in the test.
  iree_hal_allocator_t* device_allocator_ = nullptr;

  // Proactor pool shared by each short-lived device in the test.
  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
};

TEST_P(TaskQueueTest, FailurePreservesFrontierDiagnostic) {
  iree_hal_device_group_t* device_group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&device_group));
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_queue_t* queue =
      iree_hal_device_queue(device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
  auto* task_queue = reinterpret_cast<iree_hal_task_queue_t*>(queue);

  iree_async_single_frontier_t frontier;
  iree_async_single_frontier_initialize(&frontier, task_queue->axis, 1);
  std::promise<iree::Status> completion;
  auto result = completion.get_future();
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      task_queue->frontier_tracker,
      iree_async_single_frontier_as_frontier(&frontier),
      +[](void* user_data, iree_status_t status) {
        static_cast<std::promise<iree::Status>*>(user_data)->set_value(
            iree::Status(std::move(status)));
      },
      &completion, &waiter));

  // A callback failure must reach independent causal consumers of this queue,
  // even when the operation has no signal semaphores.
  iree::Status failure(iree_status_annotate_f(
      iree_make_status(IREE_STATUS_DATA_LOSS, "queue operation failed"),
      "operation diagnostic context"));
  const std::string expected = failure.ToString();
  auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t[4], iree_hal_host_call_context_t*) {
        return iree_status_clone(static_cast<iree::Status*>(user_data)->get());
      },
      &failure);
  uint64_t args[4] = {};
  IREE_ASSERT_OK(iree_hal_queue_host_call(
      queue, iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      call, args, IREE_HAL_HOST_CALL_FLAG_NONE));

  iree::Status status = result.get();
  iree_hal_device_group_release(device_group);
  EXPECT_EQ(status.ToString(), expected);
  EXPECT_EQ(status.code(), iree::StatusCode::kDataLoss);
}

// Native memory dependency with deliberately separate CPU and device views.
// The actual Task recorder, dependency scheduler and cache executor are tested.
struct CacheBuffer : iree_hal_buffer_t {
  // CPU cache contents used by prepared execution bindings.
  alignas(64) std::array<uint8_t, 128> host = {};
  // Bytes observed by the remote execution domain.
  std::array<uint8_t, 128> remote = {};
  // Stable prepared CPU view.
  iree_hal_buffer_native_binding_t binding = {};
  // Native invalidations completed before CPU reads.
  size_t invalidations = 0;
  // Native publications completed before remote reads.
  size_t publications = 0;
  // Injected native failure propagated through the operation's completion.
  iree_status_code_t failure = IREE_STATUS_OK;
  // Worker performing the last native maintenance callback.
  std::thread::id maintenance_thread;

  CacheBuffer() {
    static const iree_hal_buffer_vtable_t vtable = {
        .destroy =
            [](iree_hal_buffer_t* base) {
              delete static_cast<CacheBuffer*>(base);
            },
        .export_range = nullptr,
        .map_range = nullptr,
        .unmap_range = nullptr,
        .invalidate_range =
            [](iree_hal_buffer_t* base, iree_device_size_t offset,
               iree_device_size_t length) {
              auto* buffer = static_cast<CacheBuffer*>(base);
              ++buffer->invalidations;
              buffer->maintenance_thread = std::this_thread::get_id();
              if (buffer->failure) {
                return iree_status_from_code(buffer->failure);
              }
              memcpy(buffer->host.data() + offset,
                     buffer->remote.data() + offset, length);
              return iree_ok_status();
            },
        .flush_range =
            [](iree_hal_buffer_t* base, iree_device_size_t offset,
               iree_device_size_t length) {
              auto* buffer = static_cast<CacheBuffer*>(base);
              ++buffer->publications;
              buffer->maintenance_thread = std::this_thread::get_id();
              if (buffer->failure) {
                return iree_status_from_code(buffer->failure);
              }
              memcpy(buffer->remote.data() + offset,
                     buffer->host.data() + offset, length);
              return iree_ok_status();
            },
        .query_memory = nullptr,
        .allocation = nullptr,
    };
    iree_hal_buffer_initialize(
        iree_hal_buffer_placement_undefined(), this, host.size(), 0,
        host.size(), IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER, &vtable,
        this);
    binding.host_pointer = host.data();
    memory.bindings = &binding;
    host_binding_index = 0;
  }
};

// An owned descriptor that can be destroyed or changed immediately after
// capture.
struct CacheBarrier {
  // Single native action.
  iree_hal_memory_transition_recipe_info_t action = {};
  // Prepared recipe borrowing the action above.
  iree_hal_memory_transition_recipe_t recipe = {};
  // Logical operand borrowing the recipe above.
  iree_hal_buffer_barrier_t range = {};
  // Ordered boundary passed to the HAL.
  iree_hal_barrier_t barrier = {};
  // One-shot list borrowing the boundary above.
  iree_hal_barrier_list_t list = {};

  CacheBarrier(iree_hal_buffer_ref_t ref,
               iree_hal_memory_transition_operation_t operation) {
    action.kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
    action.executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API;
    action.operation = operation;
    action.range_granularity = 1;
    recipe.effects.bits =
        operation == IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH
            ? IREE_HAL_MEMORY_EFFECT_HOST_FLUSH
            : IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE;
    recipe.operation_count = 1;
    recipe.operations = &action;
    range.buffer_ref = ref;
    range.recipe = &recipe;
    barrier.source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER;
    barrier.target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER;
    barrier.effects = recipe.effects;
    barrier.buffer_barrier_count = 1;
    barrier.buffer_barriers = &range;
    list = {1, &barrier};
  }
};

TEST_P(TaskQueueTest, RecordedCacheRecipesSurviveBindingAndReplay) {
  iree_hal_device_group_t* group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&group));
  auto* device = iree_hal_device_group_device_at(group, 0);
  auto* queue = iree_hal_device_queue(device, 0, 0);
  auto* source = new CacheBuffer();
  auto* target = new CacheBuffer();
  // Explicit cache actions must still execute for external protocols whose
  // backing is advertised as coherent to ordinary CPU accesses.
  source->memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  target->memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &done));
  uint64_t value = 0;
  for (iree_hal_command_buffer_mode_t mode :
       {0u, static_cast<uint32_t>(IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)}) {
    for (iree_host_size_t binding_count : {0, 2}) {
      iree_hal_command_buffer_t* commands = nullptr;
      IREE_ASSERT_OK(iree_hal_command_buffer_create(
          iree_hal_queue_family(queue), mode,
          IREE_HAL_COMMAND_CATEGORY_TRANSFER, binding_count, &commands));
      auto source_ref = binding_count
                            ? iree_hal_make_indirect_buffer_ref(0, 3, 24)
                            : iree_hal_make_buffer_ref(source, 10, 24);
      auto target_ref = binding_count
                            ? iree_hal_make_indirect_buffer_ref(1, 2, 24)
                            : iree_hal_make_buffer_ref(target, 7, 24);
      CacheBarrier acquire(
          source_ref, IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE);
      CacheBarrier release(target_ref,
                           IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH);
      IREE_ASSERT_OK(iree_hal_command_buffer_begin(commands));
      IREE_ASSERT_OK(
          iree_hal_command_buffer_barrier(commands, &acquire.barrier));
      IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
          commands, source_ref, target_ref, IREE_HAL_COPY_FLAG_NONE));
      IREE_ASSERT_OK(
          iree_hal_command_buffer_barrier(commands, &release.barrier));
      IREE_ASSERT_OK(iree_hal_command_buffer_end(commands));
      EXPECT_EQ(iree_atomic_load(&source->resource.ref_count,
                                 iree_memory_order_relaxed),
                !binding_count && !mode ? 2 : 1);
      acquire.action = {};
      release.action = {};
      const iree_hal_buffer_binding_t values[] = {{source, 7, 32},
                                                  {target, 5, 32}};
      for (uint8_t replay = 1; replay <= 3; ++replay) {
        source->remote.fill(replay);
        source->host.fill(0);
        target->host.fill(0xA5);
        target->remote.fill(0xA5);
        ++value;
        const iree_hal_semaphore_list_t signals = {1, &done, &value};
        IREE_ASSERT_OK(iree_hal_queue_execute(
            queue, {}, signals, commands, {binding_count, values},
            IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
        IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
            signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
        for (size_t i = 0; i < target->remote.size(); ++i) {
          EXPECT_EQ(target->remote[i], i >= 7 && i < 31 ? replay : 0xA5) << i;
        }
        EXPECT_EQ(source->invalidations, value);
        EXPECT_EQ(target->publications, value);
      }
      iree_hal_command_buffer_release(commands);
    }
  }
  iree_hal_semaphore_release(done);
  iree_hal_buffer_release(source);
  iree_hal_buffer_release(target);
  iree_hal_device_group_release(group);
}

TEST_P(TaskQueueTest, OneShotCacheDefaultsAndExplicitEmptyBoundaries) {
  iree_hal_device_group_t* group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&group));
  auto* device = iree_hal_device_group_device_at(group, 0);
  auto* queue = iree_hal_device_queue(device, 0, 0);
  auto* source = new CacheBuffer();
  auto* target = new CacheBuffer();
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &done));
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_queue_barriers_t no_cache = {&empty, &empty};
  CacheBarrier acquire(iree_hal_make_buffer_ref(source, 10, 24),
                       IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE);
  CacheBarrier release(iree_hal_make_buffer_ref(target, 7, 24),
                       IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH);
  const iree_hal_queue_barriers_t explicit_cache = {&acquire.list,
                                                    &release.list};
  uint64_t value = 0;
  for (const iree_hal_queue_barriers_t* barriers :
       {static_cast<const iree_hal_queue_barriers_t*>(nullptr), &no_cache,
        &explicit_cache}) {
    source->remote.fill(0x37);
    source->host.fill(0x19);
    target->host.fill(0xA5);
    target->remote.fill(0xA5);
    ++value;
    const iree_hal_semaphore_list_t signals = {1, &done, &value};
    IREE_ASSERT_OK(iree_hal_queue_copy(queue, {}, signals, source, 10, target,
                                       7, 24, barriers,
                                       IREE_HAL_COPY_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(target->host[7], barriers == &no_cache ? 0x19 : 0x37);
    EXPECT_EQ(target->remote[7], barriers == &no_cache ? 0xA5 : 0x37);
    EXPECT_EQ(target->remote[6], 0xA5);
    EXPECT_EQ(target->remote[31], 0xA5);
  }
  EXPECT_EQ(source->invalidations, 2u);
  EXPECT_EQ(target->publications, 2u);
  iree_hal_semaphore_release(done);
  iree_hal_buffer_release(source);
  iree_hal_buffer_release(target);
  iree_hal_device_group_release(group);
}

TEST_P(TaskQueueTest, CacheFailureFailsCompletionBeforeDependentWork) {
  for (bool recorded : {false, true}) {
    for (bool fail_acquire : {true, false}) {
      iree_hal_device_group_t* group = nullptr;
      IREE_ASSERT_OK(CreateDeviceGroup(&group));
      auto* device = iree_hal_device_group_device_at(group, 0);
      auto* queue = iree_hal_device_queue(device, 0, 0);
      auto* source = new CacheBuffer();
      auto* target = new CacheBuffer();
      source->remote.fill(0x37);
      (fail_acquire ? source : target)->failure = IREE_STATUS_DATA_LOSS;
      iree_hal_semaphore_t* done = nullptr;
      IREE_ASSERT_OK(
          iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                    0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &done));
      uint64_t value = 1;
      const iree_hal_semaphore_list_t signals = {1, &done, &value};
      if (recorded) {
        iree_hal_command_buffer_t* commands = nullptr;
        IREE_ASSERT_OK(iree_hal_command_buffer_create(
            iree_hal_queue_family(queue), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
            IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0, &commands));
        CacheBarrier acquire(
            iree_hal_make_buffer_ref(source, 10, 24),
            IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE);
        CacheBarrier release(iree_hal_make_buffer_ref(target, 7, 24),
                             IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH);
        IREE_ASSERT_OK(iree_hal_command_buffer_begin(commands));
        IREE_ASSERT_OK(
            iree_hal_command_buffer_barrier(commands, &acquire.barrier));
        IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
            commands, acquire.range.buffer_ref, release.range.buffer_ref,
            IREE_HAL_COPY_FLAG_NONE));
        IREE_ASSERT_OK(
            iree_hal_command_buffer_barrier(commands, &release.barrier));
        IREE_ASSERT_OK(iree_hal_command_buffer_end(commands));
        IREE_ASSERT_OK(
            iree_hal_queue_execute(queue, {}, signals, commands, {},
                                   IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
        iree_hal_command_buffer_release(commands);
      } else {
        IREE_ASSERT_OK(iree_hal_queue_copy(queue, {}, signals, source, 10,
                                           target, 7, 24, nullptr,
                                           IREE_HAL_COPY_FLAG_NONE));
      }
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_DATA_LOSS,
          iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE));
      EXPECT_EQ(target->host[7], fail_acquire ? 0 : 0x37);
      EXPECT_EQ(target->remote[7], 0);
      iree_hal_semaphore_release(done);
      iree_hal_buffer_release(source);
      iree_hal_buffer_release(target);
      iree_hal_device_group_release(group);
    }
  }
}

TEST_P(TaskQueueTest, AsyncFileReadPublishesBeforeCompletion) {
  iree_hal_device_group_t* group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&group));
  auto* device = iree_hal_device_group_device_at(group, 0);
  auto* queue = iree_hal_device_queue(device, 0, 0);
  auto* target = new CacheBuffer();
  std::array<uint8_t, 32> input;
  input.fill(0x67);
  iree::testing::TempFilePath path("task_cache_file");
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(), iree_make_const_byte_span(input.data(), input.size()),
      iree_allocator_system()));
  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS |
          IREE_IO_FILE_MODE_ASYNC,
      path.path_view(), iree_allocator_system(), &handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(device, nullptr,
                                      IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                      IREE_HAL_MEMORY_ACCESS_READ, handle,
                                      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(handle);
  ASSERT_NE(iree_hal_file_async_handle(file), nullptr);
  iree_hal_semaphore_t* done = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &done));
  uint64_t value = 1;
  const iree_hal_semaphore_list_t signals = {1, &done, &value};
  IREE_ASSERT_OK(iree_hal_queue_read_linear(queue, {}, signals, file, 3, target,
                                            7, 24, nullptr,
                                            IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(target->publications, 1u);
  for (size_t i = 0; i < target->remote.size(); ++i) {
    EXPECT_EQ(target->remote[i], i >= 7 && i < 31 ? 0x67 : 0) << i;
  }
  std::thread::id control_worker;
  const auto capture_worker = iree_hal_make_host_call(
      [](void* user_data, const uint64_t[4], iree_hal_host_call_context_t*) {
        *static_cast<std::thread::id*>(user_data) = std::this_thread::get_id();
        return iree_ok_status();
      },
      &control_worker);
  uint64_t captured = 2;
  const uint64_t args[4] = {};
  const iree_hal_semaphore_list_t captured_signals = {1, &done, &captured};
  IREE_ASSERT_OK(iree_hal_queue_host_call(queue, signals, captured_signals,
                                          capture_worker, args,
                                          IREE_HAL_HOST_CALL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      captured_signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  if (GetParam() == 1) {
    EXPECT_EQ(target->maintenance_thread, control_worker);
  }
  iree_hal_file_release(file);
  iree_hal_semaphore_release(done);
  iree_hal_buffer_release(target);
  iree_hal_device_group_release(group);
}

TEST_P(TaskQueueTest, FileRectanglesCaptureOperandsAndPreservePadding) {
  iree_hal_device_group_t* group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&group));
  auto* device = iree_hal_device_group_device_at(group, 0);
  auto* queue = reinterpret_cast<iree_hal_task_queue_t*>(
      iree_hal_device_queue(device, 0, 0));
  for (bool native : {false, true}) {
    for (bool reading : {false, true}) {
      SCOPED_TRACE(native);
      SCOPED_TRACE(reading);
      std::array<uint8_t, 1024> file_bytes, buffer_bytes;
      for (size_t i = 0; i < file_bytes.size(); ++i) {
        file_bytes[i] = reading ? static_cast<uint8_t>(i * 17 + 3) : 0xDA;
        buffer_bytes[i] = reading ? 0xBD : static_cast<uint8_t>(i * 13 + 1);
      }
      auto expected_file = file_bytes;
      auto expected_buffer = buffer_bytes;
      iree::testing::TempFilePath path("task_file_rectangles");
      iree_hal_file_t* file = nullptr;
      IREE_ASSERT_OK(OpenTransferFile(
          device, path.path_view(),
          iree_make_byte_span(file_bytes.data(), file_bytes.size()), native,
          &file));
      std::atomic<uint32_t> releases{0};
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(WrapTransferBuffer(
          iree_make_byte_span(buffer_bytes.data(), buffer_bytes.size()),
          {[](void* user_data, iree_hal_buffer_t*) {
             static_cast<std::atomic<uint32_t>*>(user_data)->fetch_add(1);
           },
           &releases},
          &buffer));
      iree_hal_semaphore_t* semaphores[2] = {};
      for (auto& semaphore : semaphores) {
        IREE_ASSERT_OK(iree_hal_semaphore_create(
            device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
            IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
      }
      uint64_t value = 1;
      const iree_hal_semaphore_list_t waits = {1, &semaphores[0], &value};
      const iree_hal_semaphore_list_t signals = {1, &semaphores[1], &value};
      {
        const iree_hal_file_binding_t file_binding = {file, 32, 512};
        const iree_hal_buffer_binding_t buffer_binding = {buffer, 7, 768};
        const iree_hal_file_transfer_t transfers[] = {
            {
                .file_offset = 3,
                .buffer_offset = 5,
                .row_length = 7,
                .row_count = 3,
                .slice_count = 2,
                .file_strides = {16, 64},
                .buffer_strides = {11, 43},
            },
            {
                .file_offset = 256,
                .buffer_offset = 256,
                .row_length = 90,
                .row_count = 1,
                .slice_count = 1,
            },
            {
                .file_binding = UINT32_MAX,
                .buffer_binding = UINT32_MAX,
                .row_length = 0,
            },
        };
        const iree_hal_file_transfer_list_t list = {
            .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST,
            .host = {IREE_ARRAYSIZE(transfers), transfers},
        };
        for (const auto& transfer : transfers) {
          for (uint64_t slice = 0; slice < transfer.slice_count; ++slice) {
            for (uint64_t row = 0; row < transfer.row_count; ++row) {
              const size_t file_offset =
                  file_binding.offset + transfer.file_offset +
                  slice * transfer.file_strides.slice_stride +
                  row * transfer.file_strides.row_stride;
              const size_t buffer_offset =
                  buffer_binding.offset + transfer.buffer_offset +
                  slice * transfer.buffer_strides.slice_stride +
                  row * transfer.buffer_strides.row_stride;
              for (size_t byte = 0; byte < transfer.row_length; ++byte) {
                if (reading) {
                  expected_buffer[buffer_offset + byte] =
                      file_bytes[file_offset + byte];
                } else {
                  expected_file[file_offset + byte] =
                      buffer_bytes[buffer_offset + byte];
                }
              }
            }
          }
        }
        IREE_ASSERT_OK(
            SubmitFileTransfer(queue, waits, signals, {1, &file_binding},
                               {1, &buffer_binding}, &list, nullptr,
                               reading ? IREE_HAL_MEMORY_ACCESS_READ
                                       : IREE_HAL_MEMORY_ACCESS_WRITE));
      }
      // The call owns captures and references while its dependency is
      // unresolved.
      iree_hal_file_release(file);
      iree_hal_buffer_release(buffer);
      EXPECT_EQ(releases.load(), 0u);
      IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores[0], value, nullptr));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      EXPECT_EQ(releases.load(), 1u);
      if (native) {
        iree_io_file_contents_t* contents = nullptr;
        IREE_ASSERT_OK(iree_io_file_contents_read(
            path.path_view(), iree_allocator_system(), &contents));
        ASSERT_EQ(contents->buffer.data_length, expected_file.size());
        EXPECT_EQ(std::memcmp(contents->buffer.data, expected_file.data(),
                              expected_file.size()),
                  0);
        iree_io_file_contents_free(contents);
      } else {
        EXPECT_EQ(file_bytes, expected_file);
      }
      EXPECT_EQ(buffer_bytes, expected_buffer);
      for (auto* semaphore : semaphores) {
        iree_hal_semaphore_release(semaphore);
      }
    }
  }
  iree_hal_device_group_release(group);
}

TEST_P(TaskQueueTest, IndirectFilesUsePublishedPrefixAndAllowIndependentWork) {
  iree_hal_device_group_t* group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&group));
  auto* device = iree_hal_device_group_device_at(group, 0);
  auto* queue = reinterpret_cast<iree_hal_task_queue_t*>(
      iree_hal_device_queue(device, 0, 0));
  for (bool native : {false, true}) {
    for (bool reading : {false, true}) {
      SCOPED_TRACE(native);
      SCOPED_TRACE(reading);
      constexpr size_t kCapacity = 128;
      std::array<uint8_t, 17 + kCapacity * 90> file_bytes;
      std::array<uint8_t, 13 + kCapacity * 96 + 11> buffer_bytes;
      alignas(64) std::array<uint64_t, 8 + kCapacity * 8> metadata;
      for (size_t i = 0; i < file_bytes.size(); ++i) {
        file_bytes[i] = reading ? static_cast<uint8_t>(i * 17 + 3) : 0xDA;
      }
      for (size_t i = 0; i < buffer_bytes.size(); ++i) {
        buffer_bytes[i] = reading ? 0xBD : static_cast<uint8_t>(i * 13 + 1);
      }
      metadata.fill(UINT64_MAX);
      auto expected_buffer = buffer_bytes;
      auto expected_file = file_bytes;
      iree::testing::TempFilePath path("task_file_indirect");
      iree_hal_file_t* file = nullptr;
      IREE_ASSERT_OK(OpenTransferFile(
          device, path.path_view(),
          iree_make_byte_span(file_bytes.data(), file_bytes.size()), native,
          &file));
      iree_hal_buffer_t* buffer = nullptr;
      iree_hal_buffer_t* metadata_buffer = nullptr;
      IREE_ASSERT_OK(WrapTransferBuffer(
          iree_make_byte_span(buffer_bytes.data(), buffer_bytes.size()),
          iree_hal_buffer_release_callback_null(), &buffer));
      IREE_ASSERT_OK(WrapTransferBuffer(
          iree_make_byte_span(metadata.data(), sizeof(metadata)),
          iree_hal_buffer_release_callback_null(), &metadata_buffer));
      iree_hal_semaphore_t* semaphores[4] = {};
      for (auto& semaphore : semaphores) {
        IREE_ASSERT_OK(iree_hal_semaphore_create(
            device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
            IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
      }
      uint64_t value = 1;
      const iree_hal_semaphore_list_t waits = {1, &semaphores[0], &value};
      const iree_hal_semaphore_list_t signals = {1, &semaphores[1], &value};
      const iree_hal_semaphore_list_t independent = {1, &semaphores[2], &value};
      const iree_hal_semaphore_list_t consumed = {1, &semaphores[3], &value};
      std::array<uint8_t, 13 + kCapacity * 96 + 11> consumed_bytes = {};
      iree_hal_buffer_t* consumed_buffer = nullptr;
      IREE_ASSERT_OK(WrapTransferBuffer(
          iree_make_byte_span(consumed_bytes.data(), consumed_bytes.size()),
          iree_hal_buffer_release_callback_null(), &consumed_buffer));
      const iree_hal_file_binding_t file_binding = {file, 17, kCapacity * 90};
      const iree_hal_buffer_binding_t buffer_binding = {buffer, 13,
                                                        kCapacity * 96};
      const iree_hal_file_transfer_list_t list = {
          .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT,
          .indirect = {{metadata_buffer, 64, kCapacity * 64},
                       metadata_buffer,
                       0},
      };
      IREE_ASSERT_OK(
          SubmitFileTransfer(queue, waits, signals, {1, &file_binding},
                             {1, &buffer_binding}, &list, nullptr,
                             reading ? IREE_HAL_MEMORY_ACCESS_READ
                                     : IREE_HAL_MEMORY_ACCESS_WRITE));
      IREE_ASSERT_OK(iree_hal_queue_copy(
          &queue->base, signals, consumed, buffer, 0, consumed_buffer, 0,
          buffer_bytes.size(), nullptr, IREE_HAL_COPY_FLAG_NONE));
      iree_hal_buffer_release(consumed_buffer);
      IREE_ASSERT_OK(iree_hal_queue_barrier(&queue->base, {}, independent,
                                            nullptr,
                                            IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          independent, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      // Publish only the live prefix after enqueue. Inactive records remain
      // invalid.
      iree_unaligned_store_le_u64(&metadata[0], kCapacity - 1);
      for (size_t i = 0; i < kCapacity - 1; ++i) {
        iree_unaligned_store_le_u64(&metadata[8 + i * 8 + 0], i * 90);
        iree_unaligned_store_le_u64(&metadata[8 + i * 8 + 1], i * 96);
        iree_unaligned_store_le_u64(&metadata[8 + i * 8 + 2], i % 7 ? 90 : 0);
        if (i % 7) {
          if (reading) {
            std::memcpy(expected_buffer.data() + 13 + i * 96,
                        file_bytes.data() + 17 + i * 90, 90);
          } else {
            std::memcpy(expected_file.data() + 17 + i * 90,
                        buffer_bytes.data() + 13 + i * 96, 90);
          }
        }
      }
      iree_hal_file_release(file);
      iree_hal_buffer_release(buffer);
      iree_hal_buffer_release(metadata_buffer);
      IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores[0], value, nullptr));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          consumed, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      EXPECT_EQ(buffer_bytes, expected_buffer);
      EXPECT_EQ(consumed_bytes, expected_buffer);
      if (native) {
        iree_io_file_contents_t* contents = nullptr;
        IREE_ASSERT_OK(iree_io_file_contents_read(
            path.path_view(), iree_allocator_system(), &contents));
        ASSERT_EQ(contents->buffer.data_length, expected_file.size());
        EXPECT_EQ(std::memcmp(contents->buffer.data, expected_file.data(),
                              expected_file.size()),
                  0);
        iree_io_file_contents_free(contents);
      } else {
        EXPECT_EQ(file_bytes, expected_file);
      }
      for (auto* semaphore : semaphores) {
        iree_hal_semaphore_release(semaphore);
      }
    }
  }
  iree_hal_device_group_release(group);
}

TEST_P(TaskQueueTest, FileGroupReadsAndWritesMixedNativeAndMemoryFiles) {
  iree_hal_device_group_t* group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&group));
  auto* device = iree_hal_device_group_device_at(group, 0);
  auto* queue = reinterpret_cast<iree_hal_task_queue_t*>(
      iree_hal_device_queue(device, 0, 0));
  for (bool reading : {false, true}) {
    SCOPED_TRACE(reading);
    std::array<std::array<uint8_t, 11 + 64 * 90>, 2> file_bytes;
    std::array<std::array<uint8_t, 17 + 64 * 96>, 2> buffer_bytes;
    for (size_t binding = 0; binding < 2; ++binding) {
      for (size_t i = 0; i < file_bytes[binding].size(); ++i) {
        file_bytes[binding][i] =
            reading ? static_cast<uint8_t>(i + binding * 13) : 0xDA;
      }
      for (size_t i = 0; i < buffer_bytes[binding].size(); ++i) {
        buffer_bytes[binding][i] =
            reading ? 0xBD : static_cast<uint8_t>(i * 7 + binding * 31);
      }
    }
    auto expected_files = file_bytes;
    auto expected_buffers = buffer_bytes;
    iree::testing::TempFilePath path("task_file_mixed");
    iree_hal_file_binding_t files[2] = {};
    iree_hal_buffer_binding_t buffers[2] = {};
    for (size_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(OpenTransferFile(
          device, path.path_view(),
          iree_make_byte_span(file_bytes[i].data(), file_bytes[i].size()),
          i == 1, &files[i].file));
      files[i].offset = 11;
      files[i].length = 64 * 90;
      IREE_ASSERT_OK(WrapTransferBuffer(
          iree_make_byte_span(buffer_bytes[i].data(), buffer_bytes[i].size()),
          iree_hal_buffer_release_callback_null(), &buffers[i].buffer));
      buffers[i].offset = 17;
      buffers[i].length = 64 * 96;
    }
    std::array<iree_hal_file_transfer_t, 128> transfers = {};
    for (size_t i = 0; i < transfers.size(); ++i) {
      const uint32_t binding = i % 2;
      const size_t row = i / 2;
      transfers[i] = {
          .file_binding = binding,
          .buffer_binding = 1 - binding,
          .file_offset = row * 90,
          .buffer_offset = row * 96,
          .row_length = 90,
          .row_count = 1,
          .slice_count = 1,
      };
      if (reading) {
        std::memcpy(expected_buffers[1 - binding].data() + 17 + row * 96,
                    file_bytes[binding].data() + 11 + row * 90, 90);
      } else {
        std::memcpy(expected_files[binding].data() + 11 + row * 90,
                    buffer_bytes[1 - binding].data() + 17 + row * 96, 90);
      }
    }
    iree_hal_semaphore_t* done = nullptr;
    IREE_ASSERT_OK(
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &done));
    uint64_t value = 1;
    const iree_hal_semaphore_list_t signals = {1, &done, &value};
    auto invalid = transfers[0];
    invalid.buffer_offset = buffers[1].length;
    iree_hal_file_transfer_list_t list = {
        .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST,
        .host = {1, &invalid},
    };
    const auto access =
        reading ? IREE_HAL_MEMORY_ACCESS_READ : IREE_HAL_MEMORY_ACCESS_WRITE;
    EXPECT_THAT(
        iree::Status(SubmitFileTransfer(queue, {}, signals, {2, files},
                                        {2, buffers}, &list, nullptr, access)),
        iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
    uint64_t observed = UINT64_MAX;
    IREE_ASSERT_OK(iree_hal_semaphore_query(done, &observed));
    EXPECT_EQ(observed, 0u);
    list.host = {transfers.size(), transfers.data()};
    IREE_ASSERT_OK(SubmitFileTransfer(queue, {}, signals, {2, files},
                                      {2, buffers}, &list, nullptr, access));
    for (size_t i = 0; i < 2; ++i) {
      iree_hal_file_release(files[i].file);
      iree_hal_buffer_release(buffers[i].buffer);
    }
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(buffer_bytes, expected_buffers);
    EXPECT_EQ(file_bytes[0], expected_files[0]);
    iree_io_file_contents_t* contents = nullptr;
    IREE_ASSERT_OK(iree_io_file_contents_read(
        path.path_view(), iree_allocator_system(), &contents));
    ASSERT_EQ(contents->buffer.data_length, expected_files[1].size());
    EXPECT_EQ(std::memcmp(contents->buffer.data, expected_files[1].data(),
                          expected_files[1].size()),
              0);
    iree_io_file_contents_free(contents);
    iree_hal_semaphore_release(done);
  }
  iree_hal_device_group_release(group);
}

TEST_P(TaskQueueTest,
       FailedFileGroupRetiresAcceptedReadsBeforeReleasingBuffers) {
  iree_hal_device_group_t* group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&group));
  auto* device = iree_hal_device_group_device_at(group, 0);
  auto* queue = reinterpret_cast<iree_hal_task_queue_t*>(
      iree_hal_device_queue(device, 0, 0));
  std::array<uint8_t, 1024> file_bytes;
  file_bytes.fill(0x37);
  struct Payload {
    // Every read writes a separate row followed by untouched padding.
    std::array<uint8_t, 2048> bytes;
    // Captured prefix at final buffer release, before terminal signaling.
    std::array<uint8_t, 512> released_prefix;
    // Proves the batch dropped its final buffer reference before signaling.
    std::atomic<bool> released{false};
  } payload;
  payload.bytes.fill(0xDA);
  iree::testing::TempFilePath path("task_file_failed_group");
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(OpenTransferFile(
      device, path.path_view(),
      iree_make_byte_span(file_bytes.data(), file_bytes.size()), true, &file));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(WrapTransferBuffer(
      iree_make_byte_span(payload.bytes.data(), payload.bytes.size()),
      {[](void* user_data, iree_hal_buffer_t*) {
         auto* payload = static_cast<Payload*>(user_data);
         std::copy_n(payload->bytes.data(), payload->released_prefix.size(),
                     payload->released_prefix.data());
         payload->released.store(true);
       },
       &payload},
      &buffer));
  iree_hal_semaphore_t* semaphores[2] = {};
  for (auto& semaphore : semaphores) {
    IREE_ASSERT_OK(
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  }
  uint64_t value = 1;
  const iree_hal_semaphore_list_t waits = {1, &semaphores[0], &value};
  const iree_hal_semaphore_list_t signals = {1, &semaphores[1], &value};
  const iree_hal_file_binding_t file_binding = {file, 0, file_bytes.size()};
  const iree_hal_buffer_binding_t buffer_binding = {buffer, 0,
                                                    IREE_HAL_WHOLE_BUFFER};
  const iree_hal_file_transfer_t transfer = {
      .row_length = 16,
      .row_count = 64,
      .slice_count = 1,
      .file_strides = {16, 0},
      .buffer_strides = {32, 0},
  };
  const iree_hal_file_transfer_list_t list = {
      .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST,
      .host = {1, &transfer},
  };
  IREE_ASSERT_OK(SubmitFileTransfer(queue, waits, signals, {1, &file_binding},
                                    {1, &buffer_binding}, &list, nullptr,
                                    IREE_HAL_MEMORY_ACCESS_READ));
  // File size can change after capture. The first 16 rows remain readable;
  // later accepted requests encounter EOF and must drain before failure.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(), iree_make_const_byte_span(file_bytes.data(), 256),
      iree_allocator_system()));
  iree_hal_file_release(file);
  iree_hal_buffer_release(buffer);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores[0], value, nullptr));
  EXPECT_THAT(iree::Status(iree_hal_semaphore_list_wait(
                  signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE)),
              iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
  EXPECT_TRUE(payload.released.load());
  for (size_t i = 0; i < payload.released_prefix.size(); ++i) {
    EXPECT_EQ(payload.released_prefix[i], i % 32 < 16 ? 0x37 : 0xDA) << i;
  }
  for (size_t i = payload.released_prefix.size(); i < payload.bytes.size();
       ++i) {
    EXPECT_EQ(payload.bytes[i], 0xDA) << i;
  }
  for (auto* semaphore : semaphores) {
    iree_hal_semaphore_release(semaphore);
  }
  iree_hal_device_group_release(group);
}

TEST_P(TaskQueueTest, IndirectFileCountIsCheckedAfterProducerAndBeforeRecords) {
  for (uint64_t live_count : {UINT64_C(0), UINT64_MAX}) {
    SCOPED_TRACE(live_count);
    iree_hal_device_group_t* group = nullptr;
    IREE_ASSERT_OK(CreateDeviceGroup(&group));
    auto* device = iree_hal_device_group_device_at(group, 0);
    auto* queue = reinterpret_cast<iree_hal_task_queue_t*>(
        iree_hal_device_queue(device, 0, 0));
    alignas(64) std::array<uint64_t, 16> metadata;
    metadata.fill(UINT64_MAX);
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(WrapTransferBuffer(
        iree_make_byte_span(metadata.data(), sizeof(metadata)),
        iree_hal_buffer_release_callback_null(), &buffer));
    iree_hal_semaphore_t* semaphores[2] = {};
    for (auto& semaphore : semaphores) {
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
    }
    uint64_t value = 1;
    const iree_hal_semaphore_list_t waits = {1, &semaphores[0], &value};
    const iree_hal_semaphore_list_t signals = {1, &semaphores[1], &value};
    const iree_hal_file_transfer_list_t list = {
        .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT,
        .indirect = {{buffer, 64, 64}, buffer, 0},
    };
    IREE_ASSERT_OK(SubmitFileTransfer(queue, waits, signals, {}, {}, &list,
                                      nullptr, IREE_HAL_MEMORY_ACCESS_READ));
    iree_unaligned_store_le_u64(&metadata[0], live_count);
    iree_hal_buffer_release(buffer);
    IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores[0], value, nullptr));
    EXPECT_THAT(
        iree::Status(iree_hal_semaphore_list_wait(
            signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE)),
        iree::testing::status::StatusIs(live_count
                                            ? iree::StatusCode::kOutOfRange
                                            : iree::StatusCode::kOk));
    for (auto* semaphore : semaphores) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_hal_device_group_release(group);
  }
}

TEST_P(TaskQueueTest, ExecutesBindingsWithDeclaredPermissions) {
  iree_hal_device_group_t* device_group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&device_group));
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_queue_t* queue = iree_hal_device_queue(device, 0, 0);
  alignas(64) std::array<uint8_t, 18> source = {};
  alignas(64) std::array<uint8_t, 18> target = {};
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i);
  }
  target.fill(0xA5);

  iree_hal_buffer_t* source_buffer = nullptr;
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING |
          IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      16, iree_make_byte_span(source.data() + 1, 16),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &source_buffer));
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING |
          IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      16, iree_make_byte_span(target.data() + 1, 16),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &target_buffer));
  iree_hal_semaphore_t* completion = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion));

  const iree_hal_memory_transition_recipe_info_t release_operation = {
      .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
      .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE,
      .operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM,
      .range_granularity = 64,
  };
  const iree_hal_memory_transition_recipe_t release_recipe = {
      .effects = {IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM},
      .operation_count = 1,
      .operations = &release_operation,
  };

  for (iree_host_size_t binding_count : {0, 2}) {
    SCOPED_TRACE(binding_count);
    target.fill(0xA5);
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, binding_count, &command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
        command_buffer,
        binding_count ? iree_hal_make_indirect_buffer_ref(0, 0, 16)
                      : iree_hal_make_buffer_ref(source_buffer, 0, 16),
        binding_count ? iree_hal_make_indirect_buffer_ref(1, 0, 16)
                      : iree_hal_make_buffer_ref(target_buffer, 0, 16),
        IREE_HAL_COPY_FLAG_NONE));
    const iree_hal_buffer_ref_t barrier_ref =
        binding_count ? iree_hal_make_indirect_buffer_ref(1, 0, 16)
                      : iree_hal_make_buffer_ref(target_buffer, 0, 16);
    const iree_hal_buffer_barrier_t buffer_barrier = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
        .target_scope = 0,
        .buffer_ref = barrier_ref,
        .recipe = &release_recipe,
    };
    const iree_hal_barrier_t barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_NONE,
        .effects = release_recipe.effects,
        .memory_barrier_count = 0,
        .memory_barriers = nullptr,
        .buffer_barrier_count = 1,
        .buffer_barriers = &buffer_barrier,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(command_buffer, &barrier));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
    const iree_hal_buffer_binding_t binding_values[] = {
        {source_buffer, 0, 16},
        {target_buffer, 0, 16},
    };
    const iree_hal_buffer_binding_table_t bindings = {
        binding_count,
        binding_values,
    };
    uint64_t completion_value = binding_count + 1;
    const iree_hal_semaphore_list_t signal_list = {
        1,
        &completion,
        &completion_value,
    };
    IREE_ASSERT_OK(iree_hal_queue_execute(
        queue, iree_hal_semaphore_list_empty(), signal_list, command_buffer,
        bindings, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(0xA5, target.front());
    EXPECT_EQ(0xA5, target.back());
    for (size_t i = 1; i <= 16; ++i) {
      EXPECT_EQ(source[i], target[i]);
    }
    iree_hal_command_buffer_release(command_buffer);
  }
  iree_hal_semaphore_release(completion);
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(source_buffer);
  iree_hal_device_group_release(device_group);
}

TEST_P(TaskQueueTest, TransfersReleaseBuffersBeforeTerminalSignal) {
  for (bool fail_dependency : {false, true}) {
    for (size_t length : {16u, 512u * 1024u}) {
      SCOPED_TRACE(fail_dependency);
      SCOPED_TRACE(length);
      iree_hal_device_group_t* device_group = nullptr;
      IREE_ASSERT_OK(CreateDeviceGroup(&device_group));
      auto* device = iree_hal_device_group_device_at(device_group, 0);
      auto* queue = iree_hal_device_queue(device, 0, 0);
      iree_hal_semaphore_t* ready = nullptr;
      iree_hal_semaphore_t* completion = nullptr;
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion));
      struct ReleaseState {
        // Borrowed signal queried while the final buffer reference retires.
        iree_hal_semaphore_t* completion;
        // Number of completed release callbacks.
        std::atomic<uint32_t> count{0};
      } release_state = {completion};
      iree_hal_buffer_release_callback_t release_callback = {
          [](void* user_data, iree_hal_buffer_t* buffer) {
            auto* state = static_cast<ReleaseState*>(user_data);
            uint64_t value = UINT64_MAX;
            IREE_EXPECT_OK(iree_hal_semaphore_query(state->completion, &value));
            EXPECT_EQ(value, 0u);
            state->count.fetch_add(1, std::memory_order_release);
          },
          &release_state,
      };
      std::vector<uint8_t> source(length, 0xA5);
      std::vector<uint8_t> target(length, 0x00);
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
          iree_hal_buffer_placement_undefined(),
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          IREE_HAL_MEMORY_ACCESS_WRITE, IREE_HAL_BUFFER_USAGE_TRANSFER, length,
          iree_make_byte_span(target.data(), target.size()), release_callback,
          iree_allocator_system(), &buffer));
      uint64_t value = 1;
      const iree_hal_semaphore_list_t waits = {1, &ready, &value};
      const iree_hal_semaphore_list_t signals = {1, &completion, &value};
      {
        iree_hal_buffer_barrier_t range = {
            .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
            .target_scope = IREE_HAL_ACCESS_SCOPE_HOST_READ,
            .buffer_ref = iree_hal_make_buffer_ref(buffer, 0, length),
        };
        iree_hal_barrier_t after = {
            .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
            .target_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
            .buffer_barrier_count = 1,
            .buffer_barriers = &range,
        };
        const iree_hal_barrier_list_t list = {1, &after};
        const iree_hal_queue_barriers_t barriers = {nullptr, &list};
        IREE_ASSERT_OK(iree_hal_queue_upload(queue, waits, signals,
                                             source.data(), buffer, 0, length,
                                             &barriers));
      }
      // Descriptor storage has expired while the wait is unresolved. The
      // backend owns captured actions, not references to the caller's arrays.
      // The accepted operation owns the final buffer reference. Its explicit
      // dependency keeps that ownership alive until the caller drops its copy.
      iree_hal_buffer_release(buffer);
      if (fail_dependency) {
        iree_hal_semaphore_fail(ready,
                                iree_status_from_code(IREE_STATUS_CANCELLED));
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_CANCELLED,
            iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
        EXPECT_EQ(target, std::vector<uint8_t>(length, 0x00));
      } else {
        IREE_ASSERT_OK(iree_hal_semaphore_signal(ready, value, nullptr));
        IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
            signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
        EXPECT_EQ(target, source);
      }
      EXPECT_EQ(release_state.count.load(std::memory_order_acquire), 1u);
      iree_hal_semaphore_release(completion);
      iree_hal_semaphore_release(ready);
      iree_hal_device_group_release(device_group);
    }
  }
}

TEST_P(TaskQueueTest, ReleasesDeviceGroupWithAcceptedExecuteInFlight) {
  for (bool fail_dependency : {false, true}) {
    SCOPED_TRACE(fail_dependency);
    iree_hal_device_group_t* device_groups[2] = {};
    iree_hal_queue_t* queues[2] = {};
    iree_hal_command_buffer_t* command_buffers[2] = {};
    iree_hal_semaphore_t* completions[2] = {};
    std::array<std::array<uint8_t, 64>, 2> outputs = {};
    const uint8_t patterns[2] = {0xA5, 0x3C};
    for (size_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(CreateDeviceGroup(&device_groups[i]));
      auto* device = iree_hal_device_group_device_at(device_groups[i], 0);
      queues[i] = iree_hal_device_queue(device, 0, 0);
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completions[i]));
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
          iree_hal_buffer_placement_undefined(),
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          IREE_HAL_MEMORY_ACCESS_WRITE, IREE_HAL_BUFFER_USAGE_TRANSFER,
          outputs[i].size(),
          iree_make_byte_span(outputs[i].data(), outputs[i].size()),
          iree_hal_buffer_release_callback_null(), iree_allocator_system(),
          &buffer));
      IREE_ASSERT_OK(iree_hal_command_buffer_create(
          iree_hal_queue_family(queues[i]),
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
          IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0, &command_buffers[i]));
      IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffers[i]));
      IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
          command_buffers[i],
          iree_hal_make_buffer_ref(buffer, 0, outputs[i].size()), &patterns[i],
          sizeof(patterns[i]), IREE_HAL_FILL_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffers[i]));
      iree_hal_buffer_release(buffer);
    }
    iree_hal_semaphore_t* ready = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        iree_hal_device_group_device_at(device_groups[0], 0),
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &ready));
    uint64_t value = 1;
    const iree_hal_semaphore_list_t waits = {1, &ready, &value};
    const iree_hal_semaphore_list_t signals_a = {1, &completions[0], &value};
    const iree_hal_semaphore_list_t signals_b = {1, &completions[1], &value};
    IREE_ASSERT_OK(iree_hal_queue_execute(queues[0], waits, signals_a,
                                          command_buffers[0],
                                          iree_hal_buffer_binding_table_empty(),
                                          IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
    iree_hal_command_buffer_release(command_buffers[0]);

    auto* task_queue = reinterpret_cast<iree_hal_task_queue_t*>(queues[0]);
    std::thread destroying(
        [&] { iree_hal_device_group_release(device_groups[0]); });
    // The unresolved wait owns the sole accepted operation. Observe admission
    // closing before running B, so B proves the shared executor remains usable
    // while A's actual destructor is waiting for its external dependency.
    while (iree_atomic_load(&task_queue->pending_operation_count,
                            iree_memory_order_acquire) != 1) {
      std::this_thread::yield();
    }
    iree_status_t submit_status = iree_hal_queue_execute(
        queues[1], iree_hal_semaphore_list_empty(), signals_b,
        command_buffers[1], iree_hal_buffer_binding_table_empty(),
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
    const bool submitted_b = iree_status_is_ok(submit_status);
    IREE_EXPECT_OK(submit_status);
    iree_hal_command_buffer_release(command_buffers[1]);
    iree_hal_device_group_release(device_groups[1]);
    if (submitted_b) {
      IREE_EXPECT_OK(iree_hal_semaphore_list_wait(
          signals_b, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      std::array<uint8_t, 64> expected_b;
      expected_b.fill(patterns[1]);
      EXPECT_EQ(outputs[1], expected_b);
    }

    iree::Status failure(iree_status_annotate_f(
        iree_make_status(IREE_STATUS_DATA_LOSS, "accepted dependency failed"),
        "preserve dependency context during queue destruction"));
    if (fail_dependency) {
      iree_hal_semaphore_fail(ready, iree_status_clone(failure.get()));
    } else {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(ready, value, nullptr));
    }
    destroying.join();
    iree::Status result(iree_hal_semaphore_list_wait(
        signals_a, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(result.ToString(), fail_dependency ? failure.ToString() : "OK");
    std::array<uint8_t, 64> expected_a;
    expected_a.fill(fail_dependency ? 0 : patterns[0]);
    EXPECT_EQ(outputs[0], expected_a);
    iree_hal_semaphore_release(ready);
    for (auto* completion : completions) {
      iree_hal_semaphore_release(completion);
    }
  }
}

TEST_P(TaskQueueTest, AbortOnFailurePreservesDiagnostic) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  auto diagnostic =
      ::testing::AllOf(::testing::HasSubstr("DATA_LOSS"),
                       ::testing::HasSubstr("queue operation failed"),
                       ::testing::HasSubstr("operation diagnostic context"));
#else
  const char* diagnostic = "DATA_LOSS";
#endif
  EXPECT_DEATH_IF_SUPPORTED(
      {
        iree_hal_device_group_t* device_group = nullptr;
        IREE_ASSERT_OK(
            CreateDeviceGroup(/*abort_on_failure=*/true, &device_group));
        auto* device = iree_hal_device_group_device_at(device_group, 0);
        auto* queue = iree_hal_device_queue(device, 0, 0);
        auto call = iree_hal_make_host_call(
            +[](void*, const uint64_t[4], iree_hal_host_call_context_t*) {
              return iree_status_annotate_f(
                  iree_make_status(IREE_STATUS_DATA_LOSS,
                                   "queue operation failed"),
                  "operation diagnostic context");
            },
            nullptr);
        uint64_t args[4] = {};
        IREE_ASSERT_OK(
            iree_hal_queue_host_call(queue, iree_hal_semaphore_list_empty(),
                                     iree_hal_semaphore_list_empty(), call,
                                     args, IREE_HAL_HOST_CALL_FLAG_NONE));
        iree_hal_device_group_release(device_group);
      },
      diagnostic);
}

TEST_P(TaskQueueTest, ReusedDynamicQueueSlotAdvancesIncarnation) {
  iree_hal_device_group_t* device_group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&device_group));
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  const iree_hal_queue_family_t* queue_family =
      iree_hal_device_queue_family(device, /*family_ordinal=*/0);
  ASSERT_NE(nullptr, queue_family);

  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* queue_a = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_acquire(queue_family, &params, &queue_a));
  iree_hal_queue_t* queue_b = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_acquire(queue_family, &params, &queue_b));
  const iree_async_axis_t axis_a = ((iree_hal_task_queue_t*)queue_a)->axis;
  const iree_async_axis_t axis_b = ((iree_hal_task_queue_t*)queue_b)->axis;
  EXPECT_NE(iree_async_axis_queue_index(axis_a),
            iree_async_axis_queue_index(axis_b));
  EXPECT_NE(axis_a, axis_b);

  iree_hal_queue_release(queue_a);
  queue_a = nullptr;

  iree_hal_queue_t* queue_c = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_acquire(queue_family, &params, &queue_c));
  const iree_async_axis_t axis_c = ((iree_hal_task_queue_t*)queue_c)->axis;
  EXPECT_EQ(iree_async_axis_queue_index(axis_a),
            iree_async_axis_queue_index(axis_c));
  EXPECT_EQ(iree_async_axis_queue_incarnation(axis_a) + 1,
            iree_async_axis_queue_incarnation(axis_c));
  EXPECT_NE(axis_a, axis_c);

  iree_hal_queue_release(queue_c);
  iree_hal_queue_release(queue_b);
  iree_hal_device_group_release(device_group);
}

TEST_P(TaskQueueTest, ConcurrentQueueAcquisitionWithQueueProgress) {
  constexpr int kThreadCount = 4;
  constexpr int kQueuesPerThread = 16;
  iree_hal_device_group_t* device_group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&device_group));
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_queue_t* provisioned_queue =
      iree_hal_device_queue(device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
  const iree_hal_queue_family_t* queue_family =
      iree_hal_queue_family(provisioned_queue);
  iree_async_frontier_tracker_t* frontier_tracker =
      ((iree_hal_task_queue_t*)provisioned_queue)->frontier_tracker;

  std::atomic<bool> start{false};
  std::atomic<int> remaining{kThreadCount};
  std::array<iree_async_axis_t, kThreadCount * kQueuesPerThread> axes = {};
  std::vector<std::thread> threads;
  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index]() {
      iree_hal_semaphore_t* completion = nullptr;
      iree_status_t status = iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion);
      iree_hal_queue_params_t params;
      iree_hal_queue_params_initialize(&params);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kQueuesPerThread && iree_status_is_ok(status); ++i) {
        iree_hal_queue_t* queue = nullptr;
        status = iree_hal_queue_acquire(queue_family, &params, &queue);
        uint64_t completion_value = i + 1;
        const iree_hal_semaphore_list_t signal_list = {1, &completion,
                                                       &completion_value};
        if (iree_status_is_ok(status)) {
          axes[thread_index * kQueuesPerThread + i] =
              ((iree_hal_task_queue_t*)queue)->axis;
          status = iree_hal_queue_barrier(
              queue, iree_hal_semaphore_list_empty(), signal_list,
              /*barriers=*/NULL, IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
        }
        if (iree_status_is_ok(status)) {
          status = iree_hal_semaphore_list_wait(
              signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
        }
        iree_hal_queue_release(queue);
      }
      iree_hal_semaphore_release(completion);
      IREE_EXPECT_OK(status);
      remaining.fetch_sub(1, std::memory_order_release);
    });
  }

  // Keep an existing queue completing work while other threads acquire, use
  // and release queues in this same sealed device group.
  iree_hal_semaphore_t* completion = nullptr;
  iree_status_t status = iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion);
  start.store(true, std::memory_order_release);
  uint64_t completion_value = 0;
  do {
    const iree_hal_semaphore_list_t signal_list = {1, &completion,
                                                   &completion_value};
    ++completion_value;
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_barrier(
          provisioned_queue, iree_hal_semaphore_list_empty(), signal_list,
          /*barriers=*/NULL, IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_list_wait(
          signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    }
  } while (iree_status_is_ok(status) &&
           remaining.load(std::memory_order_acquire) != 0);

  for (auto& thread : threads) {
    thread.join();
  }
  IREE_EXPECT_OK(status);
  for (iree_async_axis_t axis : axes) {
    EXPECT_TRUE(
        iree_async_frontier_tracker_query_epoch(frontier_tracker, axis, 1));
  }
  std::sort(axes.begin(), axes.end());
  EXPECT_EQ(std::adjacent_find(axes.begin(), axes.end()), axes.end());
  iree_hal_semaphore_release(completion);
  iree_hal_device_group_release(device_group);
}

INSTANTIATE_TEST_SUITE_P(WorkerCounts, TaskQueueTest, ::testing::Values(1, 4));

}  // namespace
