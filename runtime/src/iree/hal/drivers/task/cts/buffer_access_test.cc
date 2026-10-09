// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/passthrough_pool.h"

namespace iree::hal::cts {
namespace {

enum class DispatchMode { kQueue, kInlineQueue, kDirect, kIndirect };

class TaskBufferAccessTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    LoadExecutableOrSkipUnsupported("elementwise_mul.bin", &executable_);
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    CtsTestBase::TearDown();
  }

  // Real ELF executable loaded by the device's native executable loader.
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(TaskBufferAccessTest, DirectionalStorageBindings) {
  iree_hal_queue_t* queue =
      QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
  ASSERT_NE(queue, nullptr);
  std::array<float, 4> values[3] = {
      {1.0f, 2.0f, 3.0f, 4.0f},
      {100.0f, 200.0f, 300.0f, 400.0f},
      {},
  };
  Ref<iree_hal_buffer_t> buffers[3];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    iree_hal_buffer_params_t params = {};
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    params.access =
        i == 2 ? IREE_HAL_MEMORY_ACCESS_WRITE : IREE_HAL_MEMORY_ACCESS_READ;
    params.usage = i == 2 ? IREE_HAL_BUFFER_USAGE_STORAGE_WRITE
                          : IREE_HAL_BUFFER_USAGE_STORAGE_READ;
    iree_hal_external_buffer_t external = {
        .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
        .size = sizeof(values[i])};
    external.handle.host_allocation.ptr = values[i].data();
    IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
        device_allocator_, params, &external,
        iree_hal_buffer_release_callback_null(), buffers[i].out()));
    EXPECT_EQ(iree_hal_buffer_allowed_usage(buffers[i]), params.usage);
    iree_hal_buffer_mapping_t mapping = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_PERMISSION_DENIED,
        iree_hal_buffer_map_range(buffers[i], IREE_HAL_MAPPING_MODE_SCOPED,
                                  params.access, IREE_HAL_BUFFER_MAP_FLAG_NONE,
                                  0, sizeof(values[i]), &mapping));
  }

  for (DispatchMode mode : {DispatchMode::kQueue, DispatchMode::kInlineQueue,
                            DispatchMode::kDirect, DispatchMode::kIndirect}) {
    SCOPED_TRACE(static_cast<int>(mode));
    values[2].fill(-1.0f);
    const bool indirect = mode == DispatchMode::kIndirect;
    iree_hal_buffer_ref_t refs[3] = {};
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(refs); ++i) {
      refs[i] =
          indirect ? iree_hal_make_indirect_buffer_ref(i, 0, sizeof(values[i]))
                   : iree_hal_make_buffer_ref(buffers[i], 0, sizeof(values[i]));
    }
    const iree_hal_buffer_ref_list_t bindings = {IREE_ARRAYSIZE(refs), refs};
    SemaphoreList signal(device_, {0}, {1});
    if (mode == DispatchMode::kQueue || mode == DispatchMode::kInlineQueue) {
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queue, iree_hal_semaphore_list_empty(), signal, executable_,
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), bindings, /*barriers=*/NULL,
          mode == DispatchMode::kInlineQueue
              ? IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION
              : IREE_HAL_DISPATCH_FLAG_NONE));
    } else {
      Ref<iree_hal_command_buffer_t> command_buffer;
      IREE_ASSERT_OK(iree_hal_command_buffer_create(
          iree_hal_queue_family(queue), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
          IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_ATOMIC,
          indirect ? 3 : 0, command_buffer.out()));
      IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
      const iree_hal_buffer_ref_t atomic_ref =
          indirect ? iree_hal_make_indirect_buffer_ref(2, 0, sizeof(uint32_t))
                   : iree_hal_make_buffer_ref(buffers[2], 0, sizeof(uint32_t));
      iree_hal_atomic_store_params_t store = {};
      store.width = IREE_HAL_ATOMIC_WIDTH_32;
      store.flags = IREE_HAL_ATOMIC_FLAG_RELEASE;
      IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
          command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
          IREE_HAL_EXECUTION_STAGE_DISPATCH, atomic_ref, store));
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          command_buffer, executable_,
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(1, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
      iree_hal_buffer_binding_t entries[] = {
          {buffers[0], 0, sizeof(values[0])},
          {buffers[1], 0, sizeof(values[1])},
          {buffers[2], 0, sizeof(values[2])},
      };
      const iree_hal_buffer_binding_table_t table = {
          indirect ? IREE_ARRAYSIZE(entries) : 0,
          indirect ? entries : nullptr,
      };
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
      if (indirect) {
        // The opaque dispatch must preserve the atomic store's exact WRITE
        // requirement on the same slot. Rejection occurs before submission.
        iree_hal_buffer_params_t params = {};
        params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
        params.usage = IREE_HAL_BUFFER_USAGE_STORAGE_READ;
        params.access = IREE_HAL_MEMORY_ACCESS_ALL;
        iree_hal_external_buffer_t external = {
            .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
            .size = sizeof(values[2])};
        external.handle.host_allocation.ptr = values[2].data();
        Ref<iree_hal_buffer_t> storage_read_only;
        IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
            device_allocator_, params, &external,
            iree_hal_buffer_release_callback_null(), storage_read_only.out()));
        entries[2].buffer = storage_read_only;
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_PERMISSION_DENIED,
            iree_hal_command_buffer_validate_submission(command_buffer, table));
        entries[2].buffer = buffers[2];
      }
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queue, iree_hal_semaphore_list_empty(), signal, command_buffer, table,
          IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
    }
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_THAT(values[2],
                ::testing::ElementsAre(100.0f, 400.0f, 900.0f, 1600.0f));
  }
}

// Releases a deliberately parked allocation even when a recording assertion
// fails, so fixture teardown can drain the queue's retained operations.
class AllocationGate {
 public:
  explicit AllocationGate(iree_hal_semaphore_t* semaphore)
      : semaphore_(semaphore) {}
  ~AllocationGate() {
    if (semaphore_) {
      IREE_EXPECT_OK(Signal());
    }
  }
  iree_status_t Signal() {
    auto* semaphore = semaphore_;
    semaphore_ = nullptr;
    return iree_hal_semaphore_signal(semaphore, 1, nullptr);
  }

 private:
  // Borrowed semaphore owned by the test until this guard is destroyed.
  iree_hal_semaphore_t* semaphore_;
};

TEST_P(TaskBufferAccessTest, ExecutionOnlyNestedViews) {
  iree_hal_queue_t* queue = QueueForCommandCategories(
      IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_TRANSFER);
  ASSERT_NE(queue, nullptr);
  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
      device_, iree_hal_queue_family(queue), &backend));
  Ref<iree_hal_pool_t> pool;
  iree_hal_passthrough_pool_options_t options = {.asan = backend.asan};
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      options, backend.slab_provider, backend.notification,
      backend.frontier_tracker, backend.maintenance, iree_allocator_system(),
      pool.out()));

  for (bool transient : {false, true}) {
    SCOPED_TRACE(transient ? "queue allocation" : "immediate allocation");
    for (DispatchMode mode : {DispatchMode::kQueue, DispatchMode::kInlineQueue,
                              DispatchMode::kDirect, DispatchMode::kIndirect}) {
      SCOPED_TRACE(static_cast<int>(mode));
      const std::array<float, 8> inputs[3] = {
          {11, 22, 1, 2, 3, 4, 77, 88},
          {11, 22, 100, 200, 300, 400, 77, 88},
          {11, 22, -1, -1, -1, -1, 77, 88},
      };
      Ref<iree_hal_buffer_t> roots[3];
      Ref<iree_hal_buffer_t> views[3];
      SemaphoreList gate(device_, {0}, {1});
      SemaphoreList allocated(device_, {0}, {1});
      SemaphoreList uploaded(device_, {0, 0, 0}, {1, 1, 1});
      SemaphoreList executed(device_, {0}, {1});
      SemaphoreList downloaded(device_, {0}, {1});
      AllocationGate release_gate(gate.semaphores[0]);
      iree_hal_buffer_params_t params = {};
      params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
      params.access = IREE_HAL_MEMORY_ACCESS_ALL;
      params.usage =
          IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
      const iree_hal_pool_reservation_request_t requests[] = {
          {params, sizeof(inputs[0])},
          {params, sizeof(inputs[1])},
          {params, sizeof(inputs[2])},
      };
      if (transient) {
        iree_hal_buffer_t* buffers[3] = {};
        IREE_ASSERT_OK(iree_hal_queue_alloca(queue, gate, allocated, pool,
                                             IREE_ARRAYSIZE(requests), requests,
                                             buffers));
        for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(roots); ++i) {
          roots[i].reset(buffers[i]);
        }
      } else {
        for (auto& root : roots) {
          IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
              device_allocator_, params, sizeof(inputs[0]), root.out()));
        }
      }
      for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(views); ++i) {
        Ref<iree_hal_buffer_t> outer;
        IREE_ASSERT_OK(
            iree_hal_buffer_subspan(roots[i], sizeof(float), 6 * sizeof(float),
                                    iree_allocator_system(), outer.out()));
        IREE_ASSERT_OK(
            iree_hal_buffer_subspan(outer, sizeof(float), 4 * sizeof(float),
                                    iree_allocator_system(), views[i].out()));
        EXPECT_EQ(iree_hal_buffer_allowed_usage(views[i]), params.usage);
        iree_hal_buffer_mapping_t mapping = {};
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_PERMISSION_DENIED,
            iree_hal_buffer_map_range(views[i], IREE_HAL_MAPPING_MODE_SCOPED,
                                      IREE_HAL_MEMORY_ACCESS_READ,
                                      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                      IREE_HAL_WHOLE_BUFFER, &mapping));
        const iree_hal_semaphore_list_t upload_done = {
            1, &uploaded.semaphores[i], &uploaded.payload_values[i]};
        IREE_ASSERT_OK(iree_hal_queue_upload(
            queue, transient ? allocated : gate, upload_done, inputs[i].data(),
            roots[i], 0, sizeof(inputs[i]), /*barriers=*/NULL));
      }

      const bool indirect = mode == DispatchMode::kIndirect;
      iree_hal_buffer_ref_t refs[3] = {};
      iree_hal_buffer_binding_t entries[3] = {};
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(refs); ++i) {
        refs[i] =
            indirect
                ? iree_hal_make_indirect_buffer_ref(i, 0, 4 * sizeof(float))
                : iree_hal_make_buffer_ref(views[i], 0, 4 * sizeof(float));
        entries[i] = {views[i], 0, 4 * sizeof(float)};
      }
      const iree_hal_buffer_ref_list_t bindings = {IREE_ARRAYSIZE(refs), refs};
      if (mode == DispatchMode::kQueue || mode == DispatchMode::kInlineQueue) {
        IREE_ASSERT_OK(iree_hal_queue_dispatch(
            queue, uploaded, executed, executable_,
            iree_hal_executable_function_from_index(0),
            iree_hal_make_static_dispatch_config(1, 1, 1),
            iree_const_byte_span_empty(), bindings, /*barriers=*/NULL,
            mode == DispatchMode::kInlineQueue
                ? IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION
                : IREE_HAL_DISPATCH_FLAG_NONE));
      } else {
        // Record before the allocation gate opens. Both direct transient views
        // and explicit binding-table slots resolve only after allocation waits.
        Ref<iree_hal_command_buffer_t> command_buffer;
        IREE_ASSERT_OK(iree_hal_command_buffer_create(
            iree_hal_queue_family(queue), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
            IREE_HAL_COMMAND_CATEGORY_DISPATCH, indirect ? 3 : 0,
            command_buffer.out()));
        IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
        IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
            command_buffer, executable_,
            iree_hal_executable_function_from_index(0),
            iree_hal_make_static_dispatch_config(1, 1, 1),
            iree_const_byte_span_empty(), bindings,
            IREE_HAL_DISPATCH_FLAG_NONE));
        IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
        const iree_hal_buffer_binding_table_t table = {
            indirect ? IREE_ARRAYSIZE(entries) : 0,
            indirect ? entries : nullptr,
        };
        IREE_ASSERT_OK(
            iree_hal_queue_execute(queue, uploaded, executed, command_buffer,
                                   table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
      }
      IREE_ASSERT_OK(release_gate.Signal());
      std::array<float, 8> result = {};
      IREE_ASSERT_OK(iree_hal_queue_download(
          queue, executed, downloaded, roots[2], 0, result.data(),
          sizeof(result), /*barriers=*/NULL));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          downloaded, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      EXPECT_THAT(result,
                  ::testing::ElementsAre(11, 22, 100, 400, 900, 1600, 77, 88));
      if (transient) {
        iree_hal_buffer_t* buffers[] = {roots[0], roots[1], roots[2]};
        SemaphoreList deallocated(device_, {0}, {1});
        IREE_ASSERT_OK(iree_hal_queue_dealloca(
            queue, downloaded, deallocated, IREE_ARRAYSIZE(buffers), buffers));
        IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
            deallocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      }
    }
  }
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(TaskBufferAccessTest);

}  // namespace
}  // namespace iree::hal::cts
