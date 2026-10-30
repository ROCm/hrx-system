// Copyright 2026 The IREE Authors
//
// Licensed under the Apache v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_pool.h"

#include <array>
#include <atomic>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/task/registration/driver_module.h"
#include "iree/hal/replay/execute_pool.h"
#include "iree/hal/replay/recorder.h"
#include "iree/hal/slab_pool.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct PoolAllocatorState {
  // Whether new allocations fail while frees continue to reach the system.
  std::atomic<bool> fail_allocations{false};
  // Number of allocation attempts, including rejected attempts.
  std::atomic<iree_host_size_t> allocation_attempt_count{0};
  // Number of successful allocations not yet freed.
  std::atomic<iree_host_size_t> live_allocation_count{0};
};

static iree_status_t PoolAllocatorCtl(void* self,
                                      iree_allocator_command_t command,
                                      const void* params, void** inout_ptr) {
  auto* state = static_cast<PoolAllocatorState*>(self);
  if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
      command == IREE_ALLOCATOR_COMMAND_CALLOC) {
    state->allocation_attempt_count.fetch_add(1, std::memory_order_relaxed);
    if (state->fail_allocations.load(std::memory_order_relaxed)) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
  }
  iree_allocator_t system_allocator = iree_allocator_system();
  iree_status_t status =
      system_allocator.ctl(system_allocator.self, command, params, inout_ptr);
  if (iree_status_is_ok(status)) {
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC) {
      state->live_allocation_count.fetch_add(1, std::memory_order_relaxed);
    } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      state->live_allocation_count.fetch_sub(1, std::memory_order_relaxed);
    }
  }
  return status;
}

static iree_allocator_t PoolAllocator(PoolAllocatorState* state) {
  return iree_allocator_t{state, PoolAllocatorCtl};
}

static iree_hal_device_group_t* CreateTaskDeviceGroup() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_driver_registry_t* registry = nullptr;
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_CHECK_OK(iree_hal_task_driver_module_register(registry));
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  iree_hal_device_t* device = nullptr;
  iree_status_t status =
      iree_hal_create_device(registry, IREE_SV("task"), &create_params,
                             iree_allocator_system(), &device);
  iree_hal_driver_registry_free(registry);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));
  iree_hal_device_group_t* group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &group));
  iree_async_frontier_tracker_release(frontier_tracker);
  iree_hal_device_release(device);
  return group;
}

class ReplayRecorderPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_io_file_handle_t* file_handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(recording_.data(), recording_.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &file_handle));
    iree_status_t status = iree_hal_replay_recorder_create(
        file_handle, nullptr, iree_allocator_system(), &recorder_);
    iree_io_file_handle_release(file_handle);
    IREE_ASSERT_OK(status);

    source_group_ = CreateTaskDeviceGroup();
    IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
        recorder_, source_group_, iree_allocator_system(), &wrapped_group_));

    const iree_hal_external_storage_class_t host_pointer_class =
        iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
    const iree_hal_pool_import_class_t import_class = {
        .storage_class = host_pointer_class,
        .exports = {},
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 0,
        .families = nullptr,
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_ALL,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
                .properties = IREE_HAL_HOST_MEMORY_PROPERTY_NONE,
            },
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
    options.imports.class_count = 1;
    options.imports.classes = &import_class;
    options.trace_name = IREE_SV("replay.import_ownership");
    IREE_ASSERT_OK(iree_hal_slab_pool_create(wrapped_group_, scope, &options,
                                             PoolAllocator(&allocator_state_),
                                             &pool_));
  }

  void TearDown() override {
    iree_hal_pool_release(pool_);
    iree_hal_device_group_release(wrapped_group_);
    iree_hal_device_group_release(source_group_);
    if (recorder_) {
      IREE_EXPECT_OK(iree_hal_replay_recorder_close(recorder_));
      iree_hal_replay_recorder_release(recorder_);
    }
    EXPECT_EQ(0u, allocator_state_.live_allocation_count.load(
                      std::memory_order_relaxed));
  }

  iree_status_t Import(iree_hal_external_storage_class_t storage_class,
                       iree_hal_buffer_t** out_buffer) {
    iree_hal_external_storage_t external_storage = {
        .storage_class = storage_class,
        .byte_length = sizeof(storage_),
    };
    external_storage.payload.host_pointer = storage_;
    external_storage.host_properties =
        (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
         IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK);
    external_storage.maintenance_range =
        iree_hal_make_buffer_range(0, sizeof(storage_));
    external_storage.release =
        [](void* user_data, iree_hal_external_storage_class_t storage_class,
           iree_hal_external_storage_payload_t payload) {
          (void)storage_class;
          (void)payload;
          ++*static_cast<int*>(user_data);
        };
    external_storage.release_user_data = &release_count_;
    return iree_hal_pool_import_buffer(
        pool_, &external_storage,
        iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER), out_buffer);
  }

  iree_status_t ImportHostPointer(iree_hal_buffer_t** out_buffer) {
    return Import(iree_hal_make_external_storage_class(
                      IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
                  out_buffer);
  }

  void VerifySuccessfulImport() {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(ImportHostPointer(&buffer));
    EXPECT_EQ(0, release_count_);
    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, IREE_HAL_WHOLE_BUFFER, &mapping));
    EXPECT_EQ(storage_, mapping.contents.data);
    EXPECT_EQ(sizeof(storage_), mapping.contents.data_length);
    mapping.contents.data[7] = 0xA7;
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    EXPECT_EQ(0xA7, storage_[7]);
    EXPECT_EQ(0, release_count_);
    iree_hal_buffer_release(buffer);

    // Pool destruction joins deferred native retirement and external release.
    iree_hal_pool_release(pool_);
    pool_ = nullptr;
    EXPECT_EQ(1, release_count_);
  }

  // Storage owned by the test caller, never freed by the release counter.
  alignas(64) uint8_t storage_[64] = {};
  // Number of times an imported buffer relinquished the caller's storage.
  int release_count_ = 0;
  // Host allocation control for the recording and native pool wrappers.
  PoolAllocatorState allocator_state_;
  // Destination for the actual capture stream.
  std::array<uint8_t, 32768> recording_ = {};
  // Recorder retained until its wrappers have been released.
  iree_hal_replay_recorder_t* recorder_ = nullptr;
  // Native devices used by the recording group.
  iree_hal_device_group_t* source_group_ = nullptr;
  // Recording wrappers owning the captured pool identity.
  iree_hal_device_group_t* wrapped_group_ = nullptr;
  // Import-only recording pool under test.
  iree_hal_pool_t* pool_ = nullptr;
};

TEST_F(ReplayRecorderPoolTest, PreservesZeroComputeRequirementsInEveryCell) {
  iree_hal_pool_family_access_t access = {
      .family = iree_hal_device_queue_family(
          iree_hal_device_group_device_at(source_group_, 0), 0),
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .atomics =
          {
              .queue_zero_compute = {IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT,
                                     IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
                                     IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD,
                                     IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR},
          },
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &access,
  };
  IREE_ASSERT_OK(iree_hal_pool_scope_validate(source_group_, scope));
  iree_hal_replay_pool_family_access_payload_t payload;
  iree_hal_replay_recorder_pool_capture_families(&scope, &payload);
  iree_hal_replay_executor_t executor = {
      .file_version_minor = IREE_HAL_REPLAY_FILE_VERSION_MINOR,
      .device_group = source_group_,
  };
  iree_hal_pool_family_access_t decoded;
  IREE_ASSERT_OK(iree_hal_replay_executor_pool_decode_families(
      &executor, 1, reinterpret_cast<const uint8_t*>(&payload), &decoded));
  EXPECT_EQ(decoded.family, access.family);
  EXPECT_EQ(decoded.atomics.queue_zero_compute.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
  EXPECT_EQ(decoded.atomics.queue_zero_compute.device_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(decoded.atomics.queue_zero_compute.system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD);
  EXPECT_EQ(decoded.atomics.queue_zero_compute.system_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR);
  // Replay must preserve the hard requirement even when this executor cannot
  // satisfy it. The CPU backend must not silently perform the work on a worker.
  const iree_hal_pool_scope_t decoded_scope = {
      .family_count = 1,
      .families = &decoded,
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  iree_hal_pool_t* rejected = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_slab_pool_create(source_group_, decoded_scope, &options,
                                iree_allocator_system(), &rejected));
  EXPECT_EQ(rejected, nullptr);
  executor.file_version_minor =
      IREE_HAL_REPLAY_ZERO_COMPUTE_ATOMICS_VERSION_MINOR - 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_replay_executor_pool_decode_families(
          &executor, 1, reinterpret_cast<const uint8_t*>(&payload), &decoded));
  payload.queue_zero_compute_atomic_operations = 0;
  IREE_ASSERT_OK(iree_hal_replay_executor_pool_decode_families(
      &executor, 1, reinterpret_cast<const uint8_t*>(&payload), &decoded));
}

TEST_F(ReplayRecorderPoolTest, ProxyFailurePreservesCallerOwnership) {
  const auto allocation_attempt_count =
      allocator_state_.allocation_attempt_count.load(std::memory_order_relaxed);
  const auto live_allocation_count =
      allocator_state_.live_allocation_count.load(std::memory_order_relaxed);
  allocator_state_.fail_allocations.store(true, std::memory_order_relaxed);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        ImportHostPointer(&buffer));
  EXPECT_EQ(nullptr, buffer);
  EXPECT_EQ(0, release_count_);
  EXPECT_EQ(allocation_attempt_count + 1,
            allocator_state_.allocation_attempt_count.load(
                std::memory_order_relaxed));
  EXPECT_EQ(live_allocation_count, allocator_state_.live_allocation_count.load(
                                       std::memory_order_relaxed));
  allocator_state_.fail_allocations.store(false, std::memory_order_relaxed);
  VerifySuccessfulImport();
}

TEST_F(ReplayRecorderPoolTest, NativeFailureReleasesTentativeProxy) {
  const auto live_allocation_count =
      allocator_state_.live_allocation_count.load(std::memory_order_relaxed);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        Import(iree_hal_make_external_storage_class(
                                   IREE_HAL_EXTERNAL_STORAGE_TYPE_OPAQUE_FD),
                               &buffer));
  EXPECT_EQ(nullptr, buffer);
  EXPECT_EQ(0, release_count_);
  EXPECT_EQ(live_allocation_count, allocator_state_.live_allocation_count.load(
                                       std::memory_order_relaxed));
  VerifySuccessfulImport();
}

TEST_F(ReplayRecorderPoolTest, SuccessfulImportReleasesCallerOnce) {
  VerifySuccessfulImport();
}

}  // namespace
