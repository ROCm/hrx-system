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
#include "iree/hal/drivers/amd/fabric/memory_backend.h"
#include "iree/hal/drivers/amd/gpu/device.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

IREE_ASYNC_FIXED_FRONTIER_TYPE(CompletedFrontier, 2);

// Observe actual resource reclamation, including both worker handles. Returning
// from device_release alone does not prove asynchronous cleanup has finished.
struct DeviceAllocator {
  // Live allocations made by the concrete device and its children.
  std::atomic<size_t> live_count = 0;
  // Set after successful construction, while the device still owns storage.
  std::atomic<bool> armed = false;
  // Final callback returns ownership of this test allocator to the test.
  std::binary_semaphore reclaimed{0};

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* self_pointer,
                               iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* self = static_cast<DeviceAllocator*>(self_pointer);
    const bool had_pointer = *inout_pointer != nullptr;
    auto system = iree_allocator_system();
    iree_status_t status =
        system.ctl(system.self, command, params, inout_pointer);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    if (command == IREE_ALLOCATOR_COMMAND_FREE && had_pointer) {
      if (self->live_count.fetch_sub(1) == 1 && self->armed.load()) {
        // No access to self follows publication; the waiter may destroy it.
        self->reclaimed.release();
      }
    } else if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_pointer) {
      self->live_count.fetch_add(1);
    }
    return status;
  }
};

class GpuNativeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const amdf_api_t* api = nullptr;
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api),
        "amdf_query_api"));
    IREE_ASSERT_OK(iree_hal_amd_fabric_create(
        api, AMDF_NATIVE_LIFETIME_INSTANCE, iree_allocator_system(), &fabric_));
    for (uint32_t i = 0; i < fabric_->endpoint_count && !context_; ++i) {
      if (fabric_->endpoints[i].engine_kind == AMDF_ENGINE_KIND_GPU) {
        iree::Status status = iree_hal_amd_gpu_context_allocate(
            fabric_, &fabric_->endpoints[i].id,
            iree_hal_device_event_sink_stderr(), iree_allocator_system(),
            &context_);
        if (status.code() == iree::StatusCode::kUnimplemented ||
            status.code() == iree::StatusCode::kUnavailable) {
          GTEST_SKIP() << status.ToString();
        }
        IREE_ASSERT_OK(status);
      }
    }
    if (!context_) {
      GTEST_SKIP() << "No native AMD GPU endpoint";
    }
    iree_hal_amd_gpu_device_options_initialize(&options_);
    options_.queue_family_ordinal = UINT32_MAX;
    options_.queue_count = 2;
    for (uint32_t i = 0; i < context_->endpoint_info.queue_family_count; ++i) {
      const auto& family = context_->families[i];
      if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
          (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) &&
          (family.user_queue_capabilities &
           AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER)) {
        options_.queue_family_ordinal = i;
        break;
      }
    }
    if (options_.queue_family_ordinal == UINT32_MAX) {
      GTEST_SKIP() << "No PM4 USER host-producer family";
    }
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactors_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker_));
  }

  void TearDown() override {
    iree_hal_device_group_release(group_);
    if (device_) {
      iree_hal_device_release(device_);
      allocator_.reclaimed.acquire();
    }
    EXPECT_EQ(allocator_.live_count.load(), 0u);
    iree_hal_amd_gpu_context_free(context_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_proactor_pool_release(proactors_);
    iree_hal_amd_fabric_release(fabric_);
  }

  void Create() {
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    IREE_ASSERT_OK(iree_hal_amd_gpu_device_create(
        context_, IREE_SV("PM4 native CTS"), &options_, &params,
        allocator_.allocator(), &device_));
    context_ = nullptr;
    allocator_.armed.store(true);
    IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
        device_, tracker_, iree_allocator_system(), &group_));
    queues_[0] = iree_hal_device_queue(device_, 0, 0);
    queues_[1] = iree_hal_device_queue(device_, 0, 1);
    ASSERT_NE(queues_[0], nullptr);
    ASSERT_NE(queues_[1], nullptr);
  }

  void CreateSemaphore(iree_hal_semaphore_t** out_semaphore) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, out_semaphore));
  }

  // Independent native discovery ownership, released after device cleanup.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Constructing native owner, transferred to device on success.
  iree_hal_amd_gpu_context_t* context_ = nullptr;
  // Explicit provisioning used by each test.
  iree_hal_amd_gpu_device_options_t options_ = {};
  // Process-shared progress owners supplied through the public creation API.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // External tracker reference used to observe actual queue progress.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Allocator lifetime extends through the final worker cleanup.
  DeviceAllocator allocator_;
  // Owning public device reference, independently of its group.
  iree_hal_device_t* device_ = nullptr;
  // Immutable group assigning real memory scopes and queue axes.
  iree_hal_device_group_t* group_ = nullptr;
  // Borrowed provisioned queues, never separately retained by this fixture.
  std::array<iree_hal_queue_t*, 2> queues_ = {};
};

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

TEST_F(GpuNativeTest, DirectDispatchThroughExplicitPoolAndSealedGroup) {
  constexpr uint32_t kThreads = 4;
  constexpr uint32_t kSteps = 256;
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
            queues_[i & 1], {i ? 1u : 0u, &timelines[t], &before},
            {1, &timelines[t], &after}, executables[t], functions[t], config,
            t & 1 ? iree_make_const_byte_span(&arguments, 12)
                  : iree_make_const_byte_span(&arguments.value, 4),
            t & 1 ? iree_hal_buffer_ref_list_empty()
                  : iree_hal_buffer_ref_list_t{1, &binding},
            nullptr,
            t & 1 ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS : 0));
        std::memset(&arguments, 0xCD, sizeof(arguments));
      };
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
    ASSERT_EQ(count, 2);
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

}  // namespace
