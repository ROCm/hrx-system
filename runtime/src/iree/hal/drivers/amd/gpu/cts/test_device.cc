// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/cts/test_device.h"

#include "iree/hal/device_group.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {

iree_status_t DeviceAllocator::Control(void* self_pointer,
                                       iree_allocator_command_t command,
                                       const void* params,
                                       void** inout_pointer) {
  auto* self = static_cast<DeviceAllocator*>(self_pointer);
  const bool had_pointer = (command == IREE_ALLOCATOR_COMMAND_FREE ||
                            command == IREE_ALLOCATOR_COMMAND_REALLOC) &&
                           *inout_pointer != nullptr;
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

NativeMemoryGate::~NativeMemoryGate() {
  if (active_ == this) {
    active_ = nullptr;
  }
}

iree_status_t NativeMemoryGate::CreateFabric(
    iree_hal_amd_fabric_t** out_fabric) {
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST,
                     &native_api_),
      "amdf_query_api"));
  forwarding_api_ = *native_api_;
  forwarding_api_.memory_create = CreateMemory;
  active_ = this;
  return iree_hal_amd_fabric_create(&forwarding_api_,
                                    AMDF_NATIVE_LIFETIME_INSTANCE,
                                    iree_allocator_system(), out_fabric);
}

amdf_status_t AMDF_CALL NativeMemoryGate::CreateMemory(
    amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
    amdf_memory_t** out_memory) {
  auto* self = active_;
  const bool held = self->hold_.exchange(false);
  if (held) {
    self->native_thread_ = std::this_thread::get_id();
    self->entered_.release();
    self->resume_.acquire();
  }
  const amdf_status_t status =
      self->native_api_->memory_create(scope, info, out_memory);
  if (held) {
    self->returned_.release();
  }
  return status;
}

NativeMemoryGate::Hold::Hold(NativeMemoryGate* owner) : owner_(owner) {
  owner_->hold_ = true;
}

NativeMemoryGate::Hold::~Hold() {
  // An admission failure can leave the gate unused. A consumed gate must join
  // its actual native call before any callback state is destroyed.
  if (!owner_->hold_.exchange(false)) {
    owner_->resume_.release();
    owner_->returned_.acquire();
  }
}

void NativeMemoryGate::Hold::WaitEntered() { owner_->entered_.acquire(); }

void NativeMemoryGate::Hold::ReleaseAndHoldNext() {
  // Arm the next native step before releasing the current one. A mapping's
  // consumer can progress during the subsequent allocation.
  owner_->hold_ = true;
  owner_->resume_.release();
  owner_->returned_.acquire();
  owner_->entered_.acquire();
}

void GpuNativeTest::SetUp() {
  const amdf_api_t* api = nullptr;
  IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
      amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api),
      "amdf_query_api"));
  if (!fabric_) {
    IREE_ASSERT_OK(iree_hal_amd_fabric_create(
        api, AMDF_NATIVE_LIFETIME_INSTANCE, iree_allocator_system(), &fabric_));
  }
  for (uint32_t i = 0; i < fabric_->endpoint_count && !context_; ++i) {
    if (fabric_->endpoints[i].engine_kind == AMDF_ENGINE_KIND_GPU) {
      iree::Status status =
          iree_hal_amd_gpu_context_allocate(fabric_, &fabric_->endpoints[i].id,
                                            iree_hal_device_event_sink_stderr(),
                                            iree_allocator_system(), &context_);
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
  options_.queue_count = 2;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactors_));
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker_));
}

void GpuNativeTest::TearDown() {
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

void GpuNativeTest::Create(
    iree_hal_device_runtime_feature_flags_t runtime_features) {
  auto params = iree_hal_device_create_params_default();
  params.proactor_pool = proactors_;
  params.runtime_features = runtime_features;
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

void GpuNativeTest::CreateSemaphore(iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, out_semaphore));
}

}  // namespace iree::hal::amd::gpu::cts
