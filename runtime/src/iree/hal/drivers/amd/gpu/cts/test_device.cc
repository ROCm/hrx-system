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
  EXPECT_EQ(usage_.live_bytes, 0u);
  EXPECT_TRUE(allocations_.empty());
  if (active_ == this) {
    active_ = nullptr;
  }
}

iree_status_t NativeMemoryGate::CreateFabric(iree_hal_amd_fabric_t** out_fabric,
                                             uint64_t maximum_byte_length,
                                             amdf_native_lifetime_t lifetime) {
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST,
                     &native_api_),
      "amdf_query_api"));
  forwarding_api_ = *native_api_;
  maximum_byte_length_ = maximum_byte_length;
  forwarding_api_.memory_scope_query_device_profile = QueryDeviceProfile;
  forwarding_api_.memory_create = CreateMemory;
  forwarding_api_.memory_destroy = DestroyMemory;
  active_ = this;
  if (lifetime == AMDF_NATIVE_LIFETIME_PROCESS) {
    return AcquireProcessFabric(out_fabric);
  }
  return iree_hal_amd_fabric_create(&forwarding_api_, lifetime,
                                    iree_allocator_system(), out_fabric);
}

iree_status_t NativeMemoryGate::AcquireProcessFabric(
    iree_hal_amd_fabric_t** out_fabric) {
  struct ProcessFabric {
    ProcessFabric() {
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST,
                         &native_api_),
          "amdf_query_api");
      if (iree_status_is_ok(status)) {
        api = *native_api_;
        api.memory_scope_query_device_profile = QueryDeviceProfile;
        api.memory_create = CreateMemory;
        api.memory_destroy = DestroyMemory;
        status = iree_hal_amd_fabric_create(&api, AMDF_NATIVE_LIFETIME_PROCESS,
                                            iree_allocator_system(), &fabric);
      }
    }
    ~ProcessFabric() {
      iree_hal_amd_fabric_release(fabric);
      iree_status_free(status);
    }
    // Forwarding table outlives every process-scoped native binding.
    amdf_api_t api = {};
    // One owning reference remains through all serial fixtures.
    iree_hal_amd_fabric_t* fabric = nullptr;
    // Construction failure is reported by every fixture using this owner.
    iree_status_t status = iree_ok_status();
  };
  static ProcessFabric shared;
  *out_fabric = nullptr;
  IREE_RETURN_IF_ERROR(iree_status_clone(shared.status));
  iree_hal_amd_fabric_retain(shared.fabric);
  *out_fabric = shared.fabric;
  return iree_ok_status();
}

amdf_status_t AMDF_CALL NativeMemoryGate::QueryDeviceProfile(
    amdf_memory_scope_t* scope, uint32_t profile_ordinal, uint32_t access_count,
    const amdf_memory_device_access_t* accesses,
    amdf_memory_profile_t* out_profile,
    amdf_memory_access_capabilities_t* out_access_capabilities) {
  auto* self = active_;
  const amdf_status_t status = native_api_->memory_scope_query_device_profile(
      scope, profile_ordinal, access_count, accesses, out_profile,
      out_access_capabilities);
  if (self && amdf_status_is_ok(status)) {
    out_profile->allocation.maximum_byte_length =
        iree_min(out_profile->allocation.maximum_byte_length,
                 self->maximum_byte_length_);
  }
  return status;
}

amdf_status_t AMDF_CALL NativeMemoryGate::CreateMemory(
    amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
    amdf_memory_t** out_memory) {
  auto* self = active_;
  if (!self) {
    return native_api_->memory_create(scope, info, out_memory);
  }
  if (info->byte_length > self->maximum_byte_length_) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const bool held = self->hold_.exchange(false);
  if (held) {
    self->native_thread_ = std::this_thread::get_id();
    self->entered_.release();
    self->resume_.acquire();
  }
  amdf_status_t status =
      self->native_api_->memory_create(scope, info, out_memory);
  if (amdf_status_is_ok(status)) {
    amdf_memory_info_t native = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
        .structure_size = sizeof(native),
    };
    status = self->native_api_->memory_query_info(*out_memory, &native);
    if (amdf_status_is_ok(status)) {
      std::lock_guard<std::mutex> lock(self->usage_mutex_);
      self->allocations_.emplace(*out_memory,
                                 native.native_allocation_byte_length);
      self->usage_.live_bytes += native.native_allocation_byte_length;
      self->usage_.peak_bytes =
          iree_max(self->usage_.peak_bytes, self->usage_.live_bytes);
      ++self->usage_.allocation_count;
      self->usage_.last_allocation = native;
    } else {
      EXPECT_EQ(self->native_api_->memory_destroy(*out_memory), AMDF_STATUS_OK);
      *out_memory = nullptr;
    }
  }
  if (held) {
    self->returned_.release();
  }
  return status;
}

amdf_status_t AMDF_CALL NativeMemoryGate::DestroyMemory(amdf_memory_t* memory) {
  auto* self = active_;
  if (!self) {
    return native_api_->memory_destroy(memory);
  }
  uint64_t native_bytes = 0;
  {
    std::lock_guard<std::mutex> lock(self->usage_mutex_);
    const auto entry = self->allocations_.find(memory);
    if (entry != self->allocations_.end()) {
      native_bytes = entry->second;
      self->allocations_.erase(entry);
    }
  }
  const amdf_status_t status = self->native_api_->memory_destroy(memory);
  {
    std::lock_guard<std::mutex> lock(self->usage_mutex_);
    if (amdf_status_is_ok(status)) {
      self->usage_.live_bytes -= native_bytes;
    } else {
      ++self->usage_.destroy_failures;
    }
    self->usage_changed_.notify_all();
  }
  return status;
}

NativeMemoryGate::MemoryUsage NativeMemoryGate::ResetPeak() {
  std::lock_guard<std::mutex> lock(usage_mutex_);
  usage_.peak_bytes = usage_.live_bytes;
  return usage_;
}

NativeMemoryGate::MemoryUsage NativeMemoryGate::QueryUsage() {
  std::lock_guard<std::mutex> lock(usage_mutex_);
  return usage_;
}

NativeMemoryGate::MemoryUsage NativeMemoryGate::WaitForBytes(
    uint64_t byte_count) {
  std::unique_lock<std::mutex> lock(usage_mutex_);
  usage_changed_.wait(lock, [&] {
    return usage_.live_bytes == byte_count || usage_.destroy_failures;
  });
  return usage_;
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
