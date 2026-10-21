// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/cts/test_device.h"

#include <cstdlib>
#include <optional>

#include "iree/hal/device_group.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/queue.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/queue.h"
#include "iree/hal/drivers/amd/gpu/queue/sdma/queue.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/testing/status_matchers.h"
#include "libamdf/cts/util/native_identity.h"

namespace iree::hal::amd::gpu::cts {

void GpuNativeTest::SelectControlMemory(
    amdf_memory_device_access_t* access,
    iree_hal_amd_fabric_memory_options_t* policy) {
  *access = {
      .device = context_->device,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                        AMDF_MEMORY_ACCESS_EXECUTE,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                       AMDF_MEMORY_FLAG_HOST_COHERENT |
                       AMDF_MEMORY_FLAG_QUEUE_STORAGE,
              .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
          },
  };
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info),
  };
  IREE_ASSERT_OK(
      IREE_HAL_AMD_STATUS_FROM_AMDF(context_->api->memory_scope_query_info(
                                        context_->fabric->system_scope, &info),
                                    "query system storage"));
  for (uint32_t i = 0; i < info.memory_profile_count; ++i) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const auto status = context_->api->memory_scope_query_device_profile(
        context_->fabric->system_scope, i, 1, access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    IREE_ASSERT_OK(
        IREE_HAL_AMD_STATUS_FROM_AMDF(status, "qualify program storage"));
    if (iree_all_bits_set(profile.roles,
                          AMDF_MEMORY_PROFILE_ROLE_CREATE |
                              AMDF_MEMORY_PROFILE_ROLE_HOST_MAP)) {
      *policy = {
          .scope = context_->fabric->system_scope,
          .profile = profile,
          .access_count = 1,
          .accesses = access,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
          .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
      };
      return;
    }
  }
  FAIL() << "No admitted host control profile";
}

void GpuNativeTest::PublishCode(uint32_t compute_family_ordinal,
                                iree_hal_amd_fabric_memory_t* memory,
                                iree_hal_amd_gpu_code_upload_t** upload) {
  if (!*upload) {
    return;
  }
  uint32_t family = UINT32_MAX;
  amdf_queue_publication_modes_t mode = 0;
  for (uint32_t i = 0; i < context_->endpoint_info.queue_family_count; ++i) {
    const auto modes =
        iree_hal_amd_gpu_sdma_queue_publication_modes(&context_->families[i]);
    if (modes) {
      family = i;
      mode = modes & AMDF_QUEUE_PUBLICATION_MODE_USER
                 ? AMDF_QUEUE_PUBLICATION_MODE_USER
                 : AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
      break;
    }
  }
  ASSERT_NE(family, UINT32_MAX);
  amdf_memory_device_access_t access;
  iree_hal_amd_fabric_memory_options_t control;
  ASSERT_NO_FATAL_FAILURE(SelectControlMemory(&access, &control));
  iree_hal_amd_gpu_publication_t publication = {};
  iree_hal_amd_gpu_sdma_queue_t* queue = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_queue_allocate(
      context_, family, mode, &control, &publication, 0,
      iree_allocator_system(), &queue));
  auto publish_source = iree_hal_amd_gpu_sdma_queue_publication_source(queue);
  auto completion_source = iree_hal_amd_gpu_sdma_queue_completion_source(queue);
  iree_hal_amd_gpu_completion_t completion = {};
  iree_status_t status = iree_hal_amd_gpu_publication_initialize(
      1, &publish_source, {}, iree_allocator_system(), &publication);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_completion_initialize(
        context_, 1, &completion_source, 0, {}, iree_allocator_system(),
        &completion);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_code_publish(
        context_, queue, family, compute_family_ordinal, memory, upload);
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion));
  iree_hal_amd_gpu_publication_deinitialize(&publication);
  IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
      iree_hal_amd_gpu_sdma_queue_free(queue), "join code upload queue"));
  IREE_ASSERT_OK(status);
}

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
    self->live_count.fetch_sub(1);
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
  forwarding_api_.memory_scope_query_pair_info = QueryProfilePair;
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
        api.memory_scope_query_pair_info = QueryProfilePair;
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

amdf_status_t AMDF_CALL NativeMemoryGate::QueryProfilePair(
    amdf_memory_scope_t* scope, const amdf_memory_profile_pair_query_t* query,
    amdf_memory_pair_info_t* out_info) {
  if (active_ && active_->require_code_upload_ &&
      (query->required_flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL) &&
      query->producer.kind == AMDF_MEMORY_SITE_KIND_HOST) {
    for (uint32_t i = 0; i < query->access_count; ++i) {
      if (query->accesses[i].requirements.access & AMDF_MEMORY_ACCESS_EXECUTE) {
        return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
      }
    }
  }
  return native_api_->memory_scope_query_pair_info(scope, query, out_info);
}

amdf_status_t AMDF_CALL NativeMemoryGate::CreateMemory(
    amdf_memory_scope_t* scope, const amdf_memory_create_info_t* info,
    amdf_memory_t** out_memory, amdf_memory_release_info_t* out_release_info) {
  if (out_release_info) {
    *out_release_info = {};
  }
  auto* self = active_;
  if (!self) {
    return native_api_->memory_create(scope, info, out_memory,
                                      out_release_info);
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
  amdf_status_t status = self->native_api_->memory_create(
      scope, info, out_memory, out_release_info);
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
      EXPECT_EQ(
          self->native_api_->memory_destroy(*out_memory, out_release_info),
          AMDF_STATUS_OK);
      *out_memory = nullptr;
    }
  }
  if (held) {
    self->returned_.release();
  }
  return status;
}

amdf_status_t AMDF_CALL NativeMemoryGate::DestroyMemory(
    amdf_memory_t* memory, amdf_memory_release_info_t* out_release_info) {
  auto* self = active_;
  if (!self) {
    return native_api_->memory_destroy(memory, out_release_info);
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
  amdf_memory_release_info_t release_info = {};
  const amdf_status_t status =
      self->native_api_->memory_destroy(memory, &release_info);
  if (out_release_info) {
    *out_release_info = release_info;
  }
  {
    std::lock_guard<std::mutex> lock(self->usage_mutex_);
    if (native_bytes) {
      EXPECT_LE(release_info.abandoned_byte_length, native_bytes);
      self->usage_.live_bytes -=
          native_bytes - release_info.abandoned_byte_length;
    }
    if (!amdf_status_is_ok(status)) {
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

// Runner selection uses the same stable native identity as libamdf CTS. Query
// before activation so an exclusive hardware lease never touches another GPU.
iree_status_t MatchesNativeIdentity(iree_hal_amd_fabric_t* fabric,
                                    const amdf_endpoint_id_t* id,
                                    const char* selected, bool* out_matches) {
  *out_matches = true;
  if (!selected) {
    return iree_ok_status();
  }
  amdf_endpoint_native_identity_t identity = {};
  if (!amdf::cts::ParseNativeIdentity(selected, &identity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid native GPU identity: %s", selected);
  }
  amdf_endpoint_t* endpoint = nullptr;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      fabric->api->endpoint_open(fabric->instance, id, &endpoint),
      "endpoint_open"));
  amdf_endpoint_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(info),
  };
  const amdf_status_t status =
      fabric->api->endpoint_query_info(endpoint, &info);
  const amdf_status_t close_status = fabric->api->endpoint_close(endpoint);
  if (!amdf_status_is_ok(close_status)) {
    iree_hal_amd_fabric_retain(fabric);
  }
  IREE_RETURN_IF_ERROR(iree_status_join(
      IREE_HAL_AMD_STATUS_FROM_AMDF(status, "endpoint_query_info"),
      IREE_HAL_AMD_STATUS_FROM_AMDF(close_status, "endpoint_close")));
  *out_matches =
      amdf::cts::NativeIdentityIsEqual(info.native_identity, identity);
  return iree_ok_status();
}

void GpuNativeTest::SetUp() {
  const char* selected_identity_text =
      std::getenv("AMDF_CTS_GPU_NATIVE_IDENTITY");
  std::optional<amdf_endpoint_native_identity_t> selected_identity;
  if (selected_identity_text) {
    selected_identity.emplace();
    ASSERT_TRUE(amdf::cts::ParseNativeIdentity(selected_identity_text,
                                               &*selected_identity))
        << "invalid AMDF_CTS_GPU_NATIVE_IDENTITY: " << selected_identity_text;
  }

  const amdf_api_t* api = nullptr;
  IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
      amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api),
      "amdf_query_api"));
  if (!fabric_) {
    if (command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
      IREE_ASSERT_OK(NativeMemoryGate::AcquireProcessFabric(&fabric_));
    } else {
      IREE_ASSERT_OK(
          iree_hal_amd_fabric_create(api, AMDF_NATIVE_LIFETIME_INSTANCE,
                                     iree_allocator_system(), &fabric_));
    }
  }
  bool selected_endpoint_found = false;
  for (uint32_t i = 0; i < fabric_->endpoint_count && !context_; ++i) {
    if (fabric_->endpoints[i].engine_kind == AMDF_ENGINE_KIND_GPU) {
      amdf_endpoint_t* endpoint = nullptr;
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          api->endpoint_open(fabric_->instance, &fabric_->endpoints[i].id,
                             &endpoint),
          "endpoint_open"));
      iree_status_t status = iree_ok_status();
      bool is_selected = true;
      if (selected_identity) {
        amdf_endpoint_info_t endpoint_info = {
            .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
            .structure_size = sizeof(endpoint_info),
        };
        status = IREE_HAL_AMD_STATUS_FROM_AMDF(
            api->endpoint_query_info(endpoint, &endpoint_info),
            "endpoint_query_info");
        if (iree_status_is_ok(status)) {
          is_selected = amdf::cts::NativeIdentityIsEqual(
              endpoint_info.native_identity, *selected_identity);
          selected_endpoint_found |= is_selected;
        }
      }
      bool supported = false;
      if (iree_status_is_ok(status) && is_selected) {
        amdf_endpoint_info_t info = {
            .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
            .structure_size = sizeof(info),
        };
        status = IREE_HAL_AMD_STATUS_FROM_AMDF(
            api->endpoint_query_info(endpoint, &info), "endpoint_query_info");
        for (uint32_t family = 0; family < info.queue_family_count &&
                                  !supported && iree_status_is_ok(status);
             ++family) {
          amdf_queue_family_info_t native = {
              .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
              .structure_size = sizeof(native),
          };
          status = IREE_HAL_AMD_STATUS_FROM_AMDF(
              api->endpoint_query_queue_family_info(endpoint, family, &native),
              "endpoint_query_queue_family_info");
          if (iree_status_is_ok(status)) {
            supported =
                command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                    ? iree_hal_amd_gpu_aql_queue_publication_modes(&native) != 0
                    : iree_hal_amd_gpu_pm4_queue_publication_modes(&native) !=
                          0;
          }
        }
      }
      const amdf_status_t close_status = api->endpoint_close(endpoint);
      if (!amdf_status_is_ok(close_status)) {
        iree_hal_amd_fabric_retain(fabric_);
      }
      IREE_ASSERT_OK(iree_status_join(
          status,
          IREE_HAL_AMD_STATUS_FROM_AMDF(close_status, "endpoint_close")));
      if (!is_selected || !supported) {
        continue;
      }
      IREE_ASSERT_OK(iree_hal_amd_gpu_context_allocate(
          fabric_, &fabric_->endpoints[i].id,
          iree_hal_device_event_sink_stderr(), iree_allocator_system(),
          &context_));
    }
  }
  if (!context_) {
    if (selected_identity) {
      ASSERT_TRUE(selected_endpoint_found)
          << "AMDF_CTS_GPU_NATIVE_IDENTITY does not identify an enumerated "
             "GPU: "
          << selected_identity_text;
      GTEST_SKIP() << "Selected native AMD GPU has no supported requested "
                      "packet family: "
                   << selected_identity_text;
    }
    GTEST_SKIP()
        << "No native AMD GPU endpoint with the requested packet family";
  }
  iree_hal_amd_gpu_device_options_initialize(&options_);
  options_.queue_count = 2;
  if (command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
    for (uint32_t i = 0; i < context_->endpoint_info.queue_family_count; ++i) {
      if (iree_hal_amd_gpu_aql_queue_publication_modes(
              &context_->families[i])) {
        options_.queue_family_ordinal = i;
        break;
      }
    }
  }
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, nullptr, proactor_options_, iree_allocator_system(), &proactors_));
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker_));
}

void GpuNativeTest::TearDown() {
  iree_hal_device_release(device_);
  iree_hal_device_group_release(group_);
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
  auto* context = context_;
  context_ = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_device_create(
      context, IREE_SV("AMD GPU native CTS"), &options_, &params,
      allocator_.allocator(), &device_));
  IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
      device_, tracker_, iree_allocator_system(), &group_));
  queues_[0] = iree_hal_device_queue(device_, 0, 0);
  queues_[1] = iree_hal_device_queue(device_, 0, 1);
  ASSERT_NE(queues_[0], nullptr);
  ASSERT_NE(queues_[1], nullptr);
}

iree_status_t GpuNativeTest::AcquireQueue(iree_hal_queue_t** out_queue) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  const auto* resource = reinterpret_cast<const iree_hal_resource_t*>(device_);
  const auto* vtable =
      static_cast<const iree_hal_device_vtable_t*>(resource->vtable);
  return vtable->acquire_queue(device_, iree_hal_queue_family(queues_[0]),
                               &params, out_queue);
}

void GpuNativeTest::CreateSemaphore(iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, out_semaphore));
}

}  // namespace iree::hal::amd::gpu::cts
