// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/memory_spec.h"

#include <array>
#include <cstring>

#include "iree/testing/status_matchers.h"

namespace {

struct Scope {
  // Immutable public physical-storage metadata returned by the provider.
  amdf_memory_scope_info_t info = {};
  // Alternative construction routes over the same physical bytes.
  std::array<amdf_memory_profile_t, 3> profiles = {};
  // Explicit native-query result; unsupported profiles remain distinguishable.
  amdf_status_t profile_status = AMDF_STATUS_OK;
  // Explicit sampling failure, independent of immutable metadata.
  amdf_status_t observation_status = AMDF_STATUS_OK;
  // Current allocation headroom reported by the native provider.
  uint64_t available = 0;
  // Number of explicit native observations, excluding immutable queries.
  uint32_t samples = 0;

  amdf_memory_scope_t* handle() {
    return reinterpret_cast<amdf_memory_scope_t*>(this);
  }
};

static amdf_status_t AMDF_CALL QueryInfo(amdf_memory_scope_t* scope,
                                         amdf_memory_scope_info_t* out_info) {
  *out_info = reinterpret_cast<Scope*>(scope)->info;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL
QueryProfile(amdf_memory_scope_t* scope, uint32_t ordinal,
             uint32_t access_count, const amdf_memory_device_access_t* accesses,
             amdf_memory_profile_t* out_profile,
             amdf_memory_access_capabilities_t* out_access) {
  EXPECT_EQ(access_count, 1u);
  EXPECT_NE(accesses[0].device, nullptr);
  Scope& native = *reinterpret_cast<Scope*>(scope);
  if (!amdf_status_is_ok(native.profile_status)) {
    return native.profile_status;
  }
  *out_profile = native.profiles[ordinal];
  out_access->supported_flags = AMDF_MEMORY_FLAG_HOST_COHERENT;
  out_access->atomic_operations_32 = AMDF_ATOMIC_OPERATION_ADD;
  out_access->atomic_operations_64 = AMDF_ATOMIC_OPERATION_STORE;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL QueryPair(
    amdf_memory_scope_t* scope, const amdf_memory_profile_pair_query_t* query,
    amdf_memory_pair_info_t* out_info) {
  EXPECT_EQ(query->access_count, 1u);
  EXPECT_EQ(query->producer.kind, AMDF_MEMORY_SITE_KIND_DEVICE);
  EXPECT_EQ(query->producer.value.device.queue_family_ordinal, 3u);
  EXPECT_EQ(query->consumer.value.device.queue_family_ordinal, 3u);
  out_info->atomic_reach = {AMDF_ATOMIC_SCOPE_SYSTEM, AMDF_ATOMIC_SCOPE_DEVICE};
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL
SampleObservation(amdf_memory_scope_t* scope,
                  amdf_memory_scope_observation_t* out_observation) {
  Scope& native = *reinterpret_cast<Scope*>(scope);
  ++native.samples;
  if (!amdf_status_is_ok(native.observation_status)) {
    return native.observation_status;
  }
  out_observation->capacity_byte_length = native.info.capacity_byte_length;
  out_observation->available_byte_length = native.available;
  return AMDF_STATUS_OK;
}

class MemorySpecTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_.memory_scope_query_info = QueryInfo;
    api_.memory_scope_query_device_profile = QueryProfile;
    api_.memory_scope_query_pair_info = QueryPair;
    api_.memory_scope_sample_observation = SampleObservation;
    local_.info = {
        .kind = AMDF_MEMORY_SCOPE_KIND_LOCAL,
        .memory_profile_count = 2,
        .capacity_byte_length = UINT64_C(8) << 30,
        .flags = AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES,
    };
    local_.available = UINT64_C(6) << 30;
    system_.info = {
        .kind = AMDF_MEMORY_SCOPE_KIND_SYSTEM,
        .memory_profile_count = 2,
        .capacity_byte_length = UINT64_C(32) << 30,
        .flags = AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED |
                 AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES,
    };
    system_.available = UINT64_C(20) << 30;
    for (Scope* scope : {&local_, &system_}) {
      for (uint32_t i = 0; i < scope->info.memory_profile_count; ++i) {
        scope->profiles[i] = {
            .ordinal = i,
            .memory_class = static_cast<amdf_memory_class_t>(
                scope == &local_ ? AMDF_MEMORY_CLASS_LOCAL
                                 : AMDF_MEMORY_CLASS_SYSTEM),
            .roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                     AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
            .supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                               (scope == &local_ ? AMDF_MEMORY_FLAG_DEVICE_LOCAL
                                                 : UINT64_C(0)),
            .allocation =
                {
                    .maximum_byte_length = UINT64_C(4) << 30,
                    .byte_length_granularity = 4096,
                    .minimum_alignment = 4096,
                },
            .host_mapping =
                {
                    .supported_access =
                        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
                },
        };
      }
    }
    scopes_ = {local_.handle()};
    params_ = {
        .api = &api_,
        .access = {.device = reinterpret_cast<amdf_device_t*>(this)},
        .queue_family_ordinal = 3,
        .local_scope_count = scopes_.size(),
        .local_scopes = scopes_.data(),
        .system_scope = system_.handle(),
        .supported_usage =
            IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
    iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder_);
    const iree_hal_physical_device_spec_t physical = {
        .partition_count = 1,
        .physical_device_affinity = 1,
    };
    const iree_hal_device_identity_spec_t identity = {
        .logical_device_id = IREE_SV("test"),
        .physical_device_count = 1,
        .physical_devices = &physical,
    };
    IREE_ASSERT_OK(
        iree_hal_device_spec_builder_set_identity(&builder_, &identity));
  }

  void TearDown() override {
    iree_hal_device_spec_release(spec_);
    iree_hal_device_spec_builder_deinitialize(&builder_);
  }

  iree_status_t Build() {
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_fabric_memory_spec_populate(&params_, &builder_));
    return iree_hal_device_spec_builder_finalize(&builder_, &spec_);
  }

  // Provider dependency exercised through the actual HAL projection functions.
  amdf_api_t api_ = {};
  // Discrete local-memory scope, with multiple profiles sharing its capacity.
  Scope local_;
  // System RAM, exposed independently of dedicated local memory.
  Scope system_;
  // Borrowed local-scope array owned through each projection call.
  std::array<amdf_memory_scope_t*, 1> scopes_;
  // Complete native metadata inputs, with no execution resources.
  iree_hal_amd_fabric_memory_spec_params_t params_ = {};
  // Production builder owning its temporary copies until teardown.
  iree_hal_device_spec_builder_t builder_;
  // Immutable result of the actual builder/finalization path.
  iree_hal_device_spec_t* spec_ = nullptr;
};

TEST_F(MemorySpecTest, ProfilesSharePhysicalHeapCapacity) {
  IREE_ASSERT_OK(Build());
  const auto& memory = *iree_hal_device_spec_memory(spec_);
  ASSERT_EQ(memory.heap_count, 2u);
  ASSERT_EQ(memory.memory_type_count, 4u);
  EXPECT_EQ(memory.heaps[0].capacity_bytes, UINT64_C(8) << 30);
  EXPECT_EQ(memory.heaps[1].capacity_bytes, UINT64_C(32) << 30);
  for (uint32_t i = 0; i < memory.memory_type_count; ++i) {
    const auto& type = memory.memory_types[i];
    EXPECT_EQ(type.heap_index, i / 2);
    EXPECT_EQ(type.minimum_alignment, 4096u);
    EXPECT_EQ(type.allowed_memory_access, IREE_HAL_MEMORY_ACCESS_ALL);
    EXPECT_TRUE(iree_all_bits_set(type.memory_type,
                                  i < 2 ? IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL
                                        : IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
    EXPECT_TRUE(iree_all_bits_set(type.atomic_operations.system_scope_32,
                                  IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD));
    EXPECT_EQ(type.atomic_operations.system_scope_64, 0u);
    EXPECT_EQ(type.atomic_operations.device_scope_64,
              IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  }
  EXPECT_EQ(local_.samples, 0u);
  EXPECT_EQ(system_.samples, 0u);
}

TEST_F(MemorySpecTest, IntegratedLocalViewAndSystemShareOneHeap) {
  local_.info.flags |= AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED;
  local_.info.capacity_byte_length = system_.info.capacity_byte_length;
  IREE_ASSERT_OK(Build());
  const auto& memory = *iree_hal_device_spec_memory(spec_);
  ASSERT_EQ(memory.heap_count, 1u);
  ASSERT_EQ(memory.memory_type_count, 4u);
  EXPECT_EQ(memory.heaps[0].capacity_bytes, UINT64_C(32) << 30);
  for (uint32_t i = 0; i < memory.memory_type_count; ++i) {
    EXPECT_EQ(memory.memory_types[i].heap_index, 0u);
  }
}

TEST_F(MemorySpecTest,
       UnavailableAndRegistrationOnlyProfilesAreNotAllocations) {
  local_.profiles[0].availability =
      amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
  local_.profiles[1].roles = AMDF_MEMORY_PROFILE_ROLE_REGISTER;
  IREE_ASSERT_OK(Build());
  const auto& memory = *iree_hal_device_spec_memory(spec_);
  ASSERT_EQ(memory.memory_type_count, 2u);
  EXPECT_EQ(memory.memory_types[0].heap_index, 1u);
  EXPECT_EQ(local_.samples, 0u);
}

TEST_F(MemorySpecTest, NativeMetadataFailurePreservesFailureCause) {
  local_.profile_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Build());
  EXPECT_EQ(spec_, nullptr);
}

TEST_F(MemorySpecTest, DiscreteObservationExcludesUnrelatedSystemRam) {
  iree_hal_device_observation_t observation = {};
  IREE_ASSERT_OK(iree_hal_amd_fabric_memory_spec_sample_observation(
      &params_, &observation));
  EXPECT_EQ(observation.memory.total_bytes, UINT64_C(8) << 30);
  EXPECT_EQ(observation.memory.available_bytes, UINT64_C(6) << 30);
  EXPECT_EQ(local_.samples, 1u);
  EXPECT_EQ(system_.samples, 0u);
}

TEST_F(MemorySpecTest, IntegratedObservationUsesSystemMemory) {
  params_.local_scope_count = 0;
  iree_hal_device_observation_t observation = {};
  IREE_ASSERT_OK(iree_hal_amd_fabric_memory_spec_sample_observation(
      &params_, &observation));
  EXPECT_EQ(observation.memory.total_bytes, UINT64_C(32) << 30);
  EXPECT_EQ(observation.memory.available_bytes, UINT64_C(20) << 30);
  EXPECT_EQ(system_.samples, 1u);
}

TEST_F(MemorySpecTest, CapacityWithoutAvailabilityDoesNotInventHeadroom) {
  local_.info.flags = 0;
  iree_hal_device_observation_t observation = {};
  IREE_ASSERT_OK(iree_hal_amd_fabric_memory_spec_sample_observation(
      &params_, &observation));
  EXPECT_EQ(observation.memory.total_bytes, UINT64_C(8) << 30);
  EXPECT_EQ(observation.memory.flags,
            IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_TOTAL_BYTES);
  EXPECT_EQ(local_.samples, 0u);
}

TEST_F(MemorySpecTest, SharedLocalScopesSamplePhysicalRamOnce) {
  local_.info.flags |= AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED;
  local_.info.capacity_byte_length = system_.info.capacity_byte_length;
  std::array<amdf_memory_scope_t*, 2> aliases = {local_.handle(),
                                                 system_.handle()};
  params_.local_scope_count = aliases.size();
  params_.local_scopes = aliases.data();
  iree_hal_device_observation_t observation = {};
  IREE_ASSERT_OK(iree_hal_amd_fabric_memory_spec_sample_observation(
      &params_, &observation));
  EXPECT_EQ(observation.memory.total_bytes, UINT64_C(32) << 30);
  EXPECT_EQ(observation.memory.available_bytes, local_.available);
  EXPECT_EQ(local_.samples, 1u);
  EXPECT_EQ(system_.samples, 0u);
}

TEST_F(MemorySpecTest, MissingHeadroomForOneStorageDoesNotPublishPartialSum) {
  system_.info.flags = 0;
  std::array<amdf_memory_scope_t*, 2> scopes = {local_.handle(),
                                                system_.handle()};
  params_.local_scope_count = scopes.size();
  params_.local_scopes = scopes.data();
  iree_hal_device_observation_t observation = {};
  IREE_ASSERT_OK(iree_hal_amd_fabric_memory_spec_sample_observation(
      &params_, &observation));
  EXPECT_EQ(observation.memory.total_bytes, UINT64_C(40) << 30);
  EXPECT_EQ(observation.memory.flags,
            IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_TOTAL_BYTES);
}

TEST_F(MemorySpecTest, NativeSamplingFailureLeavesCallerOutputUnchanged) {
  local_.observation_status =
      amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
  iree_hal_device_observation_t observation;
  std::memset(&observation, 0x5A, sizeof(observation));
  const auto before = observation;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_amd_fabric_memory_spec_sample_observation(
                            &params_, &observation));
  EXPECT_EQ(std::memcmp(&observation, &before, sizeof(observation)), 0);
}

}  // namespace
