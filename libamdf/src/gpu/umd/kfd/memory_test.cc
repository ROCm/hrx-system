// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory.h"

#include <cstring>

#include "amdf/gpu.h"
#include "gtest/gtest.h"
#include "libamdf/src/gpu/endpoint_profile.h"
#include "libamdf/src/gpu/umd/kfd/device.h"
#include "libamdf/src/gpu/umd/kfd/memory_profile.h"
#include "libamdf/src/gpu/umd/kfd/target/user_queue.h"

namespace {

static amdf_memory_native_profile_t QueryProfile(amdf_gpu_umd_device_t* device,
                                                 uint32_t ordinal) {
  amdf_memory_native_profile_t profile = {};
  EXPECT_EQ(amdf_gpu_umd_device_query_memory_profile(device, ordinal, &profile),
            AMDF_STATUS_OK);
  return profile;
}

TEST(LinuxGpuMemoryProfileTest, InstanceLifetimeExposesOwnedSystemMemory) {
  amdf_gpu_umd_device_t device = {
      .native_lifetime = AMDF_NATIVE_LIFETIME_INSTANCE, .page_size = 4096};
  device.topology.virtual_address.begin = UINT64_C(0x10000);
  device.topology.virtual_address.end = UINT64_C(1) << 48;

  const amdf_memory_native_profile_t profile = QueryProfile(&device, 0);
  EXPECT_EQ(profile.ordinal, 0u);
  EXPECT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_NE(profile.construction.query_access, nullptr);
  EXPECT_EQ(profile.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE |
                               AMDF_MEMORY_PROFILE_ROLE_EXPORT |
                               AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(profile.guaranteed_flags, AMDF_MEMORY_FLAG_HOST_VISIBLE |
                                          AMDF_MEMORY_FLAG_SHAREABLE |
                                          AMDF_MEMORY_FLAG_HOST_COHERENT |
                                          AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
  EXPECT_EQ(profile.guaranteed_device_access, AMDF_MEMORY_ACCESS_READ);
  EXPECT_EQ(profile.supported_device_access, AMDF_MEMORY_ACCESS_READ |
                                                 AMDF_MEMORY_ACCESS_WRITE |
                                                 AMDF_MEMORY_ACCESS_EXECUTE);
  EXPECT_EQ(profile.supported_flags,
            profile.guaranteed_flags | AMDF_MEMORY_FLAG_QUEUE_STORAGE);
  EXPECT_EQ(profile.allocation.minimum_alignment, 4096u);
  EXPECT_EQ(profile.allocation.byte_length_granularity, 1u);
  EXPECT_EQ(profile.allocation.native_byte_length_granularity, 4096u);
  EXPECT_EQ(profile.device_address.address_domain_ordinal, 0u);
  EXPECT_EQ(profile.device_address.address_bit_count, 48u);
  EXPECT_EQ(profile.device_address.minimum_address, UINT64_C(0x10000));
  EXPECT_EQ(profile.device_address.maximum_address, (UINT64_C(1) << 48) - 1);
  EXPECT_EQ(profile.host_mapping.byte_offset_granularity, 1u);
  EXPECT_EQ(profile.host_mapping.byte_length_granularity, 1u);
  EXPECT_EQ(profile.host_mapping.supported_access,
            AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
  ASSERT_EQ(profile.external_memory_support_count, 2u);
  EXPECT_EQ(profile.external_memory_support[0].type,
            AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD);
  EXPECT_EQ(profile.external_memory_support[0].flags,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS);

  amdf_memory_native_profile_t unavailable = {};
  unavailable.ordinal = UINT32_MAX;
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_device_query_memory_profile(
                &device, 1, &unavailable)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(unavailable.ordinal, UINT32_MAX);
}

TEST(LinuxGpuMemoryProfileTest, ProcessLifetimeUsesDenseOptionalProfiles) {
  amdf_gpu_umd_device_t device = {
      .native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS, .page_size = 4096};
  device.topology.virtual_address.begin = UINT64_C(0x10000);
  device.topology.virtual_address.end = UINT64_C(1) << 48;
  device.topology.memory_features =
      AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY |
      AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY;

  const amdf_memory_native_profile_t local_profile = QueryProfile(&device, 1);
  EXPECT_NE(local_profile.construction.query_access, nullptr);
  EXPECT_EQ(local_profile.memory_class, AMDF_MEMORY_CLASS_LOCAL);
  EXPECT_EQ(local_profile.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE |
                                     AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(local_profile.guaranteed_flags,
            AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
  EXPECT_EQ(local_profile.supported_flags, local_profile.guaranteed_flags |
                                               AMDF_MEMORY_FLAG_QUEUE_STORAGE |
                                               AMDF_MEMORY_FLAG_HOST_VISIBLE);
  EXPECT_EQ(local_profile.guaranteed_device_access, AMDF_MEMORY_ACCESS_READ);
  EXPECT_EQ(local_profile.supported_device_access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                AMDF_MEMORY_ACCESS_EXECUTE);
  EXPECT_EQ(local_profile.allocation.minimum_alignment, 4096u);
  EXPECT_EQ(local_profile.host_mapping.supported_access,
            AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);

  const amdf_memory_native_profile_t registered_profile =
      QueryProfile(&device, 2);
  EXPECT_NE(registered_profile.construction.query_access, nullptr);
  const amdf_memory_native_profile_t allocated_profile =
      QueryProfile(&device, 0);
  EXPECT_EQ(registered_profile.construction.query_access,
            allocated_profile.construction.query_access);
  amdf_memory_native_profile_t projected = {};
  EXPECT_FALSE(registered_profile.construction.query_access(
      &registered_profile, &allocated_profile, &projected));
  EXPECT_EQ(registered_profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(registered_profile.roles, AMDF_MEMORY_PROFILE_ROLE_REGISTER |
                                          AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(registered_profile.registration.minimum_alignment, 1u);
  EXPECT_EQ(registered_profile.registration.registered_host_pointer_alignment,
            1u);
  EXPECT_EQ(registered_profile.registration.native_byte_length_granularity,
            4096u);
  EXPECT_EQ(registered_profile.device_address.minimum_alignment, 1u);
  EXPECT_EQ(registered_profile.guaranteed_flags,
            AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_COHERENT |
                AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
  EXPECT_EQ(
      registered_profile.supported_flags,
      registered_profile.guaranteed_flags | AMDF_MEMORY_FLAG_QUEUE_STORAGE);
  EXPECT_EQ(registered_profile.guaranteed_device_access,
            AMDF_MEMORY_ACCESS_READ);
  EXPECT_EQ(registered_profile.supported_device_access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                AMDF_MEMORY_ACCESS_EXECUTE);

  device.topology.memory_features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  const amdf_memory_native_profile_t nonvisible_local_profile =
      QueryProfile(&device, 1);
  EXPECT_EQ(nonvisible_local_profile.memory_class, AMDF_MEMORY_CLASS_LOCAL);
  EXPECT_EQ(nonvisible_local_profile.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE);
  EXPECT_EQ(nonvisible_local_profile.supported_flags,
            nonvisible_local_profile.guaranteed_flags |
                AMDF_MEMORY_FLAG_QUEUE_STORAGE);
  EXPECT_EQ(nonvisible_local_profile.host_mapping.maximum_byte_length, 0u);
  const amdf_memory_native_profile_t registered_after_local_profile =
      QueryProfile(&device, 2);
  EXPECT_EQ(registered_after_local_profile.memory_class,
            AMDF_MEMORY_CLASS_SYSTEM);

  device.topology.memory_features = 0;
  const amdf_memory_native_profile_t dense_registered_profile =
      QueryProfile(&device, 1);
  EXPECT_EQ(dense_registered_profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
}

TEST(LinuxGpuMemoryProfileTest, QualifiesLocalBackingByHiveOrDirectedPciPeer) {
  amdf_gpu_umd_device_t source = {.page_size = 4096};
  source.topology.gpu_id = 41;
  source.topology.memory_features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  source.topology.virtual_address.begin = UINT64_C(0x10000);
  source.topology.virtual_address.end = UINT64_C(1) << 48;
  source.topology.memory_peers.hive_id = UINT64_C(11827785098739261628);
  source.topology.memory_peers.hive_sharing_enabled = true;
  amdf_gpu_umd_device_t consumer = source;
  consumer.topology.gpu_id = 73;
  consumer.topology.memory_features = 0;
  consumer.topology.virtual_address.begin = UINT64_C(0x20000);
  consumer.topology.virtual_address.end = UINT64_C(1) << 47;
  const auto backing = QueryProfile(&source, 1);
  auto candidate = QueryProfile(&consumer, 0);
  const auto original_candidate = candidate;
  ASSERT_TRUE(
      backing.construction.query_access(&backing, &candidate, &candidate));
  EXPECT_EQ(candidate.memory_class, AMDF_MEMORY_CLASS_LOCAL);
  EXPECT_EQ(candidate.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE);
  EXPECT_EQ(candidate.guaranteed_flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
  EXPECT_EQ(candidate.supported_flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
  EXPECT_EQ(candidate.device_address.maximum_address, (UINT64_C(1) << 47) - 1);
  EXPECT_EQ(candidate.allocation.maximum_byte_length,
            original_candidate.allocation.maximum_byte_length);
  EXPECT_EQ(candidate.construction.data, &consumer.topology);

  auto expect_unreachable = [&]() {
    auto output = original_candidate;
    EXPECT_FALSE(backing.construction.query_access(
        &backing, &original_candidate, &output));
    EXPECT_EQ(std::memcmp(&output, &original_candidate, sizeof(output)), 0);
  };
  consumer.topology.memory_peers.hive_id ^= UINT64_C(1) << 40;
  expect_unreachable();
  consumer.topology.memory_peers.hive_id = source.topology.memory_peers.hive_id;
  source.topology.memory_peers.hive_sharing_enabled = false;
  expect_unreachable();
  source.topology.memory_peers.hive_id = 0;
  consumer.topology.memory_peers.hive_id = 0;
  expect_unreachable();

  uint32_t backing_gpu_id = source.topology.gpu_id;
  consumer.topology.memory_peers.count = 1;
  consumer.topology.memory_peers.gpu_ids = &backing_gpu_id;
  EXPECT_TRUE(backing.construction.query_access(&backing, &original_candidate,
                                                &candidate));
  // B can access A's heap; this does not imply A can access B's heap.
  consumer.topology.memory_features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  const auto reverse_backing = QueryProfile(&consumer, 1);
  const auto reverse_candidate = QueryProfile(&source, 0);
  EXPECT_FALSE(reverse_backing.construction.query_access(
      &reverse_backing, &reverse_candidate, &candidate));

  consumer.topology.memory_peers.count = 0;
  consumer.topology.gpu_id = source.topology.gpu_id;
  EXPECT_TRUE(backing.construction.query_access(&backing, &original_candidate,
                                                &candidate));
}

static amdf_gpu_umd_device_t MakeDiscreteGfx942Device() {
  amdf_gpu_umd_device_t device = {
      .native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS, .page_size = 4096};
  device.topology.gpu_id = 41;
  device.topology.properties.gfx_ip = {9, 4, 2};
  device.topology.properties.compute.wavefront_size = 64;
  device.topology.properties.compute.compute_unit_count = 304;
  device.topology.properties.compute.maximum_wave_count_per_compute_unit = 32;
  device.topology.properties.compute
      .maximum_scratch_wave_count_per_compute_unit = 32;
  device.topology.properties.compute.local_data_share_byte_length = 65536;
  device.topology.properties.topology.xcc_count = 8;
  device.topology.properties.topology.shader_engine_count_per_xcc = 4;
  device.topology.compute_queue_count = 8;
  device.topology.sdma.engine_count = 2;
  device.topology.sdma.queue_count_per_engine = 8;
  device.topology.sdma.ip = {4, 4, 2, true};
  device.topology.memory_features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  device.topology.virtual_address.begin = UINT64_C(0x10000);
  device.topology.virtual_address.end = UINT64_C(1) << 48;
  return device;
}

static amdf_gpu_umd_device_t MakeGfx1151Device() {
  amdf_gpu_umd_device_t device = {
      .native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS, .page_size = 4096};
  device.topology.gpu_id = 73;
  device.topology.properties.gfx_ip = {11, 5, 1};
  device.topology.properties.compute.wavefront_size = 32;
  device.topology.properties.compute.compute_unit_count = 2;
  device.topology.properties.compute.maximum_wave_count_per_compute_unit = 32;
  device.topology.properties.compute
      .maximum_scratch_wave_count_per_compute_unit = 32;
  device.topology.properties.compute.local_data_share_byte_length = 65536;
  device.topology.properties.topology.xcc_count = 1;
  device.topology.properties.topology.shader_engine_count_per_xcc = 1;
  device.topology.compute_queue_count = 8;
  device.topology.context_save_restore_byte_length = 4096;
  device.topology.control_stack_byte_length = 4096;
  device.topology.sdma.engine_count = 2;
  device.topology.sdma.queue_count_per_engine = 8;
  device.topology.sdma.ip = {6, 1, 1, true};
  device.topology.virtual_address.begin = UINT64_C(0x10000);
  device.topology.virtual_address.end = UINT64_C(1) << 48;
  return device;
}

static void InitializeQueueFamilies(amdf_gpu_umd_device_t* device,
                                    amdf_gpu_endpoint_profile_t* out_profile) {
  amdf_gpu_kfd_user_queue_plans_t plans;
  amdf_gpu_kfd_target_user_queue_plans_initialize(
      &device->topology, device->page_size, 64, &plans);
  auto& properties = device->topology.properties;
  properties.queue_family_count = plans.count;
  for (uint32_t i = 0; i < plans.count; ++i) {
    properties.queue_families[i] = plans.values[i].family;
  }
  ASSERT_TRUE(amdf_gpu_endpoint_profile_initialize(&properties, out_profile));
}

static constexpr amdf_queue_family_info_t kTransferFamily = {
    .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
    .format_version = AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1,
    .roles = AMDF_QUEUE_ROLE_TRANSFER,
};

static constexpr amdf_queue_family_info_t kComputeFamily = {
    .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
    .format_version = AMDF_GPU_AQL_QUEUE_FORMAT_VERSION_1,
    .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_CACHE_CONTROL,
    .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                        AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
    .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
};

static void ExpectSiteUnsupported(const amdf_memory_native_profile_t& profile,
                                  const amdf_memory_site_query_t& query) {
  ASSERT_NE(profile.visibility.describe_site, nullptr);
  amdf_memory_site_description_t description;
  std::memset(&description, 0xA5, sizeof(description));
  unsigned char original[sizeof(description)];
  std::memcpy(original, &description, sizeof(description));
  EXPECT_EQ(
      amdf_status_code(profile.visibility.describe_site(&query, &description)),
      AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(&description, original, sizeof(description)), 0);
}

static void ExpectSiteUnsupported(const amdf_memory_native_profile_t& profile,
                                  const amdf_queue_family_info_t& family) {
  const amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = profile.guaranteed_flags,
      .queue_family_info = &family,
  };
  ExpectSiteUnsupported(profile, query);
}

static void ExpectGlobalQueueTransitions(
    const amdf_memory_site_description_t& description) {
  EXPECT_EQ(description.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  EXPECT_EQ(description.release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  EXPECT_EQ(description.release.operation,
            AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
  EXPECT_EQ(description.acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  EXPECT_EQ(description.acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  EXPECT_EQ(description.acquire.operation,
            AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);
}

static void ExpectGlobalQueueTransitions(
    const amdf_memory_native_profile_t& profile,
    const amdf_memory_site_query_t& query) {
  amdf_memory_site_description_t description = {};
  ASSERT_EQ(profile.visibility.describe_site(&query, &description),
            AMDF_STATUS_OK);
  ExpectGlobalQueueTransitions(description);
}

static void ExpectGlobalQueueTransitions(
    const amdf_memory_native_profile_t& profile,
    const amdf_queue_family_info_t& family) {
  const amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = profile.guaranteed_flags,
      .queue_family_info = &family,
  };
  ExpectGlobalQueueTransitions(profile, query);
}

static void ExpectNoCacheTransitions(
    const amdf_memory_site_description_t& description) {
  for (const auto& transition : {description.release, description.acquire}) {
    EXPECT_EQ(transition.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    EXPECT_EQ(transition.executor, AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
    EXPECT_EQ(transition.operation, AMDF_CACHE_OPERATION_NONE);
    EXPECT_EQ(transition.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
    EXPECT_EQ(transition.host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
    EXPECT_EQ(transition.host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
    EXPECT_EQ(transition.host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
    EXPECT_EQ(transition.range_granularity, 0u);
  }
  EXPECT_EQ(description.release_fixed_cost_nanoseconds, 0u);
  EXPECT_EQ(description.acquire_fixed_cost_nanoseconds, 0u);
}

TEST(LinuxGpuMemoryProfileTest, DeviceWrapperPreservesSelectedSitePolicy) {
  auto device = MakeDiscreteGfx942Device();
  for (uint32_t ordinal : {0u, 1u}) {
    SCOPED_TRACE(ordinal);
    amdf_memory_native_profile_t direct = {};
    ASSERT_EQ(amdf_gpu_kfd_query_memory_profile(
                  &device.topology, device.page_size, device.native_lifetime,
                  ordinal, &direct),
              AMDF_STATUS_OK);
    const auto wrapped = QueryProfile(&device, ordinal);
    ASSERT_NE(direct.visibility.describe_site, nullptr);
    EXPECT_EQ(wrapped.visibility.describe_site,
              direct.visibility.describe_site);
    EXPECT_NE(wrapped.visibility.describe_host, nullptr);
  }
}

TEST(LinuxGpuMemoryProfileTest, RequiresExactQueueRolesAndCacheOperations) {
  auto device = MakeDiscreteGfx942Device();
  for (uint32_t ordinal : {0u, 1u}) {
    SCOPED_TRACE(ordinal);
    const auto profile = QueryProfile(&device, ordinal);
    for (const auto& supported : {kTransferFamily, kComputeFamily}) {
      SCOPED_TRACE(supported.command_type);
      auto family = supported;
      ++family.format_version;
      ExpectSiteUnsupported(profile, family);
      family = supported;
      family.roles = 0;
      ExpectSiteUnsupported(profile, family);
    }
    auto compute_without_cache = kComputeFamily;
    compute_without_cache.roles = AMDF_QUEUE_ROLE_COMPUTE;
    ExpectSiteUnsupported(profile, compute_without_cache);
    for (amdf_cache_operations_t operations :
         {AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM,
          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM}) {
      SCOPED_TRACE(operations);
      auto family = kComputeFamily;
      family.cache_operations = operations;
      ExpectSiteUnsupported(profile, family);
    }
    auto family = kComputeFamily;
    family.cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_RANGE;
    ExpectSiteUnsupported(profile, family);
    family = kTransferFamily;
    family.command_type = AMDF_QUEUE_COMMAND_TYPE_XDNA;
    ExpectSiteUnsupported(profile, family);
  }
}

TEST(LinuxGpuMemoryProfileTest, RegisteredMemoryKeepsGenericSitePolicy) {
  auto device = MakeDiscreteGfx942Device();
  const auto registered = QueryProfile(&device, 2);
  EXPECT_NE(registered.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER, 0u);
  EXPECT_NE(registered.guaranteed_flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
  ExpectSiteUnsupported(registered, kTransferFamily);
  ExpectGlobalQueueTransitions(registered, kComputeFamily);
}

TEST(LinuxGpuMemoryProfileTest,
     SdmaLocalSiteRequiresNonHostVisibleDeviceLocalBacking) {
  auto device = MakeDiscreteGfx942Device();
  device.topology.memory_features |=
      AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY;
  const auto local = QueryProfile(&device, 1);
  ASSERT_NE(local.visibility.describe_site, nullptr);
  EXPECT_NE(local.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
  EXPECT_EQ(local.guaranteed_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
  for (const auto& family : {kTransferFamily, kComputeFamily}) {
    SCOPED_TRACE(family.command_type);
    amdf_memory_site_query_t query = {
        .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .flags = local.guaranteed_flags,
        .queue_family_info = &family,
    };
    amdf_memory_site_description_t description = {};
    ASSERT_EQ(local.visibility.describe_site(&query, &description),
              AMDF_STATUS_OK);

    // Host visibility excludes the SDMA no-cache policy even when no host
    // view is live. AQL retains its global payload fence actions.
    for (amdf_memory_flags_t flags :
         {local.guaranteed_flags | AMDF_MEMORY_FLAG_HOST_VISIBLE,
          local.guaranteed_flags & ~AMDF_MEMORY_FLAG_DEVICE_LOCAL}) {
      query.flags = flags;
      if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) {
        ExpectSiteUnsupported(local, query);
      } else {
        ExpectGlobalQueueTransitions(local, query);
      }
    }
  }
}

TEST(LinuxGpuMemoryProfileTest, SameGpuLocalGroupKeepsConsumerFacts) {
  auto source = MakeDiscreteGfx942Device();
  auto consumer = MakeDiscreteGfx942Device();
  consumer.topology.virtual_address.begin = UINT64_C(0x20000);
  consumer.topology.virtual_address.end = UINT64_C(1) << 47;
  const auto backing = QueryProfile(&source, 1);
  auto candidate = QueryProfile(&consumer, 0);
  const auto original_candidate = candidate;
  ASSERT_NE(backing.visibility.describe_site,
            candidate.visibility.describe_site);
  ASSERT_TRUE(
      backing.construction.query_access(&backing, &candidate, &candidate));
  EXPECT_EQ(candidate.memory_class, AMDF_MEMORY_CLASS_LOCAL);
  EXPECT_EQ(candidate.ordinal, original_candidate.ordinal);
  EXPECT_EQ(candidate.guaranteed_flags, backing.guaranteed_flags);
  EXPECT_EQ(candidate.supported_flags, backing.supported_flags);
  EXPECT_EQ(candidate.device_address.minimum_address,
            original_candidate.device_address.minimum_address);
  EXPECT_EQ(candidate.device_address.maximum_address,
            original_candidate.device_address.maximum_address);
  EXPECT_EQ(candidate.allocation.maximum_byte_length,
            original_candidate.allocation.maximum_byte_length);
  EXPECT_EQ(candidate.construction.query_access,
            original_candidate.construction.query_access);
  EXPECT_EQ(candidate.construction.data, &consumer.topology);
  EXPECT_EQ(candidate.visibility.describe_site,
            backing.visibility.describe_site);
  EXPECT_EQ(candidate.visibility.describe_host,
            original_candidate.visibility.describe_host);
  EXPECT_EQ(candidate.visibility.data, &consumer);
  for (const auto& family : {kTransferFamily, kComputeFamily}) {
    const amdf_memory_site_query_t query = {
        .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .flags = candidate.guaranteed_flags,
        .queue_family_info = &family,
    };
    amdf_memory_site_description_t description = {};
    ASSERT_EQ(candidate.visibility.describe_site(&query, &description),
              AMDF_STATUS_OK);
  }
}

TEST(LinuxGpuMemoryProfileTest,
     RemoteLocalGroupKeepsShaderFencesWithoutSdmaGuarantees) {
  for (bool same_hive : {false, true}) {
    SCOPED_TRACE(same_hive ? "hive" : "directed_peer");
    auto source = MakeDiscreteGfx942Device();
    auto consumer = MakeDiscreteGfx942Device();
    consumer.topology.gpu_id = 73;
    uint32_t backing_gpu_id = source.topology.gpu_id;
    if (same_hive) {
      source.topology.memory_peers.hive_id = 1;
      consumer.topology.memory_peers.hive_id = 1;
      source.topology.memory_peers.hive_sharing_enabled = true;
      consumer.topology.memory_peers.hive_sharing_enabled = true;
    } else {
      consumer.topology.memory_peers.count = 1;
      consumer.topology.memory_peers.gpu_ids = &backing_gpu_id;
    }
    const auto backing = QueryProfile(&source, 1);
    auto candidate = QueryProfile(&consumer, 0);
    const auto consumer_policy = candidate.visibility.describe_site;
    ASSERT_NE(consumer_policy, amdf_gpu_umd_memory_describe_site);
    ASSERT_TRUE(
        backing.construction.query_access(&backing, &candidate, &candidate));
    EXPECT_EQ(candidate.memory_class, AMDF_MEMORY_CLASS_LOCAL);
    EXPECT_EQ(candidate.visibility.describe_site, consumer_policy);
    EXPECT_NE(candidate.guaranteed_flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
    EXPECT_EQ(candidate.guaranteed_flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
    ExpectSiteUnsupported(candidate, kTransferFamily);
    ExpectGlobalQueueTransitions(candidate, kComputeFamily);
  }
}

TEST(LinuxGpuMemoryProfileTest, SystemGroupUsesEachConsumersSelectedPolicy) {
  auto qualified = MakeDiscreteGfx942Device();
  auto unqualified = MakeDiscreteGfx942Device();
  unqualified.topology.gpu_id = 73;
  unqualified.topology.sdma.ip.exact = false;
  const auto qualified_profile = QueryProfile(&qualified, 0);
  const auto unqualified_profile = QueryProfile(&unqualified, 0);
  auto projected = qualified_profile;
  ASSERT_TRUE(qualified_profile.construction.query_access(
      &qualified_profile, &unqualified_profile, &projected));
  ExpectSiteUnsupported(projected, kTransferFamily);
  ExpectGlobalQueueTransitions(projected, kComputeFamily);

  ASSERT_TRUE(unqualified_profile.construction.query_access(
      &unqualified_profile, &qualified_profile, &projected));
  ASSERT_NE(projected.visibility.describe_site, nullptr);
  const amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = projected.guaranteed_flags,
      .queue_family_info = &kTransferFamily,
  };
  amdf_memory_site_description_t description = {};
  ASSERT_EQ(projected.visibility.describe_site(&query, &description),
            AMDF_STATUS_OK);
  EXPECT_EQ(description.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  EXPECT_EQ(description.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
}

TEST(LinuxGpuMemoryProfileTest, SdmaSitesPreserveBackingSpecificCachePolicy) {
  auto gfx1100 = MakeGfx1151Device();
  gfx1100.topology.properties.gfx_ip = {11, 0, 0};
  gfx1100.topology.sdma.ip = {6, 0, 0, true};
  gfx1100.topology.memory_features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  for (auto device : {gfx1100, MakeGfx1151Device()}) {
    const bool system_sdma = device.topology.properties.gfx_ip.minor == 5;
    SCOPED_TRACE(device.topology.properties.gfx_ip.minor);
    const uint32_t profile_count = system_sdma ? 1 : 2;
    amdf_gpu_endpoint_profile_t endpoint = {};
    ASSERT_NO_FATAL_FAILURE(InitializeQueueFamilies(&device, &endpoint));
    ASSERT_EQ(endpoint.queue_family_count, 3u);
    ASSERT_EQ(endpoint.queue_families[0].command_type,
              AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
    ASSERT_EQ(endpoint.queue_families[2].command_type,
              AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
    for (uint32_t ordinal = 0; ordinal < profile_count; ++ordinal) {
      SCOPED_TRACE(ordinal);
      const auto profile = QueryProfile(&device, ordinal);
      ASSERT_NE(profile.visibility.describe_site, nullptr);
      amdf_memory_site_query_t query = {
          .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
          .flags = profile.guaranteed_flags,
          .queue_family_info = &endpoint.queue_families[0],
      };
      amdf_memory_site_description_t description = {};
      ASSERT_EQ(profile.visibility.describe_site(&query, &description),
                AMDF_STATUS_OK);
      ExpectGlobalQueueTransitions(description);
      if (system_sdma) {
        query.queue_family_info = &endpoint.queue_families[2];
        ASSERT_EQ(profile.visibility.describe_site(&query, &description),
                  AMDF_STATUS_OK);
        ExpectNoCacheTransitions(description);
      } else {
        ExpectGlobalQueueTransitions(profile, endpoint.queue_families[2]);
      }
    }
  }
}

TEST(LinuxGpuMemoryProfileTest, Gfx1151SystemPreservesPermissionsAndAtomicGap) {
  auto device = MakeGfx1151Device();
  amdf_gpu_endpoint_profile_t endpoint = {};
  ASSERT_NO_FATAL_FAILURE(InitializeQueueFamilies(&device, &endpoint));
  ASSERT_EQ(endpoint.queue_family_count, 3u);
  const auto& family = endpoint.queue_families[2];
  ASSERT_EQ(family.command_type, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  EXPECT_EQ(family.roles,
            AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL);
  EXPECT_EQ(family.cache_operations,
            AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM);
  EXPECT_EQ(family.cache_transition_kinds, AMDF_CACHE_TRANSITION_KINDS_GLOBAL);
  for (amdf_native_lifetime_t lifetime :
       {AMDF_NATIVE_LIFETIME_PROCESS, AMDF_NATIVE_LIFETIME_INSTANCE}) {
    SCOPED_TRACE(lifetime);
    device.native_lifetime = lifetime;
    const auto profile = QueryProfile(&device, 0);
    ASSERT_NE(profile.visibility.describe_site, nullptr);
    EXPECT_EQ(profile.atomic_operations_32, 0u);
    EXPECT_EQ(profile.atomic_operations_64, 0u);
    constexpr amdf_memory_access_t accesses[] = {
        AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_WRITE,
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE};
    for (amdf_memory_access_t access : accesses) {
      SCOPED_TRACE(access);
      const amdf_memory_site_query_t query = {
          .access = access,
          .flags = profile.guaranteed_flags,
          .queue_family_info = &family,
      };
      amdf_memory_site_description_t description;
      std::memset(&description, 0xA5, sizeof(description));
      ASSERT_EQ(profile.visibility.describe_site(&query, &description),
                AMDF_STATUS_OK);
      amdf_memory_site_capabilities_t expected =
          AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN |
          AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN;
      if ((access & AMDF_MEMORY_ACCESS_READ) != 0) {
        expected |= AMDF_MEMORY_SITE_CAPABILITY_READ;
      }
      if ((access & AMDF_MEMORY_ACCESS_WRITE) != 0) {
        expected |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
      }
      EXPECT_EQ(description.capabilities, expected);
      ExpectNoCacheTransitions(description);
      EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
      EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
      EXPECT_FALSE(amdf_memory_compatibility_domain_is_valid(
          &description.atomic_domain));
      EXPECT_FALSE(amdf_memory_compatibility_domain_is_valid(
          &description.mapping_domain));
    }
  }
}

TEST(LinuxGpuMemoryProfileTest, Gfx1151SystemRequiresExactNativeIdentity) {
  struct Case {
    // Qualification premise removed from the otherwise selected device.
    const char* name;
    // Changes native metadata without performing a device operation.
    void (*mutate)(amdf_gpu_umd_device_t* device);
  };
  const Case cases[] = {
      {"compute_major",
       [](amdf_gpu_umd_device_t* device) {
         device->topology.properties.gfx_ip.major = 12;
       }},
      {"compute_minor",
       [](amdf_gpu_umd_device_t* device) {
         device->topology.properties.gfx_ip.minor = 0;
       }},
      {"earlier_compute",
       [](amdf_gpu_umd_device_t* device) {
         device->topology.properties.gfx_ip.stepping = 0;
       }},
      {"later_compute",
       [](amdf_gpu_umd_device_t* device) {
         device->topology.properties.gfx_ip.stepping = 2;
       }},
      {"inexact_sdma",
       [](amdf_gpu_umd_device_t* device) {
         device->topology.sdma.ip.exact = false;
       }},
      {"sdma_major",
       [](amdf_gpu_umd_device_t* device) {
         device->topology.sdma.ip.major = 7;
       }},
      {"sdma_minor",
       [](amdf_gpu_umd_device_t* device) {
         device->topology.sdma.ip.minor = 0;
       }},
      {"earlier_sdma",
       [](amdf_gpu_umd_device_t* device) {
         device->topology.sdma.ip.revision = 0;
       }},
      {"later_sdma",
       [](amdf_gpu_umd_device_t* device) {
         device->topology.sdma.ip.revision = 2;
       }},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    auto device = MakeGfx1151Device();
    test_case.mutate(&device);
    ExpectSiteUnsupported(QueryProfile(&device, 0), kTransferFamily);
  }
}

TEST(LinuxGpuMemoryProfileTest,
     Gfx1151SystemRequiresCoherenceAndTransferFamily) {
  auto device = MakeGfx1151Device();
  const auto profile = QueryProfile(&device, 0);
  auto family = kTransferFamily;
  amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = profile.guaranteed_flags & ~AMDF_MEMORY_FLAG_HOST_COHERENT,
      .queue_family_info = &family,
  };
  ExpectSiteUnsupported(profile, query);
  query.flags = profile.guaranteed_flags;
  ++family.format_version;
  ExpectSiteUnsupported(profile, query);
  family = kTransferFamily;
  family.roles = 0;
  ExpectSiteUnsupported(profile, query);
  ExpectGlobalQueueTransitions(profile, kComputeFamily);

  const auto registered = QueryProfile(&device, 1);
  EXPECT_NE(registered.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER, 0u);
  ExpectSiteUnsupported(registered, kTransferFamily);
}

TEST(LinuxGpuMemoryProfileTest, Gfx1151GroupUsesBackingAndConsumerFacts) {
  auto consumer = MakeGfx1151Device();
  const auto system = QueryProfile(&consumer, 0);
  auto source = MakeDiscreteGfx942Device();
  const auto source_system = QueryProfile(&source, 0);
  auto projected = source_system;
  ASSERT_TRUE(source_system.construction.query_access(&source_system, &system,
                                                      &projected));
  EXPECT_EQ(projected.visibility.describe_site,
            system.visibility.describe_site);
  const amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = projected.guaranteed_flags,
      .queue_family_info = &kTransferFamily,
  };
  amdf_memory_site_description_t description = {};
  ASSERT_EQ(projected.visibility.describe_site(&query, &description),
            AMDF_STATUS_OK);
  ExpectNoCacheTransitions(description);

  uint32_t source_gpu_id = source.topology.gpu_id;
  consumer.topology.memory_peers.count = 1;
  consumer.topology.memory_peers.gpu_ids = &source_gpu_id;
  const auto local = QueryProfile(&source, 1);
  ASSERT_TRUE(local.construction.query_access(&local, &system, &projected));
  EXPECT_EQ(projected.visibility.describe_site,
            system.visibility.describe_site);
  EXPECT_EQ(projected.guaranteed_flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
  ExpectSiteUnsupported(projected, kTransferFamily);

  // LOCAL availability is not the SYSTEM policy's identity selector. Even
  // when a topology exposes it, the LOCAL profile cannot inherit that policy.
  consumer.topology.memory_features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  const auto owned_local = QueryProfile(&consumer, 1);
  ExpectSiteUnsupported(owned_local, kTransferFamily);
  const auto owned_system = QueryProfile(&consumer, 0);
  EXPECT_EQ(owned_system.visibility.describe_site,
            system.visibility.describe_site);
}

TEST(LinuxGpuMemoryProfileTest, SystemStoresPreserveIndependentSdmaPolicy) {
  auto device = MakeGfx1151Device();
  device.topology.gc_ip = {11, 5, 1, true};
  device.topology.host_atomics = {true, true};
  amdf_gpu_endpoint_profile_t endpoint = {};
  ASSERT_NO_FATAL_FAILURE(InitializeQueueFamilies(&device, &endpoint));
  ASSERT_EQ(endpoint.queue_family_count, 3u);
  const auto profile = QueryProfile(&device, 0);
  EXPECT_EQ(profile.atomic_operations_32, AMDF_ATOMIC_OPERATION_STORE);
  EXPECT_EQ(profile.atomic_operations_64, AMDF_ATOMIC_OPERATION_STORE);
  amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = profile.guaranteed_flags,
      .atomic_operations_32 = profile.atomic_operations_32,
      .atomic_operations_64 = profile.atomic_operations_64,
      .queue_family_info = &endpoint.queue_families[0],
  };
  amdf_memory_site_description_t description = {};
  ASSERT_EQ(profile.visibility.describe_site(&query, &description),
            AMDF_STATUS_OK);
  ExpectGlobalQueueTransitions(description);
  EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_SYSTEM);
  EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_SYSTEM);
  EXPECT_TRUE(
      amdf_memory_compatibility_domain_is_valid(&description.atomic_domain));
  const auto domain = description.atomic_domain;

  auto family = endpoint.queue_families[0];
  family.atomic_capabilities.operations_64 = 0;
  query.queue_family_info = &family;
  ASSERT_EQ(profile.visibility.describe_site(&query, &description),
            AMDF_STATUS_OK);
  EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_SYSTEM);
  EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
  EXPECT_TRUE(amdf_memory_compatibility_domain_is_equal(
      &description.atomic_domain, &domain));

  for (amdf_memory_access_t access :
       {AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_WRITE}) {
    query.access = access;
    ASSERT_EQ(profile.visibility.describe_site(&query, &description),
              AMDF_STATUS_OK);
    EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
    EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
    EXPECT_FALSE(
        amdf_memory_compatibility_domain_is_valid(&description.atomic_domain));
  }
  query.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  query.queue_family_info = &endpoint.queue_families[2];
  ASSERT_EQ(profile.visibility.describe_site(&query, &description),
            AMDF_STATUS_OK);
  ExpectNoCacheTransitions(description);
  EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
  EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
}

TEST(LinuxGpuMemoryProfileTest, SystemStoresDoNotRequireSdma) {
  auto device = MakeGfx1151Device();
  device.topology.gc_ip = {11, 5, 1, true};
  device.topology.host_atomics = {true, true};
  device.topology.sdma = {};
  amdf_gpu_endpoint_profile_t endpoint = {};
  ASSERT_NO_FATAL_FAILURE(InitializeQueueFamilies(&device, &endpoint));
  ASSERT_EQ(endpoint.queue_family_count, 2u);
  const auto profile = QueryProfile(&device, 0);
  EXPECT_EQ(profile.atomic_operations_32, AMDF_ATOMIC_OPERATION_STORE);
  EXPECT_EQ(profile.atomic_operations_64, AMDF_ATOMIC_OPERATION_STORE);
  const amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = profile.guaranteed_flags,
      .atomic_operations_32 = profile.atomic_operations_32,
      .atomic_operations_64 = profile.atomic_operations_64,
      .queue_family_info = &endpoint.queue_families[0],
  };
  amdf_memory_site_description_t description = {};
  ASSERT_EQ(profile.visibility.describe_site(&query, &description),
            AMDF_STATUS_OK);
  EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_SYSTEM);
  EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_SYSTEM);
  ExpectSiteUnsupported(profile, kTransferFamily);
}

TEST(LinuxGpuMemoryProfileTest, SystemGroupStoresRemainConsumerSpecific) {
  auto qualified = MakeGfx1151Device();
  qualified.topology.gc_ip = {11, 5, 1, true};
  qualified.topology.host_atomics = {true, true};
  auto unqualified = MakeGfx1151Device();
  unqualified.topology.gpu_id = qualified.topology.gpu_id + 1;
  const auto qualified_profile = QueryProfile(&qualified, 0);
  const auto unqualified_profile = QueryProfile(&unqualified, 0);
  amdf_memory_native_profile_t projected = {};
  ASSERT_TRUE(qualified_profile.construction.query_access(
      &qualified_profile, &unqualified_profile, &projected));
  EXPECT_EQ(projected.atomic_operations_32, 0u);
  EXPECT_EQ(projected.atomic_operations_64, 0u);
  ASSERT_TRUE(unqualified_profile.construction.query_access(
      &unqualified_profile, &qualified_profile, &projected));
  EXPECT_EQ(projected.atomic_operations_32, AMDF_ATOMIC_OPERATION_STORE);
  EXPECT_EQ(projected.atomic_operations_64, AMDF_ATOMIC_OPERATION_STORE);

  unqualified.topology.memory_features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  const auto local = QueryProfile(&unqualified, 1);
  uint32_t backing_gpu_id = unqualified.topology.gpu_id;
  qualified.topology.memory_peers.count = 1;
  qualified.topology.memory_peers.gpu_ids = &backing_gpu_id;
  ASSERT_TRUE(
      local.construction.query_access(&local, &qualified_profile, &projected));
  EXPECT_EQ(projected.atomic_operations_32, 0u);
  EXPECT_EQ(projected.atomic_operations_64, 0u);
  amdf_gpu_endpoint_profile_t endpoint = {};
  ASSERT_NO_FATAL_FAILURE(InitializeQueueFamilies(&qualified, &endpoint));
  const amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = projected.guaranteed_flags,
      .atomic_operations_32 = projected.atomic_operations_32,
      .atomic_operations_64 = projected.atomic_operations_64,
      .queue_family_info = &endpoint.queue_families[0],
  };
  amdf_memory_site_description_t description = {};
  ASSERT_EQ(projected.visibility.describe_site(&query, &description),
            AMDF_STATUS_OK);
  EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
  EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
}

static amdf_gpu_umd_device_t MakeLegacySdmaMemoryDevice() {
  amdf_gpu_umd_device_t device = {
      .native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS, .page_size = 4096};
  device.topology.gpu_id = 41;
  device.topology.sdma.ip = {4, 4, 2, true};
  device.topology.memory_features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  device.topology.virtual_address.begin = UINT64_C(0x10000);
  device.topology.virtual_address.end = UINT64_C(1) << 48;
  return device;
}

static constexpr amdf_queue_family_info_t kLegacySdmaFamily = {
    .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
    .format_version = AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1,
    .roles = AMDF_QUEUE_ROLE_TRANSFER,
};

TEST(LinuxGpuMemoryProfileTest,
     LegacySdmaVisibilityUsesNativeEngineAndBacking) {
  const amdf_memory_access_t accesses[] = {
      AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_WRITE,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE};
  // Compiler ISA is intentionally absent: it does not select SDMA cache policy.
  for (uint32_t revision : {2u, 4u, 5u}) {
    SCOPED_TRACE(revision);
    auto device = MakeLegacySdmaMemoryDevice();
    device.topology.sdma.ip.revision = revision;
    for (amdf_gpu_device_features_t features :
         {amdf_gpu_device_features_t{0},
          amdf_gpu_device_features_t{AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY}}) {
      SCOPED_TRACE(features);
      device.topology.memory_features = features;
      const uint32_t count = features == 0 ? 1 : 2;
      for (uint32_t ordinal = 0; ordinal < count; ++ordinal) {
        SCOPED_TRACE(ordinal);
        const auto profile = QueryProfile(&device, ordinal);
        ASSERT_NE(profile.visibility.describe_site, nullptr);
        ASSERT_NE(profile.visibility.describe_host, nullptr);
        for (const auto& family : {kLegacySdmaFamily, kComputeFamily}) {
          SCOPED_TRACE(family.command_type);
          for (amdf_memory_access_t access : accesses) {
            SCOPED_TRACE(access);
            const amdf_memory_site_query_t query = {
                .access = access,
                .flags = profile.guaranteed_flags,
                .queue_family_info = &family,
            };
            amdf_memory_site_description_t description;
            std::memset(&description, 0xA5, sizeof(description));
            ASSERT_EQ(profile.visibility.describe_site(&query, &description),
                      AMDF_STATUS_OK);
            const amdf_memory_site_capabilities_t permissions =
                ((access & AMDF_MEMORY_ACCESS_READ) != 0
                     ? AMDF_MEMORY_SITE_CAPABILITY_READ
                     : 0) |
                ((access & AMDF_MEMORY_ACCESS_WRITE) != 0
                     ? AMDF_MEMORY_SITE_CAPABILITY_WRITE
                     : 0);
            if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) {
              EXPECT_EQ(description.capabilities,
                        permissions |
                            AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN |
                            AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN);
              ExpectNoCacheTransitions(description);
            } else {
              EXPECT_EQ(description.capabilities, permissions);
              ExpectGlobalQueueTransitions(description);
            }
            EXPECT_EQ(description.atomic_reach.scope_32,
                      AMDF_ATOMIC_SCOPE_NONE);
            EXPECT_EQ(description.atomic_reach.scope_64,
                      AMDF_ATOMIC_SCOPE_NONE);
            EXPECT_FALSE(amdf_memory_compatibility_domain_is_valid(
                &description.atomic_domain));
          }
        }
      }
    }
  }
}

TEST(LinuxGpuMemoryProfileTest, LegacySdmaPolicyExcludesOtherNativeEngines) {
  const amdf_gpu_kfd_ip_version_t versions[] = {
      {4, 4, 2, false}, {4, 4, 0, true}, {4, 4, 3, true}, {4, 4, 6, true},
      {4, 3, 2, true},  {5, 4, 2, true}, {6, 1, 1, true}, {7, 1, 0, true},
  };
  for (const auto& ip : versions) {
    SCOPED_TRACE(::testing::Message() << ip.major << "." << ip.minor << "."
                                      << ip.revision << " exact=" << ip.exact);
    auto device = MakeLegacySdmaMemoryDevice();
    device.topology.sdma.ip = ip;
    for (uint32_t ordinal : {0u, 1u}) {
      const auto profile = QueryProfile(&device, ordinal);
      amdf_queue_family_info_t family = {
          .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
          .format_version = AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1,
          .roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL,
          .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                              AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
          .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      };
      const amdf_memory_site_query_t query = {
          .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
          .flags = profile.guaranteed_flags,
          .queue_family_info = &family,
      };
      amdf_memory_site_description_t description = {};
      EXPECT_EQ(amdf_status_code(
                    profile.visibility.describe_site(&query, &description)),
                AMDF_STATUS_CODE_UNSUPPORTED);
      family.format_features = AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
      ASSERT_EQ(profile.visibility.describe_site(&query, &description),
                AMDF_STATUS_OK);
      EXPECT_EQ(description.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
      EXPECT_EQ(description.acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    }
  }
}

TEST(LinuxGpuMemoryProfileTest,
     LegacySdmaGroupPolicyPreservesBackingOwnership) {
  auto source = MakeLegacySdmaMemoryDevice();
  auto consumer = source;
  const auto backing = QueryProfile(&source, 1);
  const auto system = QueryProfile(&consumer, 0);
  auto projected = system;
  ASSERT_TRUE(backing.construction.query_access(&backing, &system, &projected));
  EXPECT_EQ(projected.visibility.describe_site,
            backing.visibility.describe_site);
  EXPECT_EQ(projected.visibility.data, &consumer);
  EXPECT_EQ(projected.construction.data, &consumer.topology);

  consumer.topology.gpu_id = 73;
  uint32_t source_gpu_id = source.topology.gpu_id;
  consumer.topology.memory_peers.count = 1;
  consumer.topology.memory_peers.gpu_ids = &source_gpu_id;
  ASSERT_TRUE(backing.construction.query_access(&backing, &system, &projected));
  EXPECT_EQ(projected.visibility.describe_site,
            system.visibility.describe_site);
  const amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = projected.guaranteed_flags,
      .queue_family_info = &kLegacySdmaFamily,
  };
  amdf_memory_site_description_t description;
  std::memset(&description, 0xA5, sizeof(description));
  const auto original = description;
  EXPECT_EQ(amdf_status_code(
                projected.visibility.describe_site(&query, &description)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(&description, &original, sizeof(description)), 0);
}

TEST(LinuxGpuMemoryProfileTest, LegacySdmaExcludesHostApertureAndRegistration) {
  auto device = MakeLegacySdmaMemoryDevice();
  device.topology.memory_features |=
      AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY;
  const auto local = QueryProfile(&device, 1);
  for (amdf_memory_flags_t flags :
       {local.guaranteed_flags | AMDF_MEMORY_FLAG_HOST_VISIBLE,
        local.guaranteed_flags | AMDF_MEMORY_FLAG_HOST_VISIBLE |
            AMDF_MEMORY_FLAG_HOST_COHERENT,
        local.guaranteed_flags & ~AMDF_MEMORY_FLAG_DEVICE_LOCAL}) {
    const amdf_memory_site_query_t query = {
        .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .flags = flags,
        .queue_family_info = &kLegacySdmaFamily,
    };
    amdf_memory_site_description_t description;
    std::memset(&description, 0xA5, sizeof(description));
    const auto original = description;
    EXPECT_EQ(
        amdf_status_code(local.visibility.describe_site(&query, &description)),
        AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(std::memcmp(&description, &original, sizeof(description)), 0);
  }
  const auto registered = QueryProfile(&device, 2);
  EXPECT_EQ(registered.visibility.describe_site,
            amdf_gpu_umd_memory_describe_site);
}

TEST(LinuxGpuMemoryProfileTest, SystemStoresRequireNativeMappingAndCpuRoutes) {
  amdf_gpu_umd_device_t device = {
      .native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS, .page_size = 4096};
  device.topology.virtual_address.begin = UINT64_C(0x10000);
  device.topology.virtual_address.end = UINT64_C(1) << 48;
  device.topology.gc_ip = {.major = 11, .exact = true};
  device.topology.memory_features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
  for (uint32_t widths = 0; widths < 4; ++widths) {
    SCOPED_TRACE(widths);
    device.topology.host_atomics.supports_32 = (widths & 1) != 0;
    device.topology.host_atomics.supports_64 = (widths & 2) != 0;
    for (uint32_t major : {9u, 11u, 12u}) {
      SCOPED_TRACE(major);
      device.topology.gc_ip.major = major;
      for (uint32_t ordinal = 0; ordinal < 3; ++ordinal) {
        SCOPED_TRACE(ordinal);
        const auto profile = QueryProfile(&device, ordinal);
        const bool native_mapping = ordinal == 0 && major == 11;
        EXPECT_EQ(profile.atomic_operations_32,
                  native_mapping && (widths & 1) != 0
                      ? AMDF_ATOMIC_OPERATION_STORE
                      : 0u);
        EXPECT_EQ(profile.atomic_operations_64,
                  native_mapping && (widths & 2) != 0
                      ? AMDF_ATOMIC_OPERATION_STORE
                      : 0u);
      }
    }
  }
  device.topology.gc_ip = {
      .major = 11, .minor = 5, .revision = 0, .exact = true};
  EXPECT_EQ(QueryProfile(&device, 0).atomic_operations_64,
            AMDF_ATOMIC_OPERATION_STORE);
  device.topology.gc_ip.revision = 1;
  EXPECT_EQ(QueryProfile(&device, 0).atomic_operations_64,
            AMDF_ATOMIC_OPERATION_STORE);
  device.topology.gc_ip.exact = false;
  EXPECT_EQ(QueryProfile(&device, 0).atomic_operations_64, 0u);
}

}  // namespace
